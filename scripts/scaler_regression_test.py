import argparse
import sys

import joblib
import numpy as np
from data_loading import _read_csv, load_preprocessor, sha256, validate_model
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


def validated_fit_indices(meta, n_rows):
    internal = meta.get("internal_split")
    if not isinstance(internal, dict):
        raise ValueError("ERROR: Missing internal_split")

    def indices(source, key):
        raw = source.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"ERROR: Missing or empty {key}")
        if any(type(i) is not int for i in raw):
            raise ValueError(f"ERROR: Non-integer positions in {key}")
        values = np.asarray(raw, dtype=np.int64)
        if np.any(values < 0) or np.any(values >= n_rows):
            raise ValueError(f"ERROR: Out-of-range positions in {key}")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"ERROR: Duplicate positions in {key}")
        return values

    train = indices(meta, "train_indices")
    test = indices(meta, "test_indices")
    fit = indices(internal, "fit_indices")
    validation = indices(internal, "validation_indices")
    preprocessing_fit = indices(internal, "preprocessor_fit_indices")
    if np.intersect1d(train, test).size:
        raise ValueError("ERROR: Train and test overlap")
    if not np.array_equal(np.union1d(train, test), np.arange(n_rows)):
        raise ValueError("ERROR: Train and test do not partition the dataset")
    if np.intersect1d(fit, validation).size:
        raise ValueError("ERROR: Internal fit and validation overlap")
    if not np.array_equal(np.union1d(fit, validation), np.sort(train)):
        raise ValueError("ERROR: Internal fit and validation do not partition training")
    if not np.array_equal(preprocessing_fit, fit):
        raise ValueError("ERROR: Preprocessor fit positions differ from internal fit")
    return preprocessing_fit

def numeric_fit_checks(preprocessor, df, fit_indices):
    numeric = preprocessor.named_transformers_["num"]
    scaler = numeric.named_steps["scaler"]
    if not isinstance(scaler, StandardScaler):
        return {}
    columns = next(cols for name, _, cols in preprocessor.transformers_ if name == "num")

    reference = clone(numeric).fit(df.iloc[fit_indices].loc[:, columns])
    expected_scaler = reference.named_steps["scaler"]

    def matches(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        return actual.shape == expected.shape and bool(
            np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
        )

    checks = {
        "scaler sample count matches recorded fit": bool(
            np.all(np.asarray(scaler.n_samples_seen_) == len(fit_indices))
        ),
        "imputer statistics match recorded fit": matches(
            numeric.named_steps["imputer"].statistics_,
            reference.named_steps["imputer"].statistics_,
        ),
    }
    for attribute in ("mean_", "var_", "scale_"):
        checks[f"scaler {attribute} matches recorded fit"] = matches(
            getattr(scaler, attribute), getattr(expected_scaler, attribute)
        )
    return checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--binary-dir", default="train")
    ap.add_argument("--multiclass-dir", default="train_mc")
    args = ap.parse_args()
    df = _read_csv(args.csv)
    dataset_hash = sha256(args.csv)
    failed = False

    for base, names in [(args.binary_dir, ["logreg", "rf", "xgb", "lgbm"]),
                        (args.multiclass_dir, ["logreg_mc", "rf_mc", "xgb_mc", "lgbm_mc"])]:
        for name in names:
            directory = f"{base}/models/{name}"
            validate_model(directory)
            p, meta = load_preprocessor(f"{directory}/preprocessor.pkl", f"{directory}/preprocessor_meta.json", name)
            try:
                if dataset_hash != meta.get("dataset_sha256"):
                    raise ValueError("Dataset hash does not match preprocessing metadata")
                fit_indices = validated_fit_indices(meta, len(df))
            except ValueError as exc:
                print(f"ERROR [{name}] fit provenance: {exc}")
                failed = True
                continue
            m = joblib.load(f"{base}/models/{name}/model.pkl")
            scaler = p.named_transformers_["num"].named_steps["scaler"]
            cols = [c for n, _, cs in p.transformers_ if n != "remainder" for c in cs]
            checks = {
                "dataset matches preprocessing hash": True,
                "saved fit/validation/test indices are disjoint partitions": True,
                "scaling matches map": isinstance(scaler, StandardScaler) == name.startswith("logreg"),
                **numeric_fit_checks(p, df, fit_indices),
                "no target-derived cols": not {"label", "type", "target"} & {str(c).lower() for c in cols},
                "same pipeline at inference": len(m.predict(p.transform(df.head(100)))) == min(100, len(df)),
            }
            if isinstance(scaler, StandardScaler):
                print(f"INFO [{name}] expected fit rows={len(fit_indices)}, "
                      f"scaler n_samples_seen_={scaler.n_samples_seen_}")
            for k, v in checks.items():
                print(f"{'PASS' if v else 'ERROR'} [{name}] {k}")
                failed |= not v

    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
