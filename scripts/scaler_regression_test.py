import argparse, math, sys, joblib, pandas as pd
from data_loading import load_preprocessor, validate_model
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--binary-dir", default="train")
    ap.add_argument("--multiclass-dir", default="train_mc")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    n_train = len(df) - math.ceil(0.2 * len(df))  # same split as train_test_split(test_size=0.2)
    failed = False

    for base, names in [(args.binary_dir, ["logreg", "rf", "xgb", "lgbm"]),
                        (args.multiclass_dir, ["logreg_mc", "rf_mc", "xgb_mc", "lgbm_mc"])]:
        for name in names:
            directory = f"{base}/models/{name}"
            validate_model(directory)
            p, meta = load_preprocessor(f"{directory}/preprocessor.pkl", f"{directory}/preprocessor_meta.json", name)
            m = joblib.load(f"{base}/models/{name}/model.pkl")
            scaler = p.named_transformers_["num"].named_steps["scaler"]
            cols = [c for n, _, cs in p.transformers_ if n != "remainder" for c in cs]
            checks = {
                "scaling matches map": isinstance(scaler, StandardScaler) == name.startswith("logreg"),
                "scaler fit on train only": not isinstance(scaler, StandardScaler) or scaler.n_samples_seen_ == n_train,
                "no target-derived cols": not {"label", "type"} & set(cols),
                "same pipeline at inference": len(m.predict(p.transform(df.head(100)))) == 100,
            }
            for k, v in checks.items():
                print(f"{'PASS' if v else 'ERROR'} [{name}] {k}")
                failed |= not v

    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
