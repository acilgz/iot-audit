import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from iot_audit.preprocessing import load_and_prepare_data
from iot_audit.preprocessing_mc import load_and_prepare_multiclass

N_ROWS = 75
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 40

BINARY_TEST_IDX = [7, 20, 36, 56, 35, 17, 14, 55, 28, 49, 13, 62, 44, 41, 40]
MULTICLASS_TEST_IDX = [43, 32, 59, 72, 58, 4, 27, 63, 48, 33, 25, 23, 44, 61, 20]

def _base_dataframe():
    idx = np.arange(N_ROWS)
    return pd.DataFrame(
        {
            "row_id": idx.astype(float),
            "duration": (idx * 1.5).astype(float),
            "proto": np.where(idx % 2 == 0, "tcp", "udp"),
            "label": idx % 2,
            "type": np.array(["normal", "scan", "dos"])[idx % 3],
        }
    )

def _assert_expected_split_indices(labels, expected_test_idx):
    train_idx, test_idx = train_test_split(
        np.arange(N_ROWS),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    assert set(test_idx) == set(expected_test_idx)
    assert set(train_idx) == set(range(N_ROWS)) - set(expected_test_idx)

def _make_threshold_pair(expected_test_idx):
    train_idx = sorted(set(range(N_ROWS)) - set(expected_test_idx))

    base = _base_dataframe()
    # train with 36 distinct values
    train_values = [f"cat-{i % 36:02d}" for i in range(len(train_idx))]
    base.loc[train_idx, "candidate_cat"] = train_values
    base.loc[expected_test_idx, "candidate_cat"] = [
        f"cat-{i % 36:02d}" for i in range(len(expected_test_idx))
    ]

    modified = base.copy()
    modified.loc[expected_test_idx, "candidate_cat"] = [
        f"heldout-only-{i:02d}" for i in range(len(expected_test_idx))
    ]

    assert base.loc[train_idx, "candidate_cat"].nunique(dropna=False) == 36
    assert modified.loc[train_idx, "candidate_cat"].nunique(dropna=False) == 36
    assert base["candidate_cat"].nunique(dropna=False) <= THRESHOLD
    assert modified["candidate_cat"].nunique(dropna=False) > THRESHOLD
    pd.testing.assert_series_equal(
        base.loc[train_idx, "candidate_cat"],
        modified.loc[train_idx, "candidate_cat"],
    )
    return base, modified

def _column_lists(preprocessor):
    transformers = {name: cols for name, _, cols in preprocessor.transformers_ if name != "remainder"}
    return list(transformers["num"]), list(transformers["cat"])

def _learned_signature(preprocessor):
    num_cols, cat_cols = _column_lists(preprocessor)
    num = preprocessor.named_transformers_["num"]
    cat = preprocessor.named_transformers_["cat"]
    scaler = num.named_steps["scaler"]

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": np.asarray(num.named_steps["imputer"].statistics_, dtype=object),
        "scaler_mean": None if scaler == "passthrough" else np.asarray(scaler.mean_),
        "scaler_scale": None if scaler == "passthrough" else np.asarray(scaler.scale_),
        "cat_imputer": np.asarray(cat.named_steps["imputer"].statistics_, dtype=object),
        "ohe_categories": [np.asarray(values, dtype=object) for values in cat.named_steps["onehot"].categories_],
    }

def _assert_signatures_equal(left, right):
    assert left["num_cols"] == right["num_cols"]
    assert left["cat_cols"] == right["cat_cols"]
    np.testing.assert_array_equal(left["num_imputer"], right["num_imputer"])
    np.testing.assert_array_equal(left["cat_imputer"], right["cat_imputer"])
    if left["scaler_mean"] is None or right["scaler_mean"] is None:
        assert left["scaler_mean"] is right["scaler_mean"]
        assert left["scaler_scale"] is right["scaler_scale"]
    else:
        np.testing.assert_allclose(left["scaler_mean"], right["scaler_mean"])
        np.testing.assert_allclose(left["scaler_scale"], right["scaler_scale"])
    assert len(left["ohe_categories"]) == len(right["ohe_categories"])
    for left_values, right_values in zip(left["ohe_categories"], right["ohe_categories"]):
        np.testing.assert_array_equal(left_values, right_values)

def _assert_roundtrip(preprocessor, inference_df, artifact_path):
    joblib.dump(preprocessor, artifact_path)
    reloaded = joblib.load(artifact_path)
    assert reloaded.numeric_features_ == preprocessor.numeric_features_
    assert reloaded.categorical_features_ == preprocessor.categorical_features_
    assert _column_lists(reloaded) == _column_lists(preprocessor)
    np.testing.assert_allclose(
        reloaded.transform(inference_df),
        preprocessor.transform(inference_df),
    )

@pytest.mark.parametrize(
    ("mode", "expected_test_idx"),
    [
        ("binary", BINARY_TEST_IDX),
        ("multiclass", MULTICLASS_TEST_IDX),
    ],
)
def test_low_cardinality_selection_depends_only_on_training_split(tmp_path, mode, expected_test_idx):
    base, modified = _make_threshold_pair(expected_test_idx)

    if mode == "binary":
        _assert_expected_split_indices(base["label"].to_numpy(), expected_test_idx)
        base_path = tmp_path / "binary_base.csv"
        modified_path = tmp_path / "binary_modified.csv"
        base.to_csv(base_path, index=False)
        modified.to_csv(modified_path, index=False)

        *_, base_features, base_preprocessor = load_and_prepare_data(
            str(base_path),
            leakage_base=str(tmp_path / "base_reports"),
            model_name="logreg",
        )
        *_, modified_features, modified_preprocessor = load_and_prepare_data(
            str(modified_path),
            leakage_base=str(tmp_path / "modified_reports"),
            model_name="logreg",
        )
        inference_df = modified.drop(columns=["label"])
    else:
        encoded = pd.factorize(base["type"], sort=True)[0]
        _assert_expected_split_indices(encoded, expected_test_idx)
        base_path = tmp_path / "multiclass_base.csv"
        modified_path = tmp_path / "multiclass_modified.csv"
        base.to_csv(base_path, index=False)
        modified.to_csv(modified_path, index=False)

        *_, base_features, base_preprocessor, _ = load_and_prepare_multiclass(
            str(base_path),
            base_outdir=str(tmp_path / "base_reports_mc"),
            model_name="logreg_mc",
        )
        *_, modified_features, modified_preprocessor, _ = load_and_prepare_multiclass(
            str(modified_path),
            base_outdir=str(tmp_path / "modified_reports_mc"),
            model_name="logreg_mc",
        )
        inference_df = modified.drop(columns=["type"])

    base_num, base_cat = _column_lists(base_preprocessor)
    modified_num, modified_cat = _column_lists(modified_preprocessor)

    assert "candidate_cat" in base_cat
    assert "candidate_cat" in modified_cat
    assert base_preprocessor.numeric_features_ == base_num
    assert base_preprocessor.categorical_features_ == base_cat
    assert modified_preprocessor.numeric_features_ == modified_num
    assert modified_preprocessor.categorical_features_ == modified_cat
    assert base_num == modified_num
    assert base_cat == modified_cat
    assert base_features == modified_features
    _assert_signatures_equal(
        _learned_signature(base_preprocessor),
        _learned_signature(modified_preprocessor),
    )

    _assert_roundtrip(
        modified_preprocessor,
        inference_df,
        tmp_path / f"{mode}_preprocessor.pkl",
    )

@pytest.mark.parametrize(
    ("mode", "model_name", "expects_scaler"),
    [
        ("binary", "logreg", True),
        ("binary", "rf", False),
        ("binary", "xgb", False),
        ("binary", "lgbm", False),
        ("multiclass", "logreg_mc", True),
        ("multiclass", "rf_mc", False),
        ("multiclass", "xgb_mc", False),
        ("multiclass", "lgbm_mc", False),
    ],
)
def test_scaling_policy_is_unchanged(tmp_path, mode, model_name, expects_scaler):
    df = _base_dataframe()
    df["candidate_cat"] = [f"cat-{i % 4}" for i in range(N_ROWS)]
    csv_path = tmp_path / f"{mode}_{model_name}.csv"
    df.to_csv(csv_path, index=False)

    if mode == "binary":
        *_, preprocessor = load_and_prepare_data(
            str(csv_path),
            leakage_base=str(tmp_path / "reports"),
            model_name=model_name,
        )
    else:
        *_, preprocessor, _ = load_and_prepare_multiclass(
            str(csv_path),
            base_outdir=str(tmp_path / "reports_mc"),
            model_name=model_name,
        )

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert isinstance(scaler, StandardScaler) is expects_scaler
