import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from iot_audit.preprocessing import load_and_prepare_data
from iot_audit.preprocessing_mc import load_and_prepare_multiclass

TARGET_ALIASES = {
    "type", "Type", "TYPE",
    "label", "Label", "LABEL",
    "target", "Target", "TARGET",
}

def _make_dataset():
    n = 30

    return pd.DataFrame({
        "duration": np.arange(n, dtype=float),
        "orig_bytes": np.arange(n, dtype=float) * 10,
        "proto": ["tcp", "udp", "tcp"] * 10,

        # binary
        "label": [0, 1, 0] * 10,

        # multiclass
        "type": ["benign", "scan", "dos"] * 10,

        # aliases
        "Label": [1] * n,
        "LABEL": [0] * n,
        "Type": ["leak"] * n,
        "TYPE": ["leak"] * n,
        "target": [1] * n,
        "Target": [1] * n,
        "TARGET": [1] * n,
    })

def test_binary_excludes_target_aliases(tmp_path):
    df = _make_dataset()
    csv_path = tmp_path / "binary.csv"
    df.to_csv(csv_path, index=False)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        preprocessor,
    ) = load_and_prepare_data(
        str(csv_path),
        leakage_base=str(tmp_path / "reports"),
    )

    # ensure 'label' is used as the target
    assert set(np.unique(np.concatenate([y_train, y_test]))) == {0, 1}

    assert TARGET_ALIASES.isdisjoint(feature_names)

def test_multiclass_excludes_target_aliases(tmp_path):
    df = _make_dataset()
    csv_path = tmp_path / "multiclass.csv"
    df.to_csv(csv_path, index=False)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        preprocessor,
        class_map,
    ) = load_and_prepare_multiclass(
        str(csv_path),
        base_outdir=str(tmp_path / "reports_mc"),
    )

    # ensure 'type' is used as the target
    expected_classes = set(df["type"].astype(str).str.strip())
    assert set(class_map.values()) == expected_classes

    assert TARGET_ALIASES.isdisjoint(feature_names)

def _assert_target_columns_do_not_affect_inference(df, preprocessor, model):
    inference_with_targets = df.copy()
    inference_without_targets = df.drop(
        columns=list(TARGET_ALIASES),
        errors="ignore",
    )
    inference_with_modified_targets = df.copy()
    for column in TARGET_ALIASES.intersection(df.columns):
        if pd.api.types.is_numeric_dtype(df[column]):
            inference_with_modified_targets[column] = df[column] + 1000
        else:
            inference_with_modified_targets[column] = (
                df[column].astype(str) + "-modified"
            )

    transformed_with_targets = preprocessor.transform(inference_with_targets)
    transformed_without_targets = preprocessor.transform(inference_without_targets)
    transformed_with_modified_targets = preprocessor.transform(
        inference_with_modified_targets
    )

    np.testing.assert_allclose(
        transformed_with_targets,
        transformed_without_targets,
    )
    np.testing.assert_allclose(
        transformed_with_targets,
        transformed_with_modified_targets,
    )

    predictions_with_targets = model.predict(transformed_with_targets)
    predictions_without_targets = model.predict(transformed_without_targets)
    predictions_with_modified_targets = model.predict(
        transformed_with_modified_targets
    )

    np.testing.assert_array_equal(
        predictions_with_targets,
        predictions_without_targets,
    )
    np.testing.assert_array_equal(
        predictions_with_targets,
        predictions_with_modified_targets,
    )


def test_multiclass_target_columns_do_not_affect_inference(tmp_path):
    df = _make_dataset()
    csv_path = tmp_path / "multiclass.csv"
    df.to_csv(csv_path, index=False)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        preprocessor,
        class_map,
    ) = load_and_prepare_multiclass(
        str(csv_path),
        base_outdir=str(tmp_path / "reports_mc"),
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    _assert_target_columns_do_not_affect_inference(df, preprocessor, model)


def test_binary_target_columns_do_not_affect_inference(tmp_path):
    df = _make_dataset()
    csv_path = tmp_path / "binary.csv"
    df.to_csv(csv_path, index=False)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        preprocessor,
    ) = load_and_prepare_data(
        str(csv_path),
        leakage_base=str(tmp_path / "reports"),
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    _assert_target_columns_do_not_affect_inference(df, preprocessor, model)
