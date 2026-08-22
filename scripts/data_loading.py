from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from iot_audit.preprocessing import _read_csv as _read_csv_bin, _normalize_label
from iot_audit.preprocessing_mc import _read_csv as _read_csv_mc


def _require(preproc_path, meta_path):
    if not os.path.exists(preproc_path):
        raise SystemExit(
            f"Preprocessor not found in {preproc_path}. "
        )
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"Meta not found in {meta_path}. "
        )


def load_binary_split(csv_path, preproc_path, meta_path,
                       test_size=0.2, random_state=42):
    _require(preproc_path, meta_path)
    preproc = joblib.load(preproc_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    df = _read_csv_bin(csv_path)
    y = _normalize_label(df["label"])
    X_raw = df.drop(columns=["label"])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = preproc.transform(X_train_raw)
    X_test = preproc.transform(X_test_raw)
    return X_train, X_test, y_train.values, y_test.values, feature_names, preproc


def load_multiclass_split(csv_path, preproc_path, meta_path,
                           test_size=0.2, random_state=42):
    _require(preproc_path, meta_path)
    preproc = joblib.load(preproc_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    class_map = {int(k): v for k, v in meta["class_map"].items()}

    df = _read_csv_mc(csv_path)
    y_raw = df["type"].astype(str).str.strip()
    classes, y = np.unique(y_raw, return_inverse=True)
    X = df.drop(columns=["type"])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = preproc.transform(X_train_raw)
    X_test = preproc.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, feature_names, preproc, class_map
