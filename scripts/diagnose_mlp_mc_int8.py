from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loading import load_multiclass_split


def metrics(name, y_true, probs, class_map):
    pred = np.argmax(probs, axis=1)
    rep = classification_report(y_true, pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "recall": {
            class_map.get(str(k), str(k)): float(v["recall"])
            for k, v in rep.items()
            if str(k).isdigit()
        },
        "predictions": pred.tolist(),
    }


def tflite_predict(path, X):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    scale, zero = inp["quantization"]
    oscale, ozero = out["quantization"]

    result = []
    for row in X:
        x = row.reshape(1, -1)
        if inp["dtype"] == np.int8:
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            y = (y.astype(np.float32) - ozero) * oscale
        result.append(y[0])
    return np.asarray(result), {
        "input": {
            "dtype": str(inp["dtype"]),
            "scale": float(scale),
            "zero_point": int(zero),
        },
        "output": {
            "dtype": str(out["dtype"]),
            "scale": float(oscale),
            "zero_point": int(ozero),
        }
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv", default="data/train_test_network.csv")
    args = p.parse_args()

    base = Path(args.run_dir) / "multiclass" / "models"
    keras_dir = base / "mlp_mc"
    int8_dir = base / "mlp_mc_int8"
    out = Path(args.run_dir) / "int8_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, features, preproc, class_map = load_multiclass_split(
        args.csv,
        str(keras_dir / "preprocessor.pkl"),
        str(keras_dir / "preprocessor_meta.json"),
        model_name="mlp_mc"
    )

    scaler = joblib.load(keras_dir / "scaler.pkl")
    X_test = scaler.transform(X_test).astype(np.float32)

    keras = tf.keras.models.load_model(keras_dir / "model.keras")
    keras_probs = keras.predict(X_test, verbose=0)

    int8_probs, q = tflite_predict(int8_dir / "model.tflite", X_test)

    scale = q["input"]["scale"]
    zero = q["input"]["zero_point"]
    X_qdq = (np.clip(np.round(X_test / scale + zero), -128, 127) - zero) * scale
    qdq_probs = keras.predict(X_qdq, verbose=0)

    saved_model_dir = out / "saved_model_fp32"

    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)

    keras.export(saved_model_dir)

    fp32_converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_dir)
    )

    fp32_tflite = fp32_converter.convert()

    fp32_path = out / "model_fp32.tflite"

    fp32_path.write_bytes(fp32_tflite)

    fp32_probs, _ = tflite_predict(
        fp32_path,
        X_test
    )

    data = [
        metrics("keras_fp32", y_test, keras_probs, class_map),
        metrics("tflite_fp32", y_test, fp32_probs, class_map),
        metrics("tflite_int8", y_test, int8_probs, class_map),
        metrics("keras_qdq", y_test, qdq_probs, class_map),
    ]

    (out / "metrics.json").write_text(json.dumps(data, indent=2))
    (out / "quantization.json").write_text(json.dumps(q, indent=2))
    (out / "input_statistics.json").write_text(json.dumps({
        "min": float(X_test.min()),
        "max": float(X_test.max()),
        "percentile_99": float(np.percentile(X_test, 99)),
        "abs_gt_100": int(np.sum(np.abs(X_test) > 100)),
        "abs_gt_200": int(np.sum(np.abs(X_test) > 200)),
    }, indent=2))

if __name__ == "__main__":
    main()
