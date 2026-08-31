from __future__ import annotations
import os, sys, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.metrics import evaluate_model
from data_loading import load_binary_split
import numpy as np
import joblib
import argparse
import tensorflow as tf


class TFLiteInt8Wrapper:
    def __init__(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.in_d = self.interpreter.get_input_details()[0]
        self.out_d = self.interpreter.get_output_details()[0]
        self.feature_importances_ = None

    def predict_proba(self, X):
        in_scale, in_zero = self.in_d["quantization"]
        out_scale, out_zero = self.out_d["quantization"]
        qmin, qmax = np.iinfo(self.in_d["dtype"]).min, np.iinfo(self.in_d["dtype"]).max

        X = np.asarray(X, dtype=np.float32)
        X_q = np.clip(np.round(X / in_scale + in_zero), qmin, qmax).astype(self.in_d["dtype"])

        p1 = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            self.interpreter.set_tensor(self.in_d["index"], X_q[i:i + 1])
            self.interpreter.invoke()
            out_q = self.interpreter.get_tensor(self.out_d["index"])
            p1[i] = (out_q.astype(np.float32) - out_zero) * out_scale
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--calib_samples", type=int, default=1000)
    args = ap.parse_args()

    input_dir = args.input_dir.rstrip("/")
    model_name = os.path.basename(input_dir)
    models_root = os.path.dirname(input_dir)
    outdir = os.path.dirname(models_root)
    output_dir = os.path.join(models_root, f"{model_name}_int8")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[quantize] {input_dir} -> {output_dir}")

    model_float_path = os.path.join(input_dir, "model.keras")
    scaler_path = os.path.join(input_dir, "scaler.pkl")
    if not os.path.exists(model_float_path):
        raise SystemExit(f"[quantize] {model_float_path} not found")
    if not os.path.exists(scaler_path):
        raise SystemExit(f"[quantize] {scaler_path} not found")

    model = tf.keras.models.load_model(model_float_path)
    scaler = joblib.load(scaler_path)

    preproc_path = os.path.join(outdir, "preprocessor", "preprocessor.pkl")
    meta_path = os.path.join(os.path.dirname(preproc_path), "preprocessor_meta.json")
    X_train, X_test, y_train, y_test, feature_names, preproc = load_binary_split(
        args.csv, preproc_path, meta_path
    )
    y_test = y_test.astype(int)

    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    rng = np.random.default_rng(42)
    calib_idx = rng.choice(X_train.shape[0], size=min(args.calib_samples, X_train.shape[0]), replace=False)
    calib_data = X_train[calib_idx]

    def representative_dataset():
        for row in calib_data:
            yield [row.reshape(1, -1).astype(np.float32)]

    saved_model_dir = os.path.join(output_dir, "_saved_model_tmp")
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    tflite_path = os.path.join(output_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[quantize] int8 model saved to {tflite_path} ({os.path.getsize(tflite_path)/1024:.1f} KB)")

    joblib.dump(preproc, os.path.join(output_dir, "preprocessor.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    wrapped = TFLiteInt8Wrapper(tflite_model)
    y_pred = wrapped.predict(X_test)
    y_proba = wrapped.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(
        y_test, y_pred, y_proba, feature_names, wrapped,
        model_name=f"{model_name}_int8", base_outdir=outdir
    )
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[quantize] metrics (int8):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()