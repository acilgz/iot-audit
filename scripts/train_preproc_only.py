from __future__ import annotations
import os, sys, json
import numpy as np
import joblib
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.preprocessing import load_and_prepare_data
from iot_audit.metrics import evaluate_tflite_model

try:
    import ai_edge_litert.interpreter as tflm
except ImportError:
    try:
        import tflite_runtime.interpreter as tflm
    except ImportError:
        import tensorflow.lite as tflm


def run_tflite_inference(model_path: str, X_test: np.ndarray):
    interpreter = tflm.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    in_scale, in_zero_point = input_details[0]['quantization']
    out_scale, out_zero_point = output_details[0]['quantization']

    # match expected features dimension
    expected_features = input_details[0]['shape'][-1]
    if X_test.shape[1] > expected_features:
        X_test = X_test[:, :expected_features]

    # quantize float32 features into int8
    if in_scale > 0:
        X_test_int8 = np.round(X_test / in_scale + in_zero_point)
        X_test_formatted = np.clip(X_test_int8, -128, 127).astype(np.int8)
    else:
        X_test_formatted = X_test.astype(input_details[0]['dtype'])

    y_proba = []
    for row in X_test_formatted:
        sample = np.expand_dims(row, axis=0) # (1, 95)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

        out_int8 = interpreter.get_tensor(output_index)[0]

        # de-quantize int8 output into float32 probability [0.0, 1.0]
        if out_scale > 0:
            out_float = (out_int8.astype(np.float32) - out_zero_point) * out_scale
        else:
            out_float = out_int8.astype(np.float32)

        prob = out_float[1] if out_float.ndim > 0 and out_float.shape[0] > 1 else out_float[0]
        y_proba.append(prob)

    y_proba = np.array(y_proba, dtype=np.float32)
    y_pred = (y_proba >= 0.5).astype(int)

    return y_pred, y_proba, expected_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--model-name", default="mlp_int8")
    args = ap.parse_args()

    X_train, X_test, y_train, y_test, feature_names, preproc = load_and_prepare_data(
        csv_path=args.csv, target_col="label", test_size=0.2, random_state=42,
        leakage_base=args.outdir, model_name=args.model_name
    )

    model_dir = os.path.join(args.outdir, "models", args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    tflite_model_path = os.path.join(model_dir, "model.tflite")
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"Missing TFLite model at {tflite_model_path}")

    print(f"[{args.model_name}] Running TFLite inference on test set...")
    y_pred, y_proba, expected_features = run_tflite_inference(tflite_model_path, X_test)

    feature_names = feature_names[:expected_features]

    joblib.dump(preproc, os.path.join(model_dir, "preprocessor.pkl"))

    metrics = evaluate_tflite_model(
        y_test, y_pred, y_proba, feature_names, tflite_model_path,
        args.model_name, args.outdir
    )

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{args.model_name}] metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
