from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loading import internal_fit_validation_indices, load_multiclass_split


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_interpreter(runtime: str, model_path: Path):
    if runtime == "tflite":
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    else:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise SystemExit("ERROR: ai-edge-litert must be installed if LiteRT comparison is requested") from exc
        interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def invoke(interpreter, X, batch_size):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if tuple(inp["shape"]) != (batch_size, X.shape[1]):
        interpreter.resize_tensor_input(inp["index"], [batch_size, X.shape[1]], strict=False)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    qmin, qmax = np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
    xq_unclipped = np.round(X / scale + zero)
    xq = np.clip(xq_unclipped, qmin, qmax).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], xq)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])
    oscale, ozero = out["quantization"]
    if out["dtype"] in (np.int8, np.uint8):
        y = (y.astype(np.float32) - ozero) * oscale
    return y, {
        "input_scale": float(scale), "input_zero_point": int(zero),
        "input_quantized_min": int(qmin), "input_quantized_max": int(qmax),
        "input_real_min": float((qmin - zero) * scale),
        "input_real_max": float((qmax - zero) * scale),
        "clipped_count": int(np.count_nonzero(xq_unclipped != np.clip(xq_unclipped, qmin, qmax))),
        "input_count": int(xq_unclipped.size),
        "saturation_min_count": int(np.count_nonzero(xq == qmin)),
        "saturation_max_count": int(np.count_nonzero(xq == qmax)),
        "output_dtype": str(out["dtype"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--csv", default="data/train_test_network.csv")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    model_dir = args.run_dir / "multiclass" / "models"
    keras_dir, int8_dir = model_dir / "mlp_mc", model_dir / "mlp_mc_int8"
    model_path = int8_dir / "model.tflite"
    X_train, _, y_train, _, _, preproc, _ = load_multiclass_split(
        args.csv, str(keras_dir / "preprocessor.pkl"), str(keras_dir / "preprocessor_meta.json"), model_name="mlp_mc"
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    _, validation_idx = internal_fit_validation_indices(y_train)
    X = joblib.load(keras_dir / "scaler.pkl").transform(X_train[validation_idx]).astype(np.float32)
    y = y_train[validation_idx]
    keras = tf.keras.models.load_model(keras_dir / "model.keras")
    keras_probs = keras.predict(X, verbose=0)
    q = json.loads((int8_dir / "calibration_statistics.json").read_text())
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    results = {
        "model_sha256": model_sha,
        "runtime_versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "ai_edge_litert": _package_version("ai-edge-litert"),
        },
        "evaluation_split": "internal_validation",
        "n_samples": int(len(X)),
        "calibration": q,
        "runs": [],
    }

    for batch_size in args.batch_sizes:
        if batch_size > 1 and batch_size > len(X):
            raise SystemExit(f"batch size {batch_size} exceeds validation rows ({len(X)})")
        x_eval = X if batch_size == 1 else X[:batch_size]
        y_eval = y if batch_size == 1 else y[:batch_size]
        batch_result = {"batch_size": batch_size, "evaluated_samples": int(len(x_eval)), "runtimes": {}}
        for runtime in ("tflite", "litert"):
            interpreter = make_interpreter(runtime, model_path)
            def predict_pass():
                parts, qinfo_last, clipped, input_count, sat_min, sat_max = [], None, 0, 0, 0, 0
                step = batch_size
                for start in range(0, len(x_eval), step):
                    current = x_eval[start:start + step]
                    if len(current) < step and step > 1:
                        break
                    pred, qinfo_last = invoke(interpreter, current, len(current))
                    parts.append(pred)
                    clipped += qinfo_last["clipped_count"]
                    input_count += qinfo_last["input_count"]
                    sat_min += qinfo_last["saturation_min_count"]
                    sat_max += qinfo_last["saturation_max_count"]
                qinfo_last["clipped_count"] = clipped
                qinfo_last["input_count"] = input_count
                qinfo_last["clipped_fraction"] = float(clipped / input_count) if input_count else 0.0
                qinfo_last["saturation_min_count"] = sat_min
                qinfo_last["saturation_max_count"] = sat_max
                return np.concatenate(parts, axis=0), qinfo_last

            prediction, qinfo = predict_pass()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                prediction, qinfo = predict_pass()
                times.append(time.perf_counter() - start)
            input_scale = qinfo["input_scale"]
            input_zero = qinfo["input_zero_point"]
            qmin, qmax = qinfo["input_quantized_min"], qinfo["input_quantized_max"]
            x_q = np.clip(np.round(x_eval / input_scale + input_zero), qmin, qmax)
            x_qdq = (x_q - input_zero) * input_scale
            keras_qdq = keras.predict(x_qdq, verbose=0)
            batch_result["runtimes"][runtime] = {
                "median_ms": float(np.median(times) * 1000),
                "ms_per_1000": float(np.median(times) / len(x_eval) * 1_000_000),
                "input_quantization": qinfo,
                "accuracy": float(np.mean(np.argmax(prediction, axis=1) == y_eval)),
                "keras_fp32_max_abs_error": float(np.max(np.abs(prediction - keras_probs[:len(x_eval)]))),
                "keras_fp32_mean_abs_error": float(np.mean(np.abs(prediction - keras_probs[:len(x_eval)]))),
                "keras_qdq_accuracy": float(np.mean(np.argmax(keras_qdq, axis=1) == y_eval)),
                "keras_qdq_max_abs_error": float(np.max(np.abs(prediction - keras_qdq))),
                "keras_qdq_mean_abs_error": float(np.mean(np.abs(prediction - keras_qdq))),
                "probabilities": prediction,
            }
        tflite_probs = batch_result["runtimes"]["tflite"].pop("probabilities")
        litert_probs = batch_result["runtimes"]["litert"].pop("probabilities")
        batch_result["runtime_parity"] = {
            "max_abs_error": float(np.max(np.abs(tflite_probs - litert_probs))),
            "mean_abs_error": float(np.mean(np.abs(tflite_probs - litert_probs))),
        }
        results["runs"].append(batch_result)

    output = args.run_dir / "int8_diagnostics" / "runtime_batch_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
