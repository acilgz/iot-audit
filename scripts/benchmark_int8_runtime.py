from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
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

def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}

def git_provenance(repo_dir, run_dir=None):
    result = {
        "commit": None,
        "dirty": None,
        "status_porcelain": None,
        "status_excludes_run_dir": str(run_dir) if run_dir is not None else None,
    }
    try:
        def git(*arguments):
            return subprocess.run(
                ["git", "-C", str(repo_dir), *arguments], check=True,
                capture_output=True, text=True,
            ).stdout
        result["commit"] = git("rev-parse", "HEAD").strip()
        status_args = ["status", "--porcelain=v1", "--untracked-files=all", "--", "."]
        if run_dir is not None:
            run_path = Path(run_dir)
            if not run_path.is_absolute():
                run_path = Path.cwd() / run_path
            run_path_text = run_path.resolve().relative_to(Path(repo_dir).resolve()).as_posix()
            status_args.extend(
                [f":(exclude){run_path_text}", f":(exclude){run_path_text}/**"]
            )
        result["status_porcelain"] = git(*status_args)
        result["dirty"] = bool(result["status_porcelain"])
    except (OSError, subprocess.CalledProcessError) as exc:
        result["unavailable_reason"] = str(exc)
    return result

def collect_provenance(args, keras_dir, int8_dir):
    script = Path(__file__).resolve()
    files = {
        "dataset": Path(args.csv),
        "preprocessor": keras_dir / "preprocessor.pkl",
        "preprocessor_metadata": keras_dir / "preprocessor_meta.json",
        "scaler": keras_dir / "scaler.pkl",
        "keras_model": keras_dir / "model.keras",
        "int8_model": int8_dir / "model.tflite",
        "calibration_statistics": int8_dir / "calibration_statistics.json",
        "diagnostic_script": script,
    }
    training_records = {}
    for name in ("commit.txt", "git-status.txt", "python-version.txt", "environment.txt"):
        path = args.run_dir / name
        training_records[name] = (
            {**file_record(path), "recorded_text": path.read_text(encoding="utf-8")}
            if path.is_file() else {"available": False}
        )
    argv = [sys.executable, *sys.argv]
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": argv,
        "command_posix": shlex.join(argv),
        "cwd": str(Path.cwd()),
        "run_dir": str(args.run_dir.resolve()),
        "parameters": {"n_samples": args.n_samples, "batch_sizes": args.batch_sizes,
                       "repeats": args.repeats},
        "diagnostic_git": git_provenance(script.parent.parent, args.run_dir),
        "platform": {**platform.uname()._asdict(), "description": platform.platform()},
        "python_executable": sys.executable,
        "files": {name: file_record(path) for name, path in files.items()},
        "training_provenance_as_recorded": training_records,
    }

def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()

def compare_predictions(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Prediction shapes differ: {left.shape} != {right.shape}")
    left_labels, right_labels = left.argmax(axis=1), right.argmax(axis=1)
    different = np.flatnonzero(left_labels != right_labels)
    return {
        "max_abs_error": float(np.max(np.abs(left - right))),
        "mean_abs_error": float(np.mean(np.abs(left - right))),
        "class_disagreement_count": int(len(different)),
        "class_disagreement_fraction": float(len(different) / len(left)),
        "disagreements": [
            {"evaluation_position": int(i), "left_class_index": int(left_labels[i]),
             "right_class_index": int(right_labels[i])}
            for i in different
        ],
    }

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
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.n_samples < 1 or any(b < 1 for b in args.batch_sizes):
        parser.error("repeats, n-samples and batch sizes must be > 0")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch sizes must be unique")
    if max(args.batch_sizes) > args.n_samples:
        parser.error("batch sizes must not exceed n-samples")

    model_dir = args.run_dir / "multiclass" / "models"
    keras_dir, int8_dir = model_dir / "mlp_mc", model_dir / "mlp_mc_int8"
    model_path = int8_dir / "model.tflite"
    provenance = collect_provenance(args, keras_dir, int8_dir)
    X_train, _, y_train, _, _, _, class_map = load_multiclass_split(
        args.csv, str(keras_dir / "preprocessor.pkl"), str(keras_dir / "preprocessor_meta.json"), model_name="mlp_mc"
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    _, validation_idx = internal_fit_validation_indices(y_train)
    if args.n_samples > len(validation_idx):
        parser.error(f"n-samples ({args.n_samples}) exceeds validation rows ({len(validation_idx)})")
    evaluation_idx = np.asarray(validation_idx[:args.n_samples], dtype=np.int64)
    X = joblib.load(keras_dir / "scaler.pkl").transform(X_train[evaluation_idx]).astype(np.float32)
    y = y_train[evaluation_idx]
    class_map = {int(k): str(v) for k, v in class_map.items()}
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError("Class indices must be contiguous from zero")
    if not np.isin(y, list(class_map)).all():
        raise ValueError("Validation labels are outside the saved class map")
    keras = tf.keras.models.load_model(keras_dir / "model.keras")
    keras_probs = keras.predict(X, verbose=0)
    if keras_probs.shape != (len(X), len(class_map)):
        raise ValueError("Keras output does not match samples/classes")
    elif not np.isfinite(keras_probs).all():
        raise ValueError("Keras output contains non-finite values")
    q = json.loads((int8_dir / "calibration_statistics.json").read_text())
    model_sha = provenance["files"]["int8_model"]["sha256"]
    results = {
        "provenance": provenance,
        "model_sha256": model_sha,
        "runtime_versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "ai_edge_litert": _package_version("ai-edge-litert"),
            "numpy": np.__version__,
            "joblib": _package_version("joblib"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "evaluation_split": "internal_validation",
        "n_samples": int(len(X)),
        "evaluation": {
            "selection": "first n-samples in deterministic internal validation order",
            "available_validation_samples": int(len(validation_idx)),
            "validation_positions": list(range(len(X))),
            "outer_training_positions": evaluation_idx.tolist(),
            "inputs_sha256": array_sha256(X),
            "labels_sha256": array_sha256(y),
            "indices_sha256": array_sha256(evaluation_idx),
            "input_shape": list(X.shape),
            "input_dtype": str(X.dtype),
            "class_map": class_map,
        },
        "keras_fp32_accuracy": float(np.mean(keras_probs.argmax(axis=1) == y)),
        "timing_scope": "quantization, clipping counters, interpreter invocation, dequantization and concatenation; includes resize/allocation on shape changes; excludes one warmup pass",
        "calibration": q,
        "runs": [],
        "batch_parity": [],
    }

    predictions_by_batch = {}
    for batch_size in args.batch_sizes:
        x_eval, y_eval = X, y
        batch_result = {
            "batch_size": batch_size, "evaluated_samples": int(len(x_eval)),
            "full_batches": len(X) // batch_size,
            "remainder_batch_size": len(X) % batch_size,
            "inputs_sha256": results["evaluation"]["inputs_sha256"],
            "labels_sha256": results["evaluation"]["labels_sha256"],
            "runtimes": {},
        }
        for runtime in ("tflite", "litert"):
            interpreter = make_interpreter(runtime, model_path)
            def predict_pass():
                parts, qinfo_last, clipped, input_count, sat_min, sat_max = [], None, 0, 0, 0, 0
                step = batch_size
                for start in range(0, len(x_eval), step):
                    current = x_eval[start:start + step]
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
            if prediction.shape != keras_probs.shape:
                raise ValueError(f"ERROR: [{runtime}] output does not match samples/classes")
            elif not np.isfinite(prediction).all():
                raise ValueError(f"ERROR: [{runtime}] contains non-finite values")
            input_scale = qinfo["input_scale"]
            input_zero = qinfo["input_zero_point"]
            qmin, qmax = qinfo["input_quantized_min"], qinfo["input_quantized_max"]
            x_q = np.clip(np.round(x_eval / input_scale + input_zero), qmin, qmax)
            x_qdq = (x_q - input_zero) * input_scale
            keras_qdq = keras.predict(x_qdq, verbose=0)
            batch_result["runtimes"][runtime] = {
                "model_sha256": model_sha,
                "evaluated_samples": int(len(prediction)),
                "predictions_sha256": array_sha256(prediction),
                "repeat_ms": [float(t * 1000) for t in times],
                "median_ms": float(np.median(times) * 1000),
                "ms_per_1000": float(np.median(times) / len(x_eval) * 1_000_000),
                "input_quantization": qinfo,
                "accuracy": float(np.mean(np.argmax(prediction, axis=1) == y_eval)),
                "keras_fp32_max_abs_error": float(np.max(np.abs(prediction - keras_probs))),
                "keras_fp32_mean_abs_error": float(np.mean(np.abs(prediction - keras_probs))),
                "keras_qdq_accuracy": float(np.mean(np.argmax(keras_qdq, axis=1) == y_eval)),
                "keras_qdq_max_abs_error": float(np.max(np.abs(prediction - keras_qdq))),
                "keras_qdq_mean_abs_error": float(np.mean(np.abs(prediction - keras_qdq))),
                "probabilities": prediction,
            }
        tflite_probs = batch_result["runtimes"]["tflite"].pop("probabilities")
        litert_probs = batch_result["runtimes"]["litert"].pop("probabilities")
        batch_result["runtime_parity"] = compare_predictions(tflite_probs, litert_probs)
        batch_result["runtime_parity"].update({"left_runtime": "tflite", "right_runtime": "litert"})
        predictions_by_batch[batch_size] = {"tflite": tflite_probs, "litert": litert_probs}
        results["runs"].append(batch_result)

    for i, left_batch in enumerate(args.batch_sizes):
        for right_batch in args.batch_sizes[i + 1:]:
            for runtime in ("tflite", "litert"):
                comparison = compare_predictions(
                    predictions_by_batch[left_batch][runtime],
                    predictions_by_batch[right_batch][runtime],
                )
                comparison.update({"runtime": runtime, "left_batch_size": left_batch,
                                   "right_batch_size": right_batch})
                results["batch_parity"].append(comparison)

    output = args.run_dir / "int8_diagnostics" / "runtime_batch_comparison.json"
    results["provenance"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
