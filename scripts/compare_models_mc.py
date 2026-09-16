
from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import ai_edge_litert.interpreter as tflm
import keras

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def read_metrics(model_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m

def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

def scan_models(base_outdir: str, model_names: List[str]) -> pd.DataFrame:
    rows = []
    global_preproc = os.path.join(base_outdir, "preprocessor_mc", "preprocessor.pkl")

    for name in model_names:
        mdir = os.path.join(base_outdir, "models", name)
        if not os.path.isdir(mdir):
            continue
        m = read_metrics(mdir)
        
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        if not os.path.exists(preproc_pkl):
            preproc_pkl = global_preproc
        
        model_pkl = os.path.join(mdir, "model.pkl")
        model_keras = os.path.join(mdir, "model.keras")
        tflite_model = os.path.join(mdir, "model.tflite")
        
        if os.path.exists(tflite_model) and os.path.exists(preproc_pkl):
            model_file = tflite_model
        elif os.path.exists(model_keras) and os.path.exists(preproc_pkl):
            model_file = model_keras
        elif os.path.exists(model_pkl) and os.path.exists(preproc_pkl):
            model_file = model_pkl
        else:
            model_file = None

        if model_file:
            rows.append({
                "model": name,
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "weighted_f1": m.get("weighted_f1"),
                "roc_auc_micro": m.get("roc_auc_micro"),
                "roc_auc_macro": m.get("roc_auc_macro"),
                "pr_auc_micro": m.get("pr_auc_micro"),
                "pr_auc_macro": m.get("pr_auc_macro"),
                "model_size_mb": round(file_size_mb(model_file), 3),
                "preproc_size_mb": round(file_size_mb(preproc_pkl), 3),
                "total_size_mb": round(file_size_mb(model_file) + file_size_mb(preproc_pkl), 3),
            })
    return pd.DataFrame(rows)

def plot_bar(df: pd.DataFrame, column: str, out_png: str, title: str):
    if df.empty or column not in df.columns: 
        return
    if df[column].isna().all(): 
        return
    plt.figure(figsize=(6,4))
    plt.bar(df["model"], df[column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(column)
    _savefig(out_png)

def per_class_table(models_dir: str, model_names: List[str]) -> pd.DataFrame:
    merged = None
    for name in model_names:
        p = os.path.join(models_dir, "models", name, "per_class_report.csv")
        if not os.path.exists(p): 
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={
            "precision": f"precision_{name}",
            "recall": f"recall_{name}",
            "f1": f"f1_{name}",
            "support": f"support_{name}",
        })
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="class", how="outer")
    return merged if merged is not None else pd.DataFrame()


def _to_dense_array(X, dtype=None):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=dtype)


def _apply_scaler(X, scaler, feature_names=None):
    if scaler is None:
        return X
    if hasattr(X, "toarray"):
        X = X.toarray()
    cols = getattr(scaler, "feature_names_in_", feature_names)
    if cols is not None and len(cols) == X.shape[1]:
        X = pd.DataFrame(X, columns=cols)
    return scaler.transform(X)


def _prepare_tflite_input(X, input_detail):
    target_dtype = input_detail["dtype"]
    X_arr = _to_dense_array(X, dtype=np.float32)

    if np.issubdtype(target_dtype, np.integer):
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale is None or scale == 0:
            raise ValueError(
                f"tflite input has integer dtype {target_dtype} but no valid quantization scale"
            )
        qinfo = np.iinfo(target_dtype)
        return np.clip(
            np.round(X_arr / scale + zero_point),
            qinfo.min,
            qinfo.max,
        ).astype(target_dtype)

    return X_arr.astype(target_dtype, copy=False)



def _configure_tflite_batch(interpreter, batch_size: int, n_features: int):
    input_detail = interpreter.get_input_details()[0]
    current_shape = tuple(int(v) for v in input_detail["shape"])
    target_shape = (int(batch_size), int(n_features))

    if len(current_shape) != 2:
        raise ValueError(
            f"Expected a rank-2 tflite input, got shape {current_shape}"
        )

    if current_shape != target_shape:
        try:
            interpreter.resize_tensor_input(
                input_detail["index"], list(target_shape), strict=False
            )
            interpreter.allocate_tensors()
        except Exception as exc:
            shape_signature = tuple(
                int(v) for v in input_detail.get("shape_signature", current_shape)
            )
            raise RuntimeError(
                "Unable to resize the tflite model to the benchmark batch size "
                f"{target_shape}. Current shape: {current_shape}, "
                f"shape_signature: {shape_signature}."
            ) from exc

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return input_detail, output_detail

def _ms_per_1k(seconds: float, n_samples: int) -> float:
    return seconds / (n_samples / 1000.0) * 1000.0


def _append_average_result(results, run_results, model_name: str, n_samples: int):
    avg_result = {
        "model": model_name,
        "transform_ms_per_1k": np.mean([r["transform_ms_per_1k"] for r in run_results]),
        "predict_ms_per_1k": np.mean([r["predict_ms_per_1k"] for r in run_results]),
        "total_ms_per_1k": np.mean([r["total_ms_per_1k"] for r in run_results]),
        "n_samples": n_samples,
        "run_id": "avg",
    }
    avg_result["transform_ms_per_1k"] = (
        f"{avg_result['transform_ms_per_1k']:.6f} ± "
        f"{np.std([r['transform_ms_per_1k'] for r in run_results]):.6f}"
    )
    avg_result["predict_ms_per_1k"] = (
        f"{avg_result['predict_ms_per_1k']:.6f} ± "
        f"{np.std([r['predict_ms_per_1k'] for r in run_results]):.6f}"
    )
    avg_result["total_ms_per_1k"] = (
        f"{avg_result['total_ms_per_1k']:.6f} ± "
        f"{np.std([r['total_ms_per_1k'] for r in run_results]):.6f}"
    )
    results.append(avg_result)


def _save_mean_latency_ratio(bdf: pd.DataFrame, out_path: str, lgbm_name: str, xgb_name: str):
    raw_runs = bdf[bdf["run_id"] != "avg"].copy()
    if raw_runs.empty:
        return

    raw_runs["total_ms_per_1k"] = pd.to_numeric(raw_runs["total_ms_per_1k"], errors="coerce")
    mean_latency = raw_runs.groupby("model")["total_ms_per_1k"].mean()

    if lgbm_name not in mean_latency.index or xgb_name not in mean_latency.index:
        return

    lgbm_mean = float(mean_latency[lgbm_name])
    xgb_mean = float(mean_latency[xgb_name])
    ratio = lgbm_mean / xgb_mean

    pd.DataFrame([{
        "lgbm_mean_ms_per_1k": lgbm_mean,
        "xgb_mean_ms_per_1k": xgb_mean,
        "lgbm_xgb_ratio": ratio,
        "n_runs": int(raw_runs[raw_runs["model"] == lgbm_name].shape[0]),
    }]).to_csv(out_path, index=False)


def benchmark_inference(models_dir: str, csv_path: str, model_names: List[str], sample_size: int = 10000, random_state: int = 42, num_runs: int = 5) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="pyarrow")
    X = df.copy()

    leak_cols = [
        c for c in [
            "type", "Type", "TYPE",
            "label", "Label", "LABEL",
            "target", "Target", "TARGET",
        ] if c in X.columns
    ]
    if leak_cols:
        X = X.drop(columns=leak_cols)
        print(f"[benchmark_inference] dropped potential leakage columns: {leak_cols}")

    Xs = X.sample(n=sample_size, random_state=random_state) if len(X) > sample_size else X.copy()
    if len(Xs) == 0:
        raise ValueError("No samples available")

    preproc_pkl = os.path.join(models_dir, "preprocessor_mc", "preprocessor.pkl")
    meta_path = os.path.join(models_dir, "preprocessor_mc", "preprocessor_meta.json")
    if not os.path.exists(preproc_pkl):
        raise FileNotFoundError(f"Preprocessor not found: {preproc_pkl}")

    feature_names = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f).get("feature_names")

    results = []
    for name in model_names:
        mdir = os.path.join(models_dir, "models", name)
        model_pkl = os.path.join(mdir, "model.pkl")
        tflite_path = os.path.join(mdir, "model.tflite")
        keras_path = os.path.join(mdir, "model.keras")
        scaler_path = os.path.join(mdir, "scaler.pkl")

        if os.path.exists(tflite_path):
            try:
                preproc = joblib.load(preproc_pkl)
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

                interpreter = tflm.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()

                initial_input = interpreter.get_input_details()[0]
                expected_features = int(initial_input["shape"][-1])
                input_detail, output_detail = _configure_tflite_batch(
                    interpreter, len(Xs), expected_features
                )

                X_warm = preproc.transform(Xs)
                X_warm = _apply_scaler(X_warm, scaler, feature_names)
                X_warm = _prepare_tflite_input(X_warm, input_detail)
                interpreter.set_tensor(input_detail["index"], X_warm)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_detail["index"])

                run_results = []
                for run in range(num_runs):
                    t0 = time.perf_counter()
                    X_trans = preproc.transform(Xs)
                    X_trans = _apply_scaler(X_trans, scaler, feature_names)
                    X_trans_formatted = _prepare_tflite_input(X_trans, input_detail)
                    if X_trans_formatted.ndim != 2 or X_trans_formatted.shape[1] != expected_features:
                        raise ValueError(
                            f"tflite model {name} expects {expected_features} features, "
                            f"got shape {X_trans_formatted.shape}"
                        )
                    t1 = time.perf_counter()

                    interpreter.set_tensor(input_detail["index"], X_trans_formatted)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_detail["index"])

                    t2 = time.perf_counter()
                    run_results.append({
                        "model": name,
                        "transform_ms_per_1k": _ms_per_1k(t1 - t0, len(Xs)),
                        "predict_ms_per_1k": _ms_per_1k(t2 - t1, len(Xs)),
                        "total_ms_per_1k": _ms_per_1k(t2 - t0, len(Xs)),
                        "n_samples": len(Xs),
                        "run_id": run + 1,
                    })
            except Exception as e:
                print(f"Error running TFLite model {name}: {e}")
                continue

            results.extend(run_results)
            _append_average_result(results, run_results, name, len(Xs))

        elif os.path.exists(model_pkl):
            try:
                model = joblib.load(model_pkl)
                preproc = joblib.load(preproc_pkl)
                run_results = []

                for run in range(num_runs):
                    t0 = time.perf_counter()
                    X_trans = preproc.transform(Xs)
                    if feature_names and len(feature_names) == X_trans.shape[1]:
                        X_trans = pd.DataFrame(X_trans, columns=feature_names)
                    t1 = time.perf_counter()

                    _ = getattr(model, "predict_proba", model.predict)(X_trans)
                    t2 = time.perf_counter()

                    run_results.append({
                        "model": name,
                        "transform_ms_per_1k": _ms_per_1k(t1 - t0, len(Xs)),
                        "predict_ms_per_1k": _ms_per_1k(t2 - t1, len(Xs)),
                        "total_ms_per_1k": _ms_per_1k(t2 - t0, len(Xs)),
                        "n_samples": len(Xs),
                        "run_id": run + 1,
                    })
            except Exception as e:
                print(f"Error running model {name}: {e}")
                continue

            results.extend(run_results)
            _append_average_result(results, run_results, name, len(Xs))

        elif os.path.exists(keras_path):
            try:
                model = keras.models.load_model(keras_path)
                preproc = joblib.load(preproc_pkl)
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                X_warm = preproc.transform(Xs)
                X_warm = _apply_scaler(X_warm, scaler, feature_names)
                X_warm = _to_dense_array(X_warm, dtype=np.float32)
                _ = model(X_warm, training=False).numpy()

                run_results = []

                for run in range(num_runs):
                    t0 = time.perf_counter()
                    X_trans = preproc.transform(Xs)
                    X_trans = _apply_scaler(X_trans, scaler, feature_names)
                    X_tensor = _to_dense_array(X_trans, dtype=np.float32)
                    t1 = time.perf_counter()

                    _ = model(X_tensor, training=False).numpy()
                    t2 = time.perf_counter()

                    run_results.append({
                        "model": name,
                        "transform_ms_per_1k": _ms_per_1k(t1 - t0, len(Xs)),
                        "predict_ms_per_1k": _ms_per_1k(t2 - t1, len(Xs)),
                        "total_ms_per_1k": _ms_per_1k(t2 - t0, len(Xs)),
                        "n_samples": len(Xs),
                        "run_id": run + 1,
                    })
            except Exception as e:
                print(f"Error running Keras model {name}: {e}")
                continue

            results.extend(run_results)
            _append_average_result(results, run_results, name, len(Xs))
        else:
            print(f"No supported model found for {name} in {mdir}")

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports_mc")
    ap.add_argument("--models-dir", default="train_mc")
    ap.add_argument("--models", nargs="*", default=["rf_mc","lgbm_mc","xgb_mc","logreg_mc","mlp_mc","mlp_mc_int8"])
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark on sample of the CSV")
    ap.add_argument("--sample_size", type=int, default=10000)
    ap.add_argument("--num_runs", type=int, default=5, help="Number of runs to execute for benchmarking (default: 5)")
    args = ap.parse_args()

    base_outdir = args.outdir
    summary_dir = os.path.join(base_outdir, "summary")
    _ensure_dir(summary_dir)

    df = scan_models(args.models_dir, args.models)
    if df.empty:
        print("No model metrics found. Train some models first.")
        return
    df.to_csv(os.path.join(summary_dir, "summary_models_mc.csv"), index=False)
    print("Summary saved to", os.path.join(summary_dir, "summary_models_mc.csv"))
    print(df)

    plot_bar(df, "accuracy", os.path.join(summary_dir, "accuracy.png"), "Accuracy by Model (Multiclass)")
    plot_bar(df, "macro_f1", os.path.join(summary_dir, "macro_f1.png"), "Macro F1 by Model")
    plot_bar(df, "weighted_f1", os.path.join(summary_dir, "weighted_f1.png"), "Weighted F1 by Model")
    plot_bar(df, "roc_auc_micro", os.path.join(summary_dir, "roc_auc_micro.png"), "ROC AUC (micro) by Model")
    plot_bar(df, "roc_auc_macro", os.path.join(summary_dir, "roc_auc_macro.png"), "ROC AUC (macro) by Model")
    plot_bar(df, "pr_auc_micro", os.path.join(summary_dir, "pr_auc_micro.png"), "PR AUC (micro) by Model")
    plot_bar(df, "pr_auc_macro", os.path.join(summary_dir, "pr_auc_macro.png"), "PR AUC (macro) by Model")
    plot_bar(df, "total_size_mb", os.path.join(summary_dir, "total_size_mb.png"), "Model+Preproc Size (MB)")

    pct = per_class_table(args.models_dir, args.models)
    if not pct.empty:
        pct.to_csv(os.path.join(summary_dir, "per_class_report_merged.csv"), index=False)
        print("Per-class report merged saved to", os.path.join(summary_dir, "per_class_report_merged.csv"))
        for name in args.models:
            fcol = f"f1_{name}"
            if fcol in pct.columns:
                dd = pct[["class", fcol]].dropna()
                plt.figure(figsize=(8, max(4, 0.35*len(dd))))
                plt.barh(dd["class"], dd[fcol])
                plt.title(f"Per-class F1: {name}")
                plt.ylabel("Class"); plt.xlabel("F1")
                _savefig(os.path.join(summary_dir, f"per_class_f1_{name}.png"))

    if args.benchmark:
        bdf = benchmark_inference(args.models_dir, args.csv, args.models, sample_size=args.sample_size, num_runs=args.num_runs)
        
        for run_id in range(1, args.num_runs + 1):
            run_data = bdf[bdf["run_id"] == run_id]
            if not run_data.empty:
                run_filename = os.path.join(base_outdir, f"inference_benchmark_mc_{run_id}.csv")
                run_data_no_runid = run_data.drop(columns=["run_id"], errors="ignore")
                run_data_no_runid.to_csv(run_filename, index=False)
                print(f"Run {run_id} saved to {run_filename}")

        final_filename = os.path.join(base_outdir, "inference_benchmark_mc.csv")
        bdf.to_csv(final_filename, index=False)
        print(f"All benchmark data saved to {final_filename}")
        
        print(bdf)

        avg_rows = bdf[bdf["run_id"] == "avg"]
        if not avg_rows.empty:
            plt.figure(figsize=(6,4))
            avg_values = []
            for _, row in avg_rows.iterrows():
                if isinstance(row["total_ms_per_1k"], str) and "±" in row["total_ms_per_1k"]:
                    mean_value = float(row["total_ms_per_1k"].split("±")[0].strip())
                    avg_values.append(mean_value)
                else:
                    avg_values.append(row["total_ms_per_1k"])
            plt.bar(avg_rows["model"], avg_values)
            plt.title("Total latency (ms) per 1k flows")
            plt.xlabel("Model")
            plt.ylabel("ms per 1k")
            _savefig(os.path.join(summary_dir, "latency_total_ms_per_1k.png"))

        # save lgbm/xgb ratios
        ratio_path = os.path.join(base_outdir, "lgbm_xgb_ratios.csv")
        _save_mean_latency_ratio(bdf, ratio_path, "lgbm_mc", "xgb_mc")

if __name__ == "__main__":
    main()
