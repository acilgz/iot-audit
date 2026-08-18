
from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import ai_edge_litert.interpreter as tflm

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
    # add FP/FN
    try:
        cm = m.get("confusion_matrix", [[0,0],[0,0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        m["fp"] = fp
        m["fn"] = fn
    except Exception:
        m["fp"] = None
        m["fn"] = None
    return m

def file_size_mb(path: str) -> float:
    return os.path.getsize(path)/ (1024*1024) if os.path.exists(path) else 0.0

def scan_models(base_outdir: str, model_names: List[str]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        mdir = os.path.join(base_outdir, "models", name)
        if not os.path.isdir(mdir):
            continue
        m = read_metrics(mdir)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        tflite_model = os.path.join(mdir, "model.tflite")
        
        if os.path.exists(tflite_model):
            # TFLite model
            model_file = tflite_model
            model_type = "tflite"
        elif os.path.exists(model_pkl) and os.path.exists(preproc_pkl):
            model_file = model_pkl
            model_type = "sklearn"
        else:
            model_file = None
            model_type = None
        
        if model_file:
            rows.append({
                "model": name,
                "accuracy": m.get("accuracy"),
                "f1_pos": m.get("f1_pos"),
                "f1_neg": m.get("f1_neg"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "fp": m.get("fp"),
                "fn": m.get("fn"),
                "model_size_mb": round(file_size_mb(model_file), 3),
                "preproc_size_mb": round(file_size_mb(preproc_pkl), 3),
                "total_size_mb": round(file_size_mb(model_file)+file_size_mb(preproc_pkl), 3),
            })
    return pd.DataFrame(rows)

def plot_bar(df: pd.DataFrame, column: str, out_png: str, title: str):
    plt.figure(figsize=(6,4))
    plt.bar(df["model"], df[column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(column)
    _savefig(out_png)

def benchmark_inference(base_outdir: str, csv_path: str, model_names: List[str], sample_size: int = 10000, random_state: int = 42, num_runs: int = 5) -> pd.DataFrame:
    # Load raw CSV once
    df = pd.read_csv(csv_path, engine="pyarrow")
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    X = df.drop(columns=["label"])
    # optional: drop 'type' if present (was dropped during training by preprocessing)
    if "type" in X.columns:
        X = X.drop(columns=["type"])

    # sample
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=random_state)
    else:
        Xs = X.copy()

    results = []
    for name in model_names:
        mdir = os.path.join(base_outdir, "models", name)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        tflite_path = os.path.join(mdir, "model.tflite")

        if os.path.exists(tflite_path):
            # TFLite model
            try:
                interpreter = tflm.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                
                # get input details from the model
                input_details = interpreter.get_input_details()
                expected_input_shape = input_details[0]['shape']
                expected_features = expected_input_shape[-1]  # number of features
                
                run_results = []
                for run in range(num_runs):
                    t0 = time.time()
                    X_trans = joblib.load(preproc_pkl).transform(Xs)
                    t1 = time.time()
                    
                    if X_trans.shape[1] != expected_features:
                        print(f"Warning: Feature mismatch for {name}. Got {X_trans.shape[1]} features, expected {expected_features}")
                        # trim to match expected dimensions
                        if X_trans.shape[1] > expected_features:
                            X_trans = X_trans[:, :expected_features]
                        else:
                            print(f"Warning: Not enough features for {name}, using truncated data")
                    
                    output_details = interpreter.get_output_details()
                    
                    target_dtype = input_details[0]['dtype']
                    X_trans_formatted = np.array(X_trans, dtype=target_dtype)
                    
                    y_probs = []
                    for row in X_trans_formatted:
                        sample = np.expand_dims(row, axis=0) 
                        interpreter.set_tensor(input_details[0]['index'], sample)
                        interpreter.invoke()
                        output = interpreter.get_tensor(output_details[0]['index'])
                        y_probs.append(output[0])
                    
                    t2 = time.time()

                    run_results.append({
                        "model": name,
                        "transform_ms_per_1k": (t1 - t0) / (len(Xs)/1000.0) * 1000.0,
                        "predict_ms_per_1k": (t2 - t1) / (len(Xs)/1000.0) * 1000.0,
                        "total_ms_per_1k": (t2 - t0) / (len(Xs)/1000.0) * 1000.0,
                        "n_samples": len(Xs),
                        "run_id": run + 1
                    })
            except Exception as e:
                print(f"Error running TFLite model {name}: {e}")
                continue
            
            results.extend(run_results)
            
            avg_result = {
                "model": name,
                "transform_ms_per_1k": np.mean([r["transform_ms_per_1k"] for r in run_results]),
                "predict_ms_per_1k": np.mean([r["predict_ms_per_1k"] for r in run_results]),
                "total_ms_per_1k": np.mean([r["total_ms_per_1k"] for r in run_results]),
                "n_samples": len(Xs),
                "run_id": "avg"
            }
            
            avg_result["transform_ms_per_1k"] = f"{avg_result['transform_ms_per_1k']:.6f} ± {np.std([r['transform_ms_per_1k'] for r in run_results]):.6f}"
            avg_result["predict_ms_per_1k"] = f"{avg_result['predict_ms_per_1k']:.6f} ± {np.std([r['predict_ms_per_1k'] for r in run_results]):.6f}"
            avg_result["total_ms_per_1k"] = f"{avg_result['total_ms_per_1k']:.6f} ± {np.std([r['total_ms_per_1k'] for r in run_results]):.6f}"
            
            results.append(avg_result)
        
        elif os.path.exists(model_pkl) and os.path.exists(preproc_pkl):
            model = joblib.load(model_pkl)
            preproc = joblib.load(preproc_pkl)

            def run_inference():
                X_trans = preproc.transform(Xs)
                return getattr(model, "predict_proba", model.predict)(X_trans)
            
            run_results = []
            for run in range(num_runs):
                t0 = time.time()
                X_trans = preproc.transform(Xs)
                t1 = time.time()
                
                _ = run_inference()
                
                t2 = time.time()

                run_results.append({
                    "model": name,
                    "transform_ms_per_1k": (t1 - t0) / (len(Xs)/1000.0) * 1000.0,
                    "predict_ms_per_1k": (t2 - t1) / (len(Xs)/1000.0) * 1000.0,
                    "total_ms_per_1k": (t2 - t0) / (len(Xs)/1000.0) * 1000.0,
                    "n_samples": len(Xs),
                    "run_id": run + 1
                })
            
            results.extend(run_results)
            
            avg_result = {
                "model": name,
                "transform_ms_per_1k": np.mean([r["transform_ms_per_1k"] for r in run_results]),
                "predict_ms_per_1k": np.mean([r["predict_ms_per_1k"] for r in run_results]),
                "total_ms_per_1k": np.mean([r["total_ms_per_1k"] for r in run_results]),
                "n_samples": len(Xs),
                "run_id": "avg"
            }
            
            avg_result["transform_ms_per_1k"] = f"{avg_result['transform_ms_per_1k']:.6f} ± {np.std([r['transform_ms_per_1k'] for r in run_results]):.6f}"
            avg_result["predict_ms_per_1k"] = f"{avg_result['predict_ms_per_1k']:.6f} ± {np.std([r['predict_ms_per_1k'] for r in run_results]):.6f}"
            avg_result["total_ms_per_1k"] = f"{avg_result['total_ms_per_1k']:.6f} ± {np.std([r['total_ms_per_1k'] for r in run_results]):.6f}"
            
            results.append(avg_result)
        
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--models", nargs="*", default=["rf","lgbm","xgb","logreg"])
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark on sample of the CSV")
    ap.add_argument("--sample_size", type=int, default=10000)
    ap.add_argument("--num_runs", type=int, default=5, help="Number of runs to execute for benchmarking (default: 5)")
    args = ap.parse_args()

    base_outdir = args.outdir
    summary_dir = os.path.join(base_outdir, "summary")
    _ensure_dir(summary_dir)

    df = scan_models(base_outdir, args.models)
    if df.empty:
        print("No model metrics found. Train some models first.")
        return

    df.to_csv(os.path.join(summary_dir, "summary_models.csv"), index=False)
    print("Summary saved to", os.path.join(summary_dir, "summary_models.csv"))
    print(df)

    # plots
    plot_bar(df, "accuracy", os.path.join(summary_dir, "accuracy.png"), "Accuracy by Model")
    if df["f1_pos"].notna().any():
        plot_bar(df, "f1_pos", os.path.join(summary_dir, "f1_pos.png"), "F1 (attack) by Model")
    if df["roc_auc"].notna().any():
        plot_bar(df, "roc_auc", os.path.join(summary_dir, "roc_auc.png"), "ROC AUC by Model")
    if df["fp"].notna().any():
        plot_bar(df, "fp", os.path.join(summary_dir, "fp.png"), "False Positives by Model")
    if df["fn"].notna().any():
        plot_bar(df, "fn", os.path.join(summary_dir, "fn.png"), "False Negatives by Model")

    if args.benchmark:
        bdf = benchmark_inference(base_outdir, args.csv, args.models, sample_size=args.sample_size, num_runs=args.num_runs)
        
        for run_id in range(1, args.num_runs + 1):
            run_data = bdf[bdf["run_id"] == run_id]
            if not run_data.empty:
                run_filename = os.path.join(summary_dir, f"inference_benchmark_{run_id}.csv")
                # save individual runs without the run_id column
                run_data_no_runid = run_data.drop(columns=["run_id"], errors="ignore")
                run_data_no_runid.to_csv(run_filename, index=False)
                print(f"Run {run_id} saved to {run_filename}")

        final_filename = os.path.join(summary_dir, "inference_benchmark.csv")
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

if __name__ == "__main__":
    main()
