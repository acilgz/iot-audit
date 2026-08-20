
from __future__ import annotations
import os, json, argparse, time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
    return os.path.getsize(path)/ (1024*1024) if os.path.exists(path) else 0.0

def scan_models(models_dir: str, model_names: List[str]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        mdir = os.path.join(models_dir, name)
        if not os.path.isdir(mdir):
            continue
        m = read_metrics(mdir)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "roc_auc_micro": m.get("roc_auc_micro"),
            "roc_auc_macro": m.get("roc_auc_macro"),
            "pr_auc_micro": m.get("pr_auc_micro"),
            "pr_auc_macro": m.get("pr_auc_macro"),
            "model_size_mb": round(file_size_mb(model_pkl), 3),
            "preproc_size_mb": round(file_size_mb(preproc_pkl), 3),
            "total_size_mb": round(file_size_mb(model_pkl)+file_size_mb(preproc_pkl), 3),
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
        p = os.path.join(models_dir, name, "per_class_report.csv")
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

def benchmark_inference(models_dir: str, csv_path: str, model_names: List[str], y_col: str = "type", sample_size: int = 10000, random_state: int = 42, num_runs: int = 5) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="pyarrow")
    if y_col not in df.columns:
        raise ValueError(f"CSV must contain '{y_col}' column")
    X = df.drop(columns=[y_col])
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=random_state)
    else:
        Xs = X.copy()

    results = []
    for name in model_names:
        mdir = os.path.join(models_dir, name)
        model_pkl = os.path.join(mdir, "model.pkl")
        preproc_pkl = os.path.join(mdir, "preprocessor.pkl")
        if not (os.path.exists(model_pkl) and os.path.exists(preproc_pkl)):
            continue

        model = joblib.load(model_pkl)
        preproc = joblib.load(preproc_pkl)

        run_results = []
        for run in range(num_runs):
            t0 = time.time()
            X_trans = preproc.transform(Xs)
            t1 = time.time()
            _ = getattr(model, "predict_proba", model.predict)(X_trans)
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
    ap.add_argument("--outdir", default="reports_mc")
    ap.add_argument("--models-dir", default="reports_mc/models", help="Directory containing trained model artifacts")
    ap.add_argument("--models", nargs="*", default=["rf_mc","lgbm_mc","xgb_mc","logreg_mc"])
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark on sample of the CSV")
    ap.add_argument("--sample_size", type=int, default=10000)
    ap.add_argument("--num_runs", type=int, default=5, help="Number of runs to execute for benchmarking (default: 5)")
    args = ap.parse_args()

    base_outdir = args.outdir
    charts_dir = os.path.join(base_outdir, "charts")
    _ensure_dir(charts_dir)

    df = scan_models(args.models_dir, args.models)
    if df.empty:
        print("No model metrics found. Train some models first.")
        return
    df.to_csv(os.path.join(base_outdir, "summary_models_mc.csv"), index=False)
    print("Summary saved to", os.path.join(base_outdir, "summary_models_mc.csv"))
    print(df)

    plot_bar(df, "accuracy", os.path.join(charts_dir, "accuracy.png"), "Accuracy by Model (Multiclass)")
    plot_bar(df, "macro_f1", os.path.join(charts_dir, "macro_f1.png"), "Macro F1 by Model")
    plot_bar(df, "weighted_f1", os.path.join(charts_dir, "weighted_f1.png"), "Weighted F1 by Model")
    plot_bar(df, "roc_auc_micro", os.path.join(charts_dir, "roc_auc_micro.png"), "ROC AUC (micro) by Model")
    plot_bar(df, "roc_auc_macro", os.path.join(charts_dir, "roc_auc_macro.png"), "ROC AUC (macro) by Model")
    plot_bar(df, "pr_auc_micro", os.path.join(charts_dir, "pr_auc_micro.png"), "PR AUC (micro) by Model")
    plot_bar(df, "pr_auc_macro", os.path.join(charts_dir, "pr_auc_macro.png"), "PR AUC (macro) by Model")
    plot_bar(df, "total_size_mb", os.path.join(charts_dir, "total_size_mb.png"), "Model+Preproc Size (MB)")

    pct = per_class_table(args.models_dir, args.models)
    if not pct.empty:
        pct.to_csv(os.path.join(base_outdir, "per_class_report_merged.csv"), index=False)
        print("Per-class report merged saved to", os.path.join(base_outdir, "per_class_report_merged.csv"))
        for name in args.models:
            fcol = f"f1_{name}"
            if fcol in pct.columns:
                dd = pct[["class", fcol]].dropna()
                plt.figure(figsize=(8, max(4, 0.35*len(dd))))
                plt.barh(dd["class"], dd[fcol])
                plt.title(f"Per-class F1: {name}")
                plt.ylabel("Class"); plt.xlabel("F1")
                _savefig(os.path.join(charts_dir, f"per_class_f1_{name}.png"))

    if args.benchmark:
        bdf = benchmark_inference(args.models_dir, args.csv, args.models, y_col="type", sample_size=args.sample_size, num_runs=args.num_runs)
        
        for run_id in range(1, args.num_runs + 1):
            run_data = bdf[bdf["run_id"] == run_id]
            if not run_data.empty:
                run_filename = os.path.join(base_outdir, f"inference_benchmark_mc_{run_id}.csv")
                # save individual runs without the run_id column
                run_data_no_runid = run_data.drop(columns=["run_id"], errors="ignore")
                run_data_no_runid.to_csv(run_filename, index=False)
                print(f"Run {run_id} saved to {run_filename}")

        final_filename = os.path.join(base_outdir, "inference_benchmark_mc.csv")
        bdf.to_csv(final_filename, index=False)
        print(f"All benchmark data saved to {final_filename}")
        
        print(bdf)

    # save lgbm/xgb ratios
    raw_runs = bdf[bdf["run_id"] != "avg"].copy()
    raw_runs["total_ms_per_1k"] = raw_runs["total_ms_per_1k"].astype(float)
    pivot_df = raw_runs.pivot(index="run_id", columns="model", values="total_ms_per_1k")
    if "lgbm_mc" in pivot_df.columns and "xgb_mc" in pivot_df.columns:
        pivot_df["lgbm_xgb_ratio"] = pivot_df["lgbm_mc"] / pivot_df["xgb_mc"]
        ratio_path = os.path.join(base_outdir, "lgbm_xgb_ratios.csv")
        pivot_df.to_csv(ratio_path)

if __name__ == "__main__":
    main()
