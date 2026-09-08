from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_ratios(systems: list[str], benchmark_dir: str, subtask: str) -> list[float]:
    ratios = []
    for sys in systems:
        csv_path = os.path.join(benchmark_dir, sys, subtask, "lgbm_xgb_ratios.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "lgbm_xgb_ratio" in df.columns:
                ratios.extend(df["lgbm_xgb_ratio"].dropna().tolist())
            else:
                print(f"Warning: Missing 'lgbm_xgb_ratio' in {csv_path}")
        else:
            print(f"Warning: File not found: {csv_path}")
    return ratios

def save_chart(
    aarch64_ratios: list[float], 
    x64_ratios: list[float], 
    aarch64_systems: list[str], 
    x64_systems: list[str], 
    subtask: str, 
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 6))

    aarch64_label = "aarch64\n(" + ", ".join(aarch64_systems) + ")"
    x64_label = "x64\n(" + ", ".join(x64_systems) + ")"

    box = plt.boxplot(
        [aarch64_ratios, x64_ratios],
        labels=[aarch64_label, x64_label],
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )

    colors = ["#6baed6", "#fc9272"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    np.random.seed(42)
    for i, data in enumerate([aarch64_ratios, x64_ratios], start=1):
        x = np.random.normal(i, 0.04, size=len(data))
        plt.plot(x, data, "ro", color="black", alpha=0.6, markersize=5)

    plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Parity (Ratio = 1.0)")
    plt.title(f"LightGBM / XGBoost Ratio Distribution ({subtask.capitalize()})", fontsize=14, fontweight="bold")
    plt.xlabel("System Category", fontsize=12)
    plt.ylabel("LGBM / XGB Ratio", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    chart_filename = f"lgbm_xgb_ratio_distribution_{subtask}.png"
    chart_path = os.path.join(output_dir, chart_filename)
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {chart_path}")

def run_mann_whitney_analysis(benchmark_dir: str, aarch64_systems: list[str], x64_systems: list[str]):
    subtasks = ["binary", "multiclass"]
    charts_dir = os.path.join(benchmark_dir, "charts")

    for subtask in subtasks:
        aarch64_ratios = load_ratios(aarch64_systems, benchmark_dir, subtask)
        x64_ratios = load_ratios(x64_systems, benchmark_dir, subtask)

        if not aarch64_ratios or not x64_ratios:
            print(f"Skipped {subtask}: not enough data (Legacy: {len(aarch64_ratios)}, Modern: {len(x64_ratios)})")
            continue

        save_chart(aarch64_ratios, x64_ratios, aarch64_systems, x64_systems, subtask, charts_dir)

        stat, p_val = stats.mannwhitneyu(aarch64_ratios, x64_ratios, alternative="greater")

        n1 = len(aarch64_ratios)
        n2 = len(x64_ratios)
        
        r_rank_biserial = (2.0 * stat) / (n1 * n2) - 1.0 if (n1 * n2) > 0 else None

        mean_aarch64 = float(pd.Series(aarch64_ratios).mean())
        std_aarch64 = float(pd.Series(aarch64_ratios).std())
        median_aarch64 = float(pd.Series(aarch64_ratios).median())

        mean_x64 = float(pd.Series(x64_ratios).mean())
        std_x64 = float(pd.Series(x64_ratios).std())
        median_x64 = float(pd.Series(x64_ratios).median())

        result_df = pd.DataFrame([{
            "subtask": subtask,
            "u_statistic": stat,
            "p_value": p_val,
            "statistically_significant_0_05": bool(p_val < 0.05),
            "rank_biserial_correlation": r_rank_biserial,
            "n_aarch64_samples": n1,
            "n_x64_samples": n2,
            "aarch64_mean_ratio": mean_aarch64,
            "aarch64_std_ratio": std_aarch64,
            "aarch64_median_ratio": median_aarch64,
            "x64_mean_ratio": mean_x64,
            "x64_std_ratio": std_x64,
            "x64_median_ratio": median_x64,
            "aarch64_systems": ",".join(aarch64_systems),
            "x64_systems": ",".join(x64_systems)
        }])

        print(f"{subtask.upper()}:")
        print(f"aarch64 (n={n1}) Ratio Avg: {mean_aarch64:.4f} ± {std_aarch64:.4f} (Median: {median_aarch64:.4f})")
        print(f"x64 (n={n2}) Ratio Avg: {mean_x64:.4f} ± {std_x64:.4f} (Median: {median_x64:.4f})")
        print(f"Mann-Whitney U: {stat}, p-value: {p_val:.6e}, Rank-Biserial: {r_rank_biserial:.4f}\n")

        suffix = "_mc" if subtask == "multiclass" else ""
        out_filename = f"lgbm_xgb_mann-whitney{suffix}.csv"
        out_path = os.path.join(benchmark_dir, out_filename)
        result_df.to_csv(out_path, index=False)
        print(f"{subtask} lgbm/xgb Mann-Whitney saved to: {out_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="benchmark")
    parser.add_argument("--x64", default=",corei7-3770,corei5-7200U,ryzen7_7700")
    parser.add_argument("--aarch64", default="bcm2712,apple_m1,apple_m5")

    args = parser.parse_args()

    aarch64_systems = [s.strip() for s in args.aarch64.split(",") if s.strip()]
    x64_systems = [s.strip() for s in args.x64.split(",") if s.strip()]

    run_mann_whitney_analysis(args.benchmark, aarch64_systems, x64_systems)

if __name__ == "__main__":
    main()
