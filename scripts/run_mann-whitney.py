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
    legacy_ratios: list[float], 
    modern_ratios: list[float], 
    legacy_systems: list[str], 
    modern_systems: list[str], 
    subtask: str, 
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 6))

    legacy_label = "Legacy\n(" + ", ".join(legacy_systems) + ")"
    modern_label = "Modern\n(" + ", ".join(modern_systems) + ")"

    box = plt.boxplot(
        [legacy_ratios, modern_ratios],
        labels=[legacy_label, modern_label],
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )

    colors = ["#6baed6", "#fc9272"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    np.random.seed(42)
    for i, data in enumerate([legacy_ratios, modern_ratios], start=1):
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

def run_mann_whitney_analysis(benchmark_dir: str, legacy_systems: list[str], modern_systems: list[str]):
    subtasks = ["binary", "multiclass"]
    charts_dir = os.path.join(benchmark_dir, "charts")

    for subtask in subtasks:
        legacy_ratios = load_ratios(legacy_systems, benchmark_dir, subtask)
        modern_ratios = load_ratios(modern_systems, benchmark_dir, subtask)

        if not legacy_ratios or not modern_ratios:
            print(f"Skipped {subtask}: not enough data (Legacy: {len(legacy_ratios)}, Modern: {len(modern_ratios)})")
            continue

        save_chart(legacy_ratios, modern_ratios, legacy_systems, modern_systems, subtask, charts_dir)

        stat, p_val = stats.mannwhitneyu(legacy_ratios, modern_ratios, alternative="greater")

        n1 = len(legacy_ratios)
        n2 = len(modern_ratios)
        
        r_rank_biserial = (2.0 * stat) / (n1 * n2) - 1.0 if (n1 * n2) > 0 else None

        mean_legacy = float(pd.Series(legacy_ratios).mean())
        std_legacy = float(pd.Series(legacy_ratios).std())
        median_legacy = float(pd.Series(legacy_ratios).median())

        mean_modern = float(pd.Series(modern_ratios).mean())
        std_modern = float(pd.Series(modern_ratios).std())
        median_modern = float(pd.Series(modern_ratios).median())

        result_df = pd.DataFrame([{
            "subtask": subtask,
            "u_statistic": stat,
            "p_value": p_val,
            "statistically_significant_0_05": bool(p_val < 0.05),
            "rank_biserial_correlation": r_rank_biserial,
            "n_legacy_samples": n1,
            "n_modern_samples": n2,
            "legacy_mean_ratio": mean_legacy,
            "legacy_std_ratio": std_legacy,
            "legacy_median_ratio": median_legacy,
            "modern_mean_ratio": mean_modern,
            "modern_std_ratio": std_modern,
            "modern_median_ratio": median_modern,
            "legacy_systems": ",".join(legacy_systems),
            "modern_systems": ",".join(modern_systems)
        }])

        print(f"{subtask.upper()}:")
        print(f"Legacy (n={n1}) Ratio Avg: {mean_legacy:.4f} ± {std_legacy:.4f} (Median: {median_legacy:.4f})")
        print(f"Modern (n={n2}) Ratio Avg: {mean_modern:.4f} ± {std_modern:.4f} (Median: {median_modern:.4f})")
        print(f"Mann-Whitney U: {stat}, p-value: {p_val:.6e}, Rank-Biserial: {r_rank_biserial:.4f}\n")

        suffix = "_mc" if subtask == "multiclass" else ""
        out_filename = f"lgbm_xgb_mann-whitney{suffix}.csv"
        out_path = os.path.join(benchmark_dir, out_filename)
        result_df.to_csv(out_path, index=False)
        print(f"{subtask} lgbm/xgb Mann-Whitney saved to: {out_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="benchmark")
    parser.add_argument("--legacy", default="apple_m1,bcm2712,corei7-3770,corei5-7200U")
    parser.add_argument("--modern", default="apple_m5,ryzen7_7700")

    args = parser.parse_args()

    legacy_systems = [s.strip() for s in args.legacy.split(",") if s.strip()]
    modern_systems = [s.strip() for s in args.modern.split(",") if s.strip()]

    run_mann_whitney_analysis(args.benchmark, legacy_systems, modern_systems)

if __name__ == "__main__":
    main()
