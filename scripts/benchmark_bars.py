from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def clean_platform_name(folder_name: str) -> str:
    name = re.sub(r"[_\-]+", " ", folder_name).strip()
    return name.title() if not name.isupper() else name


def parse_val_and_std(val: float | str) -> tuple[float, float]:
    if pd.isna(val):
        return np.nan, 0.0
    if isinstance(val, (int, float)):
        return float(val), 0.0
    val_str = str(val)
    if "±" in val_str:
        parts = val_str.split("±")
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(val_str), 0.0
    except ValueError:
        return np.nan, 0.0


def load_all_benchmarks(input_dir: Path) -> pd.DataFrame:
    rows = []
    csv_files = sorted(list(set(input_dir.glob("*/*/*.csv"))))

    for file_path in csv_files:
        rel_path = file_path.relative_to(input_dir)
        parts = rel_path.parts
        if len(parts) < 3:
            continue

        platform_raw, task_raw = parts[0], parts[1].lower()
        if task_raw not in ["binary", "multiclass"]:
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if "model" not in df.columns:
            continue

        if "run_id" in df.columns:
            df_avg = df[df["run_id"].astype(str) == "avg"].copy()
            if df_avg.empty:
                df_avg = df[df["run_id"].astype(str) != "avg"].groupby("model", as_index=False).mean(numeric_only=True)
        else:
            df_avg = df.copy()

        processor_name = clean_platform_name(platform_raw)

        for _, row in df_avg.iterrows():
            record = {
                "processor": processor_name,
                "task": task_raw,
                "model": row["model"],
            }
            for metric in ["total_ms_per_1k", "predict_ms_per_1k", "transform_ms_per_1k"]:
                if metric in row:
                    mean_val, std_val = parse_val_and_std(row[metric])
                    record[f"{metric}_mean"] = mean_val
                    record[f"{metric}_std"] = std_val

            rows.append(record)

    return pd.DataFrame(rows)


def plot_task_metric(
    df_task: pd.DataFrame,
    task_name: str,
    metric: str,
    output_path: Path,
    style: str = "overlay",  # overlay, yerr, text
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if df_task.empty or mean_col not in df_task.columns:
        return

    pivot_mean = df_task.pivot_table(index="model", columns="processor", values=mean_col, aggfunc="mean")
    pivot_std = df_task.pivot_table(index="model", columns="processor", values=std_col, aggfunc="mean").fillna(0)

    if pivot_mean.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    if style == "yerr":
        pivot_mean.plot(kind="bar", width=0.8, ax=ax, yerr=pivot_std, capsize=4, ecolor="black", error_kw={"alpha": 0.7})
    else:
        pivot_mean.plot(kind="bar", width=0.8, ax=ax)

    if style == "overlay":
        for container_idx, container in enumerate(ax.containers):
            for bar_idx, bar in enumerate(container):
                m = pivot_mean.iloc[bar_idx, container_idx]
                s = pivot_std.iloc[bar_idx, container_idx]
                
                if pd.notna(m) and s > 0:
                    x = bar.get_x()
                    w = bar.get_width()
                    y_start = max(0, m - s)
                    height = (m + s) - y_start
                    
                    rect = patches.Rectangle(
                        (x, y_start),
                        w,
                        height,
                        linewidth=1.2,
                        edgecolor="black",
                        linestyle="--",
                        facecolor="black",
                        alpha=0.35,
                        zorder=3
                    )
                    ax.add_patch(rect)

    for container_idx, container in enumerate(ax.containers):
        for bar_idx, bar in enumerate(container):
            m = pivot_mean.iloc[bar_idx, container_idx]
            s = pivot_std.iloc[bar_idx, container_idx]
            if pd.isna(m):
                continue
            
            label = f"{m:.1f}\n±{s:.2f}" if s > 0 else f"{m:.1f}"
            y_pos = m + s if style in ["overlay", "yerr"] else m
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + (pivot_mean.max().max() * 0.015),
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    metric_labels = {
        "total_ms_per_1k": "Total (ms / 1000 samples)",
        "predict_ms_per_1k": "Predict (ms / 1000 samples)",
        "transform_ms_per_1k": "Transform (ms / 1000 samples)",
    }
    
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel(metric_labels.get(metric, metric), fontweight="bold")
    ax.set_title(f"Comparing {metric} - {task_name.capitalize()}", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Platform", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/charts"))
    parser.add_argument("--style", choices=["overlay", "yerr", "text"], default="overlay")
    args = parser.parse_args()

    data = load_all_benchmarks(args.input_dir)
    if data.empty:
        return

    for task in data["task"].unique():
        df_task = data[data["task"] == task]
        for metric in ["total_ms_per_1k", "predict_ms_per_1k", "transform_ms_per_1k"]:
            output_filename = args.output_dir / f"{task}_{metric}.png"
            plot_task_metric(df_task, task, metric, output_filename, style=args.style)


if __name__ == "__main__":
    main()