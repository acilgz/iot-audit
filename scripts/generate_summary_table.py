from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

METRICS = [
    ("accuracy", "Accuracy"),
    ("f1_pos", "F1-Pos"),
    ("f1_neg", "F1-Neg"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("fp", "FP"),
    ("fn", "FN"),
    ("model_size_mb", "Model Size (MB)"),
    ("preproc_size_mb", "Preproc Size (MB)"),
    ("total_size_mb", "Total Size (MB)"),
]

DEVICE_LABELS = {
    "apple_m1": "Apple M1",
    "apple_m5": "Apple M5",
    "bcm2712": "BCM2712",
    "corei5_7200u": "Core i5-7200U",
    "corei7_3770": "Core i7-3770",
    "ryzen7_7700": "Ryzen 7 7700",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def device_label(device: str) -> str:
    return DEVICE_LABELS.get(
        device.lower(),
        device.replace("_", " ").replace("-", " ").title(),
    )

def parse_mean_sd(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None

    match = re.fullmatch(
        r"\s*([-+\d.eE]+)(?:\s*±\s*([-+\d.eE]+))?\s*",
        value,
    )
    if not match:
        return None

    return float(match.group(1)), float(match.group(2) or 0.0)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_id",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmark"),
    )
    args = parser.parse_args()

    run_dir = args.benchmark_root / str(args.run_id)
    if not run_dir.is_dir():
        parser.error(f"no such run: {run_dir}")

    device_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    if not device_dirs:
        parser.error(f"no device folders found in {run_dir}")

    quality: dict[str, dict[str, str]] = {}
    quality_source: Path | None = None
    latency_by_device: dict[str, dict[str, tuple[float, float]]] = {}
    models: list[str] = []

    for device_dir in device_dirs:
        task_dir = device_dir / "binary"
        if not task_dir.is_dir():
            continue

        summary_path = task_dir / "summary" / "summary_models.csv"
        if not summary_path.is_file():
            summary_path = task_dir / "summary_models.csv"

        if summary_path.is_file():
            rows = read_csv(summary_path)

            if quality_source is None:
                quality_source = summary_path
                for row in rows:
                    model = row.get("model", "")
                    if model:
                        quality[model] = row
                        models.append(model)
            else:
                first_rows = {
                    row.get("model"): row
                    for row in read_csv(quality_source)
                }
                current_rows = {
                    row.get("model"): row
                    for row in rows
                }

                for model in set(first_rows) & set(current_rows):
                    for key, _ in METRICS:
                        if first_rows[model].get(key) != current_rows[model].get(key):
                            parser.error(
                                f"metrics for {model} do not match, column {key}: "
                                f"{summary_path}"
                            )

        inference_path = task_dir / "inference_benchmark.csv"
        if not inference_path.is_file():
            continue

        avg_rows = [
            row
            for row in read_csv(inference_path)
            if row.get("run_id", "").lower() == "avg"
        ]

        per_model: dict[str, tuple[float, float]] = {}
        for row in avg_rows:
            parsed = parse_mean_sd(row.get("total_ms_per_1k"))
            model = row.get("model", "")
            if parsed and model:
                per_model[model] = parsed

        if per_model:
            latency_by_device[device_dir.name] = per_model

    if not quality:
        parser.error(f"no summary found in {run_dir}")

    required = {
        "accuracy",
        "f1_pos",
        "f1_neg",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
        "model_size_mb",
        "preproc_size_mb",
        "total_size_mb",
    }
    missing = required - set(quality[models[0]])
    if missing:
        parser.error(
            "missing columns in summary: "
            + ", ".join(sorted(missing))
        )

    devices = list(latency_by_device)
    headers = ["Model"] + [label for _, label in METRICS]
    headers += [
        f"{device_label(device)} (1)"
        for device in devices
    ]

    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for model in models:
        metrics = quality[model]
        cells = [model]

        for key, _ in METRICS:
            value = metrics.get(key, "")
            try:
                number = float(value)
                if key in {
                    "accuracy",
                    "f1_pos",
                    "f1_neg",
                    "roc_auc",
                    "pr_auc",
                }:
                    value = f"{number:.6f}"
                else:
                    value = f"{number:g}"
            except (TypeError, ValueError):
                pass
            cells.append(value)

        for device in devices:
            result = latency_by_device[device].get(model)
            if result is None:
                cells.append("—")
            else:
                mean, sd = result
                cells.append(f"{mean:.3f} ± {sd:.3f}")

        lines.append("|" + "|".join(cells) + "|")

    lines.append("")
    lines.append("(1) ms/1k, mean ± std. deviation")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())