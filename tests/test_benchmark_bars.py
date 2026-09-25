import pandas as pd

from scripts.benchmark_bars import load_all_benchmarks

def test_aggregate_sd_is_not_divided_by_replica_count(tmp_path):
    task_dir = tmp_path / "001" / "apple_m1" / "multiclass"
    task_dir.mkdir(parents=True)
    rows = [
        {"model": "lgbm_mc", "total_ms_per_1k": 1.0 + i / 10,
         "predict_ms_per_1k": 1.0 + i / 10, "transform_ms_per_1k": 0.0,
         "n_samples": 10000, "run_id": str(i), "device_id": "apple_m1"}
        for i in range(1, 6)
    ]
    rows.append({
        "model": "lgbm_mc", "total_ms_per_1k": "1.300000 ± 0.943000",
        "predict_ms_per_1k": "1.300000 ± 0.943000",
        "transform_ms_per_1k": "0.000000 ± 0.000000", "n_samples": 10000,
        "run_id": "avg", "device_id": "apple_m1",
    })
    pd.DataFrame(rows).to_csv(task_dir / "inference_benchmark_mc.csv", index=False)
    for i in range(1, 6):
        pd.DataFrame(rows[:i]).to_csv(task_dir / f"inference_benchmark_mc_{i}.csv", index=False)

    result = load_all_benchmarks(tmp_path)
    assert len(result) == 1
    assert result.iloc[0]["total_ms_per_1k_mean"] == 1.3
    assert result.iloc[0]["total_ms_per_1k_std"] == 0.943
