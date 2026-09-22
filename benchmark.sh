#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: $0 <run_id> <device_id>" >&2
    exit 1
fi

RUN_DIR="runs/$1"
CSV="data/train_test_network.csv"

DEVICE_ID="$2"

BENCH_DIR="benchmark/$1/$DEVICE_ID"
LOG_DIR="$RUN_DIR/benchmark-logs/$DEVICE_ID"

mkdir -p "$LOG_DIR"

# Test environment
python -m pip check
python -m pytest -q 2>&1 | tee "$LOG_DIR/pytest.log"

git rev-parse HEAD > "$LOG_DIR/commit.txt"
git status --short > "$LOG_DIR/git-status.txt"
python --version > "$LOG_DIR/python-version.txt"
python -m pip freeze > "$LOG_DIR/environment.txt"

# Benchmark binary
python scripts/compare_models.py \
    --csv "$CSV" \
    --models-dir "$RUN_DIR/binary" \
    --outdir "$BENCH_DIR/binary" \
    --models rf lgbm xgb logreg mlp mlp_int8 \
    --benchmark \
    --device-id "$DEVICE_ID" \
    --sample_size 10000 \
    --num_runs 5 \
    2>&1 | tee "$LOG_DIR/binary.log"

# Benchmark multiclass
python scripts/compare_models_mc.py \
    --csv "$CSV" \
    --models-dir "$RUN_DIR/multiclass" \
    --outdir "$BENCH_DIR/multiclass" \
    --models rf_mc lgbm_mc xgb_mc logreg_mc mlp_mc mlp_mc_int8 \
    --benchmark \
    --device-id "$DEVICE_ID" \
    --sample_size 10000 \
    --num_runs 5 \
    2>&1 | tee "$LOG_DIR/multiclass.log"