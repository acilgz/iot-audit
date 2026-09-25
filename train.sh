#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1:-}" ]]; then
    echo "Usage: $0 <run_id>" >&2
    exit 1
fi

RUN_ID="$1"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run_id format" >&2
    exit 1
fi

RUN_DIR="runs/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: Run $1 already exists."
    exit 1
fi

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "ERROR: Dirty git working tree. Commit or stash your changes." >&2
    git status --short >&2
    exit 1
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"

CSV="data/train_test_network.csv"

mkdir -p "$RUN_DIR/logs"

run_logged() {
    local log_path="$1"
    shift
    "$@" 2>&1 | tee "$log_path"
}

printf '%s\n' "$SOURCE_COMMIT" > "$RUN_DIR/commit.txt"
printf '%s\n' "$SOURCE_STATUS" > "$RUN_DIR/git-status.txt"
python --version > "$RUN_DIR/python-version.txt"
python -m pip freeze > "$RUN_DIR/environment.txt"

python -m pip check

run_logged "$RUN_DIR/logs/pytest-before.log" \
    python -m pytest -q


# Binary
run_logged "$RUN_DIR/logs/prepare-binary.log" \
    python scripts/prepare_preprocessor.py \
    --csv "$CSV" \
    --outdir "$RUN_DIR/binary"

for MODEL in rf lgbm xgb logreg mlp; do
    run_logged "$RUN_DIR/logs/train-$MODEL.log" \
        python "scripts/train_${MODEL}.py" \
        --csv "$CSV" \
        --outdir "$RUN_DIR/binary"
done

run_logged "$RUN_DIR/logs/quantize-binary.log" \
    python scripts/quantize_model.py \
    --input_dir "$RUN_DIR/binary/models/mlp" \
    --csv "$CSV" \
    --calib_samples 1000

# Multiclass
run_logged "$RUN_DIR/logs/prepare-multiclass.log" \
    python scripts/prepare_preprocessor_mc.py \
    --csv "$CSV" \
    --outdir "$RUN_DIR/multiclass"

for MODEL in rf lgbm xgb logreg mlp; do
    run_logged "$RUN_DIR/logs/train-${MODEL}-mc.log" \
        python "scripts/train_mc_${MODEL}.py" \
        --csv "$CSV" \
        --outdir "$RUN_DIR/multiclass"
done

run_logged "$RUN_DIR/logs/quantize-multiclass.log" \
    python scripts/quantize_model_mc.py \
    --input_dir "$RUN_DIR/multiclass/models/mlp_mc" \
    --csv "$CSV" \
    --calib_samples 1000


# Tests
run_logged "$RUN_DIR/logs/scaler-regression.log" \
    python scripts/scaler_regression_test.py \
    --csv "$CSV" \
    --binary-dir "$RUN_DIR/binary" \
    --multiclass-dir "$RUN_DIR/multiclass"

run_logged "$RUN_DIR/logs/pytest-after.log" \
    python -m pytest -q

# EDA
run_logged "$RUN_DIR/logs/analyze-dataset.log" \
python scripts/analyze_dataset.py \
    --csv "$CSV" \
    --outdir "$RUN_DIR/eda"

run_logged "$RUN_DIR/logs/visualize-dataset.log" \
python scripts/visualize_dataset.py \
    --csv "$CSV" \
    --outdir "$RUN_DIR/eda/figures"

touch "$RUN_DIR/TRAINING_DONE"
echo "Training completed successfully."