#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PKG_ROOT/dataset}"
DATASET_NAME="${DATASET_NAME:-Final_Dataset}"
SPLIT_DIR="${SPLIT_DIR:-$PKG_ROOT/dataset/splits/kfold10_train_val_test_second_setting}"
RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/engine/runs/no_etc_kfold10_baselines_20260530}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
EFFICIENTNET_EPOCHS="${EFFICIENTNET_EPOCHS:-100}"
DIFFERNET_EPOCHS="${DIFFERNET_EPOCHS:-100}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
mkdir -p "$RUN_ROOT/logs"

run_logged() {
  local log_name="$1"
  shift
  echo "[$(date '+%F %T')] START $log_name"
  "$@" > "$RUN_ROOT/logs/${log_name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $log_name"
}

run_fold() {
  local fold="$1"
  local gpu_id="$2"
  local csv_path="$SPLIT_DIR/fold_${fold}.csv"
  local fold_root="$RUN_ROOT/fold_${fold}"
  mkdir -p "$fold_root"

  if [[ ! -f "$fold_root/local_supervised/efficientnet_b0/metrics.json" ]]; then
    run_logged "fold${fold}_efficientnet_b0" \
      env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}" python "$PROJECT_ROOT/engine/supervised_classifier/train.py" \
        --data-root "$DATA_ROOT/$DATASET_NAME" \
        --split-csv "$csv_path" \
        --split-data-root "$DATA_ROOT" \
        --output-dir "$fold_root/local_supervised" \
        --model efficientnet_b0 \
        --img-size 224 \
        --resize-mode letterbox \
        --epochs "$EFFICIENTNET_EPOCHS" \
        --batch-size 16 \
        --lr 1e-4 \
        --backbone-lr 1e-4 \
        --weight-decay 1e-3 \
        --split-strategy train_val_test \
        --device cuda \
        --amp \
        --hide-progress
  else
    echo "[$(date '+%F %T')] SKIP fold${fold}_efficientnet_b0"
  fi

  if ! find "$fold_root/patchcore" -mindepth 2 -maxdepth 2 -name metrics.json -print -quit 2>/dev/null | grep -q .; then
    run_logged "fold${fold}_patchcore" \
      env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}" python -m normal_only.patchcore \
        --data-root "$DATA_ROOT/$DATASET_NAME" \
        --split-csv "$csv_path" \
        --split-data-root "$DATA_ROOT" \
        --output-dir "$fold_root/patchcore" \
        --backbone wide_resnet50_2 \
        --img-size 224 \
        --resize-mode letterbox \
        --split-strategy train_val_test \
        --batch-size 8 \
        --num-workers 4 \
        --max-memory-patches 10000 \
        --score-top-k-frac 0.01 \
        --device cuda \
        --hide-progress
  else
    echo "[$(date '+%F %T')] SKIP fold${fold}_patchcore"
  fi

  if [[ ! -f "$fold_root/differnet/none/metrics.json" ]]; then
    run_logged "fold${fold}_differnet_none" \
      env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}" python -m attent_differnet.train \
        --data-root "$DATA_ROOT/$DATASET_NAME" \
        --split-csv "$csv_path" \
        --split-data-root "$DATA_ROOT" \
        --output-dir "$fold_root/differnet" \
        --attention none \
        --img-size 448 \
        --resize-mode letterbox \
        --epochs "$DIFFERNET_EPOCHS" \
        --sub-epochs 1 \
        --batch-size 8 \
        --batch-size-test 4 \
        --n-transforms 4 \
        --n-transforms-test 16 \
        --num-workers 4 \
        --device cuda \
        --hide-progress
  else
    echo "[$(date '+%F %T')] SKIP fold${fold}_differnet_none"
  fi
}

pids=()
fold_index=0
for fold in $FOLDS; do
  gpu_index=$(( fold_index % ${#GPU_IDS_ARR[@]} ))
  gpu_id="${GPU_IDS_ARR[$gpu_index]}"
  run_fold "$fold" "$gpu_id" > "$RUN_ROOT/logs/fold${fold}.driver.log" 2>&1 &
  pids+=("$!")
  fold_index=$((fold_index + 1))
  if (( ${#pids[@]} >= ${#GPU_IDS_ARR[@]} )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$SCRIPT_DIR/summarize_no_etc_baseline_kfold.py" \
  --run-root "$RUN_ROOT" \
  --split-dir "$SPLIT_DIR"

echo "No-ETC 10-fold baseline run completed under $RUN_ROOT"
