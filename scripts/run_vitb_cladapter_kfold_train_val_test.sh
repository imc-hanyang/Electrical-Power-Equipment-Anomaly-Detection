#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODE_DIR="$PKG_ROOT/engine/cladapter_code"
DATA_ROOT="${DATA_ROOT:-$PKG_ROOT/dataset}"
DATASET_NAME="${DATASET_NAME:-Final_Dataset}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_DIR="${SPLIT_DIR:-$PKG_ROOT/dataset/splits/kfold${N_SPLITS}_train_val_test_second_setting}"
RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/engine/runs/kfold${N_SPLITS}_vitb_cladapter_second_setting_20260530}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPLIT_SOURCE_CSV="${SPLIT_SOURCE_CSV:-$PKG_ROOT/dataset/splits/second_split.csv}"
BUILD_SPLITS="${BUILD_SPLITS:-1}"
if [[ -z "${FOLDS:-}" ]]; then
  FOLDS="$(seq 0 $((N_SPLITS - 1)))"
fi

IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
mkdir -p "$RUN_ROOT/logs"

run_logged() {
  local log_name="$1"
  shift
  echo "[$(date '+%F %T')] START $log_name"
  "$@" > "$RUN_ROOT/logs/${log_name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $log_name"
}

run_cladapter() {
  local log_name="$1"
  local gpu_id="$2"
  shift 2
  (
    cd "$CODE_DIR"
    run_logged "$log_name" env CUDA_VISIBLE_DEVICES="$gpu_id" torchrun --standalone --nproc_per_node=1 train.py "$@"
  )
}

run_fold() {
  local fold="$1"
  local gpu_id="$2"
  local csv_path="$SPLIT_DIR/fold_${fold}.csv"
  local fold_root="$RUN_ROOT/fold_${fold}"
  mkdir -p "$fold_root"

  if [[ ! -f "$fold_root/vitb_cla_stage1/metrics.json" ]]; then
    run_cladapter "fold${fold}_vitb_cla_stage1" "$gpu_id" \
      --model-mode vit \
      --finetune-mode cla \
      --image-size 224 \
      --csv-dir "$csv_path" \
      --config-name config_clip_vit \
      --data-root "$DATA_ROOT" \
      --gpu_id 0 \
      --batch-size 16 \
      --num-workers "$NUM_WORKERS" \
      --init-lr 1e-4 \
      --weight_decay 1e-4 \
      --optimizer AdamW \
      --epochs "$EPOCHS" \
      --warmup_epochs 2 \
      --nbatch_log 300 \
      --selection-metric acc \
      --backbone-name vit_base_patch16_clip_224.laion2b \
      --backbone-out-dim 768 \
      --backbone-num-patch 196 \
      --norm clip \
      --output-dir "$fold_root/vitb_cla_stage1"
  else
    echo "[$(date '+%F %T')] SKIP fold${fold}_vitb_cla_stage1"
  fi

  if [[ ! -f "$fold_root/vitb_cla_sft2/metrics.json" ]]; then
    run_cladapter "fold${fold}_vitb_cla_sft2" "$gpu_id" \
      --model-mode vit \
      --finetune-mode cla \
      --image-size 224 \
      --csv-dir "$csv_path" \
      --config-name config_clip_vit \
      --data-root "$DATA_ROOT" \
      --gpu_id 0 \
      --batch-size 16 \
      --num-workers "$NUM_WORKERS" \
      --init-lr 1e-4 \
      --weight_decay 1e-4 \
      --optimizer AdamW \
      --epochs "$EPOCHS" \
      --warmup_epochs 2 \
      --nbatch_log 300 \
      --selection-metric acc \
      --backbone-name vit_base_patch16_clip_224.laion2b \
      --backbone-out-dim 768 \
      --backbone-num-patch 196 \
      --finetune-ckpt "$fold_root/vitb_cla_stage1/vit_base_patch16_clip_224.laion2b_best.pth" \
      --norm clip \
      --output-dir "$fold_root/vitb_cla_sft2"
  else
    echo "[$(date '+%F %T')] SKIP fold${fold}_vitb_cla_sft2"
  fi
}

if [[ "$BUILD_SPLITS" == "1" ]]; then
  "$SCRIPT_DIR/make_kfold_group_train_val_test_splits.py" \
    --input-csv "$SPLIT_SOURCE_CSV" \
    --output-dir "$SPLIT_DIR" \
    --n-splits "$N_SPLITS"
fi

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

"$SCRIPT_DIR/apply_validation_thresholds.py" \
  --run-root "$RUN_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --data-root "$DATA_ROOT" \
  --models "ViT-B + CLAdapter" \
  --title "KEPCO ${N_SPLITS}-Fold ViT-B + CLAdapter Train/Validation/Test Results"

echo "ViT-B + CLAdapter ${N_SPLITS}-fold run completed under $RUN_ROOT"
