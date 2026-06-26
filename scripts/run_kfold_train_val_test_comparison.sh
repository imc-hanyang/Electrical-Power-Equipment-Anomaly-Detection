#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
CODE_DIR="$PKG_ROOT/engine/cladapter_code"
DATA_ROOT="${DATA_ROOT:-$PKG_ROOT/dataset}"
DATASET_NAME="${DATASET_NAME:-Final_Dataset}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_DIR="${SPLIT_DIR:-$PKG_ROOT/dataset/splits/kfold${N_SPLITS}_train_val_test_second_setting}"
RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/engine/runs/kfold${N_SPLITS}_train_val_test_second_setting_20260529}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
EPOCHS="${EPOCHS:-100}"
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

  run_logged "fold${fold}_resnet50_local" \
    env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}" python "$PROJECT_ROOT/engine/supervised_classifier/train.py" \
      --data-root "$DATA_ROOT/$DATASET_NAME" \
      --split-csv "$csv_path" \
      --split-data-root "$DATA_ROOT" \
      --output-dir "$fold_root/local_supervised" \
      --model resnet50 \
      --img-size 224 \
      --resize-mode letterbox \
      --epochs "$EPOCHS" \
      --batch-size 16 \
      --lr 1e-4 \
      --backbone-lr 1e-4 \
      --weight-decay 1e-3 \
      --split-strategy train_val_test \
      --device cuda \
      --amp \
      --hide-progress

  run_cladapter "fold${fold}_linear_convnextb" "$gpu_id" \
    --model-mode conv \
    --finetune-mode linear \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_convnext \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
    --init-lr 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --epochs "$EPOCHS" \
    --warmup_epochs 2 \
    --nbatch_log 300 \
    --selection-metric acc \
    --backbone-name convnext_base.clip_laion2b_augreg \
    --backbone-out-dim 1024 \
    --backbone-num-patch 49 \
    --norm clip \
    --output-dir "$fold_root/linear_convnextb"

  run_cladapter "fold${fold}_linear_vitb" "$gpu_id" \
    --model-mode vit \
    --finetune-mode linear \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_vit \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
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
    --output-dir "$fold_root/linear_vitb"

  run_cladapter "fold${fold}_resnet50_cla_stage1" "$gpu_id" \
    --model-mode res_xcep \
    --finetune-mode cla \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_convnext \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
    --init-lr 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --epochs "$EPOCHS" \
    --warmup_epochs 2 \
    --nbatch_log 300 \
    --selection-metric acc \
    --backbone-name resnet50 \
    --backbone-out-dim 2048 \
    --backbone-num-patch 49 \
    --norm imagenet \
    --output-dir "$fold_root/resnet50_cla_stage1"

  run_cladapter "fold${fold}_convnextb_cla_stage1" "$gpu_id" \
    --model-mode conv \
    --finetune-mode cla \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_convnext \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
    --init-lr 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --epochs "$EPOCHS" \
    --warmup_epochs 2 \
    --nbatch_log 300 \
    --selection-metric acc \
    --backbone-name convnext_base.clip_laion2b_augreg \
    --backbone-out-dim 1024 \
    --backbone-num-patch 49 \
    --norm clip \
    --output-dir "$fold_root/convnextb_cla_stage1"

  run_cladapter "fold${fold}_vitb_cla_stage1" "$gpu_id" \
    --model-mode vit \
    --finetune-mode cla \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_vit \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
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

  run_cladapter "fold${fold}_convnextb_cla_sft2" "$gpu_id" \
    --model-mode conv \
    --finetune-mode cla \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_convnext \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
    --init-lr 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --epochs "$EPOCHS" \
    --warmup_epochs 2 \
    --nbatch_log 300 \
    --selection-metric acc \
    --backbone-name convnext_base.clip_laion2b_augreg \
    --backbone-out-dim 1024 \
    --backbone-num-patch 49 \
    --finetune-ckpt "$fold_root/convnextb_cla_stage1/convnext_base.clip_laion2b_augreg_best.pth" \
    --norm clip \
    --output-dir "$fold_root/convnextb_cla_sft2"

  run_cladapter "fold${fold}_vitb_cla_sft2" "$gpu_id" \
    --model-mode vit \
    --finetune-mode cla \
    --image-size 224 \
    --csv-dir "$csv_path" \
    --config-name config_clip_vit \
    --data-root "$DATA_ROOT" \
    --gpu_id 0 \
    --batch-size 16 \
    --num-workers 4 \
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
  --title "KEPCO ${N_SPLITS}-Fold Train/Validation/Test Results"

echo "All ${N_SPLITS}-fold train/val/test comparison runs completed under $RUN_ROOT"
