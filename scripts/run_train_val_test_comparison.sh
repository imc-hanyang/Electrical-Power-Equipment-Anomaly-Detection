#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
CODE_DIR="$PKG_ROOT/engine/cladapter_code"
DATA_ROOT="${DATA_ROOT:-$PKG_ROOT/dataset}"
CSV_PATH="${CSV_PATH:-$PKG_ROOT/dataset/splits/train_val_test_second_setting.csv}"
RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/engine/runs/train_val_test_second_setting_20260529}"
LOG_DIR="$RUN_ROOT/logs"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "$LOG_DIR"

"$SCRIPT_DIR/make_group_train_val_test_split.py" \
  --input-csv "$PKG_ROOT/dataset/splits/second_split.csv" \
  --output-csv "$CSV_PATH"

run_logged() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  "$@" > "$LOG_DIR/${name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $name"
}

run_cladapter() {
  local log_name="$1"
  shift
  (
    cd "$CODE_DIR"
    run_logged "$log_name" env CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --standalone --nproc_per_node=1 train.py "$@"
  )
}

run_logged "resnet50_local" \
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}" python "$PROJECT_ROOT/engine/supervised_classifier/train.py" \
    --data-root "$DATA_ROOT/Final_Dataset" \
    --split-csv "$CSV_PATH" \
    --split-data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT/local_supervised" \
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

run_cladapter "linear_convnextb" \
  --model-mode conv \
  --finetune-mode linear \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --output-dir "$RUN_ROOT/linear_convnextb"

run_cladapter "linear_vitb" \
  --model-mode vit \
  --finetune-mode linear \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --output-dir "$RUN_ROOT/linear_vitb"

run_cladapter "resnet50_cla_stage1" \
  --model-mode res_xcep \
  --finetune-mode cla \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --output-dir "$RUN_ROOT/resnet50_cla_stage1"

run_cladapter "convnextb_cla_stage1" \
  --model-mode conv \
  --finetune-mode cla \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --output-dir "$RUN_ROOT/convnextb_cla_stage1"

run_cladapter "vitb_cla_stage1" \
  --model-mode vit \
  --finetune-mode cla \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --output-dir "$RUN_ROOT/vitb_cla_stage1"

run_cladapter "convnextb_cla_sft2" \
  --model-mode conv \
  --finetune-mode cla \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --finetune-ckpt "$RUN_ROOT/convnextb_cla_stage1/convnext_base.clip_laion2b_augreg_best.pth" \
  --norm clip \
  --output-dir "$RUN_ROOT/convnextb_cla_sft2"

run_cladapter "vitb_cla_sft2" \
  --model-mode vit \
  --finetune-mode cla \
  --image-size 224 \
  --csv-dir "$CSV_PATH" \
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
  --finetune-ckpt "$RUN_ROOT/vitb_cla_stage1/vit_base_patch16_clip_224.laion2b_best.pth" \
  --norm clip \
  --output-dir "$RUN_ROOT/vitb_cla_sft2"

"$SCRIPT_DIR/summarize_comparison_results.py" \
  --run-root "$RUN_ROOT" \
  --split-csv "$CSV_PATH" \
  --setting-name "Train/Validation/Test"

"$SCRIPT_DIR/apply_validation_thresholds.py" \
  --run-root "$RUN_ROOT" \
  --split-csv "$CSV_PATH" \
  --data-root "$DATA_ROOT" \
  --title "KEPCO Train/Validation/Test Results"

echo "All train/val/test comparison runs completed under $RUN_ROOT"
