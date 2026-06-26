#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODE_DIR="$PKG_ROOT/engine/cladapter_code"
DATA_ROOT="$PKG_ROOT/dataset"
CSV_PATH="$PKG_ROOT/dataset/splits/first_split.csv"
OUT_DIR="$PKG_ROOT/engine/runs/stage1_vitb_cla"

mkdir -p "$OUT_DIR"
cd "$CODE_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun --standalone --nproc_per_node=1 train.py \
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
  --epochs 100 \
  --warmup_epochs 2 \
  --nbatch_log 300 \
  --no-validation \
  --selection-metric acc \
  --backbone-name vit_base_patch16_clip_224.laion2b \
  --backbone-out-dim 768 \
  --backbone-num-patch 196 \
  --norm clip \
  --output-dir "$OUT_DIR"
