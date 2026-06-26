#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD="${1:-0}"
SPLIT="${2:-test}"

python "$ROOT_DIR/engine/scripts/infer_vitb_cladapter_sft.py" \
  --checkpoint "${CHECKPOINT:-$ROOT_DIR/engine/runs/kfold10_vitb_cladapter_second_setting/fold_${FOLD}/vitb_cla_sft2/vit_base_patch16_clip_224.laion2b_best.pth}" \
  --csv "${CSV:-$ROOT_DIR/dataset/splits/kfold10_train_val_test_second_setting/fold_${FOLD}.csv}" \
  --split "$SPLIT" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/dataset}" \
  --output "${OUTPUT:-$ROOT_DIR/engine/predictions/fold_${FOLD}_${SPLIT}_predictions.csv}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda}"

