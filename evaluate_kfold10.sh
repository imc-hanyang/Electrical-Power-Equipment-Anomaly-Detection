#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$ROOT_DIR/engine/scripts/apply_validation_thresholds.py" \
  --run-root "${RUN_ROOT:-$ROOT_DIR/engine/runs/kfold10_vitb_cladapter_second_setting}" \
  --split-dir "${SPLIT_DIR:-$ROOT_DIR/dataset/splits/kfold10_train_val_test_second_setting}" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/dataset}" \
  --models "ViT-B + CLAdapter" \
  --title "${TITLE:-KEPCO 10-Fold ViT-B + CLAdapter Re-evaluation}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda}"

