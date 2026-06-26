#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

N_SPLITS=10 \
BUILD_SPLITS="${BUILD_SPLITS:-0}" \
SPLIT_DIR="${SPLIT_DIR:-$ROOT_DIR/dataset/splits/kfold10_train_val_test_second_setting}" \
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/engine/runs/kfold10_vitb_cladapter_reproduce_${RUN_STAMP}}" \
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset}" \
GPU_IDS="${GPU_IDS:-0}" \
EPOCHS="${EPOCHS:-100}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
FOLDS="${FOLDS:-}" \
bash "$ROOT_DIR/engine/scripts/run_vitb_cladapter_kfold_train_val_test.sh"

