#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$SCRIPT_DIR/infer_vitb_cladapter_sft.py" \
  --checkpoint "$PKG_ROOT/engine/checkpoints/stage2/vitb_cla_sft2_final.pth" \
  --csv "$PKG_ROOT/dataset/splits/first_split.csv" \
  --split test \
  --data-root "$PKG_ROOT/dataset" \
  --output "$PKG_ROOT/engine/predictions/test_predictions.csv" \
  --batch-size 16
