#!/usr/bin/env bash
# Electrical Power Equipment 이상 탐지 — 10-Fold 학습 스크립트
#
# Usage:
#   bash train_kfold.sh                          # 전체 10-fold, ViT-B, GPU 0
#   bash train_kfold.sh --model convnextb         # ConvNeXt-B
#   bash train_kfold.sh --fold 3                  # fold 3만
#   bash train_kfold.sh --fold 3 --model vitb --gpu 0,1,2,3
#
# Env overrides:
#   GPU_IDS=0,1,2,3 EPOCHS=100 bash train_kfold.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"

# ── 기본값 ────────────────────────────────────────────────────
MODEL="vitb"
FOLDS_ARG=""
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPLIT_DIR="$ROOT_DIR/dataset/splits/kfold10_train_val_test_dataset_0622"
DATA_ROOT="$ROOT_DIR/dataset"
N_SPLITS=10

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2";      shift 2 ;;
    --fold)      FOLDS_ARG="$2";  shift 2 ;;
    --gpu)       GPU_IDS="$2";    shift 2 ;;
    --epochs)    EPOCHS="$2";     shift 2 ;;
    --split-dir) SPLIT_DIR="$2";  shift 2 ;;
    --data-root) DATA_ROOT="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── 모델별 파라미터 ──────────────────────────────────────────
case "$MODEL" in
  vitb)
    BACKBONE_NAME="vit_base_patch16_clip_224.laion2b"
    CONFIG_NAME="config_clip_vit"
    MODEL_MODE="vit"
    BACKBONE_OUT_DIM=768
    BACKBONE_NUM_PATCH=196
    STAGE1_SUBDIR="vitb_cla_stage1"
    STAGE2_SUBDIR="vitb_cla_sft2"
    ;;
  convnextb)
    BACKBONE_NAME="convnext_base.clip_laion2b_augreg"
    CONFIG_NAME="config_clip_convnext"
    MODEL_MODE="conv"
    BACKBONE_OUT_DIM=1024
    BACKBONE_NUM_PATCH=49
    STAGE1_SUBDIR="convnextb_cla_stage1"
    STAGE2_SUBDIR="convnextb_cla_sft2"
    ;;
  *)
    echo "ERROR: --model must be vitb or convnextb (got: $MODEL)"; exit 1 ;;
esac

# ── fold 목록 ────────────────────────────────────────────────
FOLDS="${FOLDS_ARG:-$(seq 0 $((N_SPLITS - 1)))}"

# ── 출력 경로 ────────────────────────────────────────────────
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/checkpoints/${MODEL}_kfold10_${RUN_STAMP}}"
mkdir -p "$RUN_ROOT/logs"

# ── GPU 목록 ─────────────────────────────────────────────────
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS"

# ── 헬퍼 ─────────────────────────────────────────────────────
run_fold() {
  local fold="$1"
  local gpu_id="$2"
  local csv_path="$SPLIT_DIR/fold_${fold}.csv"
  local fold_root="$RUN_ROOT/fold_${fold}"
  mkdir -p "$fold_root"

  local common_args=(
    --model-mode       "$MODEL_MODE"
    --finetune-mode    cla
    --image-size       224
    --csv-dir          "$csv_path"
    --config-name      "$CONFIG_NAME"
    --data-root        "$DATA_ROOT"
    --gpu_id           0
    --batch-size       16
    --num-workers      "$NUM_WORKERS"
    --init-lr          1e-4
    --weight_decay     1e-4
    --optimizer        AdamW
    --epochs           "$EPOCHS"
    --warmup_epochs    2
    --nbatch_log       300
    --selection-metric acc
    --backbone-name    "$BACKBONE_NAME"
    --backbone-out-dim "$BACKBONE_OUT_DIM"
    --backbone-num-patch "$BACKBONE_NUM_PATCH"
    --norm             clip
  )

  # Stage 1 (CLAdapter pretraining)
  if [[ ! -f "$fold_root/$STAGE1_SUBDIR/metrics.json" ]]; then
    echo "[$(date '+%F %T')] START fold${fold} stage1"
    (cd "$SRC_DIR" && CUDA_VISIBLE_DEVICES="$gpu_id" \
      torchrun --standalone --nproc_per_node=1 train.py \
      "${common_args[@]}" \
      --output-dir "$fold_root/$STAGE1_SUBDIR") \
      > "$RUN_ROOT/logs/fold${fold}_stage1.log" 2>&1
    echo "[$(date '+%F %T')] DONE  fold${fold} stage1"
  else
    echo "[$(date '+%F %T')] SKIP  fold${fold} stage1"
  fi

  # Stage 2 (SFT from Stage 1 checkpoint)
  if [[ ! -f "$fold_root/$STAGE2_SUBDIR/metrics.json" ]]; then
    echo "[$(date '+%F %T')] START fold${fold} stage2"
    (cd "$SRC_DIR" && CUDA_VISIBLE_DEVICES="$gpu_id" \
      torchrun --standalone --nproc_per_node=1 train.py \
      "${common_args[@]}" \
      --finetune-ckpt "$fold_root/$STAGE1_SUBDIR/${BACKBONE_NAME}_best.pth" \
      --output-dir "$fold_root/$STAGE2_SUBDIR") \
      > "$RUN_ROOT/logs/fold${fold}_stage2.log" 2>&1
    echo "[$(date '+%F %T')] DONE  fold${fold} stage2"
  else
    echo "[$(date '+%F %T')] SKIP  fold${fold} stage2"
  fi
}

# ── 실행 ─────────────────────────────────────────────────────
echo "Model:     $MODEL"
echo "Folds:     $FOLDS"
echo "GPUs:      $GPU_IDS"
echo "Split dir: $SPLIT_DIR"
echo "Run root:  $RUN_ROOT"
echo ""

pids=()
fold_index=0
for fold in $FOLDS; do
  gpu_index=$(( fold_index % ${#GPU_IDS_ARR[@]} ))
  gpu_id="${GPU_IDS_ARR[$gpu_index]}"

  run_fold "$fold" "$gpu_id" \
    > "$RUN_ROOT/logs/fold${fold}.driver.log" 2>&1 &
  pids+=("$!")
  fold_index=$(( fold_index + 1 ))

  # GPU 수만큼 채워지면 하나 끝날 때까지 대기
  if (( ${#pids[@]} >= ${#GPU_IDS_ARR[@]} )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo ""
echo "Done. Checkpoints: $RUN_ROOT"
