#!/usr/bin/env python3
"""
KEPCO OPGW 이상 탐지 Inference 스크립트

Usage:
  # 경로만 바꿔서 테스트 (fold 자동 선택)
  python inference/inference.py --test-dir /path/to/new_images --auto-best \
      --checkpoints-dir checkpoints/vitb_kfold10_20260624_123456

  # fold 직접 지정
  python inference/inference.py --test-dir /path/to/images --fold 3 \
      --checkpoints-dir checkpoints/vitb_kfold10_20260624_123456

  # 모델 선택
  python inference/inference.py --test-dir /path/to/images --auto-best \
      --model convnextb_cla_sft2 \
      --checkpoints-dir checkpoints/convnextb_kfold10_20260624_123456

체크포인트 폴더 구조 (train_kfold.sh 출력 기준):
  checkpoints-dir/
  └── fold_{N}/
      └── vitb_cla_sft2/          ← model_subdir
          ├── metrics.json
          └── vit_base_patch16_clip_224.laion2b_best.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from build_model import CLAdapter_CLIP_ViT  # noqa: E402
from utils import config_from_name           # noqa: E402

# ── 모델 설정 테이블 ───────────────────────────────────────────────────────
# model_subdir : train_kfold.sh 가 fold_N/ 아래에 생성하는 하위 폴더명
MODEL_CONFIGS = {
    # ViT-B + CLAdapter (Stage 2 SFT) — 최고 성능
    "vitb_cla_sft2": {
        "config_name":      "config_clip_vit",
        "model_mode":       "vit",
        "backbone_name":    "vit_base_patch16_clip_224.laion2b",
        "backbone_out_dim": 768,
        "backbone_num_patch": 196,
        "ckpt_filename":    "vit_base_patch16_clip_224.laion2b_best.pth",
        "model_subdir":     "vitb_cla_sft2",
        "norm":             "clip",
    },
    # ConvNeXt-B + CLAdapter (Stage 2 SFT)
    "convnextb_cla_sft2": {
        "config_name":      "config_clip_convnext",
        "model_mode":       "conv",
        "backbone_name":    "convnext_base.clip_laion2b_augreg",
        "backbone_out_dim": 1024,
        "backbone_num_patch": 49,
        "ckpt_filename":    "convnext_base.clip_laion2b_augreg_best.pth",
        "model_subdir":     "convnextb_cla_sft2",
        "norm":             "clip",
    },
    # ViT-B linear probe (kfold10_remaining 기준: linear_vitb/)
    "vitb_linear": {
        "config_name":      "config_clip_vit",
        "model_mode":       "vit",
        "backbone_name":    "vit_base_patch16_clip_224.laion2b",
        "backbone_out_dim": 768,
        "backbone_num_patch": 196,
        "ckpt_filename":    "vit_base_patch16_clip_224.laion2b_best.pth",
        "model_subdir":     "linear_vitb",
        "norm":             "clip",
        "f_mode":           "linear",   # CLAdapter 레이어 없음 (stage1 linear probe)
    },
    # ConvNeXt-B linear probe (kfold10_remaining 기준: linear_convnextb/)
    "convnextb_linear": {
        "config_name":      "config_clip_convnext",
        "model_mode":       "conv",
        "backbone_name":    "convnext_base.clip_laion2b_augreg",
        "backbone_out_dim": 1024,
        "backbone_num_patch": 49,
        "ckpt_filename":    "convnext_base.clip_laion2b_augreg_best.pth",
        "model_subdir":     "linear_convnextb",
        "norm":             "clip",
        "f_mode":           "linear",   # CLAdapter 레이어 없음 (stage1 linear probe)
    },
}

CLIP_MEAN    = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD     = [0.26862954, 0.26130258, 0.27577711]
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                ".JPG", ".JPEG", ".PNG"}

# ── Dataset ────────────────────────────────────────────────────────────────
class ImageDirDataset(Dataset):
    def __init__(self, img_dir: Path, image_size: int = 224):
        self.paths = sorted(p for p in img_dir.rglob("*") if p.suffix in IMG_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"이미지 없음: {img_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx

# ── 인자 파싱 ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="KEPCO OPGW 이상 탐지 Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test-dir", type=Path, required=True,
                   help="테스트 이미지 디렉토리 (경로만 바꾸면 됨)")
    p.add_argument("--checkpoints-dir", type=Path, required=True,
                   help="train_kfold.sh 로 생성된 체크포인트 루트 "
                        "(예: checkpoints/vitb_kfold10_20260624_123456)")
    p.add_argument("--model", type=str, default="vitb_cla_sft2",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="사용할 모델 (기본: vitb_cla_sft2)")
    p.add_argument("--fold", type=int, default=None,
                   help="사용할 fold 번호 (0~9)")
    p.add_argument("--auto-best", action="store_true",
                   help="metrics.json 기준 최고 fold 자동 선택")
    p.add_argument("--metric", type=str, default="f1",
                   choices=["f1", "auroc", "prec", "rec"],
                   help="--auto-best 기준 지표 (기본: f1)")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "predictions" / "predictions.csv")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--image-size",   type=int, default=224)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quiet", action="store_true",
                   help="verbose 출력 억제 (wire_inference.sh 용)")
    return p.parse_args()

# ── fold 선택 ──────────────────────────────────────────────────────────────
def select_fold(checkpoints_dir: Path, model_subdir: str, metric: str,
                quiet: bool = False) -> int:
    """fold_N/{model_subdir}/metrics.json 기준 최고 fold 자동 선택"""
    key_map = {"f1": "test.f1", "auroc": "test.roc",
               "prec": "test.prec", "rec": "test.reca"}
    metric_key = key_map[metric]
    log = (lambda *a: None) if quiet else print

    def get_val(m, key):
        if key in m:
            return m[key]
        parts = key.split(".", 1)
        return m.get(parts[0], {}).get(parts[1], 0.0) if len(parts) == 2 else 0.0

    best_fold, best_val = -1, -1.0
    for fold_dir in sorted(checkpoints_dir.glob("fold_*")):
        metrics_path = fold_dir / model_subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.load(open(metrics_path))
        val = get_val(m, metric_key)
        fold_num = int(fold_dir.name.split("_")[1])
        log(f"  fold {fold_num}: {metric}={val:.4f}")
        if val > best_val:
            best_val, best_fold = val, fold_num

    if best_fold < 0:
        raise FileNotFoundError(
            f"metrics.json 없음. 경로 확인: {checkpoints_dir}/fold_*/{{model_subdir}}/metrics.json"
        )
    log(f"  → 선택: fold {best_fold} ({metric}={best_val:.4f})")
    return best_fold

# ── 모델 로드 ──────────────────────────────────────────────────────────────
def build_model(cfg: dict, ckpt_path: Path, device: torch.device):
    config = config_from_name(cfg["config_name"])
    config.defrost()
    config.MODEL.num_classes             = 2
    config.MODEL.m_mode                  = cfg["model_mode"]
    config.MODEL.f_mode                  = cfg.get("f_mode", "cla")
    config.MODEL.img_size                = 224
    config.MODEL.backbone.model_name     = cfg["backbone_name"]
    config.MODEL.backbone.out_dim        = cfg["backbone_out_dim"]
    config.MODEL.backbone.num_patch      = cfg["backbone_num_patch"]
    config.MODEL.backbone.pretrained     = False
    config.MODEL.backbone.set_new_allowed(True)
    config.freeze()

    model = CLAdapter_CLIP_ViT(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model

# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device)
    cfg    = MODEL_CONFIGS[args.model]

    # fold 결정
    if args.auto_best and args.fold is not None:
        raise SystemExit("--fold 과 --auto-best 는 동시에 사용할 수 없습니다.")
    log = (lambda *a, **k: None) if args.quiet else print

    if args.auto_best:
        log(f"[auto-best] {cfg['model_subdir']} 기준 최고 fold 탐색...")
        fold = select_fold(args.checkpoints_dir, cfg["model_subdir"], args.metric,
                           quiet=args.quiet)
    elif args.fold is not None:
        fold = args.fold
    else:
        raise SystemExit("--fold N 또는 --auto-best 중 하나를 지정하세요.")

    # 체크포인트 경로: {checkpoints_dir}/fold_{N}/{model_subdir}/{ckpt_filename}
    ckpt_path = (args.checkpoints_dir / f"fold_{fold}"
                 / cfg["model_subdir"] / cfg["ckpt_filename"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    log(f"[inference] model={args.model}  fold={fold}")
    log(f"[inference] ckpt ={ckpt_path}")

    # 데이터
    dataset = ImageDirDataset(args.test_dir, args.image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=True)
    log(f"[inference] 이미지 {len(dataset)}장  ←  {args.test_dir}")

    # 추론
    model   = build_model(cfg, ckpt_path, device)
    results = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            probs  = torch.softmax(model(images).float(), dim=1).cpu()
            for prob, idx in zip(probs, indices.tolist()):
                pred = int(torch.argmax(prob))
                results.append({
                    "image_path":   str(dataset.paths[idx]),
                    "prob_normal":  round(float(prob[0]), 4),
                    "prob_anomaly": round(float(prob[1]), 4),
                    "pred_label":   pred,
                    "pred_name":    "anomaly" if pred == 1 else "normal",
                    "model":        args.model,
                    "fold":         fold,
                })

    # 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n_anomaly = sum(1 for r in results if r["pred_label"] == 1)
    if args.quiet:
        print(f"DONE {len(results)} {n_anomaly} {len(results)-n_anomaly}")
    else:
        log(f"[inference] 완료: 총 {len(results)}장  이상 {n_anomaly}장  정상 {len(results)-n_anomaly}장")
        log(f"[inference] 결과: {args.output}")

if __name__ == "__main__":
    main()
