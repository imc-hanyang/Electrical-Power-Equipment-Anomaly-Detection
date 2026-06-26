#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = PKG_ROOT / "engine" / "cladapter_code"
sys.path.insert(0, str(CODE_DIR))

from build_model import CLAdapter_CLIP_ViT  # noqa: E402
from utils import config_from_name  # noqa: E402


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ImageListDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int):
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViT-B + CLAdapter SFT inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PKG_ROOT / "engine" / "checkpoints" / "stage2" / "vitb_cla_sft2_final.pth",
    )
    parser.add_argument("--input", type=Path, default=None, help="Single image or directory. If omitted, --csv is used.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PKG_ROOT / "dataset" / "splits" / "first_split.csv",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--data-root", type=Path, default=PKG_ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=PKG_ROOT / "engine" / "predictions" / "predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collect_from_input(path: Path) -> list[dict]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)
    return [{"path": p, "image_path": str(p), "label": ""} for p in files]


def collect_from_csv(csv_path: Path, data_root: Path, split: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    if split != "all":
        df = df[df["split"] == split].copy()
    rows = []
    for row in df.to_dict("records"):
        rel = Path(row["image_path"])
        image_path = rel if rel.is_absolute() else data_root / rel
        rows.append(
            {
                "path": image_path,
                "image_path": row["image_path"],
                "label": row.get("label", ""),
                "label_name": row.get("label_name", ""),
                "split": row.get("split", ""),
            }
        )
    return rows


def build_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    config = config_from_name("config_clip_vit")
    config.defrost()
    config.MODEL.num_classes = 2
    config.MODEL.m_mode = "vit"
    config.MODEL.f_mode = "cla"
    config.MODEL.img_size = 224
    config.MODEL.backbone.model_name = "vit_base_patch16_clip_224.laion2b"
    config.MODEL.backbone.out_dim = 768
    config.MODEL.backbone.num_patch = 196
    config.MODEL.backbone.set_new_allowed(True)
    config.MODEL.backbone.pretrained = False
    config.freeze()

    model = CLAdapter_CLIP_ViT(config)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = collect_from_input(args.input) if args.input else collect_from_csv(args.csv, args.data_root, args.split)
    if not rows:
        raise SystemExit("No images found for inference.")

    missing = [str(row["path"]) for row in rows if not Path(row["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing image files, first examples: {missing[:5]}")

    dataset = ImageListDataset(rows, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.checkpoint, device)

    outputs = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images).float(), dim=1).cpu()
            for prob, idx in zip(probs, indices.tolist()):
                row = rows[idx]
                pred_label = int(torch.argmax(prob).item())
                label = row.get("label", "")
                correct = ""
                if label != "":
                    correct = int(pred_label == int(label))
                outputs.append(
                    {
                        "image_path": row["image_path"],
                        "split": row.get("split", ""),
                        "label": label,
                        "label_name": row.get("label_name", ""),
                        "prob_normal": float(prob[0].item()),
                        "prob_anomaly": float(prob[1].item()),
                        "pred_label": pred_label,
                        "pred_name": "anomaly" if pred_label == 1 else "normal",
                        "correct": correct,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(outputs[0].keys()))
        writer.writeheader()
        writer.writerows(outputs)
    print(f"Wrote {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
