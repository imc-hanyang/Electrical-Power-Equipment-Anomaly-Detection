#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = PKG_ROOT / "engine" / "cladapter_code"
sys.path.insert(0, str(CODE_DIR))

from build_model import CLAdapter_CLIP_ViT  # noqa: E402
from utils import config_from_name  # noqa: E402


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LABEL_NAMES = {0: "normal", 1: "anomaly"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ViT-B + CLAdapter SFT predictions with attention rollout and token Grad-CAM."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PKG_ROOT / "engine" / "checkpoints" / "stage2" / "vitb_cla_sft2_final.pth",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PKG_ROOT / "dataset" / "splits" / "first_split.csv",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--data-root", type=Path, default=PKG_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PKG_ROOT / "engine" / "visualizations" / "test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay strength.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument(
        "--rollout-mode",
        type=str,
        default="patch_mean",
        choices=["patch_mean", "cls"],
        help="patch_mean aligns with this model because CLAdapter receives patch tokens.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    text = re.sub(r"[\\/:\n\r\t]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:180]


def load_rows(csv_path: Path, data_root: Path, split: str, limit: int | None) -> list[dict]:
    df = pd.read_csv(csv_path)
    if split != "all":
        df = df[df["split"] == split].copy()
    if limit is not None:
        df = df.head(limit).copy()
    rows = []
    for row in df.to_dict("records"):
        rel = Path(row["image_path"])
        image_path = rel if rel.is_absolute() else data_root / rel
        rows.append(
            {
                "image_path": image_path,
                "display_path": row["image_path"],
                "label": int(row["label"]),
                "label_name": row.get("label_name", LABEL_NAMES.get(int(row["label"]), str(row["label"]))),
                "split": row.get("split", ""),
            }
        )
    return rows


def build_model(checkpoint: Path, device: torch.device) -> CLAdapter_CLIP_ViT:
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
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    for block in model.backbone.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
    model.to(device)
    model.eval()
    return model


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def load_image(path: Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    tensor = image_transform(image_size)(image).unsqueeze(0)
    return image, tensor


def compute_attention_rollout(model: CLAdapter_CLIP_ViT, image: torch.Tensor, mode: str) -> np.ndarray:
    qkv_outputs: list[torch.Tensor] = []
    handles = []

    def hook_qkv(_module, _inputs, output):
        qkv_outputs.append(output.detach())

    for block in model.backbone.blocks:
        handles.append(block.attn.qkv.register_forward_hook(hook_qkv))

    with torch.no_grad():
        _ = model(image)

    for handle in handles:
        handle.remove()

    if not qkv_outputs:
        raise RuntimeError("No qkv activations were captured.")

    eye = None
    rollout = None
    for qkv_out, block in zip(qkv_outputs, model.backbone.blocks):
        b, n_tokens, three_c = qkv_out.shape
        c = three_c // 3
        n_heads = block.attn.num_heads
        head_dim = c // n_heads
        qkv = qkv_out.reshape(b, n_tokens, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q = q * block.attn.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = attn.mean(dim=1)[0]
        if eye is None:
            eye = torch.eye(attn.shape[-1], device=attn.device, dtype=attn.dtype)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = attn if rollout is None else attn @ rollout

    if rollout is None:
        raise RuntimeError("Attention rollout failed.")
    if mode == "cls":
        score = rollout[0, 1:]
    else:
        score = rollout[1:, 1:].mean(dim=0)
    return token_scores_to_map(score)


def compute_token_gradcam(model: CLAdapter_CLIP_ViT, image: torch.Tensor, target_class: int) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def hook_post(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = model.post.register_forward_hook(hook_post)
    model.zero_grad(set_to_none=True)
    logits = model(image)
    target = logits[0, target_class]
    target.backward()
    handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM activations/gradients were not captured.")
    act = activations[-1].detach()[0]
    grad = gradients[-1].detach()[0]
    weights = grad.mean(dim=0)
    cam = torch.relu((act * weights).sum(dim=-1))
    return token_scores_to_map(cam)


def token_scores_to_map(scores: torch.Tensor) -> np.ndarray:
    scores = scores.detach().float().cpu()
    side = int(math.sqrt(scores.numel()))
    if side * side != scores.numel():
        raise ValueError(f"Cannot reshape {scores.numel()} token scores into a square map.")
    arr = scores.reshape(side, side).numpy()
    arr = arr - arr.min()
    denom = arr.max()
    if denom > 1e-12:
        arr = arr / denom
    return arr


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, image_size: int, alpha: float) -> Image.Image:
    base = np.array(image.resize((image_size, image_size), Image.Resampling.BILINEAR))
    heat = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = np.uint8(np.clip((1 - alpha) * base + alpha * color, 0, 255))
    return Image.fromarray(blended)


def make_panel(
    original: Image.Image,
    rollout_overlay: Image.Image,
    gradcam_overlay: Image.Image,
    title: str,
    image_size: int,
) -> Image.Image:
    pad = 14
    header_h = 56
    label_h = 28
    w = image_size * 3 + pad * 4
    h = header_h + image_size + label_h + pad * 2
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((pad, pad), title, fill=(20, 20, 20), font=font)
    labels = ["Original", "ViT Patch Attention Rollout", "CLAdapter Token Grad-CAM"]
    images = [original.resize((image_size, image_size), Image.Resampling.BILINEAR), rollout_overlay, gradcam_overlay]
    y = header_h
    for idx, (label, img) in enumerate(zip(labels, images)):
        x = pad + idx * (image_size + pad)
        panel.paste(img, (x, y))
        draw.text((x, y + image_size + 6), label, fill=(35, 35, 35), font=font)
    return panel


def save_contact_sheet(image_paths: list[Path], output_path: Path, thumb_width: int = 420, columns: int = 2) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumbs.append(img.resize((thumb_width, int(img.height * ratio)), Image.Resampling.BILINEAR))
    rows = math.ceil(len(thumbs) / columns)
    pad = 12
    cell_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width + (columns + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
    for idx, img in enumerate(thumbs):
        r, c = divmod(idx, columns)
        x = pad + c * (thumb_width + pad)
        y = pad + r * (cell_h + pad)
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = load_rows(args.csv, args.data_root, args.split, args.limit)
    if not rows:
        raise SystemExit("No images selected.")

    model = build_model(args.checkpoint, device)
    all_dir = args.output_dir / "all"
    correct_dir = args.output_dir / "correct"
    wrong_dir = args.output_dir / "wrong"
    for path in [all_dir, correct_dir, wrong_dir, args.output_dir / "contact_sheets"]:
        path.mkdir(parents=True, exist_ok=True)

    records = []
    groups: dict[str, list[Path]] = {
        "all": [],
        "correct": [],
        "wrong": [],
        "normal": [],
        "anomaly": [],
        "false_positive_normal_pred_anomaly": [],
        "false_negative_anomaly_pred_normal": [],
    }

    for idx, row in enumerate(rows, 1):
        original, tensor = load_image(row["image_path"], args.image_size)
        tensor = tensor.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu()
        pred = int(torch.argmax(probs).item())
        label = int(row["label"])
        is_correct = pred == label

        rollout = compute_attention_rollout(model, tensor, args.rollout_mode)
        gradcam = compute_token_gradcam(model, tensor, pred)
        rollout_overlay = overlay_heatmap(original, rollout, args.image_size, args.alpha)
        gradcam_overlay = overlay_heatmap(original, gradcam, args.image_size, args.alpha)

        stem = safe_name(Path(row["display_path"]).stem)
        title = (
            f"{idx}/{len(rows)} | true={LABEL_NAMES[label]} pred={LABEL_NAMES[pred]} "
            f"p_anomaly={probs[1].item():.4f}"
        )
        panel = make_panel(original, rollout_overlay, gradcam_overlay, title, args.image_size)
        group_dir = correct_dir if is_correct else wrong_dir
        out_name = f"{stem}__true-{LABEL_NAMES[label]}__pred-{LABEL_NAMES[pred]}.jpg"
        out_path = all_dir / out_name
        panel.save(out_path, quality=94)
        panel.save(group_dir / out_name, quality=94)

        groups["all"].append(out_path)
        groups["correct" if is_correct else "wrong"].append(out_path)
        groups[LABEL_NAMES[label]].append(out_path)
        if label == 0 and pred == 1:
            groups["false_positive_normal_pred_anomaly"].append(out_path)
        if label == 1 and pred == 0:
            groups["false_negative_anomaly_pred_normal"].append(out_path)

        records.append(
            {
                "image_path": row["display_path"],
                "visualization": str(out_path),
                "label": label,
                "label_name": LABEL_NAMES[label],
                "pred_label": pred,
                "pred_name": LABEL_NAMES[pred],
                "prob_normal": float(probs[0].item()),
                "prob_anomaly": float(probs[1].item()),
                "correct": int(is_correct),
            }
        )
        print(f"[{idx:03d}/{len(rows):03d}] {row['display_path']} -> {LABEL_NAMES[pred]} ({'correct' if is_correct else 'wrong'})")

    csv_path = args.output_dir / "visualization_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    sheet_dir = args.output_dir / "contact_sheets"
    for name, paths in groups.items():
        save_contact_sheet(paths, sheet_dir / f"{name}.jpg")

    summary_path = args.output_dir / "summary.txt"
    n_correct = sum(r["correct"] for r in records)
    summary_path.write_text(
        "\n".join(
            [
                f"split={args.split}",
                f"total={len(records)}",
                f"correct={n_correct}",
                f"wrong={len(records) - n_correct}",
                f"accuracy={n_correct / len(records):.4f}",
                f"output_dir={args.output_dir}",
                "visualization_columns=Original | ViT Patch Attention Rollout | CLAdapter Token Grad-CAM",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote visualizations to {args.output_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
