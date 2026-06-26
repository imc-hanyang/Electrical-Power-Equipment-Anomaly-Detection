from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .data import IMAGE_EXTENSIONS, make_fixed_transform
from .model import AttentDifferNet, DifferNetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score images with a trained DifferNet/AttentDifferNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-transforms-test", type=int, default=16)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: Path, device: torch.device) -> tuple[AttentDifferNet, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = DifferNetConfig(**checkpoint["config"])
    model = AttentDifferNet(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_threshold(args: argparse.Namespace, checkpoint: dict) -> float | None:
    if args.threshold is not None:
        return args.threshold
    metrics = checkpoint.get("metrics") or {}
    threshold = metrics.get("threshold")
    return float(threshold) if threshold is not None else None


def load_resize_mode(args: argparse.Namespace, checkpoint: dict) -> str:
    if args.resize_mode is not None:
        return args.resize_mode
    metrics_args = checkpoint.get("args") or {}
    if isinstance(metrics_args, dict) and "resize_mode" in metrics_args:
        return str(metrics_args["resize_mode"])
    return "stretch"


def score_image(
    model: AttentDifferNet,
    path: Path,
    transforms_list,
    device: torch.device,
) -> float:
    image = Image.open(path).convert("RGB")
    views = torch.stack([transform(image) for transform in transforms_list], dim=0).to(device)
    with torch.no_grad():
        z = model(views)
        score = torch.mean(z**2)
    return float(score.detach().cpu())


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    threshold = load_threshold(args, checkpoint)
    resize_mode = load_resize_mode(args, checkpoint)

    fixed_degrees = [index * 360.0 / args.n_transforms_test for index in range(args.n_transforms_test)]
    transforms_list = [
        make_fixed_transform(model.config.img_size, degrees, resize_mode=resize_mode)
        for degrees in fixed_degrees
    ]
    rows = []
    for image_path in tqdm(iter_images(args.input), desc="predict"):
        score = score_image(model, image_path, transforms_list, device)
        prediction = None
        if threshold is not None:
            prediction = "anomaly" if score >= threshold else "normal"
        rows.append(
            {
                "path": str(image_path),
                "score": score,
                "threshold": threshold,
                "prediction": prediction,
            }
        )
        if prediction is None:
            print(f"{image_path}\tscore={score:.6f}")
        else:
            print(f"{image_path}\tscore={score:.6f}\tprediction={prediction}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "score", "threshold", "prediction"])
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"count": len(rows), "threshold": threshold}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
