#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a binary image index CSV from normal/anomaly folders.")
    parser.add_argument("--dataset-name", required=True, help="Dataset folder name under --data-root.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
        help="Package dataset root.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV with image_path,label,split,group,label_name columns.",
    )
    return parser.parse_args()


def group_from_filename(path: Path) -> str:
    stem = path.stem
    match = re.match(r"(.+)_crop\d+$", stem)
    return match.group(1) if match else stem


def iter_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    args = parse_args()
    dataset_root = args.data_root / args.dataset_name
    labels = [("normal", 0), ("anomaly", 1)]
    rows = []
    for label_name, label in labels:
        label_dir = dataset_root / label_name
        if not label_dir.is_dir():
            raise SystemExit(f"Missing label directory: {label_dir}")
        for image_path in iter_images(label_dir):
            rows.append(
                {
                    "image_path": str(Path(args.dataset_name) / label_name / image_path.name),
                    "label": label,
                    "split": "all",
                    "group": group_from_filename(image_path),
                    "label_name": label_name,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split", "group", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    normal = sum(1 for row in rows if row["label"] == 0)
    anomaly = sum(1 for row in rows if row["label"] == 1)
    groups = len({row["group"] for row in rows})
    print(f"wrote {args.output_csv}")
    print(f"normal={normal} anomaly={anomaly} total={len(rows)} groups={groups}")


if __name__ == "__main__":
    main()
