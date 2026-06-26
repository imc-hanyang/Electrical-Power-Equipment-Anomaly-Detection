from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KEPCO split.csv to CLAdapter official CSV format.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Package root used to make image paths relative. Defaults to this package root.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--merge-val-into-train",
        action="store_true",
        help="Write validation rows as train rows, matching no-validation train/test benchmark setups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | int]] = []
    with args.split_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = "val" if row["split"] == "val" else row["split"]
            if args.merge_val_into_train and split == "val":
                split = "train"
            label = 0 if row["label"] == "normal" else 1
            image_path = Path(row["path"])
            try:
                image_path_text = str(image_path.relative_to(args.project_root))
            except ValueError:
                image_path_text = str(image_path)
            rows.append(
                {
                    "image_path": image_path_text,
                    "label": label,
                    "split": split,
                    "group": row["group"],
                    "label_name": row["label"],
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split", "group", "label_name"])
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["split"]), int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {args.output_csv}")
    print(counts)


if __name__ == "__main__":
    main()
