#!/usr/bin/env python3
"""Export split images and model mistakes for one fold."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABEL_NAMES = {0: "normal", 1: "anomaly"}


@dataclass(frozen=True)
class Record:
    source_path: Path
    split: str
    label: int
    group: str
    label_name: str
    prob_anomaly: float | None = None
    pred_at_val_threshold: int | None = None
    pred_at_0_5: int | None = None

    @property
    def canonical_split(self) -> str:
        return "valid" if self.split == "val" else self.split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="ViT-B + CLAdapter worst fold")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--sheet-cols", type=int, default=6)
    return parser.parse_args()


def read_split_records(split_csv: Path, data_root: Path) -> list[Record]:
    rows: list[Record] = []
    with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            rel = Path(row["image_path"])
            source = rel if rel.is_absolute() else data_root / rel
            rows.append(
                Record(
                    source_path=source,
                    split=row["split"],
                    label=label,
                    group=row.get("group", ""),
                    label_name=row.get("label_name") or LABEL_NAMES[label],
                )
            )
    return rows


def read_predictions(predictions_csv: Path) -> dict[str, dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    with predictions_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_path[str(Path(row["path"]).resolve())] = row
    return by_path


def merge_predictions(records: list[Record], predictions: dict[str, dict[str, str]]) -> list[Record]:
    merged: list[Record] = []
    for rec in records:
        row = predictions.get(str(rec.source_path.resolve()))
        if row is None:
            merged.append(rec)
            continue
        merged.append(
            Record(
                source_path=rec.source_path,
                split=rec.split,
                label=rec.label,
                group=rec.group,
                label_name=rec.label_name,
                prob_anomaly=float(row["prob_anomaly"]),
                pred_at_val_threshold=int(row["pred_at_val_threshold"]),
                pred_at_0_5=int(row["pred_at_0_5"]),
            )
        )
    return merged


def safe_name(index: int, path: Path) -> str:
    return f"{index:04d}__{path.name}"


def copy_records(records: list[Record], dest_dir: Path) -> list[dict[str, str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for idx, rec in enumerate(records, 1):
        dest = dest_dir / safe_name(idx, rec.source_path)
        shutil.copy2(rec.source_path, dest)
        manifest_rows.append(
            {
                "copied_path": str(dest),
                "source_path": str(rec.source_path),
                "split": rec.canonical_split,
                "label": str(rec.label),
                "label_name": rec.label_name,
                "group": rec.group,
                "prob_anomaly": "" if rec.prob_anomaly is None else f"{rec.prob_anomaly:.8f}",
                "pred_at_val_threshold": ""
                if rec.pred_at_val_threshold is None
                else str(rec.pred_at_val_threshold),
                "pred_at_0_5": "" if rec.pred_at_0_5 is None else str(rec.pred_at_0_5),
            }
        )
    return manifest_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def make_contact_sheet(image_paths: list[Path], out_path: Path, title: str, thumb_size: int, cols: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(14)
    title_font = load_font(18)
    if not image_paths:
        canvas = Image.new("RGB", (900, 120), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), f"{title}: no images", fill="black", font=title_font)
        canvas.save(out_path, quality=92)
        return

    label_h = 44
    title_h = 44
    gap = 10
    rows = math.ceil(len(image_paths) / cols)
    cell_w = thumb_size
    cell_h = thumb_size + label_h
    width = cols * cell_w + (cols + 1) * gap
    height = title_h + rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, fill="black", font=title_font)

    for idx, path in enumerate(image_paths):
        r, c = divmod(idx, cols)
        x = gap + c * (cell_w + gap)
        y = title_h + gap + r * (cell_h + gap)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                bg = Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
                px = x + (thumb_size - img.width) // 2
                py = y + (thumb_size - img.height) // 2
                canvas.paste(bg, (x, y))
                canvas.paste(img, (px, py))
        except Exception:
            draw.rectangle((x, y, x + thumb_size, y + thumb_size), outline="red", width=2)
            draw.text((x + 4, y + 4), "load error", fill="red", font=font)
        label = path.name
        if len(label) > 34:
            label = label[:31] + "..."
        draw.text((x, y + thumb_size + 4), label, fill="black", font=font)
    canvas.save(out_path, quality=92)


def summarize_counts(records: list[Record]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ["train", "valid", "test"]:
        subset = [r for r in records if r.canonical_split == split]
        normal = sum(r.label == 0 for r in subset)
        anomaly = sum(r.label == 1 for r in subset)
        rows.append(
            {
                "split": split,
                "normal": str(normal),
                "anomaly": str(anomaly),
                "total": str(len(subset)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_records = read_split_records(args.split_csv, args.data_root)
    records = merge_predictions(split_records, read_predictions(args.predictions_csv))

    manifests_dir = args.output_dir / "manifests"
    sheets_dir = args.output_dir / "contact_sheets"

    all_manifest_rows: list[dict[str, str]] = []
    for split in ["train", "valid", "test"]:
        for label, label_name in [(0, "normal"), (1, "anomaly")]:
            subset = [r for r in records if r.canonical_split == split and r.label == label]
            dest_dir = args.output_dir / split / label_name
            rows = copy_records(subset, dest_dir)
            all_manifest_rows.extend(rows)
            write_csv(manifests_dir / f"{split}_{label_name}.csv", rows)
            make_contact_sheet(
                [Path(r["copied_path"]) for r in rows],
                sheets_dir / f"{split}_{label_name}.jpg",
                f"{args.title} | {split}/{label_name} ({len(rows)})",
                args.thumb_size,
                args.sheet_cols,
            )

    test_records = [r for r in records if r.canonical_split == "test" and r.pred_at_val_threshold is not None]
    true_negative = [r for r in test_records if r.label == 0 and r.pred_at_val_threshold == 0]
    true_positive = [r for r in test_records if r.label == 1 and r.pred_at_val_threshold == 1]
    false_positive = [r for r in test_records if r.label == 0 and r.pred_at_val_threshold == 1]
    false_negative = [r for r in test_records if r.label == 1 and r.pred_at_val_threshold == 0]
    decision_groups = [
        ("true_negative_normal_pred_normal", true_negative),
        ("true_positive_anomaly_pred_anomaly", true_positive),
        ("false_positive_normal_pred_anomaly", false_positive),
        ("false_negative_anomaly_pred_normal", false_negative),
    ]
    for name, subset in decision_groups:
        rows = copy_records(subset, args.output_dir / "model_decision" / name)
        write_csv(manifests_dir / f"model_decision_{name}.csv", rows)
        make_contact_sheet(
            [Path(r["copied_path"]) for r in rows],
            sheets_dir / f"model_decision_{name}.jpg",
            f"{args.title} | {name} ({len(rows)})",
            args.thumb_size,
            args.sheet_cols,
        )

    write_csv(manifests_dir / "all_split_images.csv", all_manifest_rows)
    write_csv(manifests_dir / "split_counts.csv", summarize_counts(records))
    write_csv(
        manifests_dir / "test_predictions_with_decision.csv",
        [
            {
                "source_path": str(r.source_path),
                "label": str(r.label),
                "label_name": r.label_name,
                "prob_anomaly": "" if r.prob_anomaly is None else f"{r.prob_anomaly:.8f}",
                "pred_at_val_threshold": ""
                if r.pred_at_val_threshold is None
                else str(r.pred_at_val_threshold),
                "decision": "correct"
                if r.pred_at_val_threshold == r.label
                else ("false_positive_normal_pred_anomaly" if r.label == 0 else "false_negative_anomaly_pred_normal"),
            }
            for r in test_records
        ],
    )

    summary = [
        f"# {args.title}",
        "",
        f"- split csv: `{args.split_csv}`",
        f"- predictions csv: `{args.predictions_csv}`",
        f"- output dir: `{args.output_dir}`",
        "",
        "## Split Counts",
        "",
        "| split | normal | anomaly | total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summarize_counts(records):
        summary.append(f"| {row['split']} | {row['normal']} | {row['anomaly']} | {row['total']} |")
    summary.extend(
        [
            "",
            "## Test Decisions",
            "",
            "| decision type | count |",
            "| --- | ---: |",
            f"| normal predicted normal | {len(true_negative)} |",
            f"| anomaly predicted anomaly | {len(true_positive)} |",
            f"| normal predicted anomaly | {len(false_positive)} |",
            f"| anomaly predicted normal | {len(false_negative)} |",
            "",
            "See `manifests/` for original file paths and copied file paths.",
            "See `contact_sheets/` for arranged image sheets.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
