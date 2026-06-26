#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = [
    ("ResNet50", "local_supervised/resnet50", "local"),
    ("ResNet50 + CLAdapter", "resnet50_cla_stage1", "cladapter"),
    ("ConvNeXt-B", "linear_convnextb", "cladapter"),
    ("ConvNeXt-B + CLAdapter", "convnextb_cla_sft2", "cladapter"),
    ("ViT-B", "linear_vitb", "cladapter"),
    ("ViT-B + CLAdapter", "vitb_cla_sft2", "cladapter"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one KEPCO comparison run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--setting-name", default="Train/Validation/Test")
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def read_local_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    val = metrics.get("val")
    test_at_val = metrics.get("test_at_val_best_threshold")
    return {
        "accuracy": test["accuracy_at_0_5"]["accuracy"] * 100.0,
        "f1": macro_f1_from_confusion(test["accuracy_at_0_5"]["confusion"]),
        "auroc": test["auroc"],
        "ap": test["average_precision"],
        "test_best_accuracy": test["best_threshold"]["accuracy"] * 100.0,
        "test_best_f1": macro_f1_from_confusion(test["best_threshold"]["confusion"]),
        "test_best_threshold": test["best_threshold"]["threshold"],
        "test_at_val_threshold_accuracy": test_at_val["accuracy"] * 100.0 if test_at_val else None,
        "test_at_val_threshold_f1": macro_f1_from_confusion(test_at_val["confusion"]) if test_at_val else None,
        "val_auroc": val["auroc"] if val else None,
        "val_accuracy": val["accuracy_at_0_5"]["accuracy"] * 100.0 if val else None,
        "best_epoch": metrics.get("best_epoch"),
    }


def macro_f1_from_confusion(confusion: dict) -> float:
    tn = float(confusion["tn"])
    fp = float(confusion["fp"])
    fn = float(confusion["fn"])
    tp = float(confusion["tp"])
    normal_precision = tn / (tn + fn + 1e-7)
    normal_recall = tn / (tn + fp + 1e-7)
    anomaly_precision = tp / (tp + fp + 1e-7)
    anomaly_recall = tp / (tp + fn + 1e-7)
    precision = (normal_precision + anomaly_precision) / 2.0 * 100.0
    recall = (normal_recall + anomaly_recall) / 2.0 * 100.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_cladapter_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    val = metrics.get("val")
    return {
        "accuracy": float(test["acc"]),
        "f1": float(test["f1"]),
        "auroc": test["roc"],
        "ap": test["ap"],
        "test_best_accuracy": None,
        "test_best_f1": None,
        "test_best_threshold": None,
        "test_at_val_threshold_accuracy": None,
        "test_at_val_threshold_f1": None,
        "val_auroc": val["roc"] if val else None,
        "val_accuracy": float(val["acc"]) if val else None,
        "best_epoch": metrics.get("best_epoch"),
    }


def split_summary(split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    rows = []
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        total = len(part)
        normal = int((part["label"] == 0).sum())
        anomaly = int((part["label"] == 1).sum())
        rows.append(
            {
                "split": split,
                "normal": normal,
                "anomaly": anomaly,
                "total": total,
                "normal_ratio": normal / total if total else 0.0,
                "groups": int(part["group"].nunique()) if "group" in part.columns else None,
            }
        )
    return pd.DataFrame(rows)


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.2f}%"


def fmt_float(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.run_root / "RESULTS.md"

    rows = []
    for model_name, rel_dir, kind in MODELS:
        metrics_path = args.run_root / rel_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        values = read_local_metrics(metrics_path) if kind == "local" else read_cladapter_metrics(metrics_path)
        rows.append({"model": model_name, **values, "metrics_path": str(metrics_path)})

    if not rows:
        raise RuntimeError(f"No metrics found under {args.run_root}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.run_root / "summary.csv", index=False)
    (args.run_root / "summary.json").write_text(
        json.dumps({"run_root": str(args.run_root), "split_csv": str(args.split_csv), "summary": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_df = split_summary(args.split_csv)
    lines = [
        f"# KEPCO {args.setting_name} Results",
        "",
        "## Setting",
        "",
        "- Dataset: `Final_Dataset/normal`, `Final_Dataset/anomaly`",
        "- Split: group-preserving train/validation/test",
        "- Selection: validation-best checkpoint",
        "- Decision rule: fixed classifier decision / argmax",
        "- Epochs: 100",
        "- Image size: 224",
        "",
        "## Split Counts",
        "",
        "| Split | Normal | Anomaly | Total | Normal Ratio | Groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in split_df.to_dict("records"):
        lines.append(
            f"| {row['split']} | {int(row['normal'])} | {int(row['anomaly'])} | {int(row['total'])} | {float(row['normal_ratio']):.2%} | {int(row['groups'])} |"
        )

    lines += [
        "",
        "## Test Performance",
        "",
        "| Model | Accuracy | F1-score | AUROC | AP | Val AUROC | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {fmt_pct(row['accuracy'])} | {fmt_pct(row['f1'])} | "
            f"{fmt_float(row['auroc'])} | {fmt_float(row['ap'])} | {fmt_float(row['val_auroc'])} | {row['best_epoch']} |"
        )

    lines += [
        "",
        "## ResNet Threshold Reference",
        "",
        "The ResNet trainer also stores threshold-derived references. The main table above still uses fixed 0.5/argmax.",
        "",
        "| Model | Test-best Acc. | Test-best F1 | Test-best Threshold | Val-threshold Test Acc. | Val-threshold Test F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["test_best_accuracy"] is None and row["test_at_val_threshold_accuracy"] is None:
            continue
        lines.append(
            f"| {row['model']} | {fmt_pct(row['test_best_accuracy'])} | {fmt_pct(row['test_best_f1'])} | "
            f"{fmt_float(row['test_best_threshold'])} | {fmt_pct(row['test_at_val_threshold_accuracy'])} | "
            f"{fmt_pct(row['test_at_val_threshold_f1'])} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
