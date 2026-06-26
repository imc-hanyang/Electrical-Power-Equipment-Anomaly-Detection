#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize no-etc 10-fold EfficientNet/PatchCore/DifferNet results.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def macro_f1_from_confusion(confusion: dict) -> float:
    tn = float(confusion["tn"])
    fp = float(confusion["fp"])
    fn = float(confusion["fn"])
    tp = float(confusion["tp"])
    normal_precision = tn / (tn + fn + 1e-7)
    normal_recall = tn / (tn + fp + 1e-7)
    anomaly_precision = tp / (tp + fp + 1e-7)
    anomaly_recall = tp / (tp + fn + 1e-7)
    precision = (normal_precision + anomaly_precision) / 2.0
    recall = (normal_recall + anomaly_recall) / 2.0
    return float(2.0 * precision * recall / (precision + recall) * 100.0) if precision + recall else 0.0


def metric_from_threshold_block(block: dict) -> tuple[float, float]:
    accuracy = float(block["accuracy"] * 100.0)
    return accuracy, macro_f1_from_confusion(block["confusion"])


def read_efficientnet(fold_dir: Path) -> dict:
    metrics = json.loads((fold_dir / "local_supervised/efficientnet_b0/metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    threshold_block = metrics["test_at_val_best_threshold"] or test["accuracy_at_0_5"]
    acc, f1 = metric_from_threshold_block(threshold_block)
    fixed_acc, _ = metric_from_threshold_block(test["accuracy_at_0_5"])
    return {
        "test_accuracy": acc,
        "test_f1": f1,
        "fixed_accuracy": fixed_acc,
        "auroc": float(test["auroc"]),
        "ap": float(test["average_precision"]),
    }


def read_patchcore(fold_dir: Path) -> dict:
    candidates = sorted((fold_dir / "patchcore").glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No PatchCore metrics under {fold_dir / 'patchcore'}")
    metrics = json.loads(candidates[0].read_text(encoding="utf-8"))
    test = metrics["test"]
    threshold_block = metrics["test_at_val_best_threshold"] or test["best_threshold"]
    acc, f1 = metric_from_threshold_block(threshold_block)
    fixed_acc, _ = metric_from_threshold_block(test["normal_p95_threshold"]) if "normal_p95_threshold" in test else (float("nan"), float("nan"))
    return {
        "test_accuracy": acc,
        "test_f1": f1,
        "fixed_accuracy": fixed_acc,
        "auroc": float(test["auroc"]),
        "ap": float(test["average_precision"]),
    }


def read_differnet(fold_dir: Path) -> dict:
    metrics = json.loads((fold_dir / "differnet/none/metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    threshold_block = metrics["test_at_val_best_threshold"] or test["confusion"]
    if "confusion" not in threshold_block:
        threshold_block = {"accuracy": float("nan"), "confusion": test["confusion"]}
    acc, f1 = metric_from_threshold_block(threshold_block)
    fixed_acc = float("nan")
    return {
        "test_accuracy": acc,
        "test_f1": f1,
        "fixed_accuracy": fixed_acc,
        "auroc": float(test["auroc"]),
        "ap": float(test["average_precision"]),
    }


READERS = {
    "EfficientNet-B0": read_efficientnet,
    "PatchCore": read_patchcore,
    "DifferNet": read_differnet,
}


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.array([value for value in values if not np.isnan(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, float(arr.min()), float(arr.max())


def fmt_pct(values: tuple[float, float, float, float]) -> str:
    mean, std, min_value, max_value = values
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.2f}% +/- {std:.2f} ({min_value:.2f}~{max_value:.2f})"


def fmt_float(values: tuple[float, float, float, float]) -> str:
    mean, std, min_value, max_value = values
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.4f} +/- {std:.4f} ({min_value:.4f}~{max_value:.4f})"


def count_rows(split_dir: Path) -> list[dict]:
    rows = []
    fold_paths = [
        path
        for path in split_dir.glob("fold_*.csv")
        if path.stem.split("_")[-1].isdigit()
    ]
    for csv_path in sorted(fold_paths, key=lambda p: int(p.stem.split("_")[1])):
        df = pd.read_csv(csv_path)
        row = {"fold": csv_path.stem}
        for split in ["train", "val", "test"]:
            part = df[df["split"] == split]
            row[f"{split}_normal"] = int((part["label"] == 0).sum())
            row[f"{split}_anomaly"] = int((part["label"] == 1).sum())
            row[f"{split}_total"] = int(len(part))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for fold_dir in sorted(args.run_root.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])):
        if not fold_dir.is_dir():
            continue
        for model, reader in READERS.items():
            rows.append({"unit": fold_dir.name, "model": model, **reader(fold_dir)})
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(args.run_root / "baseline_10fold_metrics.csv", index=False)

    summary_rows = []
    for model in READERS:
        part = fold_df[fold_df["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "units": int(part["unit"].nunique()),
                "test_accuracy": summarize(part["test_accuracy"].tolist()),
                "test_f1": summarize(part["test_f1"].tolist()),
                "fixed_accuracy": summarize(part["fixed_accuracy"].tolist()),
                "auroc": summarize(part["auroc"].tolist()),
                "ap": summarize(part["ap"].tolist()),
            }
        )

    output = args.output or (args.run_root / "BASELINE_10FOLD_SUMMARY.md")
    lines = [
        "# No-ETC 10-Fold Baseline Summary",
        "",
        "Dataset: `Final_Dataset` without `etc`.",
        "",
        "Metric format: `mean +/- std (min~max)`.",
        "",
        "| Model | Test Acc. @ Val Threshold | Test F1 @ Val Threshold | Fixed/Reference Acc. | AUROC | AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {fmt_pct(row['test_accuracy'])} | {fmt_pct(row['test_f1'])} | "
            f"{fmt_pct(row['fixed_accuracy'])} | {fmt_float(row['auroc'])} | {fmt_float(row['ap'])} |"
        )

    lines += [
        "",
        "## Fold Counts",
        "",
        "`N/A/T` means `normal/anomaly/total`.",
        "",
        "| Fold | Train N/A/T | Val N/A/T | Test N/A/T |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in count_rows(args.split_dir):
        lines.append(
            f"| {row['fold']} | "
            f"{row['train_normal']}/{row['train_anomaly']}/{row['train_total']} | "
            f"{row['val_normal']}/{row['val_anomaly']}/{row['val_total']} | "
            f"{row['test_normal']}/{row['test_anomaly']}/{row['test_total']} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.run_root / "baseline_10fold_summary.json").write_text(
        json.dumps({"summary": summary_rows, "folds": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
