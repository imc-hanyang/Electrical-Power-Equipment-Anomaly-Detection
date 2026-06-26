#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Summarize KEPCO k-fold comparison results.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def custom_macro_pr_f1(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    precisions = []
    recalls = []
    for cls in [0, 1]:
        pred_mask = preds == cls
        label_mask = labels == cls
        tp = float(np.sum(pred_mask & label_mask))
        precisions.append(tp / (float(np.sum(pred_mask)) + 1e-7))
        recalls.append(tp / (float(np.sum(label_mask)) + 1e-7))
    precision = float(np.mean(precisions) * 100.0)
    recall = float(np.mean(recalls) * 100.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def read_local_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    pred_path = path.parent / "predictions_test_at_0_5.csv"
    labels = []
    preds = []
    with pred_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
    labels_np = np.array(labels, dtype=np.int64)
    preds_np = np.array(preds, dtype=np.int64)
    _precision, _recall, f1 = custom_macro_pr_f1(labels_np, preds_np)
    return {
        "accuracy": test["accuracy_at_0_5"]["accuracy"] * 100.0,
        "f1": f1,
        "auroc": test["auroc"],
        "ap": test["average_precision"],
        "num_samples": test["num_samples"],
        "num_normal": test["num_normal"],
        "num_anomaly": test["num_anomaly"],
    }


def read_cladapter_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test = metrics["test"]
    return {
        "accuracy": float(test["acc"]),
        "f1": float(test["f1"]),
        "auroc": test["roc"],
        "ap": test["ap"],
        "num_samples": None,
        "num_normal": None,
        "num_anomaly": None,
    }


def fmt_pct_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f}% ± {std:.2f}"


def fmt_float_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def summarize(values: list[float | None]) -> tuple[float, float]:
    clean = np.array([float(value) for value in values if value is not None and not math.isnan(float(value))], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.run_root / "RESULTS.md"
    rows = []
    for fold_dir in sorted(args.run_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = int(fold_dir.name.split("_")[-1])
        for model_name, rel_dir, kind in MODELS:
            metrics_path = fold_dir / rel_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            values = read_local_metrics(metrics_path) if kind == "local" else read_cladapter_metrics(metrics_path)
            rows.append({"fold": fold, "model": model_name, **values, "metrics_path": str(metrics_path)})

    if not rows:
        raise RuntimeError(f"No metrics.json files found under {args.run_root}")

    fold_df = pd.DataFrame(rows).sort_values(["model", "fold"])
    fold_df.to_csv(args.run_root / "fold_metrics.csv", index=False)

    summary_rows = []
    for model_name, _rel_dir, _kind in MODELS:
        part = fold_df[fold_df["model"] == model_name]
        if part.empty:
            continue
        acc_mean, acc_std = summarize(part["accuracy"].tolist())
        f1_mean, f1_std = summarize(part["f1"].tolist())
        auroc_mean, auroc_std = summarize(part["auroc"].tolist())
        ap_mean, ap_std = summarize(part["ap"].tolist())
        summary_rows.append(
            {
                "model": model_name,
                "folds": int(part["fold"].nunique()),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "ap_mean": ap_mean,
                "ap_std": ap_std,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.run_root / "summary.csv", index=False)
    (args.run_root / "summary.json").write_text(
        json.dumps({"run_root": str(args.run_root), "split_dir": str(args.split_dir), "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_summary = pd.read_csv(args.split_dir / "fold_summary.csv")
    n_folds = int(split_summary["fold"].nunique())
    lines = [
        f"# KEPCO {n_folds}-Fold Group CV Results",
        "",
        "## Setting",
        "",
        "- Dataset: `Final_Dataset/normal`, `Final_Dataset/anomaly`",
        "- Split: 5-fold group-stratified train/test cross-validation",
        "- Group key: original image name, so crops from the same original image stay in the same fold",
        "- Validation: not used",
        "- Selection: final epoch",
        "- Decision rule: fixed classifier decision / argmax, no test-threshold search",
        "- Epochs: 100",
        "- Image size: 224",
        "",
        "## Fold Split Counts",
        "",
        "| Fold | Train Normal | Train Anomaly | Train Total | Test Normal | Test Anomaly | Test Total | Test Normal Ratio | Group Overlap |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in split_summary.to_dict("records"):
        lines.append(
            "| {fold} | {train_normal} | {train_anomaly} | {train_total} | {test_normal} | {test_anomaly} | {test_total} | {ratio:.2%} | {overlap} |".format(
                fold=int(row["fold"]),
                train_normal=int(row["train_normal"]),
                train_anomaly=int(row["train_anomaly"]),
                train_total=int(row["train_total"]),
                test_normal=int(row["test_normal"]),
                test_anomaly=int(row["test_anomaly"]),
                test_total=int(row["test_total"]),
                ratio=float(row["test_normal_ratio"]),
                overlap=int(row["group_overlap"]),
            )
        )

    lines += [
        "",
        "## Mean Performance",
        "",
        "| Model | Accuracy | F1-score | AUROC | AP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | "
            f"{fmt_pct_mean_std(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{fmt_pct_mean_std(row['f1_mean'], row['f1_std'])} | "
            f"{fmt_float_mean_std(row['auroc_mean'], row['auroc_std'])} | "
            f"{fmt_float_mean_std(row['ap_mean'], row['ap_std'])} |"
        )

    lines += [
        "",
        "## Per-Fold Metrics",
        "",
        "| Fold | Model | Accuracy | F1-score | AUROC | AP |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_df.sort_values(["fold", "model"]).to_dict("records"):
        lines.append(
            f"| {int(row['fold'])} | {row['model']} | {row['accuracy']:.2f}% | {row['f1']:.2f}% | {row['auroc']:.4f} | {row['ap']:.4f} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
