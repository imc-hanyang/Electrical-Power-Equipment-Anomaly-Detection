#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PKG_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ViT-B + CLAdapter over multiple k-fold runs.")
    parser.add_argument("--folds", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--run-template", default="kfold{k}_vitb_cladapter_second_setting_20260530")
    parser.add_argument("--fallback-run-template", default="kfold{k}_train_val_test_second_setting_20260529")
    parser.add_argument("--split-template", default="kfold{k}_train_val_test_second_setting")
    parser.add_argument("--output", type=Path, default=PKG_ROOT / "engine/runs/vitb_cladapter_5to10fold_summary_20260530.md")
    return parser.parse_args()


def summarize(values: pd.Series) -> tuple[float, float, float, float]:
    arr = values.astype(float).to_numpy()
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, float(arr.min()), float(arr.max())


def fmt_pct(mean: float, std: float, min_value: float, max_value: float) -> str:
    return f"{mean:.2f}% +/- {std:.2f} ({min_value:.2f}~{max_value:.2f})"


def fmt_float(mean: float, std: float, min_value: float, max_value: float) -> str:
    return f"{mean:.4f} +/- {std:.4f} ({min_value:.4f}~{max_value:.4f})"


def count_rows(split_dir: Path, k: int) -> list[dict]:
    rows = []
    for fold in range(k):
        df = pd.read_csv(split_dir / f"fold_{fold}.csv")
        row = {"k": k, "fold": f"fold_{fold}"}
        for split in ["train", "val", "test"]:
            part = df[df["split"] == split]
            normal = int((part["label"] == 0).sum())
            anomaly = int((part["label"] == 1).sum())
            row[f"{split}_normal"] = normal
            row[f"{split}_anomaly"] = anomaly
            row[f"{split}_total"] = int(len(part))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    perf_rows = []
    count_all = []
    for k in args.folds:
        run_dir = PKG_ROOT / "engine/runs" / args.run_template.format(k=k)
        if not (run_dir / "validation_threshold_metrics.csv").exists():
            run_dir = PKG_ROOT / "engine/runs" / args.fallback_run_template.format(k=k)
        metrics_path = run_dir / "validation_threshold_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = pd.read_csv(metrics_path)
        part = metrics[metrics["model"] == "ViT-B + CLAdapter"]
        if part.empty:
            raise ValueError(f"ViT-B + CLAdapter is missing in {metrics_path}")
        acc = summarize(part["test_accuracy"])
        f1 = summarize(part["test_f1"])
        fixed = summarize(part["fixed_accuracy"])
        auroc = summarize(part["auroc"])
        ap = summarize(part["ap"])
        perf_rows.append(
            {
                "k": k,
                "units": int(part["unit"].nunique()),
                "test_accuracy": acc,
                "test_f1": f1,
                "fixed_accuracy": fixed,
                "auroc": auroc,
                "ap": ap,
                "run_dir": str(run_dir),
            }
        )
        split_dir = PKG_ROOT / "dataset/splits" / args.split_template.format(k=k)
        count_all.extend(count_rows(split_dir, k))

    lines = [
        "# ViT-B + CLAdapter 5-10 Fold Summary",
        "",
        "Dataset: `Final_Dataset` without `etc`.",
        "",
        "Metric format: `mean +/- std (min~max)`.",
        "",
        "## Performance",
        "",
        "| Fold | Test Acc. @ Val Threshold | Test F1 @ Val Threshold | Fixed 0.5 Acc. | AUROC | AP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    json_rows = []
    for row in perf_rows:
        lines.append(
            f"| {row['k']} | "
            f"{fmt_pct(*row['test_accuracy'])} | "
            f"{fmt_pct(*row['test_f1'])} | "
            f"{fmt_pct(*row['fixed_accuracy'])} | "
            f"{fmt_float(*row['auroc'])} | "
            f"{fmt_float(*row['ap'])} |"
        )
        json_rows.append({key: value for key, value in row.items() if key != "run_dir"} | {"run_dir": row["run_dir"]})

    lines += [
        "",
        "## Fold Counts",
        "",
        "`N/A/T` means `normal/anomaly/total`.",
        "",
        "| K | Fold | Train N/A/T | Val N/A/T | Test N/A/T |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in count_all:
        lines.append(
            f"| {row['k']} | {row['fold']} | "
            f"{row['train_normal']}/{row['train_anomaly']}/{row['train_total']} | "
            f"{row['val_normal']}/{row['val_anomaly']}/{row['val_total']} | "
            f"{row['test_normal']}/{row['test_anomaly']}/{row['test_total']} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output.with_suffix(".json")).write_text(
        json.dumps({"performance": json_rows, "counts": count_all}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
