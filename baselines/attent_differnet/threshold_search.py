from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search score thresholds for maximum classification accuracy.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/home/opgw/KEPCO_May/engine/runs/gpu3_20260528"),
        help="Run directory containing none/se/cbam subdirectories.",
    )
    parser.add_argument("--models", nargs="+", default=["none", "se", "cbam"])
    return parser.parse_args()


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    unique = np.unique(scores)
    thresholds = [float(unique[0] - 1e-12)]
    thresholds.extend(float(value) for value in (unique[:-1] + unique[1:]) / 2)
    thresholds.append(float(unique[-1] + 1e-12))
    return thresholds


def score_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    normal_acc = tn / (tn + fp) if (tn + fp) else 0.0
    anomaly_acc = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (normal_acc + anomaly_acc) / 2,
        "normal_acc": normal_acc,
        "anomaly_acc": anomaly_acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def search_model(run_root: Path, model_name: str) -> dict:
    scores_path = run_root / model_name / "scores_best.json"
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    labels = np.array([row["label"] for row in data["scores"]], dtype=np.int64)
    scores = np.array([row["score"] for row in data["scores"]], dtype=np.float64)

    rows = [score_threshold(labels, scores, threshold) for threshold in candidate_thresholds(scores)]
    best = max(rows, key=lambda row: (row["accuracy"], row["balanced_accuracy"]))
    best = {"model": model_name, **best, "num_samples": int(len(labels))}

    detail_path = run_root / model_name / "threshold_accuracy_search.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *rows[0].keys(), "num_samples"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": model_name, **row, "num_samples": int(len(labels))})

    return best


def main() -> int:
    args = parse_args()
    summary = [search_model(args.run_root, model_name) for model_name in args.models]
    summary_path = args.run_root / "threshold_accuracy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(summary_path)
    for row in summary:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
