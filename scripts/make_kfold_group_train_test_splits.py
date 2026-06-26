#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make group-stratified KEPCO k-fold train/test CSVs without validation."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "second_split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "kfold_second_setting",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=20260529)
    parser.add_argument("--seed-trials", type=int, default=300)
    return parser.parse_args()


def split_score(df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    total = len(df)
    target_test = total / len(folds)
    target_normal = int((df["label"] == 0).sum()) / len(folds)
    target_anomaly = int((df["label"] == 1).sum()) / len(folds)
    target_ratio = int((df["label"] == 0).sum()) / total
    score = 0.0
    for _train_idx, test_idx in folds:
        part = df.iloc[test_idx]
        normal = int((part["label"] == 0).sum())
        anomaly = int((part["label"] == 1).sum())
        count = len(part)
        ratio = normal / count if count else 0.0
        score += abs(count - target_test) * 2.0
        score += abs(normal - target_normal) * 8.0
        score += abs(anomaly - target_anomaly) * 8.0
        score += abs(ratio - target_ratio) * 100.0
    return score


def make_folds(df: pd.DataFrame, n_splits: int, seed_start: int, seed_trials: int) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    labels = df["label"].astype(int).to_numpy()
    groups = df["group"].astype(str).to_numpy()
    best_seed = seed_start
    best_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_score = float("inf")
    for offset in range(seed_trials):
        seed = seed_start + offset
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(df)), labels, groups))
        score = split_score(df, folds)
        if score < best_score:
            best_seed = seed
            best_folds = folds
            best_score = score
    if best_folds is None:
        raise RuntimeError("Could not build k-fold splits.")
    return best_seed, best_folds


def write_fold_csv(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_indices = set(int(index) for index in test_idx)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split", "group", "label_name"])
        writer.writeheader()
        for index, row in df.reset_index(drop=True).iterrows():
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "label": int(row["label"]),
                    "split": "test" if int(index) in test_indices else "train",
                    "group": row["group"],
                    "label_name": row["label_name"],
                }
            )


def split_summary(df: pd.DataFrame, fold: int, train_idx: np.ndarray, test_idx: np.ndarray, csv_path: Path) -> dict:
    row: dict[str, int | float | str] = {"fold": fold, "csv": str(csv_path)}
    for split_name, indices in [("train", train_idx), ("test", test_idx)]:
        part = df.iloc[indices]
        total = len(part)
        normal = int((part["label"] == 0).sum())
        anomaly = int((part["label"] == 1).sum())
        row[f"{split_name}_total"] = total
        row[f"{split_name}_normal"] = normal
        row[f"{split_name}_anomaly"] = anomaly
        row[f"{split_name}_normal_ratio"] = normal / total if total else 0.0
        row[f"{split_name}_groups"] = int(part["group"].nunique())
    train_groups = set(df.iloc[train_idx]["group"].astype(str))
    test_groups = set(df.iloc[test_idx]["group"].astype(str))
    row["group_overlap"] = len(train_groups & test_groups)
    return row


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    seed, folds = make_folds(df, args.n_splits, args.seed_start, args.seed_trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        output_csv = args.output_dir / f"fold_{fold}.csv"
        write_fold_csv(df, train_idx, test_idx, output_csv)
        summaries.append(split_summary(df, fold, train_idx, test_idx, output_csv))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "fold_summary.csv", index=False)
    metadata = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "n_splits": args.n_splits,
        "selected_seed": seed,
        "rows": len(df),
        "normal": int((df["label"] == 0).sum()),
        "anomaly": int((df["label"] == 1).sum()),
        "summaries": summaries,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output_dir}")
    print(f"selected_seed={seed}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
