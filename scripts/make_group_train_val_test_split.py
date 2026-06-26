#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a group-preserving KEPCO train/val/test split.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "second_split.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "dataset"
        / "splits"
        / "train_val_test_second_setting.csv",
    )
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed-start", type=int, default=20260529)
    parser.add_argument("--seed-trials", type=int, default=500)
    return parser.parse_args()


def group_stats(df: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    stats = []
    for group, rows in df.groupby("group", sort=False):
        normal_count = int((rows["label"] == 0).sum())
        anomaly_count = int((rows["label"] == 1).sum())
        stats.append((str(group), len(rows), normal_count, anomaly_count))
    return stats


def choose_groups(
    stats: list[tuple[str, int, int, int]],
    target_total: int,
    target_normal: int,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    shuffled = stats[:]
    rng.shuffle(shuffled)

    max_total = max(target_total + 12, target_total)
    max_normal = max(target_normal + 12, target_normal)
    dp: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for idx, (_group, total, normal, _anomaly) in enumerate(shuffled):
        additions = []
        for (cur_total, cur_normal), selected in dp.items():
            next_total = cur_total + total
            next_normal = cur_normal + normal
            if next_total <= max_total and next_normal <= max_normal and (next_total, next_normal) not in dp:
                additions.append(((next_total, next_normal), selected + (idx,)))
        for key, value in additions:
            dp.setdefault(key, value)

    if (target_total, target_normal) in dp:
        selected_indices = dp[(target_total, target_normal)]
    else:
        total_images = sum(row[1] for row in stats)
        total_normal = sum(row[2] for row in stats)
        overall_normal_ratio = total_normal / total_images if total_images else 0.0
        target_anomaly = target_total - target_normal
        best_key = None
        best_score = float("inf")
        for total, normal in dp:
            if total < max(1, target_total - 12):
                continue
            anomaly = total - normal
            ratio = normal / total if total else 0.0
            score = (
                abs(total - target_total) * 2.0
                + abs(normal - target_normal) * 8.0
                + abs(anomaly - target_anomaly) * 8.0
                + abs(ratio - overall_normal_ratio) * 100.0
            )
            if score < best_score:
                best_score = score
                best_key = (total, normal)
        if best_key is None:
            raise RuntimeError("Could not choose split groups.")
        selected_indices = dp[best_key]

    return {shuffled[idx][0] for idx in selected_indices}


def split_score(df: pd.DataFrame) -> float:
    target_ratio = int((df["label"] == 0).sum()) / len(df)
    score = 0.0
    for split, weight in [("train", 0.5), ("val", 1.0), ("test", 1.0)]:
        part = df[df["split"] == split]
        if part.empty:
            score += 9999.0
            continue
        ratio = int((part["label"] == 0).sum()) / len(part)
        score += abs(ratio - target_ratio) * 100.0 * weight
    return score


def build_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    total = len(df)
    total_normal = int((df["label"] == 0).sum())
    test_groups = choose_groups(group_stats(df), round(total * test_ratio), round(total_normal * test_ratio), seed)
    remaining = df[~df["group"].astype(str).isin(test_groups)].copy()
    val_groups = choose_groups(
        group_stats(remaining),
        round(total * val_ratio),
        round(total_normal * val_ratio),
        seed + 1009,
    )
    rows = []
    for row in df.to_dict("records"):
        row = dict(row)
        group = str(row["group"])
        if group in test_groups:
            row["split"] = "test"
        elif group in val_groups:
            row["split"] = "val"
        else:
            row["split"] = "train"
        rows.append(row)
    return pd.DataFrame(rows)


def split_summary(df: pd.DataFrame) -> list[dict]:
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
                "groups": int(part["group"].nunique()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    best_seed = args.seed_start
    best_df = None
    best_score = float("inf")
    for offset in range(args.seed_trials):
        seed = args.seed_start + offset
        candidate = build_split(df, args.val_ratio, args.test_ratio, seed)
        score = split_score(candidate)
        if score < best_score:
            best_seed = seed
            best_df = candidate
            best_score = score
    if best_df is None:
        raise RuntimeError("Could not build a train/val/test split.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split", "group", "label_name"])
        writer.writeheader()
        for row in best_df.to_dict("records"):
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "label": int(row["label"]),
                    "split": row["split"],
                    "group": row["group"],
                    "label_name": row["label_name"],
                }
            )

    summaries = split_summary(best_df)
    metadata = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "selected_seed": best_seed,
        "rows": len(best_df),
        "normal": int((best_df["label"] == 0).sum()),
        "anomaly": int((best_df["label"] == 1).sum()),
        "summary": summaries,
    }
    args.output_csv.with_suffix(".summary.csv").write_text(pd.DataFrame(summaries).to_csv(index=False), encoding="utf-8")
    args.output_csv.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"wrote {args.output_csv}")
    print(f"selected_seed={best_seed}")
    print(pd.DataFrame(summaries).to_string(index=False))
    split_groups = {split: set(best_df[best_df["split"] == split]["group"].astype(str)) for split in ["train", "val", "test"]}
    print(f"train-val overlap: {len(split_groups['train'] & split_groups['val'])}")
    print(f"train-test overlap: {len(split_groups['train'] & split_groups['test'])}")
    print(f"val-test overlap: {len(split_groups['val'] & split_groups['test'])}")


if __name__ == "__main__":
    main()
