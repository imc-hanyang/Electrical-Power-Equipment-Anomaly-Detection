#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a no-validation group-stratified KEPCO train/test split.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "first_split.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "dataset"
        / "splits"
        / "kepco_group_balanced_train_test.csv",
    )
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def group_stats(df: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    stats = []
    for group, rows in df.groupby("group", sort=False):
        normal_count = int((rows["label"] == 0).sum())
        anomaly_count = int((rows["label"] == 1).sum())
        stats.append((str(group), len(rows), normal_count, anomaly_count))
    return stats


def choose_test_groups(stats: list[tuple[str, int, int, int]], target_total: int, target_normal: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    shuffled = stats[:]
    rng.shuffle(shuffled)

    # Dynamic programming over (num_images, num_normal_images), preserving complete groups.
    dp: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    max_total = max(target_total + 12, target_total)
    max_normal = max(target_normal + 12, target_normal)
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
        overall_normal_ratio = sum(row[2] for row in stats) / sum(row[1] for row in stats)
        best_key = None
        best_score = float("inf")
        for total, normal in dp:
            if total < max(1, target_total - 10):
                continue
            ratio = normal / total
            anomaly = total - normal
            target_anomaly = target_total - target_normal
            score = (
                abs(total - target_total) * 1.5
                + abs(normal - target_normal) * 5.0
                + abs(anomaly - target_anomaly) * 5.0
                + abs(ratio - overall_normal_ratio) * 100.0
            )
            if score < best_score:
                best_score = score
                best_key = (total, normal)
        if best_key is None:
            raise RuntimeError("Could not build a group-preserving train/test split.")
        selected_indices = dp[best_key]

    return {shuffled[idx][0] for idx in selected_indices}


def write_split(df: pd.DataFrame, test_groups: set[str], output_csv: Path) -> None:
    rows = []
    for row in df.to_dict("records"):
        row = dict(row)
        row["split"] = "test" if str(row["group"]) in test_groups else "train"
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split", "group", "label_name"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(df: pd.DataFrame, output_csv: Path) -> None:
    print(f"wrote {output_csv}")
    print(pd.crosstab(df["split"], df["label_name"], margins=True))
    train_groups = set(df[df["split"] == "train"]["group"])
    test_groups = set(df[df["split"] == "test"]["group"])
    print(f"train groups: {len(train_groups)}")
    print(f"test groups: {len(test_groups)}")
    print(f"group overlap train-test: {len(train_groups & test_groups)}")
    for split in ["train", "test"]:
        part = df[df["split"] == split]
        normal = int((part["label"] == 0).sum())
        total = len(part)
        print(f"{split} normal ratio: {normal}/{total} = {normal / total:.4f}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    total = len(df)
    total_normal = int((df["label"] == 0).sum())
    target_total = round(total * args.test_ratio)
    target_normal = round(total_normal * args.test_ratio)

    stats = group_stats(df)
    test_groups = choose_test_groups(stats, target_total, target_normal, args.seed)
    write_split(df, test_groups, args.output_csv)

    out_df = pd.read_csv(args.output_csv)
    print_summary(out_df, args.output_csv)


if __name__ == "__main__":
    main()
