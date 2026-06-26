#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make an old-experiment-like KEPCO train/test split for exploratory comparison."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "first_split.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "splits" / "first_split_favorable_candidate.csv",
    )
    parser.add_argument("--target-normal", type=int, default=20)
    parser.add_argument("--target-anomaly", type=int, default=26)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--tries", type=int, default=500)
    return parser.parse_args()


def source_type(group: str) -> str:
    return "dsc" if group.startswith("DSC") or group.startswith("_DSC") else "field"


def group_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for group, part in df.groupby("group", sort=False):
        group = str(group)
        normal = int((part["label"] == 0).sum())
        anomaly = int((part["label"] == 1).sum())
        rows.append(
            {
                "group": group,
                "total": len(part),
                "normal": normal,
                "anomaly": anomaly,
                "mixed": normal > 0 and anomaly > 0,
                "source": source_type(group),
            }
        )
    return rows


def solve_subset(groups: list[dict], target_normal: int, target_anomaly: int, rng: random.Random) -> set[str]:
    shuffled = groups[:]
    rng.shuffle(shuffled)

    dp: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for idx, row in enumerate(shuffled):
        additions = []
        for (normal, anomaly), selected in dp.items():
            next_key = (normal + row["normal"], anomaly + row["anomaly"])
            if next_key[0] <= target_normal and next_key[1] <= target_anomaly and next_key not in dp:
                additions.append((next_key, selected + (idx,)))
        for key, value in additions:
            dp.setdefault(key, value)

    key = (target_normal, target_anomaly)
    if key not in dp:
        raise RuntimeError("Could not find an exact group-preserving split.")
    return {shuffled[idx]["group"] for idx in dp[key]}


def score_split(df: pd.DataFrame, test_groups: set[str]) -> tuple[float, dict]:
    test = df[df["group"].astype(str).isin(test_groups)].copy()
    train = df[~df["group"].astype(str).isin(test_groups)].copy()

    test_group_df = test.groupby("group")["label"].nunique()
    mixed_groups = int((test_group_df > 1).sum())
    mixed_images = int(test[test["group"].isin(test_group_df[test_group_df > 1].index)].shape[0])
    test["source"] = test["group"].astype(str).map(source_type)
    field_normal = int(((test["label"] == 0) & (test["source"] == "field")).sum())

    train_groups = set(train["group"].astype(str))
    overlap = len(train_groups & test_groups)

    # Lower is better. This deliberately favors an easier, old-style test set:
    # few mixed-label groups and fewer field-named normal images.
    score = mixed_images * 10.0 + mixed_groups * 25.0 + field_normal * 3.0 + overlap * 1000.0
    details = {
        "mixed_groups": mixed_groups,
        "mixed_images": mixed_images,
        "field_normal": field_normal,
        "overlap": overlap,
        "test_groups": len(test_groups),
    }
    return score, details


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


def print_summary(df: pd.DataFrame, output_csv: Path, details: dict) -> None:
    print(f"wrote {output_csv}")
    print(pd.crosstab(df["split"], df["label_name"], margins=True))
    train_groups = set(df[df["split"] == "train"]["group"])
    test_groups = set(df[df["split"] == "test"]["group"])
    print(f"train groups: {len(train_groups)}")
    print(f"test groups: {len(test_groups)}")
    print(f"group overlap train-test: {len(train_groups & test_groups)}")
    print(f"test mixed-label groups: {details['mixed_groups']}")
    print(f"test mixed-label images: {details['mixed_images']}")
    print(f"test field-named normal images: {details['field_normal']}")
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
    df["group"] = df["group"].astype(str)

    groups = group_rows(df)
    best: tuple[float, set[str], dict] | None = None
    for offset in range(args.tries):
        rng = random.Random(args.seed + offset)
        test_groups = solve_subset(groups, args.target_normal, args.target_anomaly, rng)
        score, details = score_split(df, test_groups)
        if best is None or score < best[0]:
            best = (score, test_groups, details)

    if best is None:
        raise RuntimeError("No split candidate was generated.")

    _, test_groups, details = best
    write_split(df, test_groups, args.output_csv)
    out_df = pd.read_csv(args.output_csv)
    print_summary(out_df, args.output_csv, details)


if __name__ == "__main__":
    main()
