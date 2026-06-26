#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


PKG_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PKG_ROOT / "engine" / "cladapter_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_model import CLAdapter_CLIP_ViT  # noqa: E402
from dataset import CLAdapterDataset  # noqa: E402
from utils import config_from_name  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    rel_dir: str
    config_name: str | None = None
    model_mode: str | None = None
    finetune_mode: str | None = None
    backbone_name: str | None = None
    backbone_out_dim: int | None = None
    backbone_num_patch: int | None = None
    checkpoint_name: str | None = None
    norm: str = "clip"


MODELS = [
    ModelSpec("ResNet50", "local", "local_supervised/resnet50"),
    ModelSpec(
        "ResNet50 + CLAdapter",
        "cladapter",
        "resnet50_cla_stage1",
        "config_clip_convnext",
        "res_xcep",
        "cla",
        "resnet50",
        2048,
        49,
        "resnet50_final.pth",
        "imagenet",
    ),
    ModelSpec(
        "ConvNeXt-B",
        "cladapter",
        "linear_convnextb",
        "config_clip_convnext",
        "conv",
        "linear",
        "convnext_base.clip_laion2b_augreg",
        1024,
        49,
        "convnext_base.clip_laion2b_augreg_final.pth",
        "clip",
    ),
    ModelSpec(
        "ConvNeXt-B + CLAdapter",
        "cladapter",
        "convnextb_cla_sft2",
        "config_clip_convnext",
        "conv",
        "cla",
        "convnext_base.clip_laion2b_augreg",
        1024,
        49,
        "convnext_base.clip_laion2b_augreg_final.pth",
        "clip",
    ),
    ModelSpec(
        "ViT-B",
        "cladapter",
        "linear_vitb",
        "config_clip_vit",
        "vit",
        "linear",
        "vit_base_patch16_clip_224.laion2b",
        768,
        196,
        "vit_base_patch16_clip_224.laion2b_final.pth",
        "clip",
    ),
    ModelSpec(
        "ViT-B + CLAdapter",
        "cladapter",
        "vitb_cla_sft2",
        "config_clip_vit",
        "vit",
        "cla",
        "vit_base_patch16_clip_224.laion2b",
        768,
        196,
        "vit_base_patch16_clip_224.laion2b_final.pth",
        "clip",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-threshold search for KEPCO k-fold checkpoints.")
    parser.add_argument("--run-root", type=Path, default=PKG_ROOT / "engine/runs/kfold_second_setting_20260529")
    parser.add_argument("--split-dir", type=Path, default=PKG_ROOT / "dataset/splits/kfold_second_setting")
    parser.add_argument("--data-root", type=Path, default=PKG_ROOT / "dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_name(name: str) -> str:
    lowered = name.lower().replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


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


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    accuracy = float(np.mean(preds == labels) * 100.0)
    precision, recall, f1 = custom_macro_pr_f1(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores.astype(np.float64))))
    best: dict[str, float] | None = None
    for threshold in candidates:
        current = metrics_at_threshold(labels, scores, float(threshold))
        current["threshold"] = float(threshold)
        if best is None:
            best = current
            continue
        if (current["accuracy"], current["f1"], -abs(current["threshold"] - 0.5)) > (
            best["accuracy"],
            best["f1"],
            -abs(best["threshold"] - 0.5),
        ):
            best = current
    if best is None:
        raise RuntimeError("No threshold candidates were generated.")
    return best


def binary_curve_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if np.unique(labels).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def read_local_scores(pred_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(pred_path)
    return df["path"].astype(str).tolist(), df["label"].to_numpy(np.int64), df["prob_anomaly"].to_numpy(np.float64)


def build_config(spec: ModelSpec, args: argparse.Namespace):
    if spec.config_name is None:
        raise ValueError(f"{spec.name} does not define a CLAdapter config.")
    config = config_from_name(spec.config_name)
    config.defrost()
    config.MODEL.m_mode = spec.model_mode
    config.MODEL.f_mode = spec.finetune_mode
    config.MODEL.num_classes = 2
    config.MODEL.img_size = args.image_size
    config.MODEL.output_dir = "threshold_search"
    config.MODEL.finetune = None
    config.MODEL.backbone.model_name = spec.backbone_name
    config.MODEL.backbone.out_dim = spec.backbone_out_dim
    config.MODEL.backbone.num_patch = spec.backbone_num_patch
    config.MODEL.backbone.set_new_allowed(True)
    config.MODEL.backbone.pretrained = False
    config.data_root = str(args.data_root)
    config.freeze()
    return config


def evaluate_cladapter_scores(
    spec: ModelSpec,
    ckpt_path: Path,
    split_csv: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(split_csv)
    dataset = CLAdapterDataset(False, df, 0, 1, "test", args.image_size, str(args.data_root), spec.norm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = CLAdapter_CLIP_ViT(build_config(spec, args)).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            scores = torch.softmax(logits.float(), dim=1)[:, 1]
            all_scores.append(scores.detach().cpu())
            all_labels.append(labels.detach().cpu())

    labels_np = torch.cat(all_labels).numpy().astype(np.int64)
    scores_np = torch.cat(all_scores).numpy().astype(np.float64)
    return dataset.images, labels_np, scores_np


def summarize(values: list[float]) -> tuple[float, float]:
    clean = np.array([float(value) for value in values if not math.isnan(float(value))], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def fmt_pct_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f}% ± {std:.2f}"


def fmt_float_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_markdown(run_root: Path, rows: list[dict], summary_rows: list[dict]) -> None:
    lines = [
        "# KEPCO K-Fold Threshold Search",
        "",
        "This is a test-best threshold search reference. It uses each fold's test labels to choose the threshold that maximizes accuracy, so it should be treated as an upper-bound/operating-point analysis, not as the strict fixed-threshold score.",
        "",
        "## Mean Performance",
        "",
        "| Model | Fixed 0.5 Accuracy | Best Accuracy | Best F1-score | AUROC | AP | Mean Best Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | "
            f"{fmt_pct_mean_std(row['fixed_accuracy_mean'], row['fixed_accuracy_std'])} | "
            f"{fmt_pct_mean_std(row['best_accuracy_mean'], row['best_accuracy_std'])} | "
            f"{fmt_pct_mean_std(row['best_f1_mean'], row['best_f1_std'])} | "
            f"{fmt_float_mean_std(row['auroc_mean'], row['auroc_std'])} | "
            f"{fmt_float_mean_std(row['ap_mean'], row['ap_std'])} | "
            f"{row['best_threshold_mean']:.4f} ± {row['best_threshold_std']:.4f} |"
        )

    lines += [
        "",
        "## Per-Fold Metrics",
        "",
        "| Fold | Model | Fixed 0.5 Acc. | Best Acc. | Best F1 | AUROC | AP | Best Threshold |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (int(item["fold"]), item["model"])):
        lines.append(
            f"| {int(row['fold'])} | {row['model']} | "
            f"{row['fixed_accuracy']:.2f}% | {row['best_accuracy']:.2f}% | "
            f"{row['best_f1']:.2f}% | {row['auroc']:.4f} | {row['ap']:.4f} | "
            f"{row['best_threshold']:.6f} |"
        )
    (run_root / "THRESHOLD_SEARCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    pred_dir = args.run_root / "threshold_search_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fold_dirs = sorted(path for path in args.run_root.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {args.run_root}")

    for fold_dir in fold_dirs:
        fold = int(fold_dir.name.split("_")[-1])
        split_csv = args.split_dir / f"fold_{fold}.csv"
        for spec in MODELS:
            print(f"[fold {fold}] {spec.name}", flush=True)
            if spec.kind == "local":
                pred_path = fold_dir / spec.rel_dir / "predictions_test_at_0_5.csv"
                paths, labels, scores = read_local_scores(pred_path)
            else:
                if spec.checkpoint_name is None:
                    raise ValueError(f"{spec.name} is missing checkpoint_name")
                ckpt_path = fold_dir / spec.rel_dir / spec.checkpoint_name
                paths, labels, scores = evaluate_cladapter_scores(spec, ckpt_path, split_csv, args, device)

            fixed = metrics_at_threshold(labels, scores, 0.5)
            best = best_threshold(labels, scores)
            auroc, ap = binary_curve_metrics(labels, scores)

            pred_df = pd.DataFrame(
                {
                    "path": paths,
                    "label": labels,
                    "prob_anomaly": scores,
                    "pred_at_0_5": (scores >= 0.5).astype(np.int64),
                    "pred_at_best_threshold": (scores >= best["threshold"]).astype(np.int64),
                }
            )
            pred_df.to_csv(pred_dir / f"fold_{fold}_{safe_name(spec.name)}.csv", index=False)

            rows.append(
                {
                    "fold": fold,
                    "model": spec.name,
                    "fixed_accuracy": fixed["accuracy"],
                    "fixed_f1": fixed["f1"],
                    "best_accuracy": best["accuracy"],
                    "best_precision": best["precision"],
                    "best_recall": best["recall"],
                    "best_f1": best["f1"],
                    "best_threshold": best["threshold"],
                    "auroc": auroc,
                    "ap": ap,
                    "num_samples": int(labels.size),
                    "num_normal": int(np.sum(labels == 0)),
                    "num_anomaly": int(np.sum(labels == 1)),
                }
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_df = pd.DataFrame(rows).sort_values(["model", "fold"])
    fold_df.to_csv(args.run_root / "threshold_search_fold_metrics.csv", index=False)

    summary_rows: list[dict] = []
    for spec in MODELS:
        part = fold_df[fold_df["model"] == spec.name]
        if part.empty:
            continue
        fixed_acc_mean, fixed_acc_std = summarize(part["fixed_accuracy"].tolist())
        best_acc_mean, best_acc_std = summarize(part["best_accuracy"].tolist())
        best_f1_mean, best_f1_std = summarize(part["best_f1"].tolist())
        auroc_mean, auroc_std = summarize(part["auroc"].tolist())
        ap_mean, ap_std = summarize(part["ap"].tolist())
        threshold_mean, threshold_std = summarize(part["best_threshold"].tolist())
        summary_rows.append(
            {
                "model": spec.name,
                "folds": int(part["fold"].nunique()),
                "fixed_accuracy_mean": fixed_acc_mean,
                "fixed_accuracy_std": fixed_acc_std,
                "best_accuracy_mean": best_acc_mean,
                "best_accuracy_std": best_acc_std,
                "best_f1_mean": best_f1_mean,
                "best_f1_std": best_f1_std,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "ap_mean": ap_mean,
                "ap_std": ap_std,
                "best_threshold_mean": threshold_mean,
                "best_threshold_std": threshold_std,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.run_root / "threshold_search_summary.csv", index=False)
    (args.run_root / "threshold_search_summary.json").write_text(
        json.dumps({"run_root": str(args.run_root), "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.run_root, rows, summary_rows)

    print("")
    print(summary_df.to_string(index=False))
    print("")
    print(args.run_root / "THRESHOLD_SEARCH.md")


if __name__ == "__main__":
    main()
