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
        "resnet50_best.pth",
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
        "convnext_base.clip_laion2b_augreg_best.pth",
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
        "convnext_base.clip_laion2b_augreg_best.pth",
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
        "vit_base_patch16_clip_224.laion2b_best.pth",
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
        "vit_base_patch16_clip_224.laion2b_best.pth",
        "clip",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply validation-searched thresholds to KEPCO test sets.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PKG_ROOT / "dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title", default="KEPCO Validation-Threshold Results")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model names to evaluate. Defaults to all comparison models.",
    )
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
        if best is None or (current["accuracy"], current["f1"], -abs(current["threshold"] - 0.5)) > (
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


def build_config(spec: ModelSpec, args: argparse.Namespace):
    if spec.config_name is None:
        raise ValueError(f"{spec.name} does not define a CLAdapter config.")
    config = config_from_name(spec.config_name)
    config.defrost()
    config.MODEL.m_mode = spec.model_mode
    config.MODEL.f_mode = spec.finetune_mode
    config.MODEL.num_classes = 2
    config.MODEL.img_size = args.image_size
    config.MODEL.output_dir = "validation_threshold"
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
    split_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(split_csv)
    dataset = CLAdapterDataset(False, df, 0, 1, split_name, args.image_size, str(args.data_root), spec.norm)
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
    return dataset.images, torch.cat(all_labels).numpy().astype(np.int64), torch.cat(all_scores).numpy().astype(np.float64)


def read_local_metrics(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    val = metrics["val"]
    test_at_val = metrics["test_at_val_best_threshold"]
    test_precision, test_recall = macro_pr_from_confusion(test_at_val["confusion"])
    return {
        "val_threshold": val["best_threshold"]["threshold"],
        "val_accuracy_at_threshold": val["best_threshold"]["accuracy"] * 100.0,
        "test_accuracy": test_at_val["accuracy"] * 100.0,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": macro_f1_from_confusion(test_at_val["confusion"]),
        "fixed_accuracy": test["accuracy_at_0_5"]["accuracy"] * 100.0,
        "fixed_f1": macro_f1_from_confusion(test["accuracy_at_0_5"]["confusion"]),
        "auroc": test["auroc"],
        "ap": test["average_precision"],
        "best_epoch": metrics.get("best_epoch"),
    }


def macro_pr_from_confusion(confusion: dict) -> tuple[float, float]:
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
    return precision, recall


def macro_f1_from_confusion(confusion: dict) -> float:
    precision, recall = macro_pr_from_confusion(confusion)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def selected_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if not args.models:
        return MODELS
    requested = set(args.models)
    specs = [spec for spec in MODELS if spec.name in requested]
    missing = sorted(requested - {spec.name for spec in specs})
    if missing:
        raise ValueError(f"Unknown model name(s): {missing}")
    return specs


def portable_image_paths(paths: list[str], data_root: Path) -> list[str]:
    """Store prediction paths relative to the package root when possible."""
    package_root = data_root.resolve().parent
    portable = []
    for path in paths:
        p = Path(path)
        resolved = p.resolve() if p.exists() else p
        try:
            portable.append(str(resolved.relative_to(package_root)))
        except ValueError:
            portable.append(str(path))
    return portable


def evaluate_unit(
    unit_name: str,
    unit_run_dir: Path,
    split_csv: Path,
    args: argparse.Namespace,
    device: torch.device,
    specs: list[ModelSpec],
) -> list[dict]:
    pred_dir = unit_run_dir / "validation_threshold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        print(f"[{unit_name}] {spec.name}", flush=True)
        model_dir = unit_run_dir / spec.rel_dir
        if spec.kind == "local":
            metrics = read_local_metrics(model_dir)
        else:
            if spec.checkpoint_name is None:
                raise ValueError(f"{spec.name} is missing checkpoint_name")
            ckpt_path = model_dir / spec.checkpoint_name
            val_paths, val_labels, val_scores = evaluate_cladapter_scores(spec, ckpt_path, split_csv, "valid", args, device)
            test_paths, test_labels, test_scores = evaluate_cladapter_scores(spec, ckpt_path, split_csv, "test", args, device)
            val_best = best_threshold(val_labels, val_scores)
            test_at_val = metrics_at_threshold(test_labels, test_scores, val_best["threshold"])
            fixed = metrics_at_threshold(test_labels, test_scores, 0.5)
            auroc, ap = binary_curve_metrics(test_labels, test_scores)
            metrics = {
                "val_threshold": val_best["threshold"],
                "val_accuracy_at_threshold": val_best["accuracy"],
                "test_accuracy": test_at_val["accuracy"],
                "test_precision": test_at_val["precision"],
                "test_recall": test_at_val["recall"],
                "test_f1": test_at_val["f1"],
                "fixed_accuracy": fixed["accuracy"],
                "fixed_f1": fixed["f1"],
                "auroc": auroc,
                "ap": ap,
                "best_epoch": json.loads((model_dir / "metrics.json").read_text(encoding="utf-8")).get("best_epoch"),
            }
            pd.DataFrame(
                {
                    "path": portable_image_paths(val_paths, args.data_root),
                    "split": "val",
                    "label": val_labels,
                    "prob_anomaly": val_scores,
                    "pred_at_val_threshold": (val_scores >= val_best["threshold"]).astype(np.int64),
                }
            ).to_csv(pred_dir / f"{safe_name(spec.name)}_val.csv", index=False)
            pd.DataFrame(
                {
                    "path": portable_image_paths(test_paths, args.data_root),
                    "split": "test",
                    "label": test_labels,
                    "prob_anomaly": test_scores,
                    "pred_at_0_5": (test_scores >= 0.5).astype(np.int64),
                    "pred_at_val_threshold": (test_scores >= val_best["threshold"]).astype(np.int64),
                }
            ).to_csv(pred_dir / f"{safe_name(spec.name)}_test.csv", index=False)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append({"unit": unit_name, "model": spec.name, **metrics})
    return rows


def summarize(values: list[float]) -> tuple[float, float]:
    clean = np.array([float(value) for value in values if value is not None and not math.isnan(float(value))], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if clean.size > 1 else 0.0


def fmt_pct_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f}%" if std == 0 else f"{mean:.2f}% ± {std:.2f}"


def fmt_float_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f}" if std == 0 else f"{mean:.4f} ± {std:.4f}"


def write_markdown(run_root: Path, title: str, fold_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        f"# {title}",
        "",
        "Decision rule: validation-set threshold search, then fixed application to the test set.",
        "",
        "## Mean Performance",
        "",
        "| Model | Test Acc. @ Val Threshold | Test Precision | Test Recall | Test F1 @ Val Threshold | Fixed 0.5 Acc. | AUROC | AP | Val Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            f"| {row['model']} | "
            f"{fmt_pct_mean_std(row['test_accuracy_mean'], row['test_accuracy_std'])} | "
            f"{fmt_pct_mean_std(row['test_precision_mean'], row['test_precision_std'])} | "
            f"{fmt_pct_mean_std(row['test_recall_mean'], row['test_recall_std'])} | "
            f"{fmt_pct_mean_std(row['test_f1_mean'], row['test_f1_std'])} | "
            f"{fmt_pct_mean_std(row['fixed_accuracy_mean'], row['fixed_accuracy_std'])} | "
            f"{fmt_float_mean_std(row['auroc_mean'], row['auroc_std'])} | "
            f"{fmt_float_mean_std(row['ap_mean'], row['ap_std'])} | "
            f"{fmt_float_mean_std(row['val_threshold_mean'], row['val_threshold_std'])} |"
        )

    lines += [
        "",
        "## Per-Unit Metrics",
        "",
        "| Unit | Model | Test Acc. @ Val Threshold | Test Precision | Test Recall | Test F1 | Fixed 0.5 Acc. | AUROC | AP | Val Threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_df.sort_values(["unit", "model"]).to_dict("records"):
        lines.append(
            f"| {row['unit']} | {row['model']} | {row['test_accuracy']:.2f}% | {row['test_precision']:.2f}% | "
            f"{row['test_recall']:.2f}% | {row['test_f1']:.2f}% | "
            f"{row['fixed_accuracy']:.2f}% | {row['auroc']:.4f} | {row['ap']:.4f} | {row['val_threshold']:.6f} |"
        )
    (run_root / "VALIDATION_THRESHOLD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def portable_run_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PKG_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.split_csv is None and args.split_dir is None:
        raise ValueError("Either --split-csv or --split-dir is required.")
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    specs = selected_model_specs(args)

    units: list[tuple[str, Path, Path]] = []
    if args.split_dir is not None:
        for fold_dir in sorted(path for path in args.run_root.glob("fold_*") if path.is_dir()):
            fold = int(fold_dir.name.split("_")[-1])
            units.append((f"fold_{fold}", fold_dir, args.split_dir / f"fold_{fold}.csv"))
    else:
        units.append(("single", args.run_root, args.split_csv))

    rows: list[dict] = []
    for unit_name, unit_run_dir, split_csv in units:
        rows.extend(evaluate_unit(unit_name, unit_run_dir, split_csv, args, device, specs))

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(args.run_root / "validation_threshold_metrics.csv", index=False)
    summary_rows = []
    for model in [spec.name for spec in specs]:
        part = fold_df[fold_df["model"] == model]
        if part.empty:
            continue
        acc_mean, acc_std = summarize(part["test_accuracy"].tolist())
        prec_mean, prec_std = summarize(part["test_precision"].tolist())
        rec_mean, rec_std = summarize(part["test_recall"].tolist())
        f1_mean, f1_std = summarize(part["test_f1"].tolist())
        fixed_mean, fixed_std = summarize(part["fixed_accuracy"].tolist())
        auroc_mean, auroc_std = summarize(part["auroc"].tolist())
        ap_mean, ap_std = summarize(part["ap"].tolist())
        threshold_mean, threshold_std = summarize(part["val_threshold"].tolist())
        summary_rows.append(
            {
                "model": model,
                "units": int(part["unit"].nunique()),
                "test_accuracy_mean": acc_mean,
                "test_accuracy_std": acc_std,
                "test_precision_mean": prec_mean,
                "test_precision_std": prec_std,
                "test_recall_mean": rec_mean,
                "test_recall_std": rec_std,
                "test_f1_mean": f1_mean,
                "test_f1_std": f1_std,
                "fixed_accuracy_mean": fixed_mean,
                "fixed_accuracy_std": fixed_std,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "ap_mean": ap_mean,
                "ap_std": ap_std,
                "val_threshold_mean": threshold_mean,
                "val_threshold_std": threshold_std,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.run_root / "validation_threshold_summary.csv", index=False)
    (args.run_root / "validation_threshold_summary.json").write_text(
        json.dumps({"run_root": portable_run_path(args.run_root), "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.run_root, args.title, fold_df, summary_df)

    print("")
    print(summary_df.to_string(index=False))
    print("")
    print(args.run_root / "VALIDATION_THRESHOLD.md")


if __name__ == "__main__":
    main()
