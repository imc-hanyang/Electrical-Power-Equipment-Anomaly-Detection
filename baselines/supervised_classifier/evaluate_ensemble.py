from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from attent_differnet.data import ImageRecord
from supervised_classifier.train import (
    create_model,
    make_transforms,
    threshold_metrics,
)


@dataclass(frozen=True)
class Split:
    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]


class ImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), torch.tensor(record.label, dtype=torch.long), str(record.path), record.group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mean/weighted ensembles from supervised best.pt checkpoints.")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--weight-grid-step", type=float, default=0.01)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def label_id(label_name: str) -> int:
    if label_name == "normal":
        return 0
    if label_name == "anomaly":
        return 1
    raise ValueError(f"Unsupported label: {label_name}")


def read_split(path: Path) -> Split:
    buckets: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = row["split"]
            if split not in buckets:
                continue
            label_name = row["label"]
            buckets[split].append(
                ImageRecord(
                    path=Path(row["path"]),
                    label=label_id(label_name),
                    label_name=label_name,
                    group=row["group"],
                )
            )
    return Split(train=buckets["train"], val=buckets["val"], test=buckets["test"])


def assert_same_records(reference: list[ImageRecord], candidate: list[ImageRecord], split_name: str) -> None:
    ref_keys = [(str(record.path), record.label, record.group) for record in reference]
    cand_keys = [(str(record.path), record.label, record.group) for record in candidate]
    if ref_keys != cand_keys:
        raise ValueError(f"Run split mismatch for {split_name}; ensemble members must use identical split files.")


def make_loader(records: list[ImageRecord], transform: transforms.Compose, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        ImageDataset(records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    model_name = ckpt_args["model"]
    model, _ = create_model(model_name, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt_args


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    groups: list[str] = []
    for images, batch_labels, batch_paths, batch_groups in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
        probs.append(batch_probs.detach().cpu().numpy())
        labels.append(batch_labels.numpy())
        paths.extend(batch_paths)
        groups.extend(batch_groups)
    return np.concatenate(probs), paths, groups, np.concatenate(labels)


def f1_from_confusion(confusion: dict) -> float:
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def metrics_for(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    metrics = threshold_metrics(labels, probs, threshold)
    metrics["f1"] = f1_from_confusion(metrics["confusion"])
    return metrics


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probs)))
    best: dict | None = None
    for threshold in candidates:
        metrics = metrics_for(labels, probs, float(threshold))
        key = (
            metrics["accuracy"],
            metrics["f1"],
            metrics["normal_accuracy"] or 0.0,
            metrics["anomaly_accuracy"] or 0.0,
        )
        if best is None or key > (
            best["accuracy"],
            best["f1"],
            best["normal_accuracy"] or 0.0,
            best["anomaly_accuracy"] or 0.0,
        ):
            best = metrics
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def summarize(labels: np.ndarray, probs: np.ndarray) -> dict:
    summary = {
        "num_samples": int(labels.shape[0]),
        "num_normal": int(np.sum(labels == 0)),
        "num_anomaly": int(np.sum(labels == 1)),
        "accuracy_at_0_5": metrics_for(labels, probs, 0.5),
        "best_threshold": best_threshold(labels, probs),
    }
    if len(np.unique(labels)) == 2:
        summary["auroc"] = float(roc_auc_score(labels, probs))
        summary["average_precision"] = float(average_precision_score(labels, probs))
    else:
        summary["auroc"] = None
        summary["average_precision"] = None
    return summary


def grid_weights(num_models: int, step: float) -> list[np.ndarray]:
    if num_models == 1:
        return [np.array([1.0], dtype=np.float64)]
    if num_models == 2:
        values = np.arange(0.0, 1.0 + step / 2, step)
        return [np.array([value, 1.0 - value], dtype=np.float64) for value in values]
    raw: list[np.ndarray] = []
    values = np.arange(0.0, 1.0 + step / 2, step)
    for first in values:
        for second in values:
            if first + second <= 1.0 + 1e-9:
                rest = 1.0 - first - second
                weights = np.array([first, second, rest], dtype=np.float64)
                if np.all(weights >= -1e-9):
                    raw.append(weights)
    if num_models > 3:
        return [np.ones(num_models, dtype=np.float64) / num_models]
    return raw


def weighted_mean(prob_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    return prob_matrix @ weights


def select_weights(labels: np.ndarray, prob_matrix: np.ndarray, step: float) -> tuple[np.ndarray, dict]:
    candidates = grid_weights(prob_matrix.shape[1], step)
    best_weights = candidates[0]
    best_metrics = summarize(labels, weighted_mean(prob_matrix, best_weights))
    for weights in candidates[1:]:
        probs = weighted_mean(prob_matrix, weights)
        metrics = summarize(labels, probs)
        key = (
            metrics["best_threshold"]["accuracy"],
            metrics["auroc"] or 0.0,
            metrics["best_threshold"]["normal_accuracy"] or 0.0,
            metrics["best_threshold"]["anomaly_accuracy"] or 0.0,
        )
        best_key = (
            best_metrics["best_threshold"]["accuracy"],
            best_metrics["auroc"] or 0.0,
            best_metrics["best_threshold"]["normal_accuracy"] or 0.0,
            best_metrics["best_threshold"]["anomaly_accuracy"] or 0.0,
        )
        if key > best_key:
            best_weights = weights
            best_metrics = metrics
    return best_weights / best_weights.sum(), best_metrics


def write_predictions(
    path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    paths: list[str],
    groups: list[str],
    threshold: float,
    member_probs: np.ndarray,
    member_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "group", "label", "label_name", "prob_anomaly", "pred", "pred_name", "correct"]
    fieldnames.extend([f"prob_{name}" for name in member_names])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (label, prob, image_path, group) in enumerate(zip(labels, probs, paths, groups)):
            pred = int(prob >= threshold)
            row = {
                "path": image_path,
                "group": group,
                "label": int(label),
                "label_name": "anomaly" if int(label) == 1 else "normal",
                "prob_anomaly": float(prob),
                "pred": pred,
                "pred_name": "anomaly" if pred else "normal",
                "correct": int(pred == int(label)),
            }
            for member_index, name in enumerate(member_names):
                row[f"prob_{name}"] = float(member_probs[index, member_index])
            writer.writerow(row)


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    split = read_split(args.run_dirs[0] / "split.csv")
    for run_dir in args.run_dirs[1:]:
        other = read_split(run_dir / "split.csv")
        assert_same_records(split.train, other.train, "train")
        assert_same_records(split.val, other.val, "val")
        assert_same_records(split.test, other.test, "test")

    _, eval_transform = make_transforms(224, "letterbox")
    # Prefer the checkpoint transform settings when they agree; current supervised runs all use 224/letterbox.
    val_loader = make_loader(split.val, eval_transform, args) if split.val else None
    test_loader = make_loader(split.test, eval_transform, args)

    model_names: list[str] = []
    val_prob_columns: list[np.ndarray] = []
    test_prob_columns: list[np.ndarray] = []
    val_labels: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    val_paths: list[str] = []
    val_groups: list[str] = []
    test_paths: list[str] = []
    test_groups: list[str] = []

    for run_dir in args.run_dirs:
        model, ckpt_args = load_model(run_dir, device)
        model_name = ckpt_args["model"]
        model_names.append(model_name)
        if val_loader is not None:
            probs, paths, groups, labels = predict_probs(model, val_loader, device, use_amp)
            val_prob_columns.append(probs)
            val_labels = labels
            val_paths = paths
            val_groups = groups
        probs, paths, groups, labels = predict_probs(model, test_loader, device, use_amp)
        test_prob_columns.append(probs)
        test_labels = labels
        test_paths = paths
        test_groups = groups

    test_matrix = np.stack(test_prob_columns, axis=1)
    mean_weights = np.ones(len(model_names), dtype=np.float64) / len(model_names)
    test_mean = weighted_mean(test_matrix, mean_weights)
    mean_metrics = summarize(test_labels, test_mean)

    selected_weights = mean_weights
    val_metrics = None
    test_at_val = None
    if val_prob_columns:
        val_matrix = np.stack(val_prob_columns, axis=1)
        selected_weights, val_metrics = select_weights(val_labels, val_matrix, args.weight_grid_step)
        test_selected = weighted_mean(test_matrix, selected_weights)
        val_threshold = val_metrics["best_threshold"]["threshold"]
        test_at_val = metrics_for(test_labels, test_selected, val_threshold)
        selected_test_metrics = summarize(test_labels, test_selected)
        write_predictions(
            args.output_dir / "predictions_val_selected_weights.csv",
            val_labels,
            weighted_mean(val_matrix, selected_weights),
            val_paths,
            val_groups,
            val_threshold,
            val_matrix,
            model_names,
        )
        write_predictions(
            args.output_dir / "predictions_test_at_val_threshold.csv",
            test_labels,
            test_selected,
            test_paths,
            test_groups,
            val_threshold,
            test_matrix,
            model_names,
        )
    else:
        selected_test_metrics = mean_metrics

    write_predictions(
        args.output_dir / "predictions_test_mean_at_0_5.csv",
        test_labels,
        test_mean,
        test_paths,
        test_groups,
        0.5,
        test_matrix,
        model_names,
    )
    write_predictions(
        args.output_dir / "predictions_test_mean_at_test_best_threshold.csv",
        test_labels,
        test_mean,
        test_paths,
        test_groups,
        mean_metrics["best_threshold"]["threshold"],
        test_matrix,
        model_names,
    )

    result = {
        "models": model_names,
        "run_dirs": [str(path) for path in args.run_dirs],
        "mean_weights": mean_weights.tolist(),
        "mean_test": mean_metrics,
        "selected_weights": selected_weights.tolist(),
        "val_selected": val_metrics,
        "test_selected": selected_test_metrics,
        "test_at_val_selected_threshold": test_at_val,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"run dir: {args.output_dir}")


if __name__ == "__main__":
    run()
