from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from attent_differnet.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageRecord,
    collect_records,
    make_resize_transform,
)
from supervised_classifier.train import (
    SplitRecords,
    make_split_from_csv,
    make_group_split,
    make_group_train_test_split,
    split_counts,
)


class ImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(record.label, dtype=torch.long),
            str(record.path),
            record.group,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchCore-style normal-only anomaly detection.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset"))
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--split-data-root",
        type=Path,
        default=None,
        help="Root used to resolve relative image_path values in --split-csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/patchcore"))
    parser.add_argument("--backbone", choices=["resnet50", "wide_resnet50_2"], default="wide_resnet50_2")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--split-strategy", choices=["train_test", "train_val_test"], default="train_test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--max-memory-patches", type=int, default=10000)
    parser.add_argument("--score-top-k-frac", type=float, default=0.01)
    parser.add_argument("--normal-percentile", type=float, default=95.0)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-images", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_transform(img_size: int, resize_mode: str) -> transforms.Compose:
    return transforms.Compose(
        [
            make_resize_transform(img_size, resize_mode),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loader(records: list[ImageRecord], transform: transforms.Compose, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        ImageDataset(records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def write_split_csv(path: Path, splits: SplitRecords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "label", "group", "path"])
        writer.writeheader()
        for split_name in ["train", "val", "test"]:
            for record in getattr(splits, split_name):
                writer.writerow(
                    {
                        "split": split_name,
                        "label": record.label_name,
                        "group": record.group,
                        "path": str(record.path),
                    }
                )


def create_backbone(name: str) -> nn.Module:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    elif name == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.DEFAULT
        model = models.wide_resnet50_2(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    feature_model = create_feature_extractor(model, return_nodes={"layer2": "layer2", "layer3": "layer3"})
    feature_model.eval()
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)
    return feature_model


@torch.no_grad()
def patch_embeddings(model: nn.Module, images: torch.Tensor, device: torch.device, normalize: bool = True) -> torch.Tensor:
    outputs = model(images.to(device, non_blocking=True))
    layer2 = outputs["layer2"]
    layer3 = F.interpolate(outputs["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False)
    embeddings = torch.cat([layer2, layer3], dim=1)
    embeddings = F.avg_pool2d(embeddings, kernel_size=3, stride=1, padding=1)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.permute(0, 2, 3, 1).reshape(embeddings.shape[0], -1, embeddings.shape[1])


@torch.no_grad()
def build_memory_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_memory_patches: int,
    seed: int,
    normalize_features: bool,
    hide_progress: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for images, labels, _, _ in tqdm(loader, desc="memory", disable=hide_progress):
        normal_mask = labels == 0
        if not bool(normal_mask.any()):
            continue
        patches = patch_embeddings(model, images[normal_mask], device, normalize=normalize_features)
        chunks.append(patches.reshape(-1, patches.shape[-1]).cpu())
    if not chunks:
        raise ValueError("No normal training images were available for PatchCore memory bank.")
    memory = torch.cat(chunks, dim=0)
    if memory.shape[0] > max_memory_patches:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(memory.shape[0], generator=generator)[:max_memory_patches]
        memory = memory[indices]
    return memory.contiguous()


def score_image_patches(
    patches: torch.Tensor,
    memory_bank: torch.Tensor,
    top_k_frac: float,
    distance_chunk_size: int = 512,
) -> float:
    mins: list[torch.Tensor] = []
    for start in range(0, patches.shape[0], distance_chunk_size):
        distances = torch.cdist(patches[start : start + distance_chunk_size], memory_bank)
        mins.append(distances.min(dim=1).values)
    min_distances = torch.cat(mins)
    top_k = max(1, int(math.ceil(min_distances.numel() * top_k_frac)))
    score = torch.topk(min_distances, k=top_k).values.mean()
    return float(score.detach().cpu())


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    top_k_frac: float,
    normalize_features: bool,
    hide_progress: bool,
) -> list[dict]:
    model.eval()
    memory_bank = memory_bank.to(device, non_blocking=True)
    rows: list[dict] = []
    for images, labels, paths, groups in tqdm(loader, desc="score", disable=hide_progress):
        patches_batch = patch_embeddings(model, images, device, normalize=normalize_features)
        for index in range(patches_batch.shape[0]):
            score = score_image_patches(patches_batch[index], memory_bank, top_k_frac=top_k_frac)
            label = int(labels[index])
            rows.append(
                {
                    "path": paths[index],
                    "group": groups[index],
                    "label": label,
                    "label_name": "anomaly" if label == 1 else "normal",
                    "score": score,
                }
            )
    return rows


def f1_from_confusion(confusion: dict) -> float:
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = int(labels.shape[0])
    normal_total = int(np.sum(labels == 0))
    anomaly_total = int(np.sum(labels == 1))
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "normal_accuracy": float(tn / normal_total) if normal_total else None,
        "anomaly_accuracy": float(tp / anomaly_total) if anomaly_total else None,
        "f1": f1_from_confusion(confusion),
        "confusion": confusion,
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([scores.min() - 1e-8, scores.max() + 1e-8], scores)))
    best: dict | None = None
    for threshold in candidates:
        metrics = threshold_metrics(labels, scores, float(threshold))
        key = (
            metrics["accuracy"],
            metrics["f1"],
            metrics["anomaly_accuracy"] or 0.0,
            metrics["normal_accuracy"] or 0.0,
        )
        if best is None or key > (
            best["accuracy"],
            best["f1"],
            best["anomaly_accuracy"] or 0.0,
            best["normal_accuracy"] or 0.0,
        ):
            best = metrics
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    return labels, scores


def evaluate_rows(rows: list[dict], normal_percentile: float) -> dict:
    labels, scores = rows_to_arrays(rows)
    metrics = {
        "num_samples": int(labels.shape[0]),
        "num_normal": int(np.sum(labels == 0)),
        "num_anomaly": int(np.sum(labels == 1)),
        "best_threshold": best_threshold(labels, scores),
        "scores": rows,
    }
    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None
    normal_scores = scores[labels == 0]
    if normal_scores.size:
        threshold = float(np.percentile(normal_scores, normal_percentile))
        metrics[f"normal_p{normal_percentile:g}_threshold"] = threshold_metrics(labels, scores, threshold)
    return metrics


def apply_external_threshold(rows: list[dict], threshold: float) -> dict:
    labels, scores = rows_to_arrays(rows)
    return threshold_metrics(labels, scores, threshold)


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def write_predictions(path: Path, rows: list[dict], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "group", "label", "label_name", "score", "pred", "pred_name", "correct"],
        )
        writer.writeheader()
        for row in rows:
            pred = int(row["score"] >= threshold)
            writer.writerow(
                {
                    **row,
                    "pred": pred,
                    "pred_name": "anomaly" if pred else "normal",
                    "correct": int(pred == row["label"]),
                }
            )


def fit_image_thumb(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_contact_sheet(rows: list[dict], output_path: Path, threshold: float, max_images: int = 80) -> None:
    if not rows:
        return
    rows = rows[:max_images]
    thumb = 160
    label_h = 38
    cols = min(5, len(rows))
    rows_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows_count * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        image = fit_image_thumb(Image.open(row["path"]), thumb)
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        sheet.paste(image, (x, y))
        pred = "A" if row["score"] >= threshold else "N"
        truth = "A" if row["label"] == 1 else "N"
        draw.text((x + 4, y + thumb + 2), f"T:{truth} P:{pred} s={row['score']:.3f}", fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb + 18), Path(row["path"]).stem[:26], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def export_images(run_dir: Path, rows: list[dict], threshold: float, folder_name: str) -> None:
    export_dir = run_dir / folder_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    buckets: dict[str, list[dict]] = {
        "correct/normal": [],
        "correct/anomaly": [],
        "wrong/false_positive_normal_pred_anomaly": [],
        "wrong/false_negative_anomaly_pred_normal": [],
    }
    for row in rows:
        pred = int(row["score"] >= threshold)
        if pred == row["label"] == 0:
            bucket = "correct/normal"
        elif pred == row["label"] == 1:
            bucket = "correct/anomaly"
        elif row["label"] == 0 and pred == 1:
            bucket = "wrong/false_positive_normal_pred_anomaly"
        else:
            bucket = "wrong/false_negative_anomaly_pred_normal"
        buckets[bucket].append(row)
        dst_dir = export_dir / bucket
        dst_dir.mkdir(parents=True, exist_ok=True)
        safe_score = f"{row['score']:.4f}".replace(".", "p")
        shutil.copy2(row["path"], dst_dir / f"s{safe_score}__{Path(row['path']).name}")

    for bucket, bucket_rows in buckets.items():
        sorted_rows = sorted(bucket_rows, key=lambda item: item["score"], reverse=True)
        make_contact_sheet(sorted_rows, export_dir / f"{bucket.replace('/', '__')}_contact_sheet.jpg", threshold)


def select_splits(args: argparse.Namespace) -> SplitRecords:
    if args.split_csv is not None:
        return make_split_from_csv(args.split_csv, args.split_data_root, args.data_root)
    records = collect_records(args.data_root)
    if args.split_strategy == "train_test":
        return make_group_train_test_split(records, args.test_size, args.seed)
    return make_group_split(records, args.n_splits, args.test_fold, args.val_fold, args.seed)


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    norm_name = "raw" if args.no_l2_normalize else "l2"
    topk_name = str(args.score_top_k_frac).replace(".", "p")
    run_name = f"{args.backbone}_{args.img_size}_{args.resize_mode}_{norm_name}_top{topk_name}"
    if args.split_strategy == "train_val_test":
        run_name += f"_fold{args.test_fold}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = select_splits(args)
    train_records = [record for record in splits.train if record.label_name == "normal"]
    if not train_records:
        raise ValueError("PatchCore requires normal training images.")

    transform = make_transform(args.img_size, args.resize_mode)
    train_loader = make_loader(train_records, transform, args)
    val_loader = make_loader(splits.val, transform, args) if splits.val else None
    test_loader = make_loader(splits.test, transform, args)
    write_split_csv(run_dir / "split.csv", SplitRecords(train=train_records, val=splits.val, test=splits.test))

    metadata = {
        "args": serializable_args(args),
        "split": {
            "train_memory": split_counts(train_records),
            "val": split_counts(splits.val),
            "test": split_counts(splits.test),
        },
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"device: {device}")
    print(f"backbone: {args.backbone}")
    print(f"train memory: {metadata['split']['train_memory']}")
    print(f"val:          {metadata['split']['val']}")
    print(f"test:         {metadata['split']['test']}")

    model = create_backbone(args.backbone).to(device)
    memory_bank = build_memory_bank(
        model,
        train_loader,
        device=device,
        max_memory_patches=args.max_memory_patches,
        seed=args.seed,
        normalize_features=not args.no_l2_normalize,
        hide_progress=args.hide_progress,
    )
    print(f"memory patches: {memory_bank.shape[0]}, dim: {memory_bank.shape[1]}")

    val_metrics = None
    if val_loader is not None:
        val_rows = predict_scores(
            model,
            val_loader,
            memory_bank,
            device,
            args.score_top_k_frac,
            normalize_features=not args.no_l2_normalize,
            hide_progress=args.hide_progress,
        )
        val_metrics = evaluate_rows(val_rows, args.normal_percentile)

    test_rows = predict_scores(
        model,
        test_loader,
        memory_bank,
        device,
        args.score_top_k_frac,
        normalize_features=not args.no_l2_normalize,
        hide_progress=args.hide_progress,
    )
    test_metrics = evaluate_rows(test_rows, args.normal_percentile)

    test_at_val_best = None
    test_at_val_normal_percentile = None
    if val_metrics is not None:
        test_at_val_best = apply_external_threshold(test_rows, val_metrics["best_threshold"]["threshold"])
        normal_key = f"normal_p{args.normal_percentile:g}_threshold"
        if normal_key in val_metrics:
            test_at_val_normal_percentile = apply_external_threshold(test_rows, val_metrics[normal_key]["threshold"])
            write_predictions(run_dir / f"predictions_test_at_val_normal_p{args.normal_percentile:g}.csv", test_rows, val_metrics[normal_key]["threshold"])
        write_predictions(run_dir / "predictions_test_at_val_best_threshold.csv", test_rows, val_metrics["best_threshold"]["threshold"])
    write_predictions(run_dir / "predictions_test_at_test_best_threshold.csv", test_rows, test_metrics["best_threshold"]["threshold"])

    if args.export_images:
        export_images(run_dir, test_rows, test_metrics["best_threshold"]["threshold"], "test_predictions_at_test_best_threshold")
        if val_metrics is not None:
            export_images(run_dir, test_rows, val_metrics["best_threshold"]["threshold"], "test_predictions_at_val_best_threshold")

    metrics = {
        "val": {key: value for key, value in val_metrics.items() if key != "scores"} if val_metrics else None,
        "test": {key: value for key, value in test_metrics.items() if key != "scores"},
        "test_at_val_best_threshold": test_at_val_best,
        f"test_at_val_normal_p{args.normal_percentile:g}_threshold": test_at_val_normal_percentile,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"memory_bank": memory_bank, "metadata": metadata}, run_dir / "memory_bank.pt")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    run()
