from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from attent_differnet.data import (
    IMAGENET_MEAN,
    IMAGENET_MEAN_PIXEL,
    IMAGENET_STD,
    ImageRecord,
    collect_records,
    make_resize_transform,
)


@dataclass(frozen=True)
class SplitRecords:
    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]


class ClassificationDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(record.label, dtype=torch.long), str(record.path), record.group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised normal/anomaly crop classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset"))
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV with image_path,label,split,group,label_name columns. Overrides generated splits.",
    )
    parser.add_argument(
        "--split-data-root",
        type=Path,
        default=None,
        help="Root used to resolve relative image_path values in --split-csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/supervised_classifier"))
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0", "convnext_tiny"], default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--normal-loss-multiplier",
        type=float,
        default=1.0,
        help="Multiply the normal class loss weight. Values >1 penalize false positives more strongly.",
    )
    parser.add_argument(
        "--anomaly-loss-multiplier",
        type=float,
        default=1.0,
        help="Multiply the anomaly class loss weight. Values >1 penalize false negatives more strongly.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["train_val_test", "train_test"],
        default="train_val_test",
        help="Use train_val_test for validation-selected evaluation, or train_test for fixed-epoch evaluation.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--train-on-train-val",
        action="store_true",
        help="Merge the train and validation folds for exploratory final training; test is used for monitoring.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--export-images", action="store_true")
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


def labels_np(records: list[ImageRecord]) -> np.ndarray:
    return np.array([record.label for record in records], dtype=np.int64)


def groups_np(records: list[ImageRecord]) -> np.ndarray:
    return np.array([record.group for record in records])


def make_group_split(records: list[ImageRecord], n_splits: int, test_fold: int, val_fold: int, seed: int) -> SplitRecords:
    if n_splits < 3:
        raise ValueError("--n-splits must be at least 3 so train/val/test are all non-empty.")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(records)), labels_np(records), groups_np(records)))
    test_fold = test_fold % len(folds)
    train_val_index, test_index = folds[test_fold]

    train_val_records = [records[index] for index in train_val_index]
    test_records = [records[index] for index in test_index]

    inner_splits = max(2, n_splits - 1)
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1009)
    inner_folds = list(
        inner.split(
            np.zeros(len(train_val_records)),
            labels_np(train_val_records),
            groups_np(train_val_records),
        )
    )
    val_fold = val_fold % len(inner_folds)
    train_index, val_index = inner_folds[val_fold]
    train_records = [train_val_records[index] for index in train_index]
    val_records = [train_val_records[index] for index in val_index]
    return SplitRecords(train=train_records, val=val_records, test=test_records)


def make_group_train_test_split(records: list[ImageRecord], test_size: float, seed: int) -> SplitRecords:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    n_splits = max(2, round(1.0 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(np.zeros(len(records)), labels_np(records), groups_np(records)))
    train_records = [records[index] for index in train_index]
    test_records = [records[index] for index in test_index]
    return SplitRecords(train=train_records, val=[], test=test_records)


def make_split_from_csv(split_csv: Path, split_data_root: Path | None, data_root: Path) -> SplitRecords:
    root = split_data_root if split_data_root is not None else data_root.parent
    buckets: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    with split_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = "val" if row["split"] == "valid" else row["split"]
            if split not in buckets:
                continue
            raw_path = Path(row["image_path"])
            path = raw_path if raw_path.is_absolute() else root / raw_path
            label = int(row["label"])
            label_name = row.get("label_name") or ("anomaly" if label == 1 else "normal")
            buckets[split].append(
                ImageRecord(
                    path=path,
                    label=label,
                    label_name=label_name,
                    group=row["group"],
                )
            )
    return SplitRecords(train=buckets["train"], val=buckets["val"], test=buckets["test"])


def make_transforms(img_size: int, resize_mode: str) -> tuple[transforms.Compose, transforms.Compose]:
    resize = make_resize_transform(img_size, resize_mode)
    train_transform = transforms.Compose(
        [
            resize,
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.03, 0.03),
                scale=(0.92, 1.08),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=IMAGENET_MEAN_PIXEL,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def create_model(name: str, pretrained: bool, freeze_backbone: bool) -> tuple[nn.Module, list[nn.Parameter]]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        head_parameters = list(model.fc.parameters())
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        head_parameters = list(model.classifier[-1].parameters())
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        head_parameters = list(model.classifier[-1].parameters())
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone:
        set_requires_grad(model.parameters(), False)
        set_requires_grad(head_parameters, True)
    return model, head_parameters


def make_optimizer(model: nn.Module, head_parameters: list[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.insert(0, {"params": backbone_parameters, "lr": args.backbone_lr})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def class_weights(
    records: list[ImageRecord],
    device: torch.device,
    normal_multiplier: float = 1.0,
    anomaly_multiplier: float = 1.0,
) -> torch.Tensor:
    counts = np.bincount(labels_np(records), minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= normal_multiplier
    weights[1] *= anomaly_multiplier
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    records: list[ImageRecord],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ClassificationDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def threshold_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    predictions = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = int(labels.shape[0])
    accuracy = float((tp + tn) / total) if total else 0.0
    normal_total = int(np.sum(labels == 0))
    anomaly_total = int(np.sum(labels == 1))
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "normal_accuracy": float(tn / normal_total) if normal_total else None,
        "anomaly_accuracy": float(tp / anomaly_total) if anomaly_total else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probs)))
    best: dict | None = None
    for threshold in candidates:
        metrics = threshold_metrics(labels, probs, float(threshold))
        if best is None or (
            metrics["accuracy"],
            metrics["anomaly_accuracy"] or 0.0,
            metrics["normal_accuracy"] or 0.0,
        ) > (
            best["accuracy"],
            best["anomaly_accuracy"] or 0.0,
            best["normal_accuracy"] or 0.0,
        ):
            best = metrics
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    paths_all: list[str] = []
    groups_all: list[str] = []
    with torch.no_grad():
        for images, labels, paths, groups in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            losses.append(float(loss.detach().cpu()))
            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())
            paths_all.extend(paths)
            groups_all.extend(groups)

    labels_np_all = np.concatenate(labels_all)
    probs_np_all = np.concatenate(probs_all)
    metrics = {
        "loss": float(np.mean(losses)),
        "num_samples": int(labels_np_all.shape[0]),
        "num_normal": int(np.sum(labels_np_all == 0)),
        "num_anomaly": int(np.sum(labels_np_all == 1)),
        "accuracy_at_0_5": threshold_metrics(labels_np_all, probs_np_all, 0.5),
        "best_threshold": find_best_threshold(labels_np_all, probs_np_all),
        "predictions": [
            {
                "path": path,
                "group": group,
                "label": int(label),
                "label_name": "anomaly" if int(label) == 1 else "normal",
                "prob_anomaly": float(prob),
            }
            for path, group, label, prob in zip(paths_all, groups_all, labels_np_all, probs_np_all)
        ],
    }
    if len(np.unique(labels_np_all)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels_np_all, probs_np_all))
        metrics["average_precision"] = float(average_precision_score(labels_np_all, probs_np_all))
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None
    return metrics


def split_counts(records: list[ImageRecord]) -> dict:
    labels = labels_np(records)
    return {
        "total": len(records),
        "normal": int(np.sum(labels == 0)),
        "anomaly": int(np.sum(labels == 1)),
        "groups": len({record.group for record in records}),
    }


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


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


def write_predictions_csv(path: Path, metrics: dict, threshold: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "group",
                "label",
                "label_name",
                "prob_anomaly",
                "pred",
                "pred_name",
                "correct",
            ],
        )
        writer.writeheader()
        for row in metrics["predictions"]:
            pred = int(row["prob_anomaly"] >= threshold)
            writer.writerow(
                {
                    **row,
                    "pred": pred,
                    "pred_name": "anomaly" if pred == 1 else "normal",
                    "correct": int(pred == row["label"]),
                }
            )


def fit_image_thumb(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
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
        pred = "A" if row["prob_anomaly"] >= threshold else "N"
        truth = "A" if row["label"] == 1 else "N"
        text = f"T:{truth} P:{pred} p={row['prob_anomaly']:.3f}"
        draw.text((x + 4, y + thumb + 2), text, fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb + 18), Path(row["path"]).stem[:26], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def export_prediction_images(
    run_dir: Path,
    metrics: dict,
    threshold: float,
    folder_name: str = "test_predictions_at_test_best_threshold",
) -> None:
    export_dir = run_dir / folder_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    buckets: dict[str, list[dict]] = {
        "correct/normal": [],
        "correct/anomaly": [],
        "wrong/false_positive_normal_pred_anomaly": [],
        "wrong/false_negative_anomaly_pred_normal": [],
    }
    for row in metrics["predictions"]:
        pred = int(row["prob_anomaly"] >= threshold)
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
        safe_prob = f"{row['prob_anomaly']:.4f}".replace(".", "p")
        dst = dst_dir / f"p{safe_prob}__{Path(row['path']).name}"
        shutil.copy2(row["path"], dst)

    for bucket, rows in buckets.items():
        rows_sorted = sorted(rows, key=lambda item: item["prob_anomaly"], reverse=True)
        make_contact_sheet(rows_sorted, export_dir / f"{bucket.replace('/', '__')}_contact_sheet.jpg", threshold)


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.data_root)
    if args.split_csv is not None:
        splits = make_split_from_csv(args.split_csv, args.split_data_root, args.data_root)
        selection_note = f"fixed split csv: {args.split_csv}"
    elif args.split_strategy == "train_test":
        splits = make_group_train_test_split(records, args.test_size, args.seed)
        selection_note = "fixed final epoch; no validation fold"
    else:
        splits = make_group_split(records, args.n_splits, args.test_fold, args.val_fold, args.seed)
        selection_note = "validation fold"
    if args.train_on_train_val:
        if args.split_strategy == "train_test":
            raise ValueError("--train-on-train-val is only valid with --split-strategy train_val_test.")
        splits = SplitRecords(train=splits.train + splits.val, val=splits.test, test=splits.test)
        selection_note = "test fold because --train-on-train-val was used"
    write_split_csv(run_dir / "split.csv", splits)

    train_transform, eval_transform = make_transforms(args.img_size, args.resize_mode)
    train_loader = make_loader(splits.train, train_transform, args.batch_size, args.num_workers, shuffle=True)
    val_loader = (
        make_loader(splits.val, eval_transform, args.batch_size, args.num_workers, shuffle=False)
        if splits.val
        else None
    )
    test_loader = make_loader(splits.test, eval_transform, args.batch_size, args.num_workers, shuffle=False)

    model, head_parameters = create_model(args.model, pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)
    optimizer = make_optimizer(model, head_parameters, args)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            splits.train,
            device=device,
            normal_multiplier=args.normal_loss_multiplier,
            anomaly_multiplier=args.anomaly_loss_multiplier,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    metadata = {
        "args": serializable_args(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "split": {
            "train": split_counts(splits.train),
            "val": split_counts(splits.val),
            "test": split_counts(splits.test),
        },
        "selection_note": selection_note,
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"train: {metadata['split']['train']}")
    print(f"val:   {metadata['split']['val']}")
    print(f"test:  {metadata['split']['test']}")

    best_score = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"train {epoch:03d}", disable=args.hide_progress)
        for images, labels, _, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")
        scheduler.step()

        epoch_record = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
            epoch_record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_auroc": val_metrics["auroc"],
                    "val_ap": val_metrics["average_precision"],
                    "val_acc_0_5": val_metrics["accuracy_at_0_5"]["accuracy"],
                    "val_best_acc": val_metrics["best_threshold"]["accuracy"],
                }
            )
        history.append(epoch_record)
        if val_loader is not None:
            print(
                f"epoch {epoch:03d} train_loss={epoch_record['train_loss']:.4f} "
                f"val_auc={epoch_record['val_auroc']:.4f} "
                f"val_acc={epoch_record['val_acc_0_5']:.4f} "
                f"val_best_acc={epoch_record['val_best_acc']:.4f}"
            )
        else:
            print(f"epoch {epoch:03d} train_loss={epoch_record['train_loss']:.4f}")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_loader is not None:
            current_score = epoch_record["val_auroc"] if epoch_record["val_auroc"] is not None else epoch_record["val_acc_0_5"]
            if current_score > best_score:
                best_score = float(current_score)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": serializable_args(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
                        "epoch": epoch,
                        "metadata": metadata,
                        "val_metrics": {key: value for key, value in val_metrics.items() if key != "predictions"},
                    },
                    run_dir / "best.pt",
                )

    if val_loader is not None:
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        best_epoch = int(checkpoint["epoch"])
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
    else:
        best_epoch = args.epochs
        checkpoint = {
            "model_state": model.state_dict(),
            "args": serializable_args(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
            "epoch": best_epoch,
            "metadata": metadata,
            "val_metrics": None,
        }
        torch.save(checkpoint, run_dir / "best.pt")
        val_metrics = None
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
    test_at_val_threshold = None
    if val_metrics is not None:
        val_threshold = val_metrics["best_threshold"]["threshold"]
        test_at_val_threshold = threshold_metrics(
            np.array([row["label"] for row in test_metrics["predictions"]]),
            np.array([row["prob_anomaly"] for row in test_metrics["predictions"]]),
            val_threshold,
        )

    final_metrics = {
        "best_epoch": best_epoch,
        "val": {key: value for key, value in val_metrics.items() if key != "predictions"} if val_metrics is not None else None,
        "test": {key: value for key, value in test_metrics.items() if key != "predictions"},
        "test_at_val_best_threshold": test_at_val_threshold,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if val_metrics is not None:
        write_predictions_csv(run_dir / "predictions_test_at_val_threshold.csv", test_metrics, val_threshold)
    write_predictions_csv(run_dir / "predictions_test_at_0_5.csv", test_metrics, 0.5)
    write_predictions_csv(run_dir / "predictions_test_at_test_best_threshold.csv", test_metrics, test_metrics["best_threshold"]["threshold"])
    if args.export_images:
        export_prediction_images(run_dir, test_metrics, 0.5, folder_name="test_predictions_at_0_5")
        export_prediction_images(run_dir, test_metrics, test_metrics["best_threshold"]["threshold"])

    torch.save({"model_state": model.state_dict(), "args": checkpoint["args"], "epoch": checkpoint["epoch"]}, run_dir / "last_loaded_best.pt")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    train()
