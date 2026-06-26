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
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
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
from supervised_classifier.train import (
    SplitRecords,
    make_group_split,
    make_group_train_test_split,
    split_counts,
)


class CutPasteAugment:
    def __init__(self, min_area: float = 0.02, max_area: float = 0.15) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").copy()
        width, height = image.size
        area = width * height
        for _ in range(20):
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 4 <= crop_w < width and 4 <= crop_h < height:
                break
        else:
            crop_w = max(4, width // 5)
            crop_h = max(4, height // 5)

        src_x = random.randint(0, max(0, width - crop_w))
        src_y = random.randint(0, max(0, height - crop_h))
        dst_x = random.randint(0, max(0, width - crop_w))
        dst_y = random.randint(0, max(0, height - crop_h))
        patch = image.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))

        if random.random() < 0.8:
            patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.6, 1.4))
            patch = ImageEnhance.Contrast(patch).enhance(random.uniform(0.6, 1.4))
            patch = ImageEnhance.Color(patch).enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.paste(patch, (dst_x, dst_y))
        return image


class CutPasteTrainDataset(Dataset):
    def __init__(self, normal_records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.normal_records = normal_records
        self.transform = transform
        self.cutpaste = CutPasteAugment()

    def __len__(self) -> int:
        return len(self.normal_records) * 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.normal_records[index % len(self.normal_records)]
        image = Image.open(record.path).convert("RGB")
        label = 0
        if index >= len(self.normal_records):
            image = self.cutpaste(image)
            label = 1
        return self.transform(image), torch.tensor(label, dtype=torch.long), str(record.path)


class EvalDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="CutPaste normal-only anomaly detection.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/cutpaste"))
    parser.add_argument("--model", choices=["resnet50", "wide_resnet50_2", "convnext_tiny"], default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--split-strategy", choices=["train_test", "train_val_test"], default="train_test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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


def make_transforms(img_size: int, resize_mode: str) -> tuple[transforms.Compose, transforms.Compose]:
    resize = make_resize_transform(img_size, resize_mode)
    train_transform = transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15, fill=IMAGENET_MEAN_PIXEL),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [resize, transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return train_transform, eval_transform


def create_model(name: str) -> tuple[nn.Module, list[nn.Parameter]]:
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        head_parameters = list(model.fc.parameters())
    elif name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        head_parameters = list(model.fc.parameters())
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        head_parameters = list(model.classifier[-1].parameters())
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model, head_parameters


def make_optimizer(model: nn.Module, head_parameters: list[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return torch.optim.AdamW(
        [{"params": backbone_parameters, "lr": args.backbone_lr}, {"params": head_parameters, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )


def select_splits(args: argparse.Namespace) -> SplitRecords:
    records = collect_records(args.data_root)
    if args.split_strategy == "train_test":
        return make_group_train_test_split(records, args.test_size, args.seed)
    return make_group_split(records, args.n_splits, args.test_fold, args.val_fold, args.seed)


def write_split_csv(path: Path, splits: SplitRecords, train_memory_records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "label", "group", "path"])
        writer.writeheader()
        for record in train_memory_records:
            writer.writerow({"split": "train_normal_memory", "label": record.label_name, "group": record.group, "path": str(record.path)})
        for split_name in ["val", "test"]:
            for record in getattr(splits, split_name):
                writer.writerow({"split": split_name, "label": record.label_name, "group": record.group, "path": str(record.path)})


def f1_from_confusion(confusion: dict) -> float:
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = labels.shape[0]
    normal_total = int(np.sum(labels == 0))
    anomaly_total = int(np.sum(labels == 1))
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {
        "threshold": float(threshold),
        "accuracy": float((tn + tp) / total),
        "normal_accuracy": float(tn / normal_total) if normal_total else None,
        "anomaly_accuracy": float(tp / anomaly_total) if anomaly_total else None,
        "f1": f1_from_confusion(confusion),
        "confusion": confusion,
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best: dict | None = None
    for threshold in candidates:
        metrics = threshold_metrics(labels, scores, float(threshold))
        key = (metrics["accuracy"], metrics["f1"], metrics["anomaly_accuracy"] or 0.0)
        if best is None or key > (best["accuracy"], best["f1"], best["anomaly_accuracy"] or 0.0):
            best = metrics
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    return labels, scores


def evaluate_rows(rows: list[dict]) -> dict:
    labels, scores = rows_to_arrays(rows)
    metrics = {
        "num_samples": int(labels.shape[0]),
        "num_normal": int(np.sum(labels == 0)),
        "num_anomaly": int(np.sum(labels == 1)),
        "accuracy_at_0_5": threshold_metrics(labels, scores, 0.5),
        "best_threshold": best_threshold(labels, scores),
        "scores": rows,
    }
    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    rows: list[dict] = []
    for images, labels, paths, groups in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            scores = torch.softmax(logits, dim=1)[:, 1]
        for path, group, label, score in zip(paths, groups, labels, scores.detach().cpu()):
            label_int = int(label)
            rows.append(
                {
                    "path": path,
                    "group": group,
                    "label": label_int,
                    "label_name": "anomaly" if label_int else "normal",
                    "score": float(score),
                }
            )
    return evaluate_rows(rows)


def apply_threshold(rows: list[dict], threshold: float) -> dict:
    labels, scores = rows_to_arrays(rows)
    return threshold_metrics(labels, scores, threshold)


def write_predictions(path: Path, rows: list[dict], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "group", "label", "label_name", "score", "pred", "pred_name", "correct"])
        writer.writeheader()
        for row in rows:
            pred = int(row["score"] >= threshold)
            writer.writerow({**row, "pred": pred, "pred_name": "anomaly" if pred else "normal", "correct": int(pred == row["label"])})


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
        draw.text((x + 4, y + thumb + 2), f"T:{truth} P:{pred} p={row['score']:.3f}", fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb + 18), Path(row["path"]).stem[:26], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def export_images(run_dir: Path, rows: list[dict], threshold: float, folder_name: str) -> None:
    export_dir = run_dir / folder_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    buckets = {
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
        shutil.copy2(row["path"], dst_dir / f"p{safe_score}__{Path(row['path']).name}")
    for bucket, bucket_rows in buckets.items():
        make_contact_sheet(sorted(bucket_rows, key=lambda x: x["score"], reverse=True), export_dir / f"{bucket.replace('/', '__')}_contact_sheet.jpg", threshold)


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = f"{args.model}_{args.img_size}_{args.resize_mode}"
    if args.split_strategy == "train_val_test":
        run_name += f"_fold{args.test_fold}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = select_splits(args)
    train_normal_records = [record for record in splits.train if record.label_name == "normal"]
    train_transform, eval_transform = make_transforms(args.img_size, args.resize_mode)
    train_loader = DataLoader(
        CutPasteTrainDataset(train_normal_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(EvalDataset(splits.val, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if splits.val else None
    test_loader = DataLoader(EvalDataset(splits.test, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    write_split_csv(run_dir / "split.csv", splits, train_normal_records)
    metadata = {
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "split": {
            "train_normal_memory": split_counts(train_normal_records),
            "val": split_counts(splits.val),
            "test": split_counts(splits.test),
        },
        "device": str(device),
        "note": "Only real normal images are used for training; anomaly class during training is synthetic CutPaste augmentation.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    model, head_parameters = create_model(args.model)
    model.to(device)
    optimizer = make_optimizer(model, head_parameters, args)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"train real normal: {metadata['split']['train_normal_memory']}")
    print(f"val:               {metadata['split']['val']}")
    print(f"test:              {metadata['split']['test']}")

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"train {epoch:03d}", disable=args.hide_progress)
        for images, labels, _ in progress:
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
            losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")
        scheduler.step()

        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, use_amp)
            record.update(
                {
                    "val_auroc": val_metrics["auroc"],
                    "val_acc_0_5": val_metrics["accuracy_at_0_5"]["accuracy"],
                    "val_best_acc": val_metrics["best_threshold"]["accuracy"],
                }
            )
            current_score = val_metrics["auroc"] if val_metrics["auroc"] is not None else val_metrics["accuracy_at_0_5"]["accuracy"]
            if current_score > best_score:
                best_score = float(current_score)
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "metadata": metadata}, run_dir / "best.pt")
            print(
                f"epoch {epoch:03d} train_loss={record['train_loss']:.4f} "
                f"val_auc={record['val_auroc']:.4f} val_acc={record['val_acc_0_5']:.4f} val_best_acc={record['val_best_acc']:.4f}"
            )
        else:
            print(f"epoch {epoch:03d} train_loss={record['train_loss']:.4f}")
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if val_loader is not None:
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        best_epoch = int(checkpoint["epoch"])
        val_metrics = evaluate(model, val_loader, device, use_amp)
    else:
        best_epoch = args.epochs
        val_metrics = None
        torch.save({"model_state": model.state_dict(), "epoch": best_epoch, "metadata": metadata}, run_dir / "best.pt")

    test_metrics = evaluate(model, test_loader, device, use_amp)
    test_at_val_best = None
    if val_metrics is not None:
        test_at_val_best = apply_threshold(test_metrics["scores"], val_metrics["best_threshold"]["threshold"])
        write_predictions(run_dir / "predictions_test_at_val_best_threshold.csv", test_metrics["scores"], val_metrics["best_threshold"]["threshold"])
    write_predictions(run_dir / "predictions_test_at_0_5.csv", test_metrics["scores"], 0.5)
    write_predictions(run_dir / "predictions_test_at_test_best_threshold.csv", test_metrics["scores"], test_metrics["best_threshold"]["threshold"])
    if args.export_images:
        export_images(run_dir, test_metrics["scores"], 0.5, "test_predictions_at_0_5")
        export_images(run_dir, test_metrics["scores"], test_metrics["best_threshold"]["threshold"], "test_predictions_at_test_best_threshold")
        if val_metrics is not None:
            export_images(run_dir, test_metrics["scores"], val_metrics["best_threshold"]["threshold"], "test_predictions_at_val_best_threshold")

    final_metrics = {
        "best_epoch": best_epoch,
        "val": {k: v for k, v in val_metrics.items() if k != "scores"} if val_metrics else None,
        "test": {k: v for k, v in test_metrics.items() if k != "scores"},
        "test_at_val_best_threshold": test_at_val_best,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    run()

