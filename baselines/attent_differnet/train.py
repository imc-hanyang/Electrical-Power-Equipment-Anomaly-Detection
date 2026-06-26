from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from .data import make_loaders, make_loaders_from_records, split_records_from_csv, write_split_csv
from .model import AttentDifferNet, DifferNetConfig, get_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DifferNet/AttentDifferNet on Final_Dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset"))
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--split-data-root",
        type=Path,
        default=None,
        help="Root used to resolve relative image_path values in --split-csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/attent_differnet"))
    parser.add_argument("--attention", choices=["none", "se", "cbam"], default="se")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="stretch")
    parser.add_argument("--n-scales", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sub-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-test", type=int, default=4)
    parser.add_argument("--n-transforms", type=int, default=4)
    parser.add_argument("--n-transforms-test", type=int, default=16)
    parser.add_argument("--train-normal-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--no-rotation", action="store_true")
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


def flatten_views(views: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, int]:
    batch_size, n_views = views.shape[:2]
    inputs = views.reshape(batch_size * n_views, *views.shape[-3:]).to(device, non_blocking=True)
    return inputs, batch_size, n_views


def evaluate(
    model: AttentDifferNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    hide_progress: bool = False,
) -> dict:
    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    paths_all: list[str] = []

    with torch.no_grad():
        for views, labels, paths in tqdm(loader, desc="eval", disable=hide_progress):
            inputs, batch_size, n_views = flatten_views(views, device)
            z = model(inputs)
            jacobian = model.nf.jacobian(run_forward=False)
            loss = get_loss(z, jacobian)
            z_grouped = z.view(batch_size, n_views, model.n_features)
            scores = torch.mean(z_grouped**2, dim=(1, 2))

            losses.append(float(loss.detach().cpu()))
            labels_all.append(labels.numpy())
            scores_all.append(scores.detach().cpu().numpy())
            paths_all.extend(paths)

    labels_np = np.concatenate(labels_all)
    scores_np = np.concatenate(scores_all)
    metrics = {
        "loss": float(np.mean(losses)),
        "num_samples": int(labels_np.shape[0]),
        "num_normal": int(np.sum(labels_np == 0)),
        "num_anomaly": int(np.sum(labels_np == 1)),
    }
    if len(np.unique(labels_np)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels_np, scores_np))
        metrics["average_precision"] = float(average_precision_score(labels_np, scores_np))
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
        threshold_index = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[threshold_index])
        predictions = (scores_np >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(labels_np, predictions, labels=[0, 1]).ravel()
        metrics["threshold"] = threshold
        metrics["confusion"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None
        metrics["threshold"] = None
        metrics["confusion"] = None
    metrics["scores"] = [
        {"path": path, "label": int(label), "score": float(score)}
        for path, label, score in zip(paths_all, labels_np, scores_np)
    ]
    return metrics


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = int(labels.shape[0])
    normal_total = int(np.sum(labels == 0))
    anomaly_total = int(np.sum(labels == 1))
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "normal_accuracy": float(tn / normal_total) if normal_total else None,
        "anomaly_accuracy": float(tp / anomaly_total) if anomaly_total else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([scores.min() - 1e-8, scores.max() + 1e-8], scores)))
    best: dict | None = None
    for threshold in candidates:
        current = threshold_metrics(labels, scores, float(threshold))
        key = (
            current["accuracy"],
            current["anomaly_accuracy"] or 0.0,
            current["normal_accuracy"] or 0.0,
        )
        if best is None or key > (
            best["accuracy"],
            best["anomaly_accuracy"] or 0.0,
            best["normal_accuracy"] or 0.0,
        ):
            best = current
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def score_arrays(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([row["label"] for row in metrics["scores"]], dtype=np.int64)
    scores = np.array([row["score"] for row in metrics["scores"]], dtype=np.float64)
    return labels, scores


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = args.output_dir / args.attention
    run_dir.mkdir(parents=True, exist_ok=True)

    config = DifferNetConfig(
        attention=args.attention,
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
        n_scales=args.n_scales,
        freeze_backbone=not args.train_backbone,
    )
    model = AttentDifferNet(config).to(device)
    optimizer = torch.optim.Adam(
        model.optim_parameters(train_backbone=args.train_backbone),
        lr=args.lr,
        betas=(0.8, 0.8),
        eps=1e-4,
        weight_decay=1e-5,
    )

    val_loader = None
    val_records = []
    if args.split_csv is not None:
        split_root = args.split_data_root if args.split_data_root is not None else args.data_root.parent
        raw_train_records, val_records, test_records = split_records_from_csv(args.split_csv, split_root)
        train_records = [record for record in raw_train_records if record.label_name == "normal"]
        train_loader, test_loader = make_loaders_from_records(
            train_records=train_records,
            test_records=test_records,
            img_size=args.img_size,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            n_transforms=args.n_transforms,
            n_transforms_test=args.n_transforms_test,
            num_workers=args.num_workers,
            use_rotation=not args.no_rotation,
            resize_mode=args.resize_mode,
        )
        if val_records:
            _, val_loader = make_loaders_from_records(
                train_records=train_records,
                test_records=val_records,
                img_size=args.img_size,
                batch_size=args.batch_size,
                batch_size_test=args.batch_size_test,
                n_transforms=args.n_transforms,
                n_transforms_test=args.n_transforms_test,
                num_workers=args.num_workers,
                use_rotation=not args.no_rotation,
                resize_mode=args.resize_mode,
                shuffle_train=False,
            )
    else:
        train_loader, test_loader, train_records, test_records = make_loaders(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            n_transforms=args.n_transforms,
            n_transforms_test=args.n_transforms_test,
            train_normal_ratio=args.train_normal_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            use_rotation=not args.no_rotation,
            resize_mode=args.resize_mode,
        )
    write_split_csv(run_dir / "split.csv", train_records, test_records)

    metadata = {
        "args": serializable_args(args),
        "config": asdict(config),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    best_auroc = -1.0
    history = []
    print(f"device: {device}")
    print(f"attention: {args.attention}")
    print(f"train normal images: {len(train_records)}")
    print(f"val images: {len(val_records)}")
    print(f"test images: {len(test_records)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for sub_epoch in range(1, args.sub_epochs + 1):
            progress = tqdm(train_loader, desc=f"train {epoch}.{sub_epoch}", disable=args.hide_progress)
            for views, _, _ in progress:
                optimizer.zero_grad(set_to_none=True)
                inputs, _, _ = flatten_views(views, device)
                z = model(inputs)
                jacobian = model.nf.jacobian(run_forward=False)
                loss = get_loss(z, jacobian)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.4f}")

        select_loader = val_loader if val_loader is not None else test_loader
        metrics = evaluate(model, select_loader, device, hide_progress=args.hide_progress)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "eval_loss": metrics["loss"],
            "auroc": metrics["auroc"],
            "average_precision": metrics["average_precision"],
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"eval_loss={epoch_record['eval_loss']:.4f} "
            f"auroc={epoch_record['auroc']}"
        )

        current_auroc = metrics["auroc"] if metrics["auroc"] is not None else -1.0
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "args": serializable_args(args),
                    "epoch": epoch,
                    "metrics": metrics,
                    "selection_split": "val" if val_loader is not None else "test",
                },
                run_dir / "best.pt",
            )
            (run_dir / "scores_best.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "args": serializable_args(args),
            "epoch": args.epochs,
        },
        run_dir / "last.pt",
    )

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate(model, val_loader, device, hide_progress=args.hide_progress) if val_loader is not None else None
    test_metrics = evaluate(model, test_loader, device, hide_progress=args.hide_progress)
    test_at_val_threshold = None
    if val_metrics is not None:
        val_labels, val_scores = score_arrays(val_metrics)
        test_labels, test_scores = score_arrays(test_metrics)
        val_best = best_threshold(val_labels, val_scores)
        test_at_val_threshold = threshold_metrics(test_labels, test_scores, val_best["threshold"])
        val_metrics["best_threshold"] = val_best
    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_split": checkpoint.get("selection_split"),
        "val": {key: value for key, value in val_metrics.items() if key != "scores"} if val_metrics else None,
        "test": {key: value for key, value in test_metrics.items() if key != "scores"},
        "test_at_val_best_threshold": test_at_val_threshold,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best auroc: {best_auroc:.4f}")
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    train()
