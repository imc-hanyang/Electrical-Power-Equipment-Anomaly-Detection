from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cladapter.model import CLAdapterClassifier
from supervised_classifier.train import (
    SplitRecords,
    class_weights,
    collect_records,
    evaluate,
    export_prediction_images,
    make_group_split,
    make_group_train_test_split,
    make_loader,
    make_transforms,
    resolve_device,
    split_counts,
    threshold_metrics,
    write_predictions_csv,
    write_split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised CLAdapter classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset_square2x_384"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/cladapter_supervised"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--backbone", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--adapter-depth", type=int, default=1)
    parser.add_argument("--centers", type=int, default=20)
    parser.add_argument("--temp-dim", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--adapter-style", choices=["residual", "official"], default="residual")
    parser.add_argument("--no-identity-init", action="store_true")
    parser.add_argument("--split-strategy", choices=["train_val_test", "train_test"], default="train_val_test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--train-on-train-val", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--export-images", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: CLAdapterClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameter_groups = [
        {"params": model.adapter_parameters(), "lr": args.lr},
        {"params": model.head_parameters(), "lr": args.head_lr},
    ]
    backbone_parameters = [parameter for parameter in model.backbone_parameters() if parameter.requires_grad]
    if backbone_parameters:
        parameter_groups.insert(0, {"params": backbone_parameters, "lr": args.backbone_lr})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))


def select_splits(args: argparse.Namespace) -> tuple[SplitRecords, str]:
    records = collect_records(args.data_root)
    if args.split_strategy == "train_test":
        splits = make_group_train_test_split(records, args.test_size, args.seed)
        note = "fixed final epoch; no validation fold"
    else:
        splits = make_group_split(records, args.n_splits, args.test_fold, args.val_fold, args.seed)
        note = "validation fold"
    if args.train_on_train_val:
        if args.split_strategy == "train_test":
            raise ValueError("--train-on-train-val is only valid with --split-strategy train_val_test.")
        splits = SplitRecords(train=splits.train + splits.val, val=splits.test, test=splits.test)
        note = "test fold because --train-on-train-val was used"
    return splits, note


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or (
        f"{safe_name(args.backbone)}_clad{args.adapter_depth}_c{args.centers}_"
        f"{args.adapter_style}_{args.img_size}_{args.resize_mode}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits, selection_note = select_splits(args)
    write_split_csv(run_dir / "split.csv", splits)
    train_transform, eval_transform = make_transforms(args.img_size, args.resize_mode)
    train_loader = make_loader(splits.train, train_transform, args.batch_size, args.num_workers, shuffle=True)
    val_loader = (
        make_loader(splits.val, eval_transform, args.batch_size, args.num_workers, shuffle=False)
        if splits.val
        else None
    )
    test_loader = make_loader(splits.test, eval_transform, args.batch_size, args.num_workers, shuffle=False)

    model = CLAdapterClassifier(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        adapter_depth=args.adapter_depth,
        centers=args.centers,
        temp_dim=args.temp_dim,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop,
        style=args.adapter_style,
        identity_init=not args.no_identity_init,
    ).to(device)
    optimizer = make_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits.train, device=device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metadata = {
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
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
    print(f"backbone: {args.backbone}")
    print(f"train: {metadata['split']['train']}")
    print(f"val:   {metadata['split']['val']}")
    print(f"test:  {metadata['split']['test']}")

    best_score = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
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
            value = float(loss.detach().cpu())
            losses.append(value)
            progress.set_postfix(loss=f"{value:.4f}")
        scheduler.step()

        epoch_record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
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
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_loader is not None:
            print(
                f"epoch {epoch:03d} train_loss={epoch_record['train_loss']:.4f} "
                f"val_auc={epoch_record['val_auroc']:.4f} "
                f"val_acc={epoch_record['val_acc_0_5']:.4f} "
                f"val_best_acc={epoch_record['val_best_acc']:.4f}"
            )
            current_score = epoch_record["val_auroc"] if epoch_record["val_auroc"] is not None else epoch_record["val_acc_0_5"]
            if current_score > best_score:
                best_score = float(current_score)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
                        "epoch": epoch,
                        "metadata": metadata,
                        "val_metrics": {key: value for key, value in val_metrics.items() if key != "predictions"},
                    },
                    run_dir / "best.pt",
                )
        else:
            print(f"epoch {epoch:03d} train_loss={epoch_record['train_loss']:.4f}")

    if val_loader is not None:
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        best_epoch = int(checkpoint["epoch"])
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
    else:
        best_epoch = args.epochs
        checkpoint = {
            "model_state": model.state_dict(),
            "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
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
        write_predictions_csv(run_dir / "predictions_test_at_val_threshold.csv", test_metrics, val_threshold)

    final_metrics = {
        "best_epoch": best_epoch,
        "val": {key: value for key, value in val_metrics.items() if key != "predictions"} if val_metrics else None,
        "test": {key: value for key, value in test_metrics.items() if key != "predictions"},
        "test_at_val_best_threshold": test_at_val_threshold,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
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
