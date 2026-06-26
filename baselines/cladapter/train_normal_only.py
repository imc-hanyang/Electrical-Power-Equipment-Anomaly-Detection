from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from attent_differnet.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageRecord,
    collect_records,
    make_resize_transform,
    split_records,
)
from cladapter.model import CLAdapterFeatureModel
from normal_only.patchcore import evaluate_rows, threshold_metrics
from supervised_classifier.train import SplitRecords, split_counts


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
    parser = argparse.ArgumentParser(description="Normal-only anomaly scoring with CLAdapter/foundation features.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/opgw/KEPCO_May/Final_Dataset_square2x_384"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/opgw/KEPCO_May/engine/runs/cladapter_normal_only"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--backbone", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--train-normal-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--adapter-depth", type=int, default=1)
    parser.add_argument("--centers", type=int, default=20)
    parser.add_argument("--temp-dim", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--adapter-style", choices=["residual", "official"], default="residual")
    parser.add_argument("--no-identity-init", action="store_true")
    parser.add_argument("--train-adapter", action="store_true")
    parser.add_argument("--preserve-weight", type=float, default=0.2)
    parser.add_argument("--score-method", choices=["patchcore", "center", "both"], default="both")
    parser.add_argument("--max-memory-patches", type=int, default=10000)
    parser.add_argument("--score-top-k-frac", type=float, default=0.10)
    parser.add_argument("--normal-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
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


def make_loader(records: list[ImageRecord], transform: transforms.Compose, args: argparse.Namespace, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(records, transform),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def write_split_csv(path: Path, train_records: list[ImageRecord], test_records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "label", "group", "path"])
        writer.writeheader()
        for split_name, records in [("train", train_records), ("test", test_records)]:
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "label": record.label_name,
                        "group": record.group,
                        "path": str(record.path),
                    }
                )


@torch.no_grad()
def estimate_center(
    model: CLAdapterFeatureModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    model.eval()
    for images, labels, _, _ in loader:
        normal_mask = labels == 0
        if not bool(normal_mask.any()):
            continue
        images = images[normal_mask].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            embeddings = model(images)
        chunks.append(embeddings.detach().cpu())
    if not chunks:
        raise ValueError("No normal images were available for center estimation.")
    center = torch.cat(chunks, dim=0).mean(dim=0)
    return F.normalize(center, dim=0).to(device)


def train_adapter(
    model: CLAdapterFeatureModel,
    loader: DataLoader,
    center: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> list[dict]:
    trainable = list(model.adapter.parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        compact_losses: list[float] = []
        preserve_losses: list[float] = []
        progress = tqdm(loader, desc=f"adapter {epoch:03d}", disable=args.hide_progress)
        for images, labels, _, _ in progress:
            normal_mask = labels == 0
            if not bool(normal_mask.any()):
                continue
            images = images[normal_mask].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                embeddings, original = model(images, return_original=True)
                compact = ((embeddings - center.unsqueeze(0)) ** 2).sum(dim=1).mean()
                preserve = (1.0 - F.cosine_similarity(embeddings, original.detach(), dim=1)).mean()
                loss = compact + args.preserve_weight * preserve
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            compact_losses.append(float(compact.detach().cpu()))
            preserve_losses.append(float(preserve.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "compact_loss": float(np.mean(compact_losses)),
            "preserve_loss": float(np.mean(preserve_losses)),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.4f} "
            f"compact={row['compact_loss']:.4f} preserve={row['preserve_loss']:.4f}"
        )
    return history


@torch.no_grad()
def build_patch_memory(
    model: CLAdapterFeatureModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    model.eval()
    for images, labels, _, _ in tqdm(loader, desc="memory", disable=args.hide_progress):
        normal_mask = labels == 0
        if not bool(normal_mask.any()):
            continue
        images = images[normal_mask].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            tokens = model.forward_tokens(images)
            tokens = F.normalize(tokens, dim=-1)
        chunks.append(tokens.reshape(-1, tokens.shape[-1]).detach().cpu())
    if not chunks:
        raise ValueError("No normal images were available for memory bank.")
    memory = torch.cat(chunks, dim=0)
    if memory.shape[0] > args.max_memory_patches:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        indices = torch.randperm(memory.shape[0], generator=generator)[: args.max_memory_patches]
        memory = memory[indices]
    return memory.contiguous()


def score_patch_tokens(patches: torch.Tensor, memory_bank: torch.Tensor, top_k_frac: float, distance_chunk_size: int = 512) -> float:
    mins: list[torch.Tensor] = []
    for start in range(0, patches.shape[0], distance_chunk_size):
        distances = torch.cdist(patches[start : start + distance_chunk_size], memory_bank)
        mins.append(distances.min(dim=1).values)
    min_distances = torch.cat(mins)
    top_k = max(1, int(math.ceil(min_distances.numel() * top_k_frac)))
    return float(torch.topk(min_distances, k=top_k).values.mean().detach().cpu())


@torch.no_grad()
def score_rows(
    model: CLAdapterFeatureModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    method: str,
    center: torch.Tensor | None = None,
    memory_bank: torch.Tensor | None = None,
    top_k_frac: float = 0.10,
    hide_progress: bool = False,
) -> list[dict]:
    model.eval()
    if memory_bank is not None:
        memory_bank = memory_bank.to(device, non_blocking=True)
    rows: list[dict] = []
    for images, labels, paths, groups in tqdm(loader, desc=f"score-{method}", disable=hide_progress):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            if method == "center":
                embeddings = model(images)
                scores = ((embeddings - center.unsqueeze(0)) ** 2).sum(dim=1).detach().cpu().numpy()
            elif method == "patchcore":
                tokens = F.normalize(model.forward_tokens(images), dim=-1)
                scores = np.array(
                    [score_patch_tokens(tokens[index], memory_bank, top_k_frac=top_k_frac) for index in range(tokens.shape[0])],
                    dtype=np.float64,
                )
            else:
                raise ValueError(f"Unsupported score method: {method}")
        for path, group, label, score in zip(paths, groups, labels, scores):
            label_int = int(label)
            rows.append(
                {
                    "path": path,
                    "group": group,
                    "label": label_int,
                    "label_name": "anomaly" if label_int == 1 else "normal",
                    "score": float(score),
                }
            )
    return rows


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


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = args.amp and device.type == "cuda"

    run_name = args.run_name or (
        f"{safe_name(args.backbone)}_normalonly_clad{args.adapter_depth}_c{args.centers}_"
        f"{args.adapter_style}_{'trained' if args.train_adapter else 'frozen'}_{args.img_size}_{args.resize_mode}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.data_root)
    train_records, test_records = split_records(records, train_normal_ratio=args.train_normal_ratio, seed=args.seed)
    transform = make_transform(args.img_size, args.resize_mode)
    train_loader = make_loader(train_records, transform, args, shuffle=args.train_adapter)
    train_eval_loader = make_loader(train_records, transform, args, shuffle=False)
    test_loader = make_loader(test_records, transform, args, shuffle=False)
    write_split_csv(run_dir / "split.csv", train_records, test_records)

    model = CLAdapterFeatureModel(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
        adapter_depth=args.adapter_depth,
        centers=args.centers,
        temp_dim=args.temp_dim,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop,
        style=args.adapter_style,
        identity_init=not args.no_identity_init,
        normalize_output=True,
    ).to(device)

    center = estimate_center(model, train_eval_loader, device, use_amp)
    history = []
    if args.train_adapter and args.adapter_depth > 0:
        history = train_adapter(model, train_loader, center, args, device, use_amp)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        center = estimate_center(model, train_eval_loader, device, use_amp)
    elif args.train_adapter and args.adapter_depth == 0:
        print("adapter_depth=0, so --train-adapter has no trainable adapter blocks.")

    methods = ["center", "patchcore"] if args.score_method == "both" else [args.score_method]
    metrics: dict[str, dict] = {}
    for method in methods:
        memory_bank = None
        if method == "patchcore":
            memory_bank = build_patch_memory(model, train_eval_loader, device, args, use_amp)
            torch.save({"memory_bank": memory_bank, "method": method}, run_dir / "memory_bank_patchcore.pt")
        rows = score_rows(
            model,
            test_loader,
            device,
            use_amp,
            method=method,
            center=center,
            memory_bank=memory_bank,
            top_k_frac=args.score_top_k_frac,
            hide_progress=args.hide_progress,
        )
        method_metrics = evaluate_rows(rows, args.normal_percentile)
        threshold = method_metrics["best_threshold"]["threshold"]
        write_predictions(run_dir / f"predictions_{method}_test_best_threshold.csv", rows, threshold)
        metrics[method] = {key: value for key, value in method_metrics.items() if key != "scores"}

    labels = np.array([record.label for record in test_records], dtype=np.int64)
    metadata = {
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "split": {
            "train_normal": split_counts(train_records),
            "test": split_counts(test_records),
        },
        "test_labels": {
            "normal": int(np.sum(labels == 0)),
            "anomaly": int(np.sum(labels == 1)),
        },
        "device": str(device),
    }
    final = {"metadata": metadata, "metrics": metrics}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "center": center.detach().cpu(), "metadata": metadata}, run_dir / "model.pt")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    run()
