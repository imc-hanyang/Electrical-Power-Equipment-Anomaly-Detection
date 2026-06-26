from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_PIXEL = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    label_name: str
    group: str


def strip_crop_suffix(stem: str) -> str:
    return re.sub(r"_crop\d+$", "", stem)


def collect_records(data_root: Path, labels: tuple[str, ...] = ("normal", "anomaly")) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    label_to_id = {"normal": 0, "anomaly": 1}
    for label_name in labels:
        label_dir = data_root / label_name
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    ImageRecord(
                        path=path,
                        label=label_to_id[label_name],
                        label_name=label_name,
                        group=strip_crop_suffix(path.stem),
                    )
                )
    return records


def split_records(
    records: list[ImageRecord],
    train_normal_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Use only clean normal-only groups for training and keep group leakage out."""
    by_group: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_group.setdefault(record.group, []).append(record)

    normal_only_groups = [
        group
        for group, group_records in by_group.items()
        if {record.label_name for record in group_records} == {"normal"}
    ]
    if not normal_only_groups:
        raise ValueError("No normal-only groups found for unsupervised training.")

    rng = random.Random(seed)
    rng.shuffle(normal_only_groups)
    split_index = max(1, int(round(len(normal_only_groups) * train_normal_ratio)))
    split_index = min(split_index, len(normal_only_groups) - 1) if len(normal_only_groups) > 1 else 1
    train_groups = set(normal_only_groups[:split_index])

    train_records = [
        record
        for record in records
        if record.group in train_groups and record.label_name == "normal"
    ]
    test_records = [record for record in records if record.group not in train_groups]
    return train_records, test_records


class LetterboxResize:
    def __init__(self, size: int, fill: tuple[int, int, int] = IMAGENET_MEAN_PIXEL) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
            color=self.fill,
            centering=(0.5, 0.5),
        )


def make_resize_transform(img_size: int, resize_mode: str) -> Callable[[Image.Image], Image.Image]:
    if resize_mode == "stretch":
        return transforms.Resize((img_size, img_size))
    if resize_mode == "letterbox":
        return LetterboxResize(img_size)
    raise ValueError(f"Unsupported resize_mode: {resize_mode}")


def make_train_transform(
    img_size: int,
    use_rotation: bool = True,
    resize_mode: str = "stretch",
) -> Callable[[Image.Image], torch.Tensor]:
    steps: list[Callable] = [make_resize_transform(img_size, resize_mode)]
    if use_rotation:
        if resize_mode == "letterbox":
            steps.append(transforms.RandomRotation(180, fill=IMAGENET_MEAN_PIXEL))
        else:
            steps.append(transforms.RandomRotation(180))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def make_fixed_transform(
    img_size: int,
    degrees: float,
    resize_mode: str = "stretch",
) -> Callable[[Image.Image], torch.Tensor]:
    def rotate(image: Image.Image) -> Image.Image:
        if resize_mode == "letterbox":
            return TF.rotate(
                image,
                degrees,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=IMAGENET_MEAN_PIXEL,
            )
        return TF.rotate(image, degrees, interpolation=transforms.InterpolationMode.BILINEAR)

    return transforms.Compose(
        [
            make_resize_transform(img_size, resize_mode),
            rotate,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiTransformDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        transforms_list: list[Callable[[Image.Image], torch.Tensor]],
    ) -> None:
        self.records = records
        self.transforms_list = transforms_list

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        views = torch.stack([transform(image) for transform in self.transforms_list], dim=0)
        label = torch.tensor(record.label, dtype=torch.long)
        return views, label, str(record.path)


def make_loaders(
    data_root: Path,
    img_size: int,
    batch_size: int,
    batch_size_test: int,
    n_transforms: int,
    n_transforms_test: int,
    train_normal_ratio: float,
    seed: int,
    num_workers: int = 4,
    use_rotation: bool = True,
    resize_mode: str = "stretch",
) -> tuple[DataLoader, DataLoader, list[ImageRecord], list[ImageRecord]]:
    records = collect_records(data_root)
    train_records, test_records = split_records(
        records,
        train_normal_ratio=train_normal_ratio,
        seed=seed,
    )

    train_transform = make_train_transform(
        img_size,
        use_rotation=use_rotation,
        resize_mode=resize_mode,
    )
    train_dataset = MultiTransformDataset(train_records, [train_transform] * n_transforms)

    fixed_degrees = [index * 360.0 / n_transforms_test for index in range(n_transforms_test)]
    test_transforms = [make_fixed_transform(img_size, degrees, resize_mode=resize_mode) for degrees in fixed_degrees]
    test_dataset = MultiTransformDataset(test_records, test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, train_records, test_records


def make_loaders_from_records(
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    img_size: int,
    batch_size: int,
    batch_size_test: int,
    n_transforms: int,
    n_transforms_test: int,
    num_workers: int = 4,
    use_rotation: bool = True,
    resize_mode: str = "stretch",
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_transform = make_train_transform(
        img_size,
        use_rotation=use_rotation,
        resize_mode=resize_mode,
    )
    train_dataset = MultiTransformDataset(train_records, [train_transform] * n_transforms)

    fixed_degrees = [index * 360.0 / n_transforms_test for index in range(n_transforms_test)]
    test_transforms = [make_fixed_transform(img_size, degrees, resize_mode=resize_mode) for degrees in fixed_degrees]
    test_dataset = MultiTransformDataset(test_records, test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def split_records_from_csv(
    split_csv: Path,
    split_data_root: Path,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    buckets: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    with split_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = "val" if row["split"] == "valid" else row["split"]
            if split not in buckets:
                continue
            raw_path = Path(row["image_path"])
            path = raw_path if raw_path.is_absolute() else split_data_root / raw_path
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
    return buckets["train"], buckets["val"], buckets["test"]


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
