"""Data splitting and dataloader helpers."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import torch
from torchvision import datasets, transforms


def set_global_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_class_split(
    data_dir: Path,
    processed_dir: Path,
    classes: tuple[str, ...],
    seed: int,
    *,
    max_per_class: int = 400,
    train_ratio: float = 0.8,
    rebuild: bool = False,
) -> None:
    """Copy stratified train/val split into processed_dir."""
    if processed_dir.exists() and not rebuild:
        return

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    random.seed(seed)
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        (train_dir / class_name).mkdir()
        (val_dir / class_name).mkdir()

        images = [
            class_dir / f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)
        images = images[:max_per_class]
        num_train = int(len(images) * train_ratio)

        for img in images[:num_train]:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in images[num_train:]:
            shutil.copy(img, val_dir / class_name / img.name)


def build_dataloaders(processed_dir: Path, batch_size: int):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x if x == "train" else "validation": datasets.ImageFolder(
            processed_dir / x, data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=(x == "train")
        )
        for x in ["train", "validation"]
    }
    return dataloaders, dataset_sizes
