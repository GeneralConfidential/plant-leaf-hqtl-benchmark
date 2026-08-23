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


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(
        class_dir / f
        for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def prepare_class_split(
    data_dir: Path,
    processed_dir: Path,
    classes: tuple[str, ...],
    seed: int,
    *,
    max_per_class: int = 400,
    train_ratio: float = 0.8,
    test_ratio: float = 0.0,
    split_seed: int | None = None,
    rebuild: bool = False,
) -> None:
    """Copy stratified train/val(/test) split into processed_dir.

    Default (test_ratio=0): 80/20 train/val on up to max_per_class images,
    shuffled with ``seed`` (legacy protocol).

    Held-out (test_ratio>0): fixed ``split_seed`` assigns a test holdout first,
    then up to max_per_class remaining images are split train/val with ``seed``.
    When a class has at least max_per_class + int(max_per_class*test_ratio)
    images, train/val counts match the legacy 80/20 on max_per_class.
    """
    if processed_dir.exists() and not rebuild:
        return

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir: Path | None = None
    if test_ratio > 0:
        test_dir = processed_dir / "test"
        test_dir.mkdir(parents=True)

    holdout_seed = split_seed if split_seed is not None else seed
    n_test_target = int(max_per_class * test_ratio) if test_ratio > 0 else 0

    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        (train_dir / class_name).mkdir()
        (val_dir / class_name).mkdir()
        if test_dir is not None:
            (test_dir / class_name).mkdir()

        all_images = _list_images(class_dir)

        if test_ratio > 0:
            random.seed(holdout_seed)
            random.shuffle(all_images)
            extend_cap = max_per_class + n_test_target
            if len(all_images) >= extend_cap:
                test_images = all_images[:n_test_target]
                pool = all_images[n_test_target : n_test_target + max_per_class]
            else:
                n_test = min(n_test_target, max(1, int(len(all_images) * test_ratio)))
                test_images = all_images[:n_test]
                pool = all_images[n_test : n_test + max_per_class]
                if len(pool) > max_per_class:
                    pool = pool[:max_per_class]

            random.seed(seed)
            random.shuffle(pool)
            num_train = int(len(pool) * train_ratio)
            train_images = pool[:num_train]
            val_images = pool[num_train:]

            for img in test_images:
                shutil.copy(img, test_dir / class_name / img.name)
            for img in train_images:
                shutil.copy(img, train_dir / class_name / img.name)
            for img in val_images:
                shutil.copy(img, val_dir / class_name / img.name)
            continue

        random.seed(seed)
        random.shuffle(all_images)
        images = all_images[:max_per_class]
        num_train = int(len(images) * train_ratio)

        for img in images[:num_train]:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in images[num_train:]:
            shutil.copy(img, val_dir / class_name / img.name)


def build_dataloaders(processed_dir: Path, batch_size: int):
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": val_transform,
        "test": val_transform,
    }

    folder_keys = ["train", "val"]
    loader_keys = ["train", "validation"]
    if (processed_dir / "test").is_dir():
        folder_keys.append("test")
        loader_keys.append("test")

    image_datasets = {}
    for folder, key in zip(folder_keys, loader_keys):
        image_datasets[key] = datasets.ImageFolder(
            processed_dir / folder, data_transforms[folder]
        )

    dataset_sizes = {key: len(image_datasets[key]) for key in loader_keys}
    dataloaders = {
        key: torch.utils.data.DataLoader(
            image_datasets[key],
            batch_size=batch_size,
            shuffle=(key == "train"),
        )
        for key in loader_keys
    }
    return dataloaders, dataset_sizes
