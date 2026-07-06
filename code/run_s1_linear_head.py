#!/usr/bin/env python3
"""Train frozen ResNet18 + linear head on S1 and write elevation CSV."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from metrics_utils import count_trainable_parameters, evaluate_model

GLOBAL_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]

S1_CLASSES = (
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Background_without_leaves",
)

BASELINE_CSV = Path(__file__).resolve().parents[1] / "results" / "tables" / "s1_four_unrelated.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "elevation" / "s1_linear_head.csv"
DEFAULT_DATA_DIR = REPO_ROOT / "paper-release" / "code" / "data"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "paper-release" / "code" / "scl_metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root with one folder per S1 class (PlantVillage layout)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="80/20 train/val split output (created on first run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Elevation CSV path",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Recreate train/val split from --data-dir",
    )
    return parser.parse_args()


def validate_data_dir(data_dir: Path) -> None:
    missing = [c for c in S1_CLASSES if not (data_dir / c).is_dir()]
    if missing:
        raise FileNotFoundError(
            "S1 classes missing under "
            f"{data_dir}: {', '.join(missing)}. "
            "Run: uv run python scripts/download_plantvillage_data.py --only s1"
        )


def ensure_split(data_dir: Path, processed_dir: Path, rebuild: bool) -> None:
    if processed_dir.exists() and not rebuild:
        return

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    random.seed(GLOBAL_SEED)
    total_images_per_class = 400
    train_ratio = 0.8

    for class_name in S1_CLASSES:
        class_dir = data_dir / class_name
        (train_dir / class_name).mkdir()
        (val_dir / class_name).mkdir()

        images = [
            class_dir / f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)
        images = images[:total_images_per_class]
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


def build_linear_model(n_classes: int, device: torch.device) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, n_classes)
    return model.to(device)


def train_model(
    model: nn.Module,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device: torch.device,
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_corrects += torch.sum(preds == labels.data).item()

            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print(
                f"Phase: {phase:12s} Epoch: {epoch + 1}/{num_epochs} "
                f"Acc: {epoch_acc:.4f}"
            )

        scheduler.step()

    model.load_state_dict(best_model_wts)
    elapsed = time.time() - since
    print(f"Training completed in {elapsed:.1f}s; best val acc: {best_acc:.4f}")
    return model, elapsed


def load_baseline_metrics() -> dict[str, dict[str, str]]:
    by_metric: dict[str, dict[str, str]] = {}
    with BASELINE_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            by_metric[metric] = {
                "hybrid": row["hybrid"],
                "resnet18_ft": row["resnet18"],
                "densenet121_ft": row["densenet121"],
                "simple_cnn": row["simple_cnn"],
            }
    return by_metric


def write_elevation_csv(linear_metrics: dict[str, float], output: Path) -> None:
    baselines = load_baseline_metrics()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        base = baselines.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "linear_head": f"{linear_metrics[metric]:.2f}",
                "hybrid": base.get("hybrid", ""),
                "resnet18_ft": base.get("resnet18_ft", ""),
                "densenet121_ft": base.get("densenet121_ft", ""),
                "simple_cnn": base.get("simple_cnn", ""),
            }
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "linear_head",
                "hybrid",
                "resnet18_ft",
                "densenet121_ft",
                "simple_cnn",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(GLOBAL_SEED)
    os.environ["OMP_NUM_THREADS"] = "1"

    validate_data_dir(args.data_dir)
    ensure_split(args.data_dir, args.processed_dir, args.rebuild_split)
    dataloaders, dataset_sizes = build_dataloaders(args.processed_dir, args.batch_size)
    print(f"S1 classes: {', '.join(S1_CLASSES)}")
    print(f"Dataset sizes: {dataset_sizes}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_classes = len(S1_CLASSES)
    model = build_linear_model(n_classes, device)
    print(f"Linear head trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model, _ = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
    )

    metrics = evaluate_model(model, dataloaders["validation"], device)
    print(
        f"Validation — acc: {metrics['accuracy']:.4f}, "
        f"prec: {metrics['precision']:.4f}, "
        f"rec: {metrics['recall']:.4f}, "
        f"f1: {metrics['f1_score']:.4f}"
    )
    write_elevation_csv(metrics, args.output)


if __name__ == "__main__":
    main()
