#!/usr/bin/env python3
"""CPU smoke test: synthetic tiny dataset, hybrid forward/backward, metrics."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from data_utils import build_dataloaders, prepare_class_split, set_global_seed
from hybrid_model import build_hybrid_model, build_linear_head_model
from metrics_utils import evaluate_model

SMOKE_CLASSES = ("class_a", "class_b", "class_c", "class_d")
IMAGES_PER_CLASS = 3


def _write_synthetic_images(root: Path) -> None:
    rng = np.random.default_rng(0)
    for class_idx, class_name in enumerate(SMOKE_CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(IMAGES_PER_CLASS):
            arr = rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8)
            arr[:, :, class_idx % 3] = 180 + class_idx * 10
            Image.fromarray(arr).save(class_dir / f"{class_name}_{img_idx}.png")


def _one_train_step(model: nn.Module, device: torch.device) -> None:
    inputs = torch.randn(2, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1], device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()


def run_smoke() -> None:
    set_global_seed(0)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # Force CPU for portable CI smoke test.
        pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        _write_synthetic_images(raw_dir)
        prepare_class_split(
            raw_dir,
            processed_dir,
            SMOKE_CLASSES,
            seed=0,
            max_per_class=IMAGES_PER_CLASS,
            rebuild=True,
        )
        dataloaders, _ = build_dataloaders(processed_dir, batch_size=2)

        n_classes = len(SMOKE_CLASSES)
        hybrid = build_hybrid_model(n_classes, device, n_qubits=4, q_depth=2)
        _one_train_step(hybrid, device)
        hybrid_metrics = evaluate_model(hybrid, dataloaders["validation"], device)
        assert 0.0 <= hybrid_metrics["accuracy"] <= 1.0

        linear = build_linear_head_model(n_classes, device)
        _one_train_step(linear, device)
        linear_metrics = evaluate_model(linear, dataloaders["validation"], device)
        assert 0.0 <= linear_metrics["accuracy"] <= 1.0

    print("SMOKE OK: hybrid acc={:.3f} linear acc={:.3f}".format(
        hybrid_metrics["accuracy"], linear_metrics["accuracy"]
    ))


def main() -> int:
    try:
        run_smoke()
        return 0
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
