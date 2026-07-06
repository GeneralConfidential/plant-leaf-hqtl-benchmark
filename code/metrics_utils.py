"""Shared evaluation and training helpers for elevation experiments."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    return_confusion_matrix: bool = False,
) -> Dict[str, Any]:
    """Return accuracy and macro precision/recall/F1 on a dataloader."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    result: Dict[str, Any] = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    if return_confusion_matrix:
        result["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
    return result


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def timed_train(train_fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Run train_fn and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = train_fn()
    elapsed = time.perf_counter() - start
    return result, elapsed
