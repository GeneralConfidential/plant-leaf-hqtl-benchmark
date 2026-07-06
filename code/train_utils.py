"""Shared training loop."""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn


def train_model(
    model: nn.Module,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
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
