#!/usr/bin/env python3
"""Diagnose the S3 mlp_head collapse: are hidden ReLU units dead?

Reproduces the exact S3 training protocol for the given seeds, then measures
per-unit activation rates on the validation split. An activation rate of 0.0
means the unit never fires and its gradient is identically zero (dead ReLU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from data_utils import build_dataloaders, prepare_class_split, set_global_seed
from dataset_config import S3_TOMATO_CLASSES
from hybrid_model import build_mlp_head_model
from metrics_utils import evaluate_model
from paths import private_root
from train_utils import train_model

HEAD_LR = 4e-4
BATCH_SIZE = 16
EPOCHS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--activation", default="relu", choices=["relu", "tanh", "leaky_relu"])
    p.add_argument("--n-hidden", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = private_root()
    data_dir = root / "data" / "plantvillage" / "s2_tomato"
    processed_base = root / "data" / "plantvillage" / "s3_processed"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = S3_TOMATO_CLASSES

    for seed in args.seeds:
        set_global_seed(seed)
        processed = processed_base / f"seed_{seed}"
        prepare_class_split(data_dir, processed, classes, seed, rebuild=False)
        loaders, sizes = build_dataloaders(processed, BATCH_SIZE)

        model = build_mlp_head_model(
            len(classes), device, n_hidden=args.n_hidden, activation=args.activation
        )
        optimizer = optim.Adam(model.fc.parameters(), lr=HEAD_LR)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        model, _ = train_model(
            model, loaders, sizes, nn.CrossEntropyLoss(), optimizer, scheduler, EPOCHS, device
        )
        metrics = evaluate_model(model, loaders["validation"], device)

        # Capture pre-activations entering the hidden nonlinearity.
        pre_acts: list[torch.Tensor] = []
        hook = model.fc[0].register_forward_hook(
            lambda _m, _i, out: pre_acts.append(out.detach().cpu())
        )
        model.eval()
        with torch.no_grad():
            for inputs, _ in loaders["validation"]:
                model(inputs.to(device))
        hook.remove()

        pre = torch.cat(pre_acts)
        fire_rate = (pre > 0).float().mean(dim=0)
        dead = int((fire_rate == 0).sum())

        print(f"\n=== seed={seed} activation={args.activation} n_hidden={args.n_hidden} ===")
        print(f"val accuracy = {metrics['accuracy']:.4f}   macro F1 = {metrics['f1_score']:.4f}")
        print(f"per-unit fire rate = {[round(float(x), 4) for x in fire_rate]}")
        print(f"dead units = {dead} / {args.n_hidden}")
        print(f"pre-activation mean/std = {pre.mean():.4f} / {pre.std():.4f}")


if __name__ == "__main__":
    main()
