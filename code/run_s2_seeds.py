#!/usr/bin/env python3
"""Multi-seed S2 evaluation (10 tomato classes; 10 or 16 qubits)."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as qnp
from torch.optim import lr_scheduler

from data_utils import build_dataloaders, prepare_class_split, set_global_seed
from dataset_config import S1_CLASSES, S2_SEEDS, S2_TOMATO_CLASSES, S3_TOMATO_CLASSES
from hybrid_model import (
    SimpleCNN,
    build_densenet121_finetune,
    build_hybrid_model,
    build_linear_head_model,
    build_mlp_head_model,
    build_resnet18_finetune,
)
from metrics_utils import evaluate_model
from paths import (
    default_s1_data_dir,
    default_s2_data_dir,
    default_s2_processed_base,
    elevation_results_dir,
)
from train_utils import train_model

CLASS_SETS = {"s1": S1_CLASSES, "s2": S2_TOMATO_CLASSES, "s3": S3_TOMATO_CLASSES}

# MLP head variants: same widths as the hybrid head, differing only in activation.
MLP_HEAD_MODELS = {
    "mlp_head": "relu",
    "mlp_tanh_head": "tanh",
    "mlp_leaky_head": "leaky_relu",
}

HELDOUT_SPLIT_SEED = 42
HELDOUT_TEST_RATIO = 0.2

DEFAULT_DATA_DIR = default_s2_data_dir()
DEFAULT_PROCESSED_BASE = default_s2_processed_base()
DEFAULT_RUNS_CSV = elevation_results_dir() / "s2a_runs.csv"
DEFAULT_SUMMARY_CSV = elevation_results_dir() / "s2a_summary.csv"

HEAD_LR = 4e-4
CLASSICAL_LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--processed-base", type=Path, default=DEFAULT_PROCESSED_BASE)
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(S2_SEEDS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_head", "resnet18_ft", "hybrid"],
        choices=[
            "linear_head",
            "mlp_head",
            "mlp_tanh_head",
            "mlp_leaky_head",
            "resnet18_ft",
            "hybrid",
            "densenet121_ft",
            "simple_cnn",
        ],
    )
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--q-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--class-set", choices=list(CLASS_SETS), default="s2")
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Fixed test holdout (split_seed) + val checkpoint; logs test metrics.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=HELDOUT_SPLIT_SEED,
        help="Seed for fixed train/val/test image assignment (held-out mode).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=HELDOUT_TEST_RATIO,
        help="Fraction of max_per_class held out as test (held-out mode).",
    )
    return parser.parse_args()


def load_existing_runs(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {(int(row["seed"]), row["model"]) for row in csv.DictReader(f)}


def append_run(path: Path, row: dict, *, held_out: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = [
        "seed",
        "model",
        "epochs",
        "split_seed",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "train_seconds",
    ]
    if not held_out:
        fieldnames = [
            "seed",
            "model",
            "epochs",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "train_seconds",
        ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_model(name: str, n_classes: int, device: torch.device, args: argparse.Namespace):
    if name == "hybrid":
        return build_hybrid_model(
            n_classes, device, n_qubits=args.n_qubits, q_depth=args.q_depth
        )
    if name == "linear_head":
        return build_linear_head_model(n_classes, device)
    if name in MLP_HEAD_MODELS:
        return build_mlp_head_model(
            n_classes,
            device,
            n_hidden=args.n_qubits,
            activation=MLP_HEAD_MODELS[name],
        )
    if name == "resnet18_ft":
        return build_resnet18_finetune(n_classes, device)
    if name == "densenet121_ft":
        return build_densenet121_finetune(n_classes, device)
    if name == "simple_cnn":
        return SimpleCNN(n_classes).to(device)
    raise ValueError(name)


def build_optimizer(model: nn.Module, name: str):
    if name == "hybrid":
        return optim.Adam(model.fc.parameters(), lr=HEAD_LR)
    if name == "linear_head":
        return optim.Adam(model.fc.parameters(), lr=HEAD_LR)
    if name in MLP_HEAD_MODELS:
        return optim.Adam(model.fc.parameters(), lr=HEAD_LR)
    return optim.Adam(model.parameters(), lr=CLASSICAL_LR)


def run_one(
    seed: int,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_global_seed(seed)
    qnp.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = "1"

    classes = CLASS_SETS[args.class_set]
    if args.held_out:
        processed_dir = args.processed_base / f"heldout_split{args.split_seed}"
    else:
        processed_dir = args.processed_base / f"seed_{seed}"
    prepare_class_split(
        args.data_dir,
        processed_dir,
        classes,
        seed,
        test_ratio=args.test_ratio if args.held_out else 0.0,
        split_seed=args.split_seed if args.held_out else None,
        rebuild=args.rebuild_splits,
    )
    dataloaders, dataset_sizes = build_dataloaders(processed_dir, BATCH_SIZE)
    n_classes = len(classes)

    print(
        f"\n=== seed={seed} model={model_name} epochs={args.epochs} "
        f"sizes={dataset_sizes} ==="
    )
    model = build_model(model_name, n_classes, device, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, model_name)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model, train_seconds = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
    )
    val_metrics = evaluate_model(model, dataloaders["validation"], device)

    row = {
        "seed": seed,
        "model": model_name,
        "epochs": args.epochs,
        "accuracy": f"{val_metrics['accuracy']:.4f}",
        "precision": f"{val_metrics['precision']:.4f}",
        "recall": f"{val_metrics['recall']:.4f}",
        "f1": f"{val_metrics['f1_score']:.4f}",
        "train_seconds": f"{train_seconds:.1f}",
    }
    if args.held_out:
        test_metrics = evaluate_model(model, dataloaders["test"], device)
        row["split_seed"] = args.split_seed
        row["test_accuracy"] = f"{test_metrics['accuracy']:.4f}"
        row["test_precision"] = f"{test_metrics['precision']:.4f}"
        row["test_recall"] = f"{test_metrics['recall']:.4f}"
        row["test_f1"] = f"{test_metrics['f1_score']:.4f}"
        print(
            f"Test: acc={test_metrics['accuracy']:.4f} "
            f"P={test_metrics['precision']:.4f} "
            f"R={test_metrics['recall']:.4f} "
            f"F1={test_metrics['f1_score']:.4f}"
        )

    append_run(args.runs_csv, row, held_out=args.held_out)
    return row


def write_summary(runs_csv: Path, summary_csv: Path, *, held_out: bool = False) -> None:
    by_model: dict[str, list[dict[str, float]]] = {}
    latest_by_seed: dict[tuple[str, str], dict[str, str]] = {}
    with runs_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            latest_by_seed[(row["seed"], row["model"])] = row

    for (_seed, model_name), row in sorted(latest_by_seed.items()):
        entry = {
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
            "train_seconds": float(row["train_seconds"]),
        }
        if held_out:
            entry["test_accuracy"] = float(row["test_accuracy"])
            entry["test_f1"] = float(row["test_f1"])
        by_model.setdefault(model_name, []).append(entry)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if held_out:
        fieldnames = [
            "model",
            "val_accuracy_mean",
            "val_accuracy_std",
            "val_f1_mean",
            "val_f1_std",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_f1_mean",
            "test_f1_std",
            "train_seconds_mean",
        ]
    else:
        fieldnames = [
            "model",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "train_seconds_mean",
        ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, rows in sorted(by_model.items()):
            accs = [r["accuracy"] for r in rows]
            f1s = [r["f1"] for r in rows]
            times = [r["train_seconds"] for r in rows]
            if held_out:
                test_accs = [r["test_accuracy"] for r in rows]
                test_f1s = [r["test_f1"] for r in rows]
                writer.writerow(
                    {
                        "model": model_name,
                        "val_accuracy_mean": f"{statistics.mean(accs):.4f}",
                        "val_accuracy_std": f"{statistics.stdev(accs):.4f}"
                        if len(accs) > 1
                        else "0.0000",
                        "val_f1_mean": f"{statistics.mean(f1s):.4f}",
                        "val_f1_std": f"{statistics.stdev(f1s):.4f}"
                        if len(f1s) > 1
                        else "0.0000",
                        "test_accuracy_mean": f"{statistics.mean(test_accs):.4f}",
                        "test_accuracy_std": f"{statistics.stdev(test_accs):.4f}"
                        if len(test_accs) > 1
                        else "0.0000",
                        "test_f1_mean": f"{statistics.mean(test_f1s):.4f}",
                        "test_f1_std": f"{statistics.stdev(test_f1s):.4f}"
                        if len(test_f1s) > 1
                        else "0.0000",
                        "train_seconds_mean": f"{statistics.mean(times):.1f}",
                    }
                )
            else:
                writer.writerow(
                    {
                        "model": model_name,
                        "accuracy_mean": f"{statistics.mean(accs):.4f}",
                        "accuracy_std": f"{statistics.stdev(accs):.4f}"
                        if len(accs) > 1
                        else "0.0000",
                        "f1_mean": f"{statistics.mean(f1s):.4f}",
                        "f1_std": f"{statistics.stdev(f1s):.4f}" if len(f1s) > 1 else "0.0000",
                        "train_seconds_mean": f"{statistics.mean(times):.1f}",
                    }
                )
    print(f"Wrote summary {summary_csv}")


def main() -> None:
    args = parse_args()
    if args.held_out and args.class_set not in {"s1", "s2", "s3"}:
        raise SystemExit("Held-out mode is supported for --class-set s1, s2, and s3 only.")
    if args.held_out and args.class_set == "s1" and args.data_dir == DEFAULT_DATA_DIR:
        args.data_dir = default_s1_data_dir()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.held_out:
        print(
            f"Held-out mode: split_seed={args.split_seed}, "
            f"test_ratio={args.test_ratio}, train/val split seed=run seed"
        )

    existing = load_existing_runs(args.runs_csv) if args.skip_existing else set()

    for seed in args.seeds:
        for model_name in args.models:
            key = (seed, model_name)
            if args.skip_existing and key in existing:
                print(f"Skipping existing seed={seed} model={model_name}")
                continue
            run_one(seed, model_name, args, device)

    if args.runs_csv.exists():
        write_summary(args.runs_csv, args.summary_csv, held_out=args.held_out)


if __name__ == "__main__":
    main()
