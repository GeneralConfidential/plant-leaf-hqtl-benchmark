#!/usr/bin/env python3
"""Hybrid learning-rate sweep for S1, S2a, and S3 (hybrid VQC head only).

Default LRs: 1e-4, 2e-4, 1e-3, 2e-3. Paper baseline 4e-4 is skipped unless
--include-baseline is set (already reported in main tables).

Separate CSV per setting; does not overwrite s1/s2a/s3 seed tables.
Resume-safe on (seed, head_lr) within each CSV.

Examples:

    # finish all remaining jobs for one setting
    uv run python paper-release/code/run_s2a_lr_sweep.py --class-set s2

    # overnight: S2a then S1 then S3
    uv run python paper-release/code/run_s2a_lr_sweep.py --class-set all

    # daytime: one job, leave GPU headroom
    uv run python paper-release/code/run_s2a_lr_sweep.py --class-set all \\
        --max-jobs 1 --gpu-memory-fraction 0.45 --low-priority
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as qnp
from torch.optim import lr_scheduler

from data_utils import build_dataloaders, prepare_class_split, set_global_seed
from dataset_config import (
    S1_CLASSES,
    S2_SEEDS,
    S2_TOMATO_CLASSES,
    S3_TOMATO_CLASSES,
)
from hybrid_model import build_hybrid_model
from metrics_utils import evaluate_model
from paths import elevation_results_dir, is_private_monorepo, private_root
from train_utils import train_model

REPO_ROOT = private_root() if is_private_monorepo() else Path(__file__).resolve().parents[1]

DEFAULT_LRS = (1e-4, 2e-4, 1e-3, 2e-3)
PAPER_LR = 4e-4
BATCH_SIZE = 16
EPOCHS = 10

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
RUN_FIELDS = (
    "seed",
    "model",
    "head_lr",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "train_seconds",
)


@dataclass(frozen=True)
class SettingConfig:
    key: str
    label: str
    classes: tuple[str, ...]
    data_dir: Path
    processed_base: Path
    n_qubits: int
    q_depth: int
    runs_csv: Path
    summary_csv: Path
    seconds_per_job: int


def setting_configs() -> dict[str, SettingConfig]:
    elev = elevation_results_dir()
    return {
        "s1": SettingConfig(
            key="s1",
            label="S1 (4 unrelated leaves, 4q)",
            classes=S1_CLASSES,
            data_dir=REPO_ROOT / "paper-release" / "code" / "data",
            processed_base=REPO_ROOT / "data" / "plantvillage" / "s1_processed",
            n_qubits=4,
            q_depth=4,
            runs_csv=elev / "s1_lr_sweep_runs.csv",
            summary_csv=elev / "s1_lr_sweep_summary.csv",
            seconds_per_job=220,
        ),
        "s2": SettingConfig(
            key="s2",
            label="S2a (10 tomato classes, 10q)",
            classes=S2_TOMATO_CLASSES,
            data_dir=REPO_ROOT / "data" / "plantvillage" / "s2_tomato",
            processed_base=REPO_ROOT / "data" / "plantvillage" / "s2_processed",
            n_qubits=10,
            q_depth=4,
            runs_csv=elev / "s2a_lr_sweep_runs.csv",
            summary_csv=elev / "s2a_lr_sweep_summary.csv",
            seconds_per_job=2100,
        ),
        "s3": SettingConfig(
            key="s3",
            label="S3 (4 tomato classes, 4q)",
            classes=S3_TOMATO_CLASSES,
            data_dir=REPO_ROOT / "data" / "plantvillage" / "s2_tomato",
            processed_base=REPO_ROOT / "data" / "plantvillage" / "s3_processed",
            n_qubits=4,
            q_depth=4,
            runs_csv=elev / "s3_lr_sweep_runs.csv",
            summary_csv=elev / "s3_lr_sweep_summary.csv",
            seconds_per_job=240,
        ),
    }


def lr_key(lr: float) -> str:
    return f"{lr:.0e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-set",
        choices=("s1", "s2", "s3", "all"),
        default="s2",
        help="Benchmark setting (s2 = S2a ten-class tomato).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(S2_SEEDS))
    parser.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help=f"Also run {PAPER_LR:g} (paper 4e-4 baseline).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Stop after this many new runs across all selected settings.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=1.0,
        help="Cap GPU memory for this process (e.g. 0.45).",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="Set Windows process priority to BelowNormal.",
    )
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run jobs even if (seed, lr) exists in the CSV.",
    )
    return parser.parse_args()


def set_below_normal_priority() -> None:
    if sys.platform != "win32":
        return
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    if not ok:
        print("Warning: could not set BelowNormal process priority")
        return
    print("Process priority: BelowNormal")


def load_existing(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "head_lr" not in (reader.fieldnames or []):
            return set()
        return {(int(row["seed"]), row["head_lr"]) for row in reader}


def append_run(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RUN_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def planned_jobs(
    cfg: SettingConfig,
    seeds: list[int],
    lrs: list[float],
    include_baseline: bool,
) -> list[tuple[int, float]]:
    lr_list = list(lrs)
    if include_baseline and PAPER_LR not in lr_list:
        lr_list.append(PAPER_LR)
    return [(seed, lr) for lr in lr_list for seed in seeds]


def run_one(
    cfg: SettingConfig,
    seed: int,
    head_lr: float,
    rebuild_splits: bool,
    device: torch.device,
) -> dict:
    set_global_seed(seed)
    qnp.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = "1"

    processed_dir = cfg.processed_base / f"seed_{seed}"
    prepare_class_split(
        cfg.data_dir,
        processed_dir,
        cfg.classes,
        seed,
        rebuild=rebuild_splits,
    )
    dataloaders, dataset_sizes = build_dataloaders(processed_dir, BATCH_SIZE)

    print(
        f"\n=== {cfg.key} seed={seed} model=hybrid lr={lr_key(head_lr)} "
        f"sizes={dataset_sizes} ==="
    )
    model = build_hybrid_model(
        len(cfg.classes),
        device,
        n_qubits=cfg.n_qubits,
        q_depth=cfg.q_depth,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=head_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model, train_seconds = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        EPOCHS,
        device,
    )
    metrics = evaluate_model(model, dataloaders["validation"], device)
    row = {
        "seed": seed,
        "model": "hybrid",
        "head_lr": lr_key(head_lr),
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision": f"{metrics['precision']:.4f}",
        "recall": f"{metrics['recall']:.4f}",
        "f1": f"{metrics['f1_score']:.4f}",
        "train_seconds": f"{train_seconds:.1f}",
    }
    append_run(cfg.runs_csv, row)
    return row


def write_summary(cfg: SettingConfig) -> None:
    runs_csv = cfg.runs_csv
    if not runs_csv.exists():
        return
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with runs_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            latest[(row["head_lr"], row["seed"])] = row

    by_lr: dict[str, list[dict[str, float]]] = {}
    for (_lr, _seed), row in sorted(latest.items()):
        by_lr.setdefault(row["head_lr"], []).append(
            {
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
                "train_seconds": float(row["train_seconds"]),
            }
        )

    summary_csv = cfg.summary_csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "head_lr",
                "n_seeds",
                "accuracy_mean",
                "accuracy_std",
                "f1_mean",
                "f1_std",
                "train_seconds_mean",
            ],
        )
        writer.writeheader()
        for head_lr, rows in sorted(by_lr.items()):
            accs = [r["accuracy"] for r in rows]
            f1s = [r["f1"] for r in rows]
            times = [r["train_seconds"] for r in rows]
            writer.writerow(
                {
                    "head_lr": head_lr,
                    "n_seeds": len(rows),
                    "accuracy_mean": f"{statistics.mean(accs):.4f}",
                    "accuracy_std": (
                        f"{statistics.stdev(accs):.4f}" if len(accs) > 1 else "0.0000"
                    ),
                    "f1_mean": f"{statistics.mean(f1s):.4f}",
                    "f1_std": f"{statistics.stdev(f1s):.4f}" if len(f1s) > 1 else "0.0000",
                    "train_seconds_mean": f"{statistics.mean(times):.1f}",
                }
            )
    print(f"Wrote summary {summary_csv}")


def remaining_jobs(
    cfg: SettingConfig,
    args: argparse.Namespace,
) -> list[tuple[int, float]]:
    skip = not args.no_skip_existing
    existing = load_existing(cfg.runs_csv) if skip else set()
    jobs = planned_jobs(cfg, args.seeds, args.lrs, args.include_baseline)
    return [(seed, lr) for seed, lr in jobs if (seed, lr_key(lr)) not in existing]


def main() -> None:
    args = parse_args()
    if args.low_priority:
        set_below_normal_priority()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda" and 0 < args.gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(f"GPU memory fraction: {args.gpu_memory_fraction}")

    all_cfgs = setting_configs()
    keys = ["s2", "s1", "s3"] if args.class_set == "all" else [args.class_set]
    selected = [all_cfgs[k] for k in keys]

    queue: list[tuple[SettingConfig, int, float]] = []
    est_seconds = 0
    for cfg in selected:
        rem = remaining_jobs(cfg, args)
        est_seconds += len(rem) * cfg.seconds_per_job
        for seed, lr in rem:
            queue.append((cfg, seed, lr))
        print(
            f"{cfg.label}: {len(rem)} remaining "
            f"(~{len(rem) * cfg.seconds_per_job / 3600:.1f} h)"
        )

    print(f"Total queue: {len(queue)} jobs (~{est_seconds / 3600:.1f} h)")
    if args.max_jobs > 0:
        queue = queue[: args.max_jobs]
        print(f"This invocation will run at most {args.max_jobs} job(s)")

    touched: set[str] = set()
    for cfg, seed, head_lr in queue:
        run_one(cfg, seed, head_lr, args.rebuild_splits, device)
        touched.add(cfg.key)

    for key in keys:
        if key in touched or not queue:
            write_summary(all_cfgs[key])

    if not queue:
        print("Nothing left to run for the selected setting(s).")


if __name__ == "__main__":
    main()
