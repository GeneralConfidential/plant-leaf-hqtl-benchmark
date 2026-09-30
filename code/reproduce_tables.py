#!/usr/bin/env python3
"""Recompute the manuscript's summary statistics from seed-level CSVs.

Usage:
  python code/reproduce_tables.py
  python code/reproduce_tables.py --tables-dir results/tables

Prints mean ± std (sample std, ddof=1) per model for every experiment table,
plus Welch t-tests and Cohen's d for the S2a circuit-removal comparisons.
S2a rows are reported both on the three main seeds (42, 123, 456) used in the
paper's tables and on the five-seed confirmation (adds 789, 1024).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import elevation_results_dir

try:
    from scipy import stats
except ImportError:  # p-values are optional; everything else is numpy-only
    stats = None

MAIN_SEEDS = (42, 123, 456)
FIVE_SEEDS = (42, 123, 456, 789, 1024)

SEED_TABLES = (
    ("S1 (4 classes, 4 qubits)", "s1_seeds_runs.csv"),
    ("S2b (10 tomato classes, 16 qubits)", "s2b_runs.csv"),
    ("S3 (4 tomato classes, 4 qubits)", "s3_seeds_runs.csv"),
)
HELDOUT_TABLES = (
    ("S1 held-out test", "s1_heldout_runs.csv"),
    ("S2a held-out test", "s2a_heldout_runs.csv"),
    ("S3 held-out test", "s3_heldout_runs.csv"),
)
CONTROL_TABLES = (
    ("S2a 50-epoch continuation", "s2a_long_epochs_runs.csv"),
    ("S2a frozen random circuit", "s2a_frozen_circuit_runs.csv"),
    ("S2a data re-uploading, depth 4", "s2a_reupload_d4_runs.csv"),
    ("S2a data re-uploading, depth 8", "s2a_reupload_d8_runs.csv"),
)
LR_SWEEPS = (
    ("S1", "s1_lr_sweep_summary.csv"),
    ("S2a", "s2a_lr_sweep_summary.csv"),
    ("S3", "s3_lr_sweep_summary.csv"),
)
DEPTHS = (4, 6, 8, 10, 12, 14, 16)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return f"{arr[0]:.3f}"
    return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"


def by_model(
    rows: list[dict[str, str]], col: str, seeds: tuple[int, ...] | None = None
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        seed = int(row["seed"])
        if seeds is not None and seed not in seeds:
            continue
        out[row["model"]][seed] = float(row[col])
    return dict(out)


def print_models(grouped: dict[str, dict[int, float]]) -> None:
    for model in sorted(grouped):
        vals = list(grouped[model].values())
        print(f"  {model:24s} {fmt(vals)}  (n={len(vals)})")


def summarize(title: str, path: Path, seeds: tuple[int, ...] | None = None) -> None:
    print(f"\n=== {title}  [{path.name}] ===")
    if not path.exists():
        print("  [missing]")
        return
    print_models(by_model(read_rows(path), "accuracy", seeds))


def summarize_heldout(title: str, path: Path) -> None:
    print(f"\n=== {title}  [{path.name}] ===")
    if not path.exists():
        print("  [missing]")
        return
    rows = read_rows(path)
    val = by_model(rows, "accuracy")
    test = by_model(rows, "test_accuracy")
    for model in sorted(val):
        print(
            f"  {model:24s} val {fmt(list(val[model].values()))}  "
            f"test {fmt(list(test[model].values()))}  (n={len(val[model])})"
        )


def welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Welch t-test and pooled-SD Cohen's d for mean(b) - mean(a)."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (b.mean() - a.mean()) / se
    dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = float(2 * stats.t.sf(abs(t), dof)) if stats is not None else float("nan")
    s_pool = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return {
        "delta": float(b.mean() - a.mean()),
        "t": float(t),
        "df": float(dof),
        "p": p,
        "d": float((b.mean() - a.mean()) / s_pool),
    }


def removal_effects(path: Path) -> None:
    print(f"\n=== S2a circuit-removal effects (control minus hybrid)  [{path.name}] ===")
    if not path.exists():
        print("  [missing]")
        return
    rows = read_rows(path)
    for label, seeds in (("3 main seeds", MAIN_SEEDS), ("5 seeds", FIVE_SEEDS)):
        grouped = by_model(rows, "accuracy", seeds)
        hybrid = grouped.get("hybrid", {})
        for control in ("mlp_tanh_head", "linear_head"):
            ctrl = grouped.get(control, {})
            common = sorted(set(hybrid) & set(ctrl))
            if len(common) < 2:
                print(f"  {label:12s} {control:14s} [insufficient seeds: {common}]")
                continue
            eff = welch(
                np.array([hybrid[s] for s in common]),
                np.array([ctrl[s] for s in common]),
            )
            print(
                f"  {label:12s} {control:14s} delta={eff['delta']:.3f}  "
                f"Welch t={eff['t']:.2f} df={eff['df']:.1f} p={eff['p']:.4f}  "
                f"Cohen d={eff['d']:.2f}  seeds={common}"
            )


def depth_sweep(tables: Path) -> None:
    print("\n=== S2a hybrid depth sweep (10 qubits, 3 main seeds) ===")
    for depth in DEPTHS:
        path = tables / ("s2a_runs.csv" if depth == 4 else f"s2a_depth{depth}_runs.csv")
        if not path.exists():
            print(f"  d={depth:2d}  [missing {path.name}]")
            continue
        vals = list(by_model(read_rows(path), "accuracy", MAIN_SEEDS).get("hybrid", {}).values())
        print(f"  d={depth:2d}  {fmt(vals)}  (n={len(vals)})")


def lr_sweeps(tables: Path) -> None:
    print("\n=== Hybrid head learning-rate sweeps (summary CSVs) ===")
    for name, fname in LR_SWEEPS:
        path = tables / fname
        if not path.exists():
            print(f"  {name}: [missing {fname}]")
            continue
        for row in sorted(read_rows(path), key=lambda r: float(r["head_lr"])):
            print(
                f"  {name:4s} lr={float(row['head_lr']):.0e}  "
                f"{float(row['accuracy_mean']):.3f} ± {float(row['accuracy_std']):.3f}  "
                f"(n={row['n_seeds']})"
            )


def grad_variance(tables: Path) -> None:
    path = tables / "s2a_circuit_grad_variance.csv"
    print(f"\n=== Circuit-parameter gradient variance at initialisation  [{path.name}] ===")
    if not path.exists():
        print("  [missing]")
        return
    for row in read_rows(path):
        print(
            f"  {row['init_scheme']:7s} q={int(row['n_qubits']):2d} d={int(row['depth']):2d}  "
            f"Var[dL/dtheta]={float(row['grad_var_mean']):.3e}  "
            f"mean|g|={float(row['grad_abs_mean']):.3e}  (inits={row['n_inits']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=elevation_results_dir(),
        help="Directory with seed-level CSVs (default: results/tables)",
    )
    args = parser.parse_args()
    tables: Path = args.tables_dir

    print("HQTL table reproduction from seed-level CSVs")
    print(f"tables dir: {tables}")
    if stats is None:
        print("scipy not installed: Welch p-values will print as nan")

    for title, fname in SEED_TABLES:
        summarize(title, tables / fname)
    summarize("S2a (10 tomato classes, 10 qubits), 3 main seeds", tables / "s2a_runs.csv", MAIN_SEEDS)
    summarize("S2a five-seed confirmation", tables / "s2a_runs.csv", FIVE_SEEDS)
    removal_effects(tables / "s2a_runs.csv")
    for title, fname in HELDOUT_TABLES:
        summarize_heldout(title, tables / fname)
    for title, fname in CONTROL_TABLES:
        summarize(title, tables / fname)
    depth_sweep(tables)
    lr_sweeps(tables)
    grad_variance(tables)


if __name__ == "__main__":
    main()
