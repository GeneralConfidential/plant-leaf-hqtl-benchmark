#!/usr/bin/env python3
"""Plot hybrid head learning-rate sensitivity for S1 / S2a / S3.

Sweep points come from s{1,2a,3}_lr_sweep_summary.csv; the selected head LR
(4e-4) is the three-main-seed hybrid mean from the corresponding seed CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import elevation_results_dir, figures_dir, is_private_monorepo, private_root

SELECTED_LR = 4e-4
# s2a_runs.csv also holds five-seed confirmation rows (789, 1024); the sweep uses these three.
MAIN_SEEDS = frozenset({42, 123, 456})
SETTINGS = (
    ("S1 (4q)", "s1_lr_sweep_summary.csv", "s1_seeds_runs.csv", (0.94, 1.005)),
    ("S2a (10q)", "s2a_lr_sweep_summary.csv", "s2a_runs.csv", (0.45, 0.80)),
    ("S3 (4q)", "s3_lr_sweep_summary.csv", "s3_seeds_runs.csv", (0.45, 1.00)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=elevation_results_dir())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: figures/ and manuscript/figures/)",
    )
    return parser.parse_args()


def selected_lr_point(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as f:
        accs = [
            float(row["accuracy"])
            for row in csv.DictReader(f)
            if row["model"] == "hybrid" and int(row["seed"]) in MAIN_SEEDS
        ]
    if not accs:
        raise SystemExit(f"No hybrid rows found in {path}")
    return float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0


def sweep_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["head_lr"]))
    return (
        np.array([float(r["head_lr"]) for r in rows]),
        np.array([float(r["accuracy_mean"]) for r in rows]),
        np.array([float(r["accuracy_std"]) for r in rows]),
    )


def plot_setting(ax: plt.Axes, title: str, sweep_csv: Path, seeds_csv: Path, ylim: tuple[float, float]) -> None:
    lrs, mus, sds = sweep_points(sweep_csv)
    base_mu, base_sd = selected_lr_point(seeds_csv)
    xs = np.concatenate([lrs, [SELECTED_LR]])
    order = np.argsort(xs)
    ys = np.concatenate([mus, [base_mu]])[order]
    es = np.concatenate([sds, [base_sd]])[order]
    ax.errorbar(
        xs[order],
        ys,
        yerr=es,
        fmt="-o",
        capsize=2.5,
        linewidth=1.4,
        markersize=4.0,
        color="#1f78b4",
        ecolor="#1f78b4",
        label="Sweep (+ selected)",
    )
    ax.scatter(
        [SELECTED_LR],
        [base_mu],
        marker="D",
        s=48,
        color="#e31a1c",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Selected $4\\times10^{-4}$",
    )
    ax.set_xscale("log", base=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_ylim(*ylim)


def default_outputs() -> list[Path]:
    outputs = [figures_dir() / "head_lr_sweep_sensitivity.png"]
    if is_private_monorepo():
        outputs.append(private_root() / "manuscript" / "figures" / "head_lr_sweep_sensitivity.png")
    return outputs


def plot_figure(tables: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), dpi=180)
    for ax, (title, sweep, seeds, ylim) in zip(axes, SETTINGS):
        plot_setting(ax, title, tables / sweep, tables / seeds, ylim)
        ax.set_xlabel("Hybrid head LR", fontsize=9)
        ax.set_xticks([1e-4, 2e-4, 4e-4, 1e-3, 2e-3])
        ax.set_xticklabels(["1e-4", "2e-4", "4e-4", "1e-3", "2e-3"], fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Validation accuracy", fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    outputs = [args.output] if args.output else default_outputs()
    for path in outputs:
        plot_figure(args.tables_dir, path)


if __name__ == "__main__":
    main()
