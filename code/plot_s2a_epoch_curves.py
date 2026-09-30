#!/usr/bin/env python3
"""Plot S2a validation accuracy per epoch (manuscript Fig. s2a_epoch_curves).

Panel (a): hybrid, linear head, matched tanh head, ResNet18 fine-tune (10 epochs).
Panel (b): hybrid, linear, and tanh heads continued to 50 epochs.
Mean ± sample std over the three main seeds, read from the per-epoch CSVs
s2a_epoch_curves.csv and s2a_long_epochs_curves.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import elevation_results_dir, figures_dir, is_private_monorepo, private_root

SEEDS = (42, 123, 456)
LABELS = {
    "hybrid": "Hybrid (VQC, 4e-4)",
    "linear_head": "Linear head (4e-4)",
    "mlp_tanh_head": "Matched tanh head (4e-4)",
    "resnet18_ft": "ResNet18 FT (1e-3)",
}
COLORS = {
    "hybrid": "#6a3d9a",
    "linear_head": "#1f78b4",
    "mlp_tanh_head": "#ff7f00",
    "resnet18_ft": "#33a02c",
}
SHORT_MODELS = ("hybrid", "linear_head", "mlp_tanh_head", "resnet18_ft")
LONG_MODELS = ("hybrid", "linear_head", "mlp_tanh_head")
YLIM = (0.30, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    tables = elevation_results_dir()
    parser.add_argument("--short-csv", type=Path, default=tables / "s2a_epoch_curves.csv")
    parser.add_argument("--long-csv", type=Path, default=tables / "s2a_long_epochs_curves.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: figures/ and manuscript/figures/)",
    )
    return parser.parse_args()


def load_curves(path: Path, models: tuple[str, ...], n_epochs: int) -> dict[str, np.ndarray]:
    """Return model -> array of shape (len(SEEDS), n_epochs)."""
    acc: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc[(row["model"], int(row["seed"]))][int(row["epoch"])] = float(row["val_accuracy"])
    curves: dict[str, np.ndarray] = {}
    for model in models:
        per_seed = []
        for seed in SEEDS:
            by_epoch = acc.get((model, seed), {})
            if sorted(by_epoch) != list(range(1, n_epochs + 1)):
                raise SystemExit(f"{path.name}: incomplete {n_epochs}-epoch curve for {model} seed {seed}")
            per_seed.append([by_epoch[e] for e in range(1, n_epochs + 1)])
        curves[model] = np.asarray(per_seed, dtype=float)
    return curves


def plot_panel(ax: plt.Axes, curves: dict[str, np.ndarray], markevery: int) -> None:
    for model, arr in curves.items():
        epochs = np.arange(1, arr.shape[1] + 1)
        ax.errorbar(
            epochs,
            arr.mean(axis=0),
            yerr=arr.std(axis=0, ddof=1),
            marker="o",
            markersize=3.0,
            markevery=markevery,
            linewidth=1.5,
            capsize=2.0,
            elinewidth=0.9,
            color=COLORS[model],
            label=LABELS[model],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(*YLIM)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)


def default_outputs() -> list[Path]:
    outputs = [figures_dir() / "s2a_epoch_curves.png"]
    if is_private_monorepo():
        outputs.append(private_root() / "manuscript" / "figures" / "s2a_epoch_curves.png")
    return outputs


def plot_figure(short: dict[str, np.ndarray], long: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.0), dpi=150, sharey=True)
    plot_panel(axes[0], short, markevery=1)
    plot_panel(axes[1], long, markevery=5)
    axes[0].set_xticks(list(range(1, 11)))
    axes[0].set_xlim(1, 10)
    axes[1].set_xlim(1, 50)
    axes[1].axvline(10, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(0.015, 0.96, tag, transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    short = load_curves(args.short_csv, SHORT_MODELS, 10)
    long = load_curves(args.long_csv, LONG_MODELS, 50)
    for model in ("hybrid", "linear_head", "mlp_tanh_head"):
        if not np.allclose(short[model], long[model][:, :10]):
            raise SystemExit(f"{model}: 10-epoch curve differs from first 10 epochs of 50-epoch run")
    outputs = [args.output] if args.output else default_outputs()
    for path in outputs:
        plot_figure(short, long, path)
    for model, arr in short.items():
        print(f"(a) {LABELS[model]}: epoch 10 {arr[:, -1].mean():.3f} ± {arr[:, -1].std(ddof=1):.3f}")
    for model, arr in long.items():
        mu = arr.mean(axis=0)
        print(f"(b) {LABELS[model]}: epoch 50 {mu[-1]:.3f} ± {arr[:, -1].std(ddof=1):.3f}, peak {mu.max():.3f}")


if __name__ == "__main__":
    main()
