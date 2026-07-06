#!/usr/bin/env python3
"""Generate S2a accuracy + cost figure for paper and public repo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import elevation_results_dir, figures_dir, is_private_monorepo, private_root

MODEL_LABELS = {
    "linear_head": "Frozen linear head",
    "hybrid": "HQTL (10q)",
    "resnet18_ft": "ResNet18 fine-tune",
}
MODEL_COLORS = {
    "linear_head": "#2ca02c",
    "hybrid": "#9467bd",
    "resnet18_ft": "#1f77b4",
}
MODEL_ORDER = ("linear_head", "hybrid", "resnet18_ft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_summary = elevation_results_dir() / "s2a_summary.csv"
    parser.add_argument("--summary-csv", type=Path, default=default_summary)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: figures/ and manuscript/figures/)",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["model"]] = {
                "accuracy_mean": float(row["accuracy_mean"]),
                "accuracy_std": float(row["accuracy_std"]),
                "train_seconds_mean": float(row["train_seconds_mean"]),
            }
    return rows


def default_outputs() -> list[Path]:
    outputs = [figures_dir() / "s2a_accuracy_pareto.png"]
    if is_private_monorepo():
        outputs.append(private_root() / "manuscript" / "figures" / "s2a_accuracy_pareto.png")
    return outputs


def plot_figure(summary: dict[str, dict[str, float]], output: Path) -> None:
    fig, ax_bar = plt.subplots(figsize=(6.5, 3.8))

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    colors: list[str] = []
    for key in MODEL_ORDER:
        if key not in summary:
            continue
        labels.append(MODEL_LABELS[key])
        means.append(summary[key]["accuracy_mean"])
        stds.append(summary[key]["accuracy_std"])
        colors.append(MODEL_COLORS[key])

    x = np.arange(len(labels))
    ax_bar.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=12, ha="right")
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_ylabel("Validation accuracy (mean ± std, 3 seeds)")
    ax_bar.set_title("S2a: ten tomato classes (10 qubits)")
    ax_bar.axhline(0.878, color="#2ca02c", linestyle=":", alpha=0.5, linewidth=1)

    ax_time = ax_bar.twinx()
    for key in MODEL_ORDER:
        if key not in summary:
            continue
        idx = labels.index(MODEL_LABELS[key])
        t = summary[key]["train_seconds_mean"]
        ax_time.scatter(
            [idx],
            [t],
            marker="D",
            s=55,
            color="black",
            zorder=5,
        )
    ax_time.set_ylabel("10-epoch train time (s)")
    ax_time.set_ylim(0, max(summary[k]["train_seconds_mean"] for k in MODEL_ORDER) * 1.15)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_csv)
    outputs = [args.output] if args.output else default_outputs()
    for path in outputs:
        plot_figure(summary, path)


if __name__ == "__main__":
    main()
