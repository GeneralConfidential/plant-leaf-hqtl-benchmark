#!/usr/bin/env python3
"""Generate multi-seed S2a hybrid depth-ablation figure for paper and public repo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from paths import elevation_results_dir, figures_dir, is_private_monorepo, private_root

DEPTHS = (4, 6, 8, 10, 12, 14, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elevation-dir",
        type=Path,
        default=elevation_results_dir(),
        help="Directory with s2a_summary.csv and s2a_depth*_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: figures/ and manuscript/figures/)",
    )
    return parser.parse_args()


def load_depth_point(elevation_dir: Path, depth: int) -> tuple[float, float]:
    if depth == 4:
        path = elevation_dir / "s2a_summary.csv"
    else:
        path = elevation_dir / f"s2a_depth{depth}_summary.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "hybrid":
                return float(row["accuracy_mean"]), float(row["accuracy_std"])
    raise FileNotFoundError(f"hybrid row missing in {path}")


def default_outputs() -> list[Path]:
    outputs = [figures_dir() / "s2a_depth_ablation.png"]
    if is_private_monorepo():
        outputs.append(private_root() / "manuscript" / "figures" / "s2a_depth_ablation.png")
    return outputs


def plot_figure(
    depths: list[int],
    means: list[float],
    stds: list[float],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(
        depths,
        means,
        yerr=stds,
        fmt="-o",
        color="#9467bd",
        ecolor="#555555",
        capsize=4,
        linewidth=1.8,
        markersize=6,
    )
    ax.set_xlabel("Circuit depth $d$")
    ax.set_ylabel("Validation accuracy (mean ± std, 3 seeds)")
    ax.set_title("S2a hybrid depth ablation (10 tomato classes, 10 qubits)")
    ax.set_xticks(list(depths))
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    means: list[float] = []
    stds: list[float] = []
    for d in DEPTHS:
        mean, std = load_depth_point(args.elevation_dir, d)
        means.append(mean)
        stds.append(std)

    outputs = [args.output] if args.output else default_outputs()
    for path in outputs:
        plot_figure(list(DEPTHS), means, stds, path)


if __name__ == "__main__":
    main()
