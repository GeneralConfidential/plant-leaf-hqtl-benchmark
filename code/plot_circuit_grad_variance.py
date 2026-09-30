#!/usr/bin/env python3
"""Plot circuit-parameter gradient variance at initialisation (S2a dressed VQC head).

Left: Var[dL/dtheta] vs circuit depth at 10 qubits. Right: vs qubit count at depth 4.
One line per initialisation scheme from s2a_circuit_grad_variance.csv
(written by probe_circuit_grad_variance.py).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from paths import elevation_results_dir, figures_dir

SCHEME_STYLE = {
    "model": ("Training init ($0.01\\,\\mathcal{N}(0,1)$)", "#6a3d9a", "o"),
    "uniform": ("Uniform $[0, 2\\pi)$", "#ff7f00", "s"),
}
SWEEP_QUBITS = 10
SWEEP_DEPTH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=elevation_results_dir() / "s2a_circuit_grad_variance.csv"
    )
    parser.add_argument("--output", type=Path, default=figures_dir() / "s2a_circuit_grad_variance.png")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {
                "scheme": row["init_scheme"],
                "q": int(row["n_qubits"]),
                "d": int(row["depth"]),
                "var": float(row["grad_var_mean"]),
            }
            for row in csv.DictReader(f)
        ]


def series(rows: list[dict], key: str, fixed_key: str, fixed_val: int) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row[fixed_key] == fixed_val:
            out[row["scheme"]].append((row[key], row["var"]))
    return {scheme: sorted(points) for scheme, points in out.items()}


def plot_axis(ax: plt.Axes, data: dict[str, list[tuple[int, float]]], xlabel: str) -> None:
    for scheme, points in data.items():
        label, color, marker = SCHEME_STYLE.get(scheme, (scheme, "0.3", "^"))
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.4, markersize=4.5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=8)


def main() -> None:
    args = parse_args()
    rows = load(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=180, sharey=True)
    plot_axis(axes[0], series(rows, "d", "q", SWEEP_QUBITS), f"Circuit depth $d$ ({SWEEP_QUBITS} qubits)")
    plot_axis(axes[1], series(rows, "q", "d", SWEEP_DEPTH), f"Qubits ($d{{=}}{SWEEP_DEPTH}$)")
    axes[0].set_ylabel(r"Mean $\mathrm{Var}[\partial L / \partial \theta_k]$", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
