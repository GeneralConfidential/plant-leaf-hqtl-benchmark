#!/usr/bin/env python3
"""Circuit-parameter gradient variance across random initialisations (S2a).

Standard barren-plateau diagnostic: for each (n_qubits, depth, init scheme), build
many independently initialised dressed heads, compute the cross-entropy loss on one
fixed batch of frozen ResNet18 S2a features, and record dL/dtheta for the variational
circuit weights only. Reports Var over initialisations, averaged over circuit weights.

Init schemes:
  model    q_params ~ 0.01 * N(0, 1), the initialisation used in training
  uniform  q_params ~ U[0, 2*pi), the random-circuit setting of barren-plateau theory

Example:
  uv run python paper-release/code/probe_circuit_grad_variance.py
  uv run python paper-release/code/probe_circuit_grad_variance.py --inits 10 --depths 4 --qubits 10
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from data_utils import build_dataloaders, prepare_class_split, set_global_seed
from dataset_config import S2_TOMATO_CLASSES
from hybrid_model import DressedQuantumNet
from paths import default_s2_data_dir, default_s2_processed_base, elevation_results_dir

FIELDS = [
    "init_scheme",
    "n_qubits",
    "depth",
    "n_inits",
    "batch_size",
    "n_circuit_params",
    "grad_var_mean",
    "grad_var_first_layer",
    "grad_var_last_layer",
    "grad_abs_mean",
    "circuit_grad_norm_mean",
    "pre_net_grad_norm_mean",
    "seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=default_s2_data_dir())
    parser.add_argument("--processed-base", type=Path, default=default_s2_processed_base())
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inits", type=int, default=50)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14, 16])
    parser.add_argument("--depth-sweep-qubits", type=int, default=10)
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14, 16])
    parser.add_argument("--qubit-sweep-depth", type=int, default=4)
    parser.add_argument("--schemes", nargs="+", default=["model", "uniform"], choices=["model", "uniform"])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="Q:D",
        help="Explicit n_qubits:depth pairs; overrides the depth and qubit sweeps (for parallel workers)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=elevation_results_dir() / "s2a_circuit_grad_variance.csv",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def fixed_batch_features(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen ResNet18 features for one fixed batch, computed on CPU so every worker agrees."""
    processed = args.processed_base / f"seed_{args.split_seed}"
    prepare_class_split(args.data_dir, processed, S2_TOMATO_CLASSES, args.split_seed)
    loaders, _ = build_dataloaders(processed, batch_size=args.batch_size)
    dataset = loaders["train"].dataset
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(dataset), generator=gen)[: args.batch_size].tolist()
    images = torch.stack([dataset[i][0] for i in idx])
    labels = torch.tensor([dataset[i][1] for i in idx])

    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    with torch.no_grad():
        feats = backbone(images)
    return feats, labels


def probe_config(
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_qubits: int,
    depth: int,
    scheme: str,
    n_inits: int,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    grads = []
    circuit_norms = []
    pre_norms = []
    start = time.time()
    for i in range(n_inits):
        torch.manual_seed(10_000 + i)
        head = DressedQuantumNet(n_qubits, len(S2_TOMATO_CLASSES), depth)
        if scheme == "uniform":
            with torch.no_grad():
                head.q_params.uniform_(0.0, 2.0 * math.pi)
        loss = criterion(head(feats), labels)
        loss.backward()
        g = head.q_params.grad.detach().clone()
        grads.append(g)
        circuit_norms.append(float(g.norm()))
        pre_norms.append(float(head.pre_net.weight.grad.norm()))
    g_all = torch.stack(grads)  # (n_inits, depth * n_qubits)
    var = g_all.var(dim=0, unbiased=True)
    first = var[:n_qubits].mean()
    last = var[-n_qubits:].mean()
    return {
        "init_scheme": scheme,
        "n_qubits": n_qubits,
        "depth": depth,
        "n_inits": n_inits,
        "batch_size": len(labels),
        "n_circuit_params": g_all.shape[1],
        "grad_var_mean": f"{float(var.mean()):.6e}",
        "grad_var_first_layer": f"{float(first):.6e}",
        "grad_var_last_layer": f"{float(last):.6e}",
        "grad_abs_mean": f"{float(g_all.abs().mean()):.6e}",
        "circuit_grad_norm_mean": f"{sum(circuit_norms) / n_inits:.6e}",
        "pre_net_grad_norm_mean": f"{sum(pre_norms) / n_inits:.6e}",
        "seconds": f"{time.time() - start:.1f}",
    }


def load_done(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["init_scheme"], int(r["n_qubits"]), int(r["depth"])) for r in csv.DictReader(fh)}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_global_seed(args.split_seed)
    feats, labels = fixed_batch_features(args)
    print(f"Fixed batch: {tuple(feats.shape)} features, labels={labels.tolist()}")

    if args.configs:
        configs = [tuple(int(x) for x in pair.split(":")) for pair in args.configs]
    else:
        configs = [(args.depth_sweep_qubits, d) for d in args.depths]
        configs += [(q, args.qubit_sweep_depth) for q in args.qubits if (q, args.qubit_sweep_depth) not in configs]
    jobs = [(s, q, d) for s in args.schemes for q, d in configs]
    done = load_done(args.out_csv) if args.skip_existing else set()

    t0 = time.time()
    for k, (scheme, q, d) in enumerate(jobs, 1):
        if (scheme, q, d) in done:
            print(f"[{k}/{len(jobs)}] skip {scheme} q={q} d={d}")
            continue
        row = probe_config(feats, labels, n_qubits=q, depth=d, scheme=scheme, n_inits=args.inits)
        append_row(args.out_csv, row)
        print(
            f"[{k}/{len(jobs)}] {scheme:7s} q={q:2d} d={d:2d}  Var={row['grad_var_mean']}  "
            f"|g|={row['grad_abs_mean']}  ({row['seconds']} s, elapsed {time.time() - t0:.0f} s)",
            flush=True,
        )
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
