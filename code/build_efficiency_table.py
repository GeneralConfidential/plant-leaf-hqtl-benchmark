#!/usr/bin/env python3
"""Build parameter count and wall-clock efficiency table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from hybrid_model import (
    SimpleCNN,
    build_densenet121_finetune,
    build_hybrid_model,
    build_linear_head_model,
    build_resnet18_finetune,
)
from metrics_utils import count_trainable_parameters, format_params

from paths import elevation_results_dir, is_private_monorepo, private_root

REPO_ROOT = private_root() if is_private_monorepo() else Path(__file__).resolve().parents[1]
DEFAULT_OUT = elevation_results_dir() / "efficiency.csv"
DEFAULT_TEX = (
    REPO_ROOT / "results" / "elevation" / "efficiency_table.tex"
    if is_private_monorepo()
    else REPO_ROOT / "results" / "tables" / "efficiency_table.tex"
)
S1_RUNS = elevation_results_dir() / "s1_linear_head.csv"
S2_RUNS = elevation_results_dir() / "s2a_runs.csv"
S2_SUMMARY = elevation_results_dir() / "s2a_summary.csv"
S2B_RUNS = elevation_results_dir() / "s2b_runs.csv"
S2B_SUMMARY = elevation_results_dir() / "s2b_summary.csv"

# Published single-seed accuracies where multi-seed not yet available
KNOWN_S1_SECONDS = {"linear_head": 42.7}
PUBLISHED_VAL_ACC = {
    ("hybrid", "S1"): 0.99,
    ("hybrid", "S2a"): 0.69,
    ("hybrid", "S2b"): 0.79,
    ("densenet121_ft", "S1"): 0.99,
    ("densenet121_ft", "S2a"): 0.91,
    ("simple_cnn", "S1"): 0.74,
    ("simple_cnn", "S2a"): 0.47,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    return parser.parse_args()


def mean_train_seconds(model_name: str, runs_csv: Path | None) -> float | None:
    if runs_csv is None or not runs_csv.exists():
        return None
    latest_by_seed: dict[str, float] = {}
    with runs_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == model_name:
                latest_by_seed[row["seed"]] = float(row["train_seconds"])
    times = list(latest_by_seed.values())
    return sum(times) / len(times) if times else None


def mean_accuracy(model_key: str, setting: str) -> float | None:
    if setting == "S2a" and S2_SUMMARY.exists():
        with S2_SUMMARY.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == model_key:
                    return float(row["accuracy_mean"])
    if setting == "S2b" and model_key == "hybrid" and S2B_SUMMARY.exists():
        with S2B_SUMMARY.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == model_key:
                    return float(row["accuracy_mean"])
    if setting == "S1" and model_key == "linear_head" and S1_RUNS.exists():
        with S1_RUNS.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["metric"] == "accuracy":
                    return float(row["linear_head"])
    return PUBLISHED_VAL_ACC.get((model_key, setting))


def build_rows(device: torch.device) -> list[dict]:
    configs = [
        ("Hybrid (4q, d=4)", "hybrid", "S1", lambda: build_hybrid_model(4, device, n_qubits=4, q_depth=4)),
        ("Hybrid (10q, d=4)", "hybrid", "S2a", lambda: build_hybrid_model(10, device, n_qubits=10, q_depth=4)),
        ("Hybrid (16q, d=4)", "hybrid", "S2b", lambda: build_hybrid_model(10, device, n_qubits=16, q_depth=4)),
        ("Linear head (K=4)", "linear_head", "S1", lambda: build_linear_head_model(4, device)),
        ("Linear head (K=10)", "linear_head", "S2a", lambda: build_linear_head_model(10, device)),
        ("ResNet18 fine-tune", "resnet18_ft", "S1", lambda: build_resnet18_finetune(4, device)),
        ("ResNet18 fine-tune", "resnet18_ft", "S2a", lambda: build_resnet18_finetune(10, device)),
        ("DenseNet121 fine-tune", "densenet121_ft", "S1", lambda: build_densenet121_finetune(4, device)),
        ("DenseNet121 fine-tune", "densenet121_ft", "S2a", lambda: build_densenet121_finetune(10, device)),
        ("Simple CNN", "simple_cnn", "S1", lambda: SimpleCNN(4).to(device)),
        ("Simple CNN", "simple_cnn", "S2a", lambda: SimpleCNN(10).to(device)),
    ]

    rows = []
    for label, model_key, setting, builder in configs:
        model = builder()
        params = count_trainable_parameters(model)
        del model

        train_seconds = mean_train_seconds(
            model_key,
            S2B_RUNS if setting == "S2b" else (S2_RUNS if setting == "S2a" else None),
        )
        if train_seconds is None and setting == "S1":
            train_seconds = KNOWN_S1_SECONDS.get(model_key)

        val_acc = mean_accuracy(model_key, setting)

        rows.append(
            {
                "model": label,
                "setting": setting,
                "trainable_params": params,
                "trainable_params_fmt": format_params(params),
                "train_seconds_10epochs": f"{train_seconds:.1f}" if train_seconds else "",
                "val_accuracy": f"{val_acc:.2f}" if val_acc is not None else "",
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "setting",
                "trainable_params",
                "trainable_params_fmt",
                "train_seconds_10epochs",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_tex(rows: list[dict], path: Path) -> None:
    lines = [
        "% Paste into manuscript/main.tex (Step 3 efficiency table)",
        "\\vspace{-0.35em}",
        "\\begin{center}",
        "\\footnotesize",
        "\\captionof{table}{Trainable parameters and 10-epoch training time (validation accuracy reference).}",
        "\\label{tab:efficiency}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Setting} & \\textbf{Params} & \\textbf{Time (s)} & \\textbf{Val acc.} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['setting']} & {row['trainable_params_fmt']} & "
            f"{row['train_seconds_10epochs'] or '---'} & {row['val_accuracy'] or '---'} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{center}",
            "\\vspace{-0.5em}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    rows = build_rows(device)
    write_csv(rows, args.output)
    write_tex(rows, args.tex)


if __name__ == "__main__":
    main()
