# Results

Validation metrics (CSV) and figure index. See `ARTIFACTS.md` for paper mapping.

## S1 multi-seed (4 unrelated classes, 4 qubits)

| Model | Accuracy (mean ± std) | F1 (mean ± std) | Train time (mean) |
|-------|----------------------|-----------------|-------------------|
| densenet121_ft | 0.998 ± 0.002 | 0.998 ± 0.002 | 172.5 s |
| resnet18_ft | 0.995 ± 0.005 | 0.995 ± 0.005 | 65.1 s |
| hybrid | 0.994 ± 0.003 | 0.994 ± 0.003 | 219.1 s |
| linear_head | 0.992 ± 0.004 | 0.992 ± 0.004 | 36.8 s |
| simple_cnn | 0.971 ± 0.002 | 0.971 ± 0.002 | 67.6 s |

Source: `tables/s1_seeds_summary.csv`.

## S2a multi-seed (10 tomato classes, 10 qubits)

| Model | Accuracy (mean ± std) | F1 (mean ± std) | Train time (mean) |
|-------|----------------------|-----------------|-------------------|
| densenet121_ft | 0.972 ± 0.007 | 0.972 ± 0.007 | 434.7 s |
| resnet18_ft | 0.938 ± 0.017 | 0.938 ± 0.017 | 250.3 s |
| linear_head | 0.878 ± 0.019 | 0.877 ± 0.021 | 192.8 s |
| simple_cnn | 0.774 ± 0.036 | 0.773 ± 0.034 | 138.8 s |
| hybrid (d=4) | 0.695 ± 0.041 | 0.666 ± 0.051 | 3384.1 s |

Source: `tables/s2a_summary.csv`.

## S2b multi-seed (10 tomato classes, 16 qubits)

| Model | Accuracy (mean ± std) | F1 (mean ± std) | Train time (mean) |
|-------|----------------------|-----------------|-------------------|
| densenet121_ft | 0.961 ± 0.008 | 0.961 ± 0.008 | 531.1 s |
| resnet18_ft | 0.948 ± 0.015 | 0.948 ± 0.015 | 267.7 s |
| linear_head | 0.878 ± 0.019 | 0.877 ± 0.021 | 213.9 s |
| hybrid | 0.787 ± 0.031 | 0.783 ± 0.033 | 27 636.6 s |
| simple_cnn | 0.772 ± 0.029 | 0.770 ± 0.029 | 210.5 s |

Source: `tables/s2b_summary.csv`.

## S2a hybrid depth ablation (10q, 3 seeds)

| Depth | Accuracy (mean ± std) | Train time (mean) |
|------:|----------------------:|------------------:|
| 4 | 0.695 ± 0.041 | 3384 s |
| 6 | 0.666 ± 0.028 | 3422 s |
| 8 | **0.832 ± 0.011** | 6052 s |
| 10 | 0.678 ± 0.028 | 7524 s |
| 12 | 0.721 ± 0.043 | 6199 s |
| 14 | 0.717 ± 0.038 | 4004 s |
| 16 | **0.842 ± 0.024** | 10521 s |

Source: `tables/s2a_summary.csv` (d=4) and `tables/s2a_depth*_summary.csv`.

## Circuit-removal ablation (3 seeds)

The MLP head reuses the dressed head's widths — `Linear(512 → n_qubits)`, a scalar
nonlinearity, `Linear(n_qubits → K)` — so it is the hybrid with the variational circuit
removed, carrying 99.2% of the hybrid head's trainable parameters.

| Setting | hybrid (VQC) | mlp_tanh_head | mlp_head (ReLU) | mlp_leaky_head | linear_head |
|---------|-------------:|--------------:|----------------:|---------------:|------------:|
| S1 (4q) | 0.994 ± 0.003 | 0.990 ± 0.005 | 0.975 ± 0.011 | 0.951 ± 0.052 | 0.992 ± 0.004 |
| S2a (10q) | 0.695 ± 0.041 | 0.844 ± 0.019 | 0.821 ± 0.028 | 0.834 ± 0.031 | 0.878 ± 0.019 |
| S2b (16q) | 0.787 ± 0.031 | 0.864 ± 0.013 | 0.849 ± 0.012 | 0.860 ± 0.019 | 0.878 ± 0.019 |
| S3 (4q) | 0.901 ± 0.027 | 0.904 ± 0.012 | 0.720 ± 0.201 | 0.743 ± 0.195 | 0.914 ± 0.008 |

`tanh` is the primary matched baseline: it matches the dressing nonlinearity and was
stable in every setting. The unbounded ReLU/LeakyReLU variants are seed-sensitive at
hidden width 4 (S1, S3), where a single unlucky initialisation leaves most hidden units
permanently inactive. Reproduce the diagnosis with `code/diagnose_mlp_head.py`.

Source: the per-setting `*_summary.csv` files listed below.

## All tables

| File | Description |
|------|-------------|
| `tables/s1_seeds_runs.csv` / `s1_seeds_summary.csv` | S1 multi-seed |
| `tables/s1_four_unrelated.csv` | S1 legacy point estimates |
| `tables/s2a_runs.csv` / `s2a_summary.csv` | S2a multi-seed (d=4) |
| `tables/s2b_runs.csv` / `s2b_summary.csv` | S2b multi-seed |
| `tables/s2a_depth*_*.csv` | S2a hybrid depth sweep |
| `tables/efficiency.csv` | Trainable params, wall-clock, val accuracy |
| `tables/s3_four_tomato_depth4.csv` | S3 four-tomato mean accuracy (3 seeds) |
| `tables/s3_seeds_runs.csv` | S3 per-seed raw runs |
| `tables/s3_seeds_summary.csv` | S3 mean ± std summary |

Figures: `../figures/`. Reproduction: `../REPRODUCTION.md`.
