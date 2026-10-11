# Results

Validation metrics (CSV) and figure index. See `TABLES.md` for the paper file map.

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

## S2a five-seed confirmation (seeds 42, 123, 456, 789, 1024)

| Model | Accuracy (mean ± std) | Gap to hybrid | Welch p | Cohen's d |
|-------|----------------------:|--------------:|--------:|----------:|
| linear_head | 0.880 ± 0.015 | +15.7 pp | 0.001 | 4.34 |
| mlp_tanh_head | 0.844 ± 0.013 | +12.1 pp | 0.004 | 3.35 |
| hybrid (d=4) | 0.723 ± 0.049 | — | — | — |

Source: `tables/s2a_runs.csv`; printed by `code/reproduce_tables.py`. The paper's three-seed
tables and `s2a_summary.csv` use seeds 42, 123, 456 only.

## Held-out test partition (3 seeds, 10 epochs)

Fixed 20% stratified test split (split seed 42); checkpoint chosen on validation, test scored once.

| Model | S1 test | S2a test | S3 test |
|-------|--------:|---------:|--------:|
| densenet121_ft | 0.987 ± 0.005 | 0.965 ± 0.010 | 0.968 ± 0.002 |
| resnet18_ft | 0.989 ± 0.010 | 0.942 ± 0.011 | 0.990 ± 0.002 |
| linear_head | 0.990 ± 0.005 | 0.869 ± 0.004 | 0.906 ± 0.019 |
| mlp_tanh_head | 0.984 ± 0.005 | 0.839 ± 0.008 | 0.883 ± 0.016 |
| hybrid (d=4) | 0.986 ± 0.007 | 0.706 ± 0.084 | 0.893 ± 0.016 |
| simple_cnn | 0.948 ± 0.010 | 0.795 ± 0.007 | 0.862 ± 0.008 |

Source: `tables/s{1,2a,3}_heldout_runs.csv`.

## S2a circuit controls (3 seeds, 10 qubits)

| Run | Accuracy (mean ± std) |
|-----|----------------------:|
| Trained hybrid, d=4 (main table) | 0.695 ± 0.041 |
| Frozen random circuit, d=4 | 0.676 ± 0.039 |
| Data re-uploading, d=4 | 0.690 ± 0.068 |
| Data re-uploading, d=8 | 0.544 ± 0.009 |
| Hybrid, 50 epochs | 0.740 ± 0.028 |
| Linear head, 50 epochs | 0.888 ± 0.013 |
| Matched tanh head, 50 epochs | 0.859 ± 0.022 |

Source: `tables/s2a_frozen_circuit_*.csv`, `s2a_reupload_d{4,8}_*.csv`, `s2a_long_epochs_*.csv`.

## All tables

| File | Description |
|------|-------------|
| `tables/s1_seeds_runs.csv` / `s1_seeds_summary.csv` | S1 multi-seed |
| `tables/s1_four_unrelated.csv` | S1 legacy point estimates |
| `tables/s2a_runs.csv` / `s2a_summary.csv` | S2a multi-seed (d=4); runs include five-seed rows, summary is three-seed |
| `tables/s2b_runs.csv` / `s2b_summary.csv` | S2b multi-seed |
| `tables/s2a_depth*_*.csv` | S2a hybrid depth sweep |
| `tables/efficiency.csv` | Trainable params, wall-clock, val accuracy |
| `tables/s3_four_tomato_depth4.csv` | S3 four-tomato mean accuracy (3 seeds) |
| `tables/s3_seeds_runs.csv` | S3 per-seed raw runs |
| `tables/s3_seeds_summary.csv` | S3 mean ± std summary |
| `tables/s{1,2a,3}_heldout_runs.csv` / `*_heldout_summary.csv` | Held-out test evaluation |
| `tables/s{1,2a,3}_lr_sweep_runs.csv` / `*_lr_sweep_summary.csv` | Hybrid head learning-rate sweep |
| `tables/s2a_long_epochs_runs.csv` / `*_summary.csv` | S2a 50-epoch continuation |
| `tables/s2a_epoch_curves.csv`, `s2a_long_epochs_curves.csv` | Per-epoch validation accuracy |
| `tables/s2a_frozen_circuit_*.csv` | Frozen random-circuit control |
| `tables/s2a_reupload_d{4,8}_*.csv` | Data re-uploading control |
| `tables/s2a_circuit_grad_variance.csv` | Circuit-parameter gradient variance at initialisation |

Figures: `../figures/`. Reproduction: `../REPRODUCTION.md`.
