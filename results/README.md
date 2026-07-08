# Results

Validation metrics (CSV) and figure index. See `ARTIFACTS.md` for paper mapping.

## S2a multi-seed summary (10 tomato classes, 10 qubits)

| Model | Accuracy (mean ± std) | F1 (mean ± std) | Train time (mean) |
|-------|----------------------|-----------------|-------------------|
| resnet18_ft | 0.938 ± 0.017 | 0.938 ± 0.017 | 250.3 s |
| linear_head | 0.878 ± 0.019 | 0.877 ± 0.021 | 192.8 s |
| hybrid | 0.695 ± 0.041 | 0.666 ± 0.051 | 3384.1 s |

Source: `tables/s2a_summary.csv` (per-seed rows in `tables/s2a_runs.csv`).

## S2b multi-seed summary (10 tomato classes, 16 qubits — hybrid only)

| Model | Accuracy (mean ± std) | F1 (mean ± std) | Train time (mean) |
|-------|----------------------|-----------------|-------------------|
| hybrid | 0.787 ± 0.031 | 0.783 ± 0.033 | 27 636.6 s |

Source: `tables/s2b_summary.csv` (per-seed rows in `tables/s2b_runs.csv`).

## All tables

| File | Description |
|------|-------------|
| `tables/s1_four_unrelated.csv` | S1 full metrics incl. frozen linear head |
| `tables/s2a_ten_tomato_10q.csv` | S2a point estimates for paper table |
| `tables/s2a_ten_tomato_10q_summary.csv` | S2a mean ± std (alias of s2a_summary) |
| `tables/s2a_runs.csv` | Per-seed S2a runs |
| `tables/s2a_summary.csv` | S2a aggregated summary |
| `tables/efficiency.csv` | Trainable params, wall-clock, val accuracy |
| `tables/s2b_runs.csv` | Per-seed S2b hybrid runs |
| `tables/s2b_summary.csv` | S2b aggregated summary |
| `tables/s2b_ten_tomato_16q.csv` | S2b full metrics (hybrid 3-seed mean; R18/D121 single-seed) |
| `tables/s3_four_tomato_depth4.csv` | S3 depth-4 tomato subset |

Figures: `../figures/`. Reproduction: `../REPRODUCTION.md`.
