# Changelog

## v1.0.1 — 2026-07-08

### Added
- S2b multi-seed hybrid runs (16 qubits, seeds 42/123/456): `s2b_runs.csv`, `s2b_summary.csv`
- `make s2b` / `reproduce.ps1 s2b` entrypoint
- Efficiency table row for Hybrid (16q, S2b)

### Changed
- `s2b_ten_tomato_16q.csv`: hybrid accuracy updated to 3-seed mean (0.79)
- `run_s2_seeds.py`: summary dedupes by `(seed, model)` (keeps last run)
- `build_efficiency_table.py`: reads S2b CSVs; inline LaTeX table format for manuscript

### Results (S2b, hybrid only)
- HQTL hybrid (16q): **0.787 ± 0.031** accuracy (~27 637 s / 10 epochs)

## v1.0.0 — 2026-07-06

### Added
- Multi-seed S2a runner (`run_s2_seeds.py`) for linear head, ResNet18 fine-tune, and hybrid (seeds 42, 123, 456)
- S1 frozen linear-head script (`run_s1_linear_head.py`)
- Efficiency table builder (`build_efficiency_table.py`)
- S2a accuracy/cost figure generator (`plot_s2a_figure.py`)
- CPU smoke test (`run_smoke_test.py`) and GitHub Actions CI
- Artifact manifest (`results/manifest.json`) and checker (`check_artifacts.py`)
- `CITATION.cff`, `MODEL_CARD.md`, `ARTIFACTS.md`, `RELEASE_CHECKLIST.md`

### Results (S2a, 10 tomato classes, 10 qubits, 3 seeds)
- Frozen linear head: **0.878 ± 0.019** accuracy
- HQTL hybrid: **0.695 ± 0.041** accuracy (~3384 s / 10 epochs)
- ResNet18 fine-tune: **0.938 ± 0.017** accuracy

### Notes
- PlantVillage images are not bundled; see `REPRODUCTION.md`.
