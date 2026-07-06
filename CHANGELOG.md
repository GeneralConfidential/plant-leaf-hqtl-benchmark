# Changelog

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
- S2b (16 qubits) remains a single-seed exploratory result; see paper footnote.
- PlantVillage images are not bundled; see `REPRODUCTION.md`.
