# Changelog

## v1.1.0 — 2026-09-30

Artifact release matching the submitted manuscript. v1.0.3 was never tagged; its
changes are included here.

### Added
- Held-out test protocol (`run_s2_seeds.py --held-out`): fixed 20% stratified test
  partition (split seed 42) for S1, S2a, S3: `s{1,2a,3}_heldout_runs.csv` / `*_summary.csv`
- Hybrid head learning-rate sweep (`run_s2a_lr_sweep.py`): `s{1,2a,3}_lr_sweep_*.csv`
- S2a 50-epoch continuation for hybrid, linear, and tanh heads: `s2a_long_epochs_*.csv`
  and per-epoch curves `s2a_epoch_curves.csv`, `s2a_long_epochs_curves.csv`
- Frozen random-circuit control (`hybrid_frozen_circuit`): `s2a_frozen_circuit_*.csv`
- Parameter-matched data re-uploading ansatz (`hybrid_reupload`,
  `build_hybrid_model(..., ansatz="reupload")`) at d=4 and d=8: `s2a_reupload_d{4,8}_*.csv`
- Five-seed S2a confirmation (seeds 789, 1024 for hybrid, linear, tanh) in `s2a_runs.csv`
- `probe_circuit_grad_variance.py`: variance of circuit-parameter gradients over random
  initialisations (depth and qubit sweeps, training and uniform init):
  `s2a_circuit_grad_variance.csv`
- `reproduce_tables.py`: recomputes every paper summary, the five-seed Welch tests, and
  Cohen's d from the seed-level CSVs (numpy only; scipy for p-values)
- Figure generators that read only released CSVs: `plot_s2a_epoch_curves.py`,
  `plot_head_lr_sweep.py`, `plot_circuit_grad_variance.py`
- `.zenodo.json` archive metadata; `make stats` / `reproduce.ps1 stats`

### Fixed
- `build_efficiency_table.py` averaged training time over every seed in `s2a_runs.csv`;
  it now uses the three main seeds, matching the paper's efficiency table

### Headline results
- Five seeds, S2a: matched tanh head beats the d=4 hybrid by 12.1 pp
  (Welch p = 0.004, Cohen's d = 3.35); frozen linear head by 15.7 pp (p = 0.001, d = 4.34)
- Held-out test (S2a): linear 0.869 ± 0.004, tanh 0.839 ± 0.008, hybrid 0.706 ± 0.084
- 50 epochs (S2a): hybrid 0.740 ± 0.028 vs linear 0.888 ± 0.013 and tanh 0.859 ± 0.022
- Frozen random circuit 0.676 ± 0.039; re-uploading d=4 0.690 ± 0.068, d=8 0.544 ± 0.009
- Circuit-parameter gradient variance at init (200 inits, S2a features): under the
  training init ($0.01\,\mathcal{N}(0,1)$) mean $\mathrm{Var}[\partial L/\partial\theta]$
  stays $\sim 3\times 10^{-4}$ from $d{=}4$ to $d{=}16$ at 10 qubits; under uniform
  $[0,2\pi)$ it falls $\sim 9\times$ over the same depths

## v1.0.3 — never tagged (included in v1.1.0)

### Added
- S3 multi-seed (all 5 models × seeds 42/123/456): `s3_seeds_runs.csv`, `s3_seeds_summary.csv`
- Circuit-removal ablation: a parameter-matched nonlinear head that reuses the dressed
  head's widths and replaces only the variational circuit with a scalar nonlinearity,
  run across S1/S2a/S2b/S3 with `tanh`, ReLU, and LeakyReLU variants
- `build_mlp_head_model(..., activation=...)` in `hybrid_model.py`; `mlp_tanh_head`,
  `mlp_head`, `mlp_leaky_head` model keys in `run_s2_seeds.py`
- `diagnose_mlp_head.py`: measures per-unit activation rates, reproducing the dead-unit
  collapse behind the narrow ReLU head's variance

### Changed
- Efficiency table reports validation accuracy as mean ± std and includes the matched
  nonlinear head rows

### Headline results (3 seeds)
- Removing the circuit at matched width and parameter count (99.2% of hybrid head params)
  **improves** accuracy on both fine-grained settings: S2a 0.695 ± 0.041 → **0.844 ± 0.019**
  (+14.9 pp), S2b 0.787 ± 0.031 → **0.864 ± 0.013** (+7.6 pp), with non-overlapping std
- S1 and S3 are unchanged within noise (0.994 → 0.990; 0.901 → 0.904)
- Narrow unbounded activations are seed-sensitive: S3 ReLU **0.720 ± 0.201**,
  LeakyReLU **0.743 ± 0.195**; `tanh` was stable in every setting

## v1.0.2 — 2026-07-23

### Added
- S1 multi-seed (all 5 models × seeds 42/123/456): `s1_seeds_runs.csv`, `s1_seeds_summary.csv`
- S2a/S2b classical baselines filled to 3 seeds (DenseNet121, Simple CNN; S2b also linear/ResNet)
- S2a hybrid depth ablation (d ∈ {4,6,8,10,12,14,16}, 3 seeds): `s2a_depth*_runs/summary.csv`
- Depth figure generator: `plot_s2a_depth_figure.py` → `figures/s2a_depth_ablation.png`
- `run_s2_seeds.py --class-set {s1,s2}` for S1 multi-seed fills

### Changed
- Efficiency table now reads multi-seed S1/S2a/S2b summaries
- Paper depth figure replaced with multi-seed S2a ablation (was single-seed S3 notebook plot)

### Headline results (3 seeds unless noted)
- S1 hybrid: **0.994 ± 0.003**; DenseNet121: **0.998 ± 0.002**
- S2a DenseNet121: **0.972 ± 0.007**; Simple CNN: **0.774 ± 0.036**
- S2b DenseNet121: **0.961 ± 0.008**; ResNet18: **0.948 ± 0.015**; linear: **0.878 ± 0.019**
- S2a depth peaks: d=8 **0.832 ± 0.011**, d=16 **0.842 ± 0.024**

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
