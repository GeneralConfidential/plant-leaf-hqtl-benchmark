# Code

Training and analysis scripts for the public benchmark repo.

| Script | Purpose |
|--------|---------|
| `run_smoke_test.py` | CPU smoke test (synthetic data, no download) |
| `run_s1_linear_head.py` | S1 frozen linear head |
| `run_s2_seeds.py` | Multi-seed runs for every model and control (`--class-set`, `--n-qubits`, `--q-depth`, `--epochs`, `--held-out`) |
| `run_s2a_lr_sweep.py` | Hybrid head learning-rate sweep (S1 / S2a / S3) |
| `probe_circuit_grad_variance.py` | Circuit-parameter gradient variance at initialisation (depth and qubit sweeps) |
| `diagnose_mlp_head.py` | Dead-unit diagnosis for narrow ReLU heads |
| `reproduce_tables.py` | Recompute paper summaries, Welch tests, and effect sizes from the CSVs (no GPU) |
| `build_efficiency_table.py` | Params + wall-clock table |
| `plot_s2a_figure.py` | S2a accuracy/cost figure |
| `plot_s2a_depth_figure.py` | S2a depth-ablation figure |
| `plot_s2a_epoch_curves.py` | S2a 10-epoch and 50-epoch validation curves |
| `plot_head_lr_sweep.py` | Head learning-rate sensitivity figure |
| `plot_circuit_grad_variance.py` | Gradient-variance figure |
| `generate_manifest.py` | SHA256 manifest for `results/tables/` |
| `check_artifacts.py` | Verify manifest (used in CI) |
| `demo_hybrid.ipynb` | Interactive hybrid training |

Install: `pip install -r requirements.txt` (GPU) or `requirements-ci.txt` (CPU/CI).
