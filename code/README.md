# Code

| Script | Purpose |
|--------|---------|
| `run_smoke_test.py` | CPU smoke test (synthetic data) |
| `run_s1_linear_head.py` | S1 frozen linear head |
| `run_s2_seeds.py` | Multi-seed training for models and controls |
| `run_s2a_lr_sweep.py` | Hybrid head learning-rate sweep |
| `probe_circuit_grad_variance.py` | Gradient variance at initialisation |
| `diagnose_mlp_head.py` | Dead-unit check for narrow ReLU heads |
| `reproduce_tables.py` | Recompute summaries / Welch tests from CSVs |
| `build_efficiency_table.py` | Params + wall-clock table |
| `plot_s2a_figure.py` | S2a accuracy/cost figure |
| `plot_s2a_depth_figure.py` | S2a depth figure |
| `plot_s2a_epoch_curves.py` | Epoch curves |
| `plot_head_lr_sweep.py` | LR sensitivity figure |
| `plot_circuit_grad_variance.py` | Gradient-variance figure |
| `generate_manifest.py` | SHA256 manifest for `results/tables/` |
| `check_artifacts.py` | Verify manifest (CI) |
| `demo_hybrid.ipynb` | Interactive hybrid training |

Install: `pip install -r requirements.txt` (GPU) or `requirements-ci.txt` (CPU/CI).
