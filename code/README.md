# Code

Training and analysis scripts for the public benchmark repo.

| Script | Purpose |
|--------|---------|
| `run_smoke_test.py` | CPU smoke test (synthetic data, no download) |
| `run_s1_linear_head.py` | S1 frozen linear head |
| `run_s2_seeds.py` | S2a multi-seed (linear, ResNet18 FT, hybrid) |
| `build_efficiency_table.py` | Params + wall-clock table |
| `plot_s2a_figure.py` | S2a accuracy/cost figure |
| `generate_manifest.py` | SHA256 manifest for `results/tables/` |
| `check_artifacts.py` | Verify manifest (used in CI) |
| `demo_hybrid.ipynb` | Interactive hybrid training |

Install: `pip install -r requirements.txt` (GPU) or `requirements-ci.txt` (CPU/CI).
