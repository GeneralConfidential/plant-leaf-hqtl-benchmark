# plant-leaf-hqtl-benchmark

[![CI](https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Code and experiment outputs for hybrid quantum-classical transfer learning (HQTL) on PlantVillage-derived leaf subsets (SDP, VIT-AP).

**Paper:** *Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification: An Empirical Performance and Limitations Study* (manuscript maintained separately).

## Headline results (S2a: 10 tomato classes, 10 qubits, 3 seeds)

| Model | Val. accuracy | 10-epoch time |
|-------|---------------|---------------|
| ResNet18 fine-tune | **0.938 ± 0.017** | ~250 s |
| Frozen linear head | **0.878 ± 0.019** | ~193 s |
| HQTL hybrid (VQC head) | 0.695 ± 0.041 | ~3384 s |

On the same frozen ResNet18 embeddings, a linear head outperforms the VQC head by **~18 pp** at **~17× lower** training time. See `results/tables/s2a_summary.csv`.

## Quick start

**CPU smoke test (no data download, &lt; 5 min):**

```bash
git clone https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark.git
cd plant-leaf-hqtl-benchmark
pip install -r code/requirements-ci.txt
python code/run_smoke_test.py
```

**Full hybrid training:** see `REPRODUCTION.md` (requires PlantVillage download + GPU recommended).

## Repository layout

| Path | Description |
|------|-------------|
| `code/` | Training scripts, hybrid model, notebooks |
| `results/tables/` | Validation metrics (CSV) |
| `figures/` | Paper figures |
| `REPRODUCTION.md` | Splits, hyperparameters, exact commands |
| `ARTIFACTS.md` | Paper table → file mapping |
| `MODEL_CARD.md` | Scope, limits, intended use |
| `CITATION.cff` | Machine-readable citation metadata |

## Reproduce key artifacts

```bash
make smoke          # or: .\reproduce.ps1 smoke
make figures        # S2a accuracy/cost plot
make tables         # efficiency.csv
make check          # verify results/manifest.json
```

Full S2a multi-seed training (GPU, hours):

```bash
make s2a
```

## Citation

```bibtex
@software{gautam2026hqtl_plant,
  title        = {Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification: Code and Reproduction Artifacts},
  author       = {Gautam, Raag and B, Shreyas M},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark},
  note         = {Add Zenodo DOI after release}
}
```

Raag Gautam, Shreyas M B — VIT-AP. MIT license (see `LICENSE`).
