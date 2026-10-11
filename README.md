# plant-leaf-hqtl-benchmark

[![CI](https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark/actions/workflows/ci.yml)

Code and CSV results for hybrid quantum-classical transfer learning (HQTL) on PlantVillage leaf subsets. Raag Gautam and Shreyas M. B., VIT-AP.

Paper (separate repo / Overleaf): *Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification: An Empirical Performance and Limitations Study*.

Frozen ImageNet ResNet18 embeddings, PennyLane dressed VQC head, compared with a frozen linear head and a matched-width head with the circuit replaced by `tanh`.

## Results (S2a: 10 tomato classes, 10 qubits, 3 seeds)

| Model | Val. accuracy | ~10-epoch time |
|-------|---------------|----------------|
| ResNet18 fine-tune | 0.938 ± 0.017 | ~250 s |
| Frozen linear head | 0.878 ± 0.019 | ~193 s |
| Matched tanh (no VQC) | 0.844 ± 0.019 | ~169 s |
| Hybrid VQC head (d=4) | 0.695 ± 0.041 | ~3384 s |

On the same embeddings the linear and tanh heads beat the hybrid and train much faster. Start with `results/tables/s2a_summary.csv`. Other settings and controls are listed in `TABLES.md`.

S2b (16 qubits, hybrid): 0.787 ± 0.031 — see `results/tables/s2b_summary.csv`.

## Quick check (CPU, no PlantVillage)

```bash
git clone https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark.git
cd plant-leaf-hqtl-benchmark
pip install -r code/requirements-ci.txt
python code/run_smoke_test.py
```

Full training needs PlantVillage on disk and preferably a GPU. See `REPRODUCTION.md`.

```bash
make smoke    # same as the smoke test above
make stats    # recompute table summaries from CSVs
make figures  # rebuild plots from CSVs
make check    # verify results/manifest.json
```

## Files

- `code/` — training and plotting scripts
- `results/tables/` — metrics CSVs
- `figures/` — paper plots
- `REPRODUCTION.md` — data layout, hyperparameters, commands
- `TABLES.md` — which CSV/figure matches which paper table
- `LIMITATIONS.md` — scope and caveats

## Citation

```bibtex
@software{gautam2026hqtl_plant,
  title   = {Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification: Code and Reproduction Artifacts},
  author  = {Gautam, Raag and B, Shreyas M},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark}
}
```

MIT license — see `LICENSE`.
