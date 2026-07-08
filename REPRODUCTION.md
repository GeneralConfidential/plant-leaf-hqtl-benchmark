# Reproduction notes

## Model

Frozen ResNet18 (ImageNet) + dressed VQC head in PennyLane:

`Linear(512→n_qubits)` → tanh → π/2 → variational circuit → `Linear(n_qubits→n_classes)`

Defaults: depth 4, lr 4e-4, batch 16, 10 epochs (hybrid/linear head).

## Quick verify (CPU, no PlantVillage)

```bash
pip install -r code/requirements-ci.txt
python code/run_smoke_test.py
```

Uses a synthetic 4-class mini-dataset; checks hybrid + linear forward/backward and metrics.

## Experiments

| ID | What | Classes | Qubits |
|----|------|---------|--------|
| S1 | Unrelated leaf categories | 4 | 4 |
| S2a | Tomato subset (multi-seed) | 10 | 10 |
| S2b | Tomato subset (hybrid, multi-seed) | 10 | 16 |
| S3 | Tomato subset, depth sweep | 4 | 4 |

### S1 folder layout

```
code/data/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry___healthy/
└── Background_without_leaves/
```

### S2a folder layout

```
code/data/s2_tomato/
├── Tomato___Bacterial_spot/
├── ...
└── Tomato___healthy/
```

Download from [PlantVillage (Mendeley)](https://data.mendeley.com/datasets/tywbtsjrjv/1).

## Exact commands (v1.0.0)

From repo root, with GPU and data in place:

```bash
pip install -r code/requirements.txt

# S2a multi-seed (linear + ResNet18 + hybrid)
python code/run_s2_seeds.py --models linear_head resnet18_ft hybrid --seeds 42 123 456

# S2b multi-seed (hybrid only, 16 qubits)
python code/run_s2_seeds.py --models hybrid --n-qubits 16 --seeds 42 123 456 \
  --runs-csv results/tables/s2b_runs.csv --summary-csv results/tables/s2b_summary.csv

# Efficiency table
python code/build_efficiency_table.py

# S2a figure for paper
python code/plot_s2a_figure.py

# Manifest for audit
python code/generate_manifest.py
python code/check_artifacts.py
```

From the private `sdp` monorepo, use `uv run python paper-release/code/...` with data under `data/plantvillage/s2_tomato/`.

## Protocol

- Resize 256, center crop 224, ImageNet normalization
- 80/20 train/val, up to 400 images per class, seed 42 (S1) or seeds {42,123,456} (S2a; S2b hybrid only)
- Metrics: accuracy, macro precision/recall/F1
- Hybrid/linear: frozen ResNet18; ResNet18/DenseNet121 baselines: full fine-tune

## Hardware we used

RTX 3050 Ti, i7-12700H. PennyLane `default.qubit` simulator. Hybrid S2a ~56 min/seed; S2b ~7.7 h/seed.

## Precomputed outputs

Metrics: `results/tables/*.csv`  
Figures: `figures/`  
Integrity: `results/manifest.json`
