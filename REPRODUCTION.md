# Reproduction Guide

How to rerun the hybrid quantum-classical transfer learning experiments described in the paper.

## Model (hybrid)

- **Backbone:** ResNet18 (ImageNet, frozen)
- **Head:** dressed VQC — `Linear(512→n_qubits)` → tanh → π/2 → PennyLane circuit → `Linear(n_qubits→n_classes)`
- **Defaults:** 4 qubits, depth 4, Adam lr=0.0004, batch 16, 10 epochs

Implementation: `code/demo_hybrid.ipynb` (Mari et al. 2019 / PennyLane quantum transfer learning pattern).

## Dataset subsets

| ID | Task | Classes | Qubits |
|----|------|---------|--------|
| S1 | Unrelated leaf categories | 4 | 4 |
| S2 | Ten tomato classes | 10 | 10 or 16 |
| S3 | Four tomato classes (depth ablation) | 4 | 4 |

Example S1 layout:

```
code/data/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry___healthy/
└── Background_without_leaves/
```

## Protocol

1. Resize 256 → center crop 224, ImageNet normalization
2. 80/20 train/validation split, up to 400 images/class (S1), seed 42
3. Metrics: accuracy, macro precision/recall/F1
4. Baselines (run separately in private development repo): Simple CNN, ResNet18, DenseNet121

## Hardware (original runs)

NVIDIA RTX 3050 Ti, Intel i7-12700H; PennyLane `default.qubit` simulator.

## Shipped results

Precomputed validation metrics: `results/tables/*.csv`  
Paper figures: `figures/`
