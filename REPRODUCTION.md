# Reproduction notes

## Model

Frozen ResNet18 (ImageNet) + dressed VQC head in PennyLane:

`Linear(512→n_qubits)` → tanh → π/2 → variational circuit → `Linear(n_qubits→n_classes)`

Defaults in the notebook: 4 qubits, depth 4, lr 0.0004, batch 16, 10 epochs.

## Experiments

| ID | What | Classes | Qubits |
|----|------|---------|--------|
| S1 | Unrelated leaf categories | 4 | 4 |
| S2 | Tomato subset | 10 | 10 or 16 |
| S3 | Tomato subset, depth sweep | 4 | 4 |

S1 folder layout:

```
code/data/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry___healthy/
└── Background_without_leaves/
```

## Protocol

- Resize 256, center crop 224, ImageNet normalization
- 80/20 train/val, up to 400 images per class for S1, seed 42
- Metrics: accuracy, macro precision/recall/F1

ResNet18 and DenseNet121 baselines were trained in our private dev repo with the same splits.

## Hardware we used

RTX 3050 Ti, i7-12700H. PennyLane `default.qubit` simulator.

## Precomputed outputs

Metrics: `results/tables/*.csv`  
Figures: `figures/`
