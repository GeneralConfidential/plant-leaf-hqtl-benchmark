# Experiment code

Run `demo_hybrid.ipynb` for the primary hybrid model pipeline.

## Data layout

```
data/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry___healthy/
└── Background_without_leaves/
```

The notebook creates `scl_metadata/train` and `scl_metadata/val` on first run.

## Hyperparameters (defaults in notebook)

| Parameter | Value |
|-----------|-------|
| n_qubits | 4 |
| q_depth | 4 |
| batch_size | 16 |
| learning_rate | 0.0004 |
| num_epochs | 10 |
| GLOBAL_SEED | 42 |

For tomato and multi-class experiments, adjust `n_qubits`, `n_classes`, and `q_depth` to match the paper tables.
