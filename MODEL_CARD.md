# Model card — HQTL plant leaf classifier

## Model description

Frozen ImageNet ResNet18 feature extractor + PennyLane variational quantum circuit (VQC) classifier head with linear readout. “Dressed” input: `Linear(512→n_qubits)` → tanh → angle embedding → entangled `R_y` layers → `Linear(n_qubits→K)`.

## Intended use

- Research reproduction of our PlantVillage-subset benchmarks
- Teaching/demo of hybrid quantum-classical transfer learning in PennyLane + PyTorch
- Comparing VQC heads against a **parameter-matched frozen linear head** on the same embeddings

## Out of scope

- Production plant-disease deployment
- Full 38-class PlantVillage benchmark
- NISQ hardware inference (simulation only: `default.qubit`)
- Disease detection on S1 (coarse leaf/background categories only)

## Training data

Curated subsets of [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1):
- S1: 4 visually distinct classes
- S2: 10 tomato classes
- S3: 4 tomato classes (depth ablation)

80/20 stratified train/validation split; up to 400 images per class.

## Evaluation

Macro-averaged precision, recall, F1, and accuracy on the validation split. S2a reports mean ± std over seeds `{42, 123, 456}` for hybrid, linear head, and ResNet18. S2b reports mean ± std for hybrid only (16 qubits, same seeds).

## Limitations

- **Protocol asymmetry:** hybrid and linear head freeze ResNet18; ResNet18/DenseNet121 baselines fine-tune the full network.
- **No held-out test set** on multi-seed runs.
- **Simulation cost:** per-sample VQC execution is much slower than a linear head (~17× wall-clock on S2a; ~8× longer on S2b vs S2a hybrid).
- **Software:** PennyLane 0.45 / PyTorch 2.6 (see `code/requirements.txt`).

## Ethical considerations

Public agricultural image dataset; no human subjects. Results are dataset-specific and should not be extrapolated to field deployment without further validation.

## Citation

See `CITATION.cff` and `README.md`. Prefer the Zenodo DOI once published.
