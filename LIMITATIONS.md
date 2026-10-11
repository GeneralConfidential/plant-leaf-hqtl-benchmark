# Scope and limits

- Setup: frozen ImageNet ResNet18 → dress (`512 → n_qubits`) → VQC or matched `tanh` → linear readout.
- Data: capped PlantVillage subsets (S1 / S2 tomato / S3), usually ≤400 images per class. Simulator only (`default.qubit`), not NISQ hardware.
- Hybrid, linear, and tanh freeze the backbone; ResNet18 / DenseNet121 baselines fine-tune the full network.
- Held-out test CSVs exist for S1, S2a, and S3 under `results/tables/`.
- Wall-clock times are dominated by state-vector simulation; treat them as simulator cost.
- Pins: roughly PennyLane 0.45 / PyTorch 2.6 — see `code/requirements.txt`.

Not a field disease detector and not a full 38-class PlantVillage leaderboard run.
