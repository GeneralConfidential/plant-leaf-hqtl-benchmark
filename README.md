# plant-leaf-hqtl-benchmark

Public reproduction artifacts for *Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification* (Gautam & B, VIT-AP).

This repository contains **code, figures, and experiment protocols** reported in the paper. The manuscript itself is not published here.

## Contents

```
├── code/               # Training notebook + dependencies
├── figures/            # Paper figures (depth ablation, sample predictions)
├── results/tables/     # Validation metrics (CSV)
├── REPRODUCTION.md     # How to rerun experiments
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark.git
cd plant-leaf-hqtl-benchmark/code
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook demo_hybrid.ipynb
```

Download [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) images and place them under `code/data/<class_name>/`. See `REPRODUCTION.md` for splits and hyperparameters.

## Results

Validation metrics shipped with this release are in `results/tables/`. Figures are in `figures/`.

## Citation

```bibtex
@misc{gautam2024hqtl_plant,
  title={Hybrid Quantum-Classical Transfer Learning for Plant Leaf Classification:
         Code and Reproduction Artifacts},
  author={Gautam, Raag and B, Shreyas M},
  year={2024},
  howpublished={\url{https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark}}
}
```

## Authors

Raag Gautam, Shreyas M B — School of Computer Science and Engineering, VIT-AP, Amaravati, India

## License

MIT — see [LICENSE](LICENSE).
