# plant-leaf-hqtl-benchmark

Code and experiment outputs for our work on hybrid quantum-classical transfer learning for plant leaf classification (SDP, VIT-AP).

The write-up itself lives elsewhere. This repo is only for rerunning the hybrid model and checking the reported numbers.

## What's in the repo

- `code/demo_hybrid.ipynb` — ResNet18 + PennyLane VQC head, training and eval
- `code/requirements.txt` — PyTorch, PennyLane, etc.
- `figures/` — depth ablation plot and sample predictions
- `results/tables/` — validation metrics (CSV, one file per experiment)
- `REPRODUCTION.md` — splits, hyperparameters, dataset layout

## Run it

```bash
git clone https://github.com/GeneralConfidential/plant-leaf-hqtl-benchmark.git
cd plant-leaf-hqtl-benchmark/code
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
jupyter notebook demo_hybrid.ipynb
```

Grab leaf images from [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and put them under `code/data/<class_name>/`. The notebook builds the train/val split on first run. Details in `REPRODUCTION.md`.

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

Raag Gautam, Shreyas M B — VIT-AP. MIT license (see `LICENSE`).
