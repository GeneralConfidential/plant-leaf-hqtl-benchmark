# Release checklist (v1.0.0)

## Pre-release verification

- [ ] `python code/run_smoke_test.py` passes on CPU
- [ ] `python code/check_artifacts.py` passes
- [ ] `results/tables/s2a_summary.csv` matches paper Table S2a
- [ ] `figures/s2a_accuracy_pareto.png` matches paper Figure
- [ ] `manuscript/main.pdf` builds with updated numbers (private repo)

## Commands run for v1.0.0 (private `sdp` repo)

```bash
uv sync
uv run python paper-release/code/run_s2_seeds.py --models linear_head resnet18_ft hybrid --seeds 42 123 456
uv run python paper-release/code/build_efficiency_table.py
uv run python paper-release/code/plot_s2a_figure.py
```

Hardware: NVIDIA RTX 3050 Ti, Intel i7-12700H. Hybrid S2a ~56 min/seed.

## GitHub release

1. Commit and push `paper-release` to `clean-main` (or `main`)
2. Tag `v1.0.0` with release notes from `CHANGELOG.md`
3. Enable Zenodo-GitHub integration → publish deposition
4. Add DOI to `CITATION.cff`, `README.md`, `manuscript/references.bib`

## Post-Zenodo

- [ ] Update citation blocks with `10.5281/zenodo.XXXXXXX`
- [ ] Rebuild `manuscript/main.pdf`
- [ ] Bump submodule pointer in private `sdp` repo
