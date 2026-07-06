.PHONY: smoke tables figures manifest check

smoke:
	python code/run_smoke_test.py

tables:
	python code/build_efficiency_table.py

figures:
	python code/plot_s2a_figure.py

manifest:
	python code/generate_manifest.py

check:
	python code/check_artifacts.py

# Full S2a reproduction (GPU, hours). Requires PlantVillage S2 data under code/data/s2_tomato/.
s2a:
	python code/run_s2_seeds.py --models linear_head resnet18_ft hybrid --seeds 42 123 456
