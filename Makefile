.PHONY: smoke tables figures manifest check s2a s2b

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

# S2b hybrid only (16 qubits, ~7.7 h/seed). Same data requirement as s2a.
s2b:
	python code/run_s2_seeds.py --models hybrid --n-qubits 16 --seeds 42 123 456 \
		--runs-csv results/tables/s2b_runs.csv --summary-csv results/tables/s2b_summary.csv
	python code/build_efficiency_table.py
	python code/generate_manifest.py
