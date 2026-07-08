# Reproduction entrypoints (Windows). Run from paper-release/ root.
param(
    [Parameter(Position = 0)]
    [ValidateSet("smoke", "tables", "figures", "manifest", "check", "s2a", "s2b")]
    [string]$Target = "smoke"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

switch ($Target) {
    "smoke"    { python code/run_smoke_test.py }
    "tables"   { python code/build_efficiency_table.py }
    "figures"  { python code/plot_s2a_figure.py }
    "manifest" { python code/generate_manifest.py }
    "check"    { python code/check_artifacts.py }
    "s2a"      {
        python code/run_s2_seeds.py --models linear_head resnet18_ft hybrid --seeds 42 123 456
    }
    "s2b"      {
        python code/run_s2_seeds.py --models hybrid --n-qubits 16 --seeds 42 123 456 `
            --runs-csv results/tables/s2b_runs.csv --summary-csv results/tables/s2b_summary.csv
        python code/build_efficiency_table.py
        python code/generate_manifest.py
    }
}
