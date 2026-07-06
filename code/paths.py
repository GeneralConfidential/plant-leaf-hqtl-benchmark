"""Repository path helpers (paper-release standalone or sdp monorepo)."""

from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PUBLIC_ROOT = CODE_DIR.parent


def is_private_monorepo() -> bool:
    return (PUBLIC_ROOT.parent / "manuscript" / "main.tex").exists()


def private_root() -> Path:
    return PUBLIC_ROOT.parent if is_private_monorepo() else PUBLIC_ROOT


def elevation_results_dir() -> Path:
    if is_private_monorepo():
        return private_root() / "results" / "elevation"
    return PUBLIC_ROOT / "results" / "tables"


def tables_dir() -> Path:
    return PUBLIC_ROOT / "results" / "tables"


def figures_dir() -> Path:
    return PUBLIC_ROOT / "figures"


def default_s1_data_dir() -> Path:
    if is_private_monorepo():
        return private_root() / "paper-release" / "code" / "data"
    return CODE_DIR / "data"


def default_s2_data_dir() -> Path:
    if is_private_monorepo():
        return private_root() / "data" / "plantvillage" / "s2_tomato"
    return CODE_DIR / "data" / "s2_tomato"


def default_s2_processed_base() -> Path:
    if is_private_monorepo():
        return private_root() / "data" / "plantvillage" / "s2_processed"
    return CODE_DIR / "scl_metadata" / "s2_processed"
