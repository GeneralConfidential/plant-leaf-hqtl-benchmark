#!/usr/bin/env python3
"""Verify results/manifest.json matches current results/tables CSV hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from paths import tables_dir


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=tables_dir().parent / "manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"Missing manifest: {args.manifest}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected: dict[str, str] = manifest.get("files", {})
    errors: list[str] = []

    for name, want in sorted(expected.items()):
        path = tables_dir() / name
        if not path.exists():
            errors.append(f"missing file: {name}")
            continue
        got = file_sha256(path)
        if got != want:
            errors.append(f"hash mismatch: {name}")

    for path in sorted(tables_dir().glob("*.csv")):
        if path.name not in expected:
            errors.append(f"untracked csv in manifest: {path.name}")

    if errors:
        for err in errors:
            print(f"CHECK FAILED: {err}", file=sys.stderr)
        return 1

    print(f"CHECK OK: {len(expected)} CSV files match manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
