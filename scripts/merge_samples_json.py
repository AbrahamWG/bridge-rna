#!/usr/bin/env python3
"""Merge W&B-style samples.json lists [{id, species}, ...] from multiple files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate samples.json list files.")
    p.add_argument("inputs", nargs="+", type=Path, help="samples.json files to merge (order preserved)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output path")
    args = p.parse_args()

    merged: list = []
    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"Expected JSON list in {path}, got {type(data)}")
        merged.extend(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged):,} samples -> {args.output}")


if __name__ == "__main__":
    main()
