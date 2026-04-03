#!/usr/bin/env python3
"""
Extract small reference subsets from large NASA ARCHS4 H5 and OSDR tables.

Does not copy the multi-GB source files. Writes compressed NumPy archives and CSV/TSV
under reference_subset/ for shape/metadata inspection and docs.

Usage:
  python scripts/extract_reference_samples.py
  python scripts/extract_reference_samples.py --nasa-root /path/to/NASA/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _decode_h5_strings(arr: np.ndarray) -> list:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        elif hasattr(x, "decode"):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def extract_archs4(
    h5_path: Path,
    species: str,
    n_samples: int,
    seed: int,
    out_dir: Path,
) -> None:
    rng = np.random.default_rng(seed + (0 if species == "human" else 17))
    with h5py.File(h5_path, "r") as f:
        expr = f["data/expression"]
        n_genes, n_total = expr.shape
        indices = np.sort(rng.choice(n_total, size=min(n_samples, n_total), replace=False))
        block = np.asarray(expr[:, indices], dtype=np.uint32)
        genes = _decode_h5_strings(f["meta/genes/symbol"][:])

        meta_keys = [
            "geo_accession",
            "organism_ch1",
            "title",
            "library_strategy",
            "series_id",
            "source_name_ch1",
        ]
        meta = {"sample_column_index": indices.astype(np.int64)}
        for key in meta_keys:
            path = f"meta/samples/{key}"
            if path in f:
                raw = f[path][indices]
                meta[key] = _decode_h5_strings(np.asarray(raw))

    stem = f"archs4_{species}_{len(indices)}"
    np.savez_compressed(
        out_dir / f"{stem}_expression.npz",
        counts=block,
        gene_symbol=np.array(genes, dtype=object),
        sample_h5_column_index=indices,
        species=np.array(species),
    )
    pd.DataFrame(meta).to_csv(out_dir / f"{stem}_metadata.csv", index=False)

    summary = {
        "species": species,
        "source_h5": str(h5_path),
        "expression_shape_genes_by_samples": list(block.shape),
        "dtype_stored": "uint32 (ARCHS4 raw counts)",
        "n_genes": int(n_genes),
        "files": [f"{stem}_expression.npz", f"{stem}_metadata.csv"],
    }
    with open(out_dir / f"{stem}_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)


def extract_osdr(
    counts_csv: Path,
    metadata_tsv: Path,
    n_rows: int,
    out_dir: Path,
) -> None:
    df_counts = pd.read_csv(counts_csv, nrows=n_rows)
    df_meta = pd.read_csv(metadata_tsv, sep="\t", nrows=n_rows)
    df_counts.to_csv(out_dir / "osdr_counts_first_n.csv", index=False)
    df_meta.to_csv(out_dir / "osdr_sample_metadata_first_n.tsv", sep="\t", index=False)
    summary = {
        "source_counts": str(counts_csv),
        "source_metadata": str(metadata_tsv),
        "n_rows_written": int(len(df_counts)),
        "n_expression_columns": int(df_counts.shape[1]),
        "metadata_columns": int(df_meta.shape[1]),
        "note": "Rows are the first n samples in each file (same ordering as source tables).",
    }
    with open(out_dir / "osdr_subset_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract small reference subsets from NASA data.")
    parser.add_argument(
        "--nasa-root",
        type=Path,
        default=Path("/Users/abrahamwestleyguan/Documents/GitHub/NASA/data"),
        help="Directory containing human_gene_v2.5.h5, mouse_gene_v2.5.h5, osdr-selected/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_subset",
        help="Output directory (created if missing).",
    )
    parser.add_argument("--n-archs4-per-species", type=int, default=50)
    parser.add_argument("--n-osdr", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_h5 = args.nasa_root / "human_gene_v2.5.h5"
    mouse_h5 = args.nasa_root / "mouse_gene_v2.5.h5"
    if not human_h5.is_file():
        raise FileNotFoundError(f"Missing {human_h5}")
    if not mouse_h5.is_file():
        raise FileNotFoundError(f"Missing {mouse_h5}")

    extract_archs4(human_h5, "human", args.n_archs4_per_species, args.seed, out_dir)
    extract_archs4(mouse_h5, "mouse", args.n_archs4_per_species, args.seed, out_dir)

    osdr_counts = args.nasa_root / "osdr-selected" / "osdr_counts_prepped.csv"
    osdr_meta = args.nasa_root / "osdr-selected" / "metadata" / "selected_sample_metadata.tsv"
    if osdr_counts.is_file() and osdr_meta.is_file():
        extract_osdr(osdr_counts, osdr_meta, args.n_osdr, out_dir)
    else:
        print("[WARN] OSDR paths not found; skipping OSDR subset.", flush=True)

    readme = out_dir / "README.md"
    readme.write_text(
        """# Reference subset (local extraction)

Small pulls from large NASA datasets for **shape and metadata context** only.

- **ARCHS4**: random bulk RNA-seq samples (counts matrix = genes × samples) plus CSV metadata.
  Source: `human_gene_v2.5.h5` / `mouse_gene_v2.5.h5` under your NASA data root.
- **OSDR**: first *n* rows of prepped counts and `selected_sample_metadata.tsv` (same row order as source files).

Regenerate:

```bash
python scripts/extract_reference_samples.py --nasa-root /path/to/NASA/data
```

Do not commit multi‑MB artifacts if your team policy forbids it; this folder can stay local.
""",
        encoding="utf-8",
    )
    print(f"[DONE] Wrote reference files under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
