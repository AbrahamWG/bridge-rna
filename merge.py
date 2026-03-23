#!/usr/bin/env python3
"""Merge sample-major parquet batch files into a single parquet file.

Input layout (default):
  data/archs4/train_orthologs/
	batch_files/*.parquet
	samples.json
	metadata.csv
	genes.json
	canonical_genes.csv

Output layout:
  <output_dir>/expression.parquet
  <output_dir>/batch_manifest.json
  <output_dir>/samples.json            (copied if present)
  <output_dir>/metadata.csv            (copied if present)
  <output_dir>/genes.json              (copied if present)
  <output_dir>/canonical_genes.csv     (copied if present)

Notes:
  - Rows remain samples and columns remain genes.
  - The manifest maps expression.parquet to the ordered sample IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INDEX_CANDIDATES = ("geo_accession", "__index_level_0__")


def _find_index_column(columns: list[str]) -> str | None:
	for name in INDEX_CANDIDATES:
		if name in columns:
			return name
	return None


def merge_parquets(input_dir: Path, output_dir: Path, compression: str = "zstd") -> None:
	batch_dir = input_dir / "batch_files"
	if not batch_dir.exists():
		batch_dir = input_dir

	batch_files = sorted(batch_dir.glob("*.parquet"))
	if not batch_files:
		raise FileNotFoundError(f"No parquet files found in {batch_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	out_file = output_dir / "expression.parquet"

	writer: pq.ParquetWriter | None = None
	base_columns: list[str] | None = None
	sample_ids: list[str] = []
	total_rows = 0

	print(f"[MERGE] Found {len(batch_files):,} parquet files in {batch_dir}")

	try:
		for i, file_path in enumerate(batch_files, start=1):
			table = pq.read_table(file_path, use_threads=True)
			cols = table.schema.names

			if base_columns is None:
				base_columns = cols
			elif cols != base_columns:
				raise ValueError(
					f"Schema mismatch in {file_path.name}. "
					"All parquet files must have the same ordered columns."
				)

			idx_col = _find_index_column(cols)
			if idx_col is not None:
				sample_ids.extend(map(str, table.column(idx_col).to_pylist()))
			else:
				# Fallback if an explicit sample ID column is absent.
				start = len(sample_ids)
				sample_ids.extend([str(start + j) for j in range(table.num_rows)])

			if writer is None:
				writer = pq.ParquetWriter(
					out_file,
					table.schema,
					compression=compression,
					use_dictionary=True,
				)

			writer.write_table(table)
			total_rows += table.num_rows

			if i % 25 == 0 or i == len(batch_files):
				print(f"[MERGE] {i}/{len(batch_files)} files merged ({total_rows:,} rows)")
	finally:
		if writer is not None:
			writer.close()

	manifest = {"expression.parquet": sample_ids}
	manifest_path = output_dir / "batch_manifest.json"
	with open(manifest_path, "w") as f:
		json.dump(manifest, f)

	copied = []
	for name in ("samples.json", "metadata.csv", "genes.json", "canonical_genes.csv"):
		src = input_dir / name
		if src.exists():
			dst = output_dir / name
			shutil.copy2(src, dst)
			copied.append(name)

	print(f"[DONE] Wrote {out_file} ({total_rows:,} rows)")
	print(f"[DONE] Wrote {manifest_path} with {len(sample_ids):,} sample IDs")
	if copied:
		print(f"[DONE] Copied metadata files: {', '.join(copied)}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Merge all parquet files into expression.parquet and generate manifest"
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("./data/archs4/train_orthologs"),
		help="Directory containing batch_files/ and metadata",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("./data/archs4/train_orthologs_merged"),
		help="Directory where expression.parquet and manifest are written",
	)
	parser.add_argument(
		"--compression",
		default="zstd",
		help="Parquet compression codec (default: zstd)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	merge_parquets(args.input_dir, args.output_dir, compression=args.compression)


if __name__ == "__main__":
	main()
