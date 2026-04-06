#!/usr/bin/env bash
# Create a directory containing ONLY symlinks to Brian's processed_mouse_*.parquet files
# so train.py does not also load input_chunk_*.parquet from the same folder.
#
# Run on Savio (login or compute), from repo root:
#   bash scripts/setup_smoke_data_symlinks.sh
#
# Optional env (examples):
#   Smaller subset (~100k-sample tree):
#     BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/subset_100k_mouse
#   Full processed batches only (still excludes input_chunk_* — do NOT point BRIDGE_RNA_DATA_DIR at archs4_mouse itself):
#     BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/archs4_mouse
#     SMOKE_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-mouse-processed-only
#   Default SMOKE_DATA_DIR if unset:
#     /global/scratch/users/abrahamguan/bridge-rna-smoke-data

set -euo pipefail

BRIAN_PROCESSED_DIR="${BRIAN_PROCESSED_DIR:-/global/scratch/users/brianzhou/subset_100k_mouse}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"

if [[ ! -d "$BRIAN_PROCESSED_DIR" ]]; then
  echo "ERROR: Directory not found: $BRIAN_PROCESSED_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$BRIAN_PROCESSED_DIR"/processed_mouse_*.parquet)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "ERROR: No processed_mouse_*.parquet under $BRIAN_PROCESSED_DIR" >&2
  exit 1
fi

mkdir -p "$SMOKE_DATA_DIR"
# Remove stale symlinks from previous runs (only our pattern)
find "$SMOKE_DATA_DIR" -maxdepth 1 -type l -name 'processed_mouse_*.parquet' -delete 2>/dev/null || true

for f in "${files[@]}"; do
  base=$(basename "$f")
  ln -sfn "$f" "$SMOKE_DATA_DIR/$base"
done

echo ""
echo "Created ${#files[@]} symlinks in:"
echo "  $SMOKE_DATA_DIR"
echo ""
echo "For smoke training, use:"
echo "  export BRIDGE_RNA_DATA_DIR=$SMOKE_DATA_DIR"
echo "  export BRIDGE_RNA_SMOKE=1"
echo ""
