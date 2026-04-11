#!/usr/bin/env bash
# Create a directory containing ONLY symlinks to Brian's processed_*.parquet files
# so train.py does not also load input_chunk_*.parquet from the same folder.
#
# Run on Savio (login or compute), from repo root:
#   bash scripts/setup_smoke_data_symlinks.sh
# Human-only (same idea as mouse-only):
#   SPECIES=human bash scripts/setup_smoke_data_symlinks.sh
#
# Optional env (examples):
#   SPECIES=mouse|human   (default: mouse)
#   Smaller subset (~100k-sample tree), mouse:
#     BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/subset_100k_mouse
#   Full processed batches only for mouse (still excludes input_chunk_*):
#     BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/archs4_mouse
#     SMOKE_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-mouse-processed-only
#   Human full processed only (do NOT point BRIDGE_RNA_DATA_DIR at archs4_human itself
#   if that folder also has input_chunk_* — use this symlink dir or a processed-only tree):
#     SPECIES=human BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/archs4_human
#     SMOKE_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-smoke-data-human
#   If Brian provides subset_100k_human, prefer:
#     SPECIES=human BRIAN_PROCESSED_DIR=/global/scratch/users/brianzhou/subset_100k_human
#
# Defaults:
#   mouse: BRIAN_PROCESSED_DIR=.../subset_100k_mouse, SMOKE_DATA_DIR=.../bridge-rna-smoke-data
#   human: BRIAN_PROCESSED_DIR=.../archs4_human, SMOKE_DATA_DIR=.../bridge-rna-smoke-data-human

set -euo pipefail

SPECIES="${SPECIES:-mouse}"
case "$SPECIES" in
  mouse)
    PROCESSED_GLOB="processed_mouse_*.parquet"
    BRIAN_PROCESSED_DIR="${BRIAN_PROCESSED_DIR:-/global/scratch/users/brianzhou/subset_100k_mouse}"
    SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"
    ;;
  human)
    PROCESSED_GLOB="processed_human_*.parquet"
    BRIAN_PROCESSED_DIR="${BRIAN_PROCESSED_DIR:-/global/scratch/users/brianzhou/archs4_human}"
    SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data-human}"
    ;;
  *)
    echo "ERROR: SPECIES must be mouse or human, got: $SPECIES" >&2
    exit 1
    ;;
esac

if [[ ! -d "$BRIAN_PROCESSED_DIR" ]]; then
  echo "ERROR: Directory not found: $BRIAN_PROCESSED_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$BRIAN_PROCESSED_DIR"/$PROCESSED_GLOB)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "ERROR: No $PROCESSED_GLOB under $BRIAN_PROCESSED_DIR" >&2
  exit 1
fi

mkdir -p "$SMOKE_DATA_DIR"
# Drop symlinks from either species so one directory cannot mix mouse + human by mistake.
find "$SMOKE_DATA_DIR" -maxdepth 1 -type l \( -name 'processed_mouse_*.parquet' -o -name 'processed_human_*.parquet' \) -delete 2>/dev/null || true

for f in "${files[@]}"; do
  base=$(basename "$f")
  ln -sfn "$f" "$SMOKE_DATA_DIR/$base"
done

echo ""
echo "SPECIES=$SPECIES — created ${#files[@]} symlinks in:"
echo "  $SMOKE_DATA_DIR"
echo ""
echo "For training, use:"
echo "  export BRIDGE_RNA_DATA_DIR=$SMOKE_DATA_DIR"
echo "  # smoke:"
echo "  export BRIDGE_RNA_SMOKE=1"
echo ""
