#!/usr/bin/env bash
# Build a single BRIDGE_RNA_DATA_DIR view for train.py with mouse + human Parquets
# and a merged samples.json so balanced_sampling can do 50/50 human/mouse.
#
# ⚠️  CRITICAL — gene column compatibility (read before using)
#   train.py StreamingParquetMLMDataset uses the *first* Parquet file’s gene columns
#   for ALL files. Brian’s separate ARCHS4 trees have *different* vocabulary sizes
#   (human ~16734 genes, mouse ~16551 per gene_vocabulary / processed batches).
#   You CANNOT safely mix raw archs4_human + archs4_mouse parquets in one folder
#   unless Brian’s files share identical gene columns (same names, same order).
#   For true joint human+mouse training, use preprocessing output in ONE canonical
#   ortholog space (e.g. RNADatasetBuilder species=both, shared_orthologs), not two
#   independent archs4_* roots. Ask Brian for a combined batch_files + samples.json.
#
# Prerequisite: run on Savio after you know Brian's paths:
#   bash scripts/discover_brian_scratch_data.sh
#
# --- Mode A (simplest): Brian already wrote one output_dir with batch_files/ + samples.json
#    for both species. Do NOT use this script — set:
#      export BRIDGE_RNA_DATA_DIR=/path/to/combined_output
#
# --- Mode B: Two directories, each with ONLY processed *.parquet (no input_chunk_*) and
#    each with its own samples.json listing {id, species}.
#
# Usage (example):
#   MOUSE_PARQUET_DIR=/global/scratch/users/brianzhou/subset_100k_mouse \
#   HUMAN_PARQUET_DIR=/global/scratch/users/brianzhou/subset_100k_human \
#   TARGET_DIR=/global/scratch/users/abrahamguan/bridge-rna-mouse-human-symlinks \
#   bash scripts/setup_mouse_human_symlinks.sh
#
# Optional glob patterns (defaults match common Brian names):
#   MOUSE_GLOB='processed_mouse_*.parquet'
#   HUMAN_GLOB='processed_human_*.parquet'
#
# Then training:
#   export BRIDGE_RNA_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-mouse-human-symlinks
#   # balanced_sampling=True (default) + train_subset/val_subset → 50/50 per species when both present

set -euo pipefail

MOUSE_PARQUET_DIR="${MOUSE_PARQUET_DIR:-}"
HUMAN_PARQUET_DIR="${HUMAN_PARQUET_DIR:-}"
TARGET_DIR="${TARGET_DIR:-/global/scratch/users/abrahamguan/bridge-rna-mouse-human-symlinks}"
MOUSE_GLOB="${MOUSE_GLOB:-processed_mouse_*.parquet}"
HUMAN_GLOB="${HUMAN_GLOB:-processed_human_*.parquet}"

if [[ -z "$MOUSE_PARQUET_DIR" || -z "$HUMAN_PARQUET_DIR" ]]; then
  echo "Set MOUSE_PARQUET_DIR and HUMAN_PARQUET_DIR to directories containing only the Parquets you want." >&2
  echo "See script header for Mode A if Brian has a single combined output_dir." >&2
  exit 1
fi

for d in "$MOUSE_PARQUET_DIR" "$HUMAN_PARQUET_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: Not a directory: $d" >&2
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGE_PY="${REPO_ROOT}/scripts/merge_samples_json.py"

shopt -s nullglob
mouse_files=("$MOUSE_PARQUET_DIR"/$MOUSE_GLOB)
human_files=("$HUMAN_PARQUET_DIR"/$HUMAN_GLOB)
shopt -u nullglob

if [[ ${#mouse_files[@]} -eq 0 ]]; then
  echo "ERROR: No files matching $MOUSE_GLOB under $MOUSE_PARQUET_DIR" >&2
  exit 1
fi
if [[ ${#human_files[@]} -eq 0 ]]; then
  echo "ERROR: No files matching $HUMAN_GLOB under $HUMAN_PARQUET_DIR" >&2
  echo "Hint: set HUMAN_GLOB if Brian uses a different name (e.g. human_batch_*.parquet)." >&2
  exit 1
fi

M_SAMPLES="${MOUSE_PARQUET_DIR}/samples.json"
H_SAMPLES="${HUMAN_PARQUET_DIR}/samples.json"
if [[ ! -f "$M_SAMPLES" || ! -f "$H_SAMPLES" ]]; then
  echo "ERROR: Need samples.json in both dirs for species-balanced training." >&2
  echo "  Mouse: $M_SAMPLES" >&2
  echo "  Human: $H_SAMPLES" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
# Drop only our previous symlinks / merged metadata
find "$TARGET_DIR" -maxdepth 1 -type l \( -name 'processed_mouse_*.parquet' -o -name 'processed_human_*.parquet' -o -name 'human_*.parquet' -o -name 'mouse_*.parquet' \) -delete 2>/dev/null || true
rm -f "$TARGET_DIR/samples.json"

# Symlink with stable unique basenames (mouse / human prefixes if basename collision)
for f in "${mouse_files[@]}"; do
  base=$(basename "$f")
  name="mouse__${base}"
  ln -sfn "$f" "$TARGET_DIR/$name"
done
for f in "${human_files[@]}"; do
  base=$(basename "$f")
  name="human__${base}"
  ln -sfn "$f" "$TARGET_DIR/$name"
done

python3 "$MERGE_PY" "$M_SAMPLES" "$H_SAMPLES" -o "$TARGET_DIR/samples.json"

n_parquet=$(find "$TARGET_DIR" -maxdepth 1 -type l -name '*.parquet' | wc -l | tr -d ' ')
echo ""
echo "OK: $n_parquet parquet symlinks + merged samples.json"
echo "  $TARGET_DIR"
echo ""
echo "export BRIDGE_RNA_DATA_DIR=$TARGET_DIR"
echo "# train.py will use this dir as batch_files root (no batch_files subdir) — glob *.parquet"
echo "# Ensure BRIDGE_RNA_TRAIN_SUBSET + VAL_SUBSET leave room for 50/50 balance (see train.py get_sample_indices)."
