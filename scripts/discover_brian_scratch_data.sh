#!/usr/bin/env bash
# Run on Savio (login or compute) to locate Brian Zhou's Parquet outputs under scratch.
# Does not modify anything — prints paths for you to plug into setup_mouse_human_symlinks.sh
#
# Usage:
#   bash scripts/discover_brian_scratch_data.sh
# Optional:
#   BRIAN_ROOT=/global/scratch/users/brianzhou bash scripts/discover_brian_scratch_data.sh

set -euo pipefail

ROOT="${BRIAN_ROOT:-/global/scratch/users/brianzhou}"

echo "== Listing top-level (max depth 1) =="
if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Not found: $ROOT (set BRIAN_ROOT if Brian uses another path)" >&2
  exit 1
fi
ls -la "$ROOT"

echo ""
echo "== Directories named batch_files (depth ≤4) =="
find "$ROOT" -maxdepth 4 -type d -name 'batch_files' 2>/dev/null | head -40

echo ""
echo "== samples.json (depth ≤5) =="
find "$ROOT" -maxdepth 5 -type f -name 'samples.json' 2>/dev/null | head -40

echo ""
echo "== Processed Parquet name patterns (first 30 each) =="
echo "--- processed_mouse_*.parquet ---"
find "$ROOT" -maxdepth 4 -type f -name 'processed_mouse_*.parquet' 2>/dev/null | head -30
echo "--- processed_human_*.parquet ---"
find "$ROOT" -maxdepth 4 -type f -name 'processed_human_*.parquet' 2>/dev/null | head -30
echo "--- human_batch_*.parquet ---"
find "$ROOT" -maxdepth 4 -type f -name 'human_batch_*.parquet' 2>/dev/null | head -15
echo "--- mouse_batch_*.parquet ---"
find "$ROOT" -maxdepth 4 -type f -name 'mouse_batch_*.parquet' 2>/dev/null | head -15
echo "--- batch_*.parquet under batch_files ---"
find "$ROOT" -path '*/batch_files/batch_*.parquet' -type f 2>/dev/null | head -20

echo ""
echo "== Next steps =="
echo "1) If Brian gave ONE output dir with batch_files/ + samples.json (human+mouse):"
echo "     export BRIDGE_RNA_DATA_DIR=/path/to/that/dir"
echo "     (train.py uses batch_files/ under it; balanced_sampling needs species in samples.json)"
echo "2) If mouse and human live in SEPARATE dirs:"
echo "     bash scripts/setup_mouse_human_symlinks.sh   # see script header for env vars"
