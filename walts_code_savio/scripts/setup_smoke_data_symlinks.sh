#!/usr/bin/env bash
# Optional: if a folder mixes processed_mouse_*.parquet with other Parquets, this
# creates a new folder of symlinks to only processed_mouse_*.parquet for BRIDGE_RNA_DATA_DIR.
#
#   export BRIAN_PROCESSED_DIR=/path/to/mixed_folder
#   export SMOKE_DATA_DIR=/path/to/your_clean_folder
#   bash scripts/setup_smoke_data_symlinks.sh

set -euo pipefail

if [[ -z "${BRIAN_PROCESSED_DIR:-}" || -z "${SMOKE_DATA_DIR:-}" ]]; then
  echo "Set both directories, e.g.:" >&2
  echo "  export BRIAN_PROCESSED_DIR=/path/to/folder/with/processed_mouse_*.parquet" >&2
  echo "  export SMOKE_DATA_DIR=/path/to/empty_or_new_folder_for_symlinks" >&2
  exit 1
fi

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
echo "Then: export BRIDGE_RNA_DATA_DIR=$SMOKE_DATA_DIR"
echo ""
