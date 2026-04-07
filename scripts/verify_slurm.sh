#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${1:-scripts/savio_dev_train_savio2_1080ti.slurm}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: Slurm script not found: ${SCRIPT_PATH}" >&2
  echo "Usage: scripts/verify_slurm.sh [path/to/script.slurm]" >&2
  exit 1
fi

echo "== Slurm fingerprint: ${SCRIPT_PATH} =="
rg -n "^(#SBATCH --(job-name|account|partition|qos|gres|nodes|ntasks|cpus-per-task|mem|time|output|error)|module load|conda activate|export BRIDGE_RNA_|torchrun )" "${SCRIPT_PATH}" || true
echo
echo "== Syntax check (sbatch --test-only) =="
sbatch --test-only "${SCRIPT_PATH}"
echo
echo "OK: review fingerprint above, then submit with:"
echo "  sbatch ${SCRIPT_PATH}"
