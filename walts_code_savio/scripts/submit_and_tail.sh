#!/usr/bin/env bash
# Submit savio_train.slurm, wait until the job finishes, then print sacct + log tails.
#
# Usage (from folder that contains train.py):
#   export BRIDGE_RNA_DATA_DIR=/path/to/parquet
#   bash scripts/submit_and_tail.sh
#
# Wait for a job you already submitted (no new sbatch):
#   bash scripts/submit_and_tail.sh 12345678

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SLURM_SCRIPT="${SLURM_SCRIPT:-scripts/savio_train.slurm}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  JOB_ID="$1"
  echo "Waiting for existing job ${JOB_ID}"
else
  if sbatch --help 2>&1 | grep -q parsable; then
    JOB_ID="$(sbatch --parsable --export=ALL "${SLURM_SCRIPT}")"
  else
    out="$(sbatch --export=ALL "${SLURM_SCRIPT}")"
    JOB_ID="$(echo "${out}" | awk '{print $NF}')"
  fi
  echo "Submitted job ${JOB_ID}"
fi

echo "Waiting for job ${JOB_ID} (tail in another terminal: tail -f logs/slurm-${JOB_ID}.out)"
while squeue -j "${JOB_ID}" -h 2>/dev/null | grep -q .; do
  line="$(squeue -j "${JOB_ID}" -h -o '%.10i %8T %10M %R' 2>/dev/null | head -1)"
  echo "  $(date '+%H:%M:%S')  ${line:-running}"
  sleep 8
done
sleep 2

echo ""
echo "======== sacct ========"
sacct -j "${JOB_ID}" --format=JobID,NodeList,State,Elapsed,ExitCode,End 2>/dev/null || true

OUT="${REPO_ROOT}/logs/slurm-${JOB_ID}.out"
ERR="${REPO_ROOT}/logs/slurm-${JOB_ID}.err"

for label in out err; do
  f="${OUT}"
  [[ "$label" == "err" ]] && f="${ERR}"
  echo ""
  echo "======== tail ${f} ========"
  if [[ -f "${f}" ]]; then
    tail -120 "${f}"
  else
    echo "(missing)"
  fi
done

echo ""
echo "ExitCode 0:0 on the main JobID row means success."
