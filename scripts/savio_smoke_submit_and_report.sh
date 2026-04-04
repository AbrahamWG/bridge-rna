#!/usr/bin/env bash
# Submit savio2 1080 Ti smoke training, wait until the job leaves the queue, then
# print sacct + tails of Slurm .out / .err (one place to see success or failure).
#
# Usage (from repo root on Savio):
#   bash scripts/savio_smoke_submit_and_report.sh
#
# Optional: wait for an existing job only (no submit):
#   bash scripts/savio_smoke_submit_and_report.sh 33196453
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SLURM_SCRIPT="${SLURM_SCRIPT:-scripts/savio_smoke_train_savio2_1080ti.slurm}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  JOB_ID="$1"
  echo "Waiting for existing job ${JOB_ID} (no sbatch)"
else
  if sbatch --help 2>&1 | grep -q parsable; then
    JOB_ID="$(sbatch --parsable "${SLURM_SCRIPT}")"
  else
    out="$(sbatch "${SLURM_SCRIPT}")"
    JOB_ID="$(echo "${out}" | awk '{print $NF}')"
  fi
  echo "Submitted batch job ${JOB_ID}"
fi

echo "Polling until job ${JOB_ID} leaves squeue..."
while squeue -j "${JOB_ID}" -h 2>/dev/null | grep -q .; do
  sleep 8
done
# Brief pause so Slurm flushes log files
sleep 2

echo ""
echo "======== sacct ========"
sacct -j "${JOB_ID}" --format=JobID,NodeList,State,Elapsed,ExitCode,End 2>/dev/null || true

OUT="${REPO_ROOT}/logs/slurm-${JOB_ID}.out"
ERR="${REPO_ROOT}/logs/slurm-${JOB_ID}.err"

echo ""
echo "======== tail ${OUT} ========"
if [[ -f "${OUT}" ]]; then
  tail -120 "${OUT}"
else
  echo "(missing: ${OUT})"
fi

echo ""
echo "======== tail ${ERR} ========"
if [[ -f "${ERR}" ]]; then
  tail -120 "${ERR}"
else
  echo "(missing: ${ERR})"
fi

echo ""
echo "Done. Exit code line: look for ExitCode 0:0 on the main JobID row in sacct above."
