#!/usr/bin/env bash
# Submit smoke training, wait until the job leaves the queue, then print sacct +
# tails of Slurm .out / .err (one place to see success or failure).
#
# Default Slurm file: savio2 1080 Ti (known-good for ic_cdss170). Override for Savio3:
#   SLURM_SCRIPT=scripts/savio_smoke_train.slurm bash scripts/savio_smoke_submit_and_report.sh
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

echo "Waiting for job ${JOB_ID} to finish (typically 1–5+ minutes; not stuck)."
echo "Other terminal: tail -f logs/slurm-${JOB_ID}.out"
while true; do
  if ! squeue -j "${JOB_ID}" -h 2>/dev/null | grep -q .; then
    break
  fi
  line="$(squeue -j "${JOB_ID}" -h -o '%.10i %8T %10M %R' 2>/dev/null | head -1)"
  echo "  $(date '+%H:%M:%S')  ${line:-(running)}"
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
