#!/usr/bin/env bash
set -euo pipefail

JOBID="${1:-}"

if [[ -z "${JOBID}" ]]; then
  echo "Usage: scripts/archive/savio_tail_job.sh <JOBID>" >&2
  exit 1
fi

echo "== sacct summary for ${JOBID} =="
sacct -j "${JOBID}" --format=JobID,JobName,Partition,State,ExitCode,DerivedExitCode,Elapsed,Start,End,NodeList,Reason || true
echo

LOG_BASE="logs/slurm-${JOBID}"
OUT_FILE="${LOG_BASE}.out"
ERR_FILE="${LOG_BASE}.err"

echo "== Log files =="
ls -la "${OUT_FILE}" "${ERR_FILE}" 2>/dev/null || echo "No slurm-${JOBID}.* logs found under logs/"
echo

if [[ -f "${OUT_FILE}" ]]; then
  echo "== ${OUT_FILE} (head) =="
  sed -n '1,80p' "${OUT_FILE}" || true
  echo
  echo "== ${OUT_FILE} (tail) =="
  tail -n 80 "${OUT_FILE}" || true
  echo
fi

if [[ -f "${ERR_FILE}" ]]; then
  echo "== ${ERR_FILE} (head) =="
  sed -n '1,80p' "${ERR_FILE}" || true
  echo
  echo "== ${ERR_FILE} (tail) =="
  tail -n 80 "${ERR_FILE}" || true
  echo
fi
