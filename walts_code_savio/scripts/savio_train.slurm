#!/bin/bash
# One script for GPU training on Savio: smoke test, full run, or W&B sweep.
#
# Edit #SBATCH lines below for your account / partition / QoS (run: sshare -U).
#
# --- Data (required on the login node before sbatch) ---
#   export BRIDGE_RNA_DATA_DIR=/path/to/parquet/folder
#
# --- Normal training (torchrun + train.py) ---
#   Smoke test (tiny data, ~minutes): default — just set BRIDGE_RNA_DATA_DIR
#   Full training:  export BRIDGE_RNA_SMOKE=0
#                   sbatch --time=24:00:00 --export=ALL scripts/savio_train.slurm
#   (Default wall time below is short on purpose; override with sbatch --time=... for long runs.)
#
# --- W&B sweep (optional) ---
#   export WANDB_SWEEP_ID=entity/project/sweep_xxxxxx
#   sbatch --time=04:00:00 --export=ALL scripts/savio_train.slurm
#
#SBATCH --job-name=brna-train
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0228.savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Default: enough for smoke; use: sbatch --time=24:00:00 ... when BRIDGE_RNA_SMOKE=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_job_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

if [[ -z "${BRIDGE_RNA_DATA_DIR:-}" ]]; then
  echo "ERROR: Set data directory before sbatch, e.g.:" >&2
  echo "  export BRIDGE_RNA_DATA_DIR=/global/scratch/users/YOU/your-parquet-dir" >&2
  exit 1
fi
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Mode A: W&B sweep agent (runs one trial per job)
# ---------------------------------------------------------------------------
if [[ -n "${WANDB_SWEEP_ID:-}" ]]; then
  export BRIDGE_RNA_SMOKE=0
  export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-5}"
  export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-4000}"
  export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-800}"
  export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
  export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_sweep_${SLURM_JOB_ID}"
  export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna}"
  echo "MODE=sweep WANDB_SWEEP_ID=${WANDB_SWEEP_ID}"
  echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
  wandb agent --count 1 "${WANDB_SWEEP_ID}"
  echo "Done sweep job ${SLURM_JOB_ID}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Mode B: torchrun training
# ---------------------------------------------------------------------------
# BRIDGE_RNA_SMOKE=1 (default) = quick sanity check. Set to 0 for real training.
export BRIDGE_RNA_SMOKE="${BRIDGE_RNA_SMOKE:-1}"

if [[ "$BRIDGE_RNA_SMOKE" == "1" ]]; then
  export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-32}"
  export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-8}"
  export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-1}"
  export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-2}"
  export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-128}"
  export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_smoke_${SLURM_JOB_ID}"
  export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-smoke-${SLURM_JOB_ID}}"
  if [[ -z "${WANDB_API_KEY:-}" ]] && [[ ! -f "${HOME}/.netrc" ]]; then
    echo "WARN: wandb not logged in (optional). Run: wandb login" >&2
  fi
else
  export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-5}"
  export BRIDGE_RNA_USE_ALL_SAMPLES="${BRIDGE_RNA_USE_ALL_SAMPLES:-1}"
  export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-4}"
  export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
  export BRIDGE_RNA_HUBER_BETA="${BRIDGE_RNA_HUBER_BETA:-1.0}"
  export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_train_${SLURM_JOB_ID}"
  export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-train-${SLURM_JOB_ID}}"
fi

echo "MODE=torchrun BRIDGE_RNA_SMOKE=${BRIDGE_RNA_SMOKE}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

torchrun --standalone --nproc_per_node=1 train.py

echo "Done job ${SLURM_JOB_ID}"
