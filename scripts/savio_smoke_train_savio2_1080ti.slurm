#!/bin/bash
#SBATCH --job-name=bridge-rna-smoke-1080ti
# Savio GPU smoke test: savio2_1080ti + savio_normal (known-good for ic_cdss170 / CDSS).
# Ref: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:1
# Exclude nodes with GPU/CUDA flake (2026-04); remove if fixed by admins.
#SBATCH --exclude=n0227.savio2,n0228.savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# -----------------------------------------------------------------------------
# Submit: sbatch scripts/savio_smoke_train_savio2_1080ti.slurm
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_job_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

export BRIDGE_RNA_DATA_DIR="${BRIDGE_RNA_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  echo "  Run: bash scripts/setup_smoke_data_symlinks.sh" >&2
  exit 1
fi

export BRIDGE_RNA_SMOKE=1
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-32}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-8}"
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-1}"
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-2}"
export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-128}"

export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_smoke_${SLURM_JOB_ID}"

export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-smoke-1080ti-${SLURM_JOB_ID}}"

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ ! -f "${HOME}/.netrc" ]]; then
  echo "WARN: WANDB_API_KEY is unset and ~/.netrc not found. Run 'wandb login' on the login node or set WANDB_API_KEY." >&2
fi

echo "PYTHONPATH=${PYTHONPATH}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

torchrun --standalone --nproc_per_node=1 train.py

echo "Done smoke job ${SLURM_JOB_ID}"
