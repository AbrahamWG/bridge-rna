#!/bin/bash
#SBATCH --job-name=bridge-rna-smoke
# Charge account from `sshare -U` — not your username. ic_cdss170 matches Brian’s shared scratch (ic_cdss170).
#SBATCH --account=ic_cdss170
# Savio retired savio2_gpu; use savio3_gpu or savio4_gpu (see docs-research-it.berkeley.edu Savio "GPU Jobs").
# Low-priority queue often works without a faculty GPU condo QoS:
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
# savio3_gpu requires a GPU type, e.g. GTX2080TI (see: sinfo -p savio3_gpu; scheduler-examples GPU script)
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Memory on GPU partitions scales with cores; omit --mem unless you need more (request extra cores).
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# -----------------------------------------------------------------------------
# Savio smoke test for train.py (single GPU, tiny data subset via env).
#
# On Savio:
#   mkdir -p logs
#   sbatch scripts/savio_smoke_train.slurm
#
# Data default: Brian processed_mouse symlinks — run first if needed:
#   bash scripts/setup_smoke_data_symlinks.sh
#
# Override defaults when submitting:
#   BRIDGE_RNA_DATA_DIR=/other/path sbatch scripts/savio_smoke_train.slurm
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

# Repo root on Python path (train.py imports slim_performer_model)
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Fresh local W&B cache per job (does not delete cloud runs; use WANDB_PROJECT to separate)
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_job_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

# --- Savio: Anaconda (module name is anaconda3, not conda) ---
module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

# Default smoke data: symlink dir with only processed_mouse_*.parquet (see setup_smoke_data_symlinks.sh)
export BRIDGE_RNA_DATA_DIR="${BRIDGE_RNA_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  echo "  Run: bash scripts/setup_smoke_data_symlinks.sh" >&2
  exit 1
fi

# Smoke test: tiny subsets + 1 epoch (see train.py _apply_runtime_env_config)
export BRIDGE_RNA_SMOKE=1
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-32}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-8}"
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-1}"
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-2}"
export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-128}"

# Checkpoints isolated per job
export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_smoke_${SLURM_JOB_ID}"

# W&B: default project keeps your runs separate from ad-hoc local runs
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-smoke-${SLURM_JOB_ID}}"
# export WANDB_ENTITY=your-wandb-username-or-team

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ ! -f "${HOME}/.netrc" ]]; then
  echo "WARN: WANDB_API_KEY is unset and ~/.netrc not found. Run 'wandb login' on the login node or set WANDB_API_KEY." >&2
fi

echo "PYTHONPATH=${PYTHONPATH}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

# Single GPU, one process (DDP world size 1)
torchrun --standalone --nproc_per_node=1 train.py

echo "Done smoke job ${SLURM_JOB_ID}"
