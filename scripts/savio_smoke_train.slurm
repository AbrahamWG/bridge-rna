#!/bin/bash
#SBATCH --job-name=bridge-rna-smoke
#SBATCH --account=PUT_YOUR_SAVIO_ACCOUNT_HERE
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# -----------------------------------------------------------------------------
# Savio smoke test for train.py (single GPU, tiny data subset via env).
#
# BEFORE YOUR FIRST RUN (order matters):
#   1) Dependencies: Python 3.12+ (not /usr/bin/python3). Use conda env from
#      environment.yml or a venv, then install CUDA PyTorch for Savio.
#   2) Weights & Biases: on the login node run `wandb login` once, OR set
#      WANDB_API_KEY (see https://wandb.ai/authorize). Use a dedicated
#      WANDB_PROJECT for your runs so you do not mix with your mentor's.
#   3) Data: set BRIDGE_RNA_DATA_DIR to the directory that contains
#      batch_files/*.parquet (or Parquet files directly). If Brian's folder mixes
#      input_chunk_*.parquet with processed_mouse_*.parquet, run first:
#        bash scripts/setup_smoke_data_symlinks.sh
#      then export BRIDGE_RNA_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-smoke-data
#   4) Optional: remove stale local caches — `rm -rf ./wandb` in the repo
#      (already gitignored) so this job writes a fresh tree under WANDB_DIR.
#
# Submit from the repo root (or set SLURM_SUBMIT_DIR via cd below):
#   mkdir -p logs
#   export BRIDGE_RNA_DATA_DIR=/path/to/your/prepared/archs4_data
#   export WANDB_API_KEY=...   # if not using ~/.netrc from wandb login
#   export WANDB_PROJECT=bridge-rna-smoke-yourname
#   sbatch scripts/savio_smoke_train.slurm
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

# Repo root on Python path (train.py imports slim_performer_model)
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Fresh local W&B cache per job (does not delete cloud runs; use WANDB_PROJECT to separate)
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_job_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

# --- Activate your environment (uncomment and edit ONE) ---
# source /path/to/venv/bin/activate
# conda activate bridge-rna

# --- Savio: conda is provided as anaconda3 (not "conda") ---
#   module load anaconda3/2024.10-1-11.4
#   eval "$(conda shell.bash hook)"
#   conda activate bridge-rna

# Smoke test: tiny subsets + 1 epoch (see train.py _apply_runtime_env_config)
export BRIDGE_RNA_SMOKE=1
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-32}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-8}"
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-1}"
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-2}"
export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-128}"

# Point to data that exists on Savio (must contain batch Parquet layout expected by train.py)
if [[ -z "${BRIDGE_RNA_DATA_DIR:-}" ]]; then
  echo "ERROR: Set BRIDGE_RNA_DATA_DIR to your prepared data directory before sbatch." >&2
  echo "  export BRIDGE_RNA_DATA_DIR=/scratch/.../train_orthologs" >&2
  exit 1
fi

# Checkpoints isolated per job
export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_smoke_${SLURM_JOB_ID}"

# W&B: use your project/entity so runs are not confused with your mentor's
export WANDB_PROJECT="${WANDB_PROJECT:-expression-performer}"
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
