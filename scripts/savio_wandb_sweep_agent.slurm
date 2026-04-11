#!/bin/bash
# Run ONE trial from the W&B sweep (use with: wandb sweep sweeps/walt_sweep.yaml).
# Submit multiple jobs with the same WANDB_SWEEP_ID to drain the sweep queue.
#
# Required before sbatch:
#   export WANDB_SWEEP_ID="your-entity/bridge-rna/sweep_xxxxxxxx"
# Optional:
#   export BRIDGE_RNA_DATA_DIR=/path/to/parquet/dir
#   export BRIDGE_RNA_TRAIN_SUBSET=8000 BRIDGE_RNA_VAL_SUBSET=800 BRIDGE_RNA_EPOCHS=5
#   export WANDB_PROJECT=bridge-rna
#
# Sweep YAML (pick one; each needs its own `wandb sweep` → WANDB_SWEEP_ID):
#   walt_sweep.yaml — grid loss × LR × batch_size → 12 trials (see YAML)
#   walt_mse_anchor.yaml — 1 trial: optional MSE-only quick baseline (overlaps loss=mse in walt_sweep)
#   torchrun --nproc_per_node=1 (see YAML command:)
#
#SBATCH --job-name=brna-sweep
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:1
# Temporary mitigation: avoid flaky GPU nodes (CUDA init / visibility); adjust as admins confirm.
# n0227: sweep jobs saw CUDA unknown error → 0 GPUs + NCCL fail (2026-04).
#SBATCH --exclude=n0227.savio2,n0228.savio2,n0302.savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#STRICT: use --mem=60G on savio2_1080ti (64G is not allocatable on most nodes).
#SBATCH --mem=60G
# Increase walltime: recent sweep agents timed out at 4h before trial completion.
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-sweep-%j.out
#SBATCH --error=logs/slurm-sweep-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [[ -z "${WANDB_SWEEP_ID:-}" ]]; then
  echo "ERROR: Set WANDB_SWEEP_ID to entity/project/sweep_xxx from 'wandb sweep ...' output." >&2
  exit 1
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_job_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

export BRIDGE_RNA_DATA_DIR="${BRIDGE_RNA_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  exit 1
fi

export BRIDGE_RNA_SMOKE=0
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-3}"
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-10000}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-2000}"
export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_sweep_${SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna}"

echo "WANDB_SWEEP_ID=${WANDB_SWEEP_ID}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "slurm_timelimit=12:00:00"
echo "epochs=${BRIDGE_RNA_EPOCHS} train=${BRIDGE_RNA_TRAIN_SUBSET} val=${BRIDGE_RNA_VAL_SUBSET}"

# One sweep trial per job (agent pulls next hyperparams from W&B)
wandb agent --count 1 "${WANDB_SWEEP_ID}"

echo "Done sweep agent job ${SLURM_JOB_ID}"
