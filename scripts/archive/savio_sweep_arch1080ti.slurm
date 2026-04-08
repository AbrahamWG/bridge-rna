#!/bin/bash
# ARCHIVED — legacy W&B agent for architecture grid sweeps (sweeps/archive/arch1080ti_grid.yaml).
# Current workflow: sweeps/walt_sweep.yaml + scripts/savio_wandb_sweep_agent.slurm
# One Slurm job = one sweep trial. Submit many jobs with the same WANDB_SWEEP_ID to drain the queue.
# Uses 3× GPU DDP (torchrun in sweep YAML).
#
#   export WANDB_SWEEP_ID="entity/bridge-rna/sweep_xxxxxxxx"
#   export BRIDGE_RNA_DATA_DIR=/global/scratch/users/abrahamguan/bridge-rna-smoke-data
#   sbatch --export=ALL scripts/archive/savio_sweep_arch1080ti.slurm
#
# Do not set BRIDGE_RNA_DATA_DIR to a folder that mixes processed_*.parquet with input_chunk_*.parquet.
# Batch size comes from the sweep YAML (default 4 per GPU); on OOM edit arch1080ti_grid.yaml to 2 and recreate the sweep.
#
#SBATCH --job-name=brna-arch-sw
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-sweep-arch-%j.out
#SBATCH --error=logs/slurm-sweep-arch-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [[ -z "${WANDB_SWEEP_ID:-}" ]]; then
  echo "ERROR: Set WANDB_SWEEP_ID to entity/project/sweep_xxx from: wandb sweep sweeps/archive/arch1080ti_grid.yaml" >&2
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
export BRIDGE_RNA_USE_ALL_SAMPLES="${BRIDGE_RNA_USE_ALL_SAMPLES:-0}"
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-5}"
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-10000}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-2000}"
export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_sweep_${SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "WANDB_SWEEP_ID=${WANDB_SWEEP_ID}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "epochs=${BRIDGE_RNA_EPOCHS} train=${BRIDGE_RNA_TRAIN_SUBSET} val=${BRIDGE_RNA_VAL_SUBSET}"

wandb agent --count 1 "${WANDB_SWEEP_ID}"

echo "Done sweep agent job ${SLURM_JOB_ID}"
