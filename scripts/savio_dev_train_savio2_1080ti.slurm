#!/bin/bash
# Longer GPU run (not smoke): more epochs + default-sized subsets so W&B charts are useful while away.
# Same partition/account as savio_smoke_train_savio2_1080ti.slurm.
#
# Submit from repo root:
#   sbatch scripts/savio_dev_train_savio2_1080ti.slurm
#
# Tune via exports before sbatch, e.g.:
#   export BRIDGE_RNA_EPOCHS=3
#   sbatch --export=ALL scripts/savio_dev_train_savio2_1080ti.slurm
#
# Full data (all samples in BRIDGE_RNA_DATA_DIR, 80/20 train/val) is the default (USE_ALL_SAMPLES=1).
# Fixed subsets instead: export BRIDGE_RNA_USE_ALL_SAMPLES=0 BRIDGE_RNA_TRAIN_SUBSET=2000 BRIDGE_RNA_VAL_SUBSET=400
#
# Loss ablation (MSE vs Huber / SmoothL1), compare val_mse in W&B:
#   export BRIDGE_RNA_LOSS=smooth_l1
#   export WANDB_RUN_NAME=overnight-hub
#   sbatch --export=ALL scripts/savio_dev_train_savio2_1080ti.slurm
#
#SBATCH --job-name=bridge-rna-dev-1080ti
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0228.savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Full-dataset epochs are much slower than 2k-subset runs; increase if jobs hit TIMEOUT.
#SBATCH --time=24:00:00
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

export BRIDGE_RNA_DATA_DIR="${BRIDGE_RNA_DATA_DIR:-/global/scratch/users/abrahamguan/bridge-rna-smoke-data}"
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  exit 1
fi

# Not smoke: uses full CONFIG defaults unless overridden below.
export BRIDGE_RNA_SMOKE=0
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-5}"
# 1 = use all samples (80/20 train/val). 0 = use BRIDGE_RNA_TRAIN_SUBSET / VAL (default 2000/400 from train.py CONFIG if unset).
export BRIDGE_RNA_USE_ALL_SAMPLES="${BRIDGE_RNA_USE_ALL_SAMPLES:-1}"
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-4}"
export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
export BRIDGE_RNA_HUBER_BETA="${BRIDGE_RNA_HUBER_BETA:-1.0}"

export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_dev_${SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-dev-1080ti-${SLURM_JOB_ID}}"

echo "BRIDGE_RNA_SMOKE=${BRIDGE_RNA_SMOKE} EPOCHS=${BRIDGE_RNA_EPOCHS} USE_ALL_SAMPLES=${BRIDGE_RNA_USE_ALL_SAMPLES} TRAIN_SUBSET=${BRIDGE_RNA_TRAIN_SUBSET:-} VAL_SUBSET=${BRIDGE_RNA_VAL_SUBSET:-} batch=${BRIDGE_RNA_BATCH_SIZE}"
echo "BRIDGE_RNA_LOSS=${BRIDGE_RNA_LOSS} BRIDGE_RNA_HUBER_BETA=${BRIDGE_RNA_HUBER_BETA}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

torchrun --standalone --nproc_per_node=1 train.py

echo "Done dev job ${SLURM_JOB_ID}"
