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
# Defaults: 3× GPU DDP, 10k train / 2k val (same 5:1 ratio as 20k/4k). Minggang comparison runs often used 20k/4k — override TRAIN/VAL_SUBSET.
# BRIDGE_RNA_BATCH_SIZE is per GPU (per DDP process). Start with 4 per GPU; if CUDA OOM, use 2 per GPU — do not raise to 8/16 per GPU on 1080 Ti.
# Hyperparams: batch 4 per GPU, lr 2e-4, hidden 768, ffn 3072 (effective samples/step ≈ batch_size × 3 GPUs).
# This script runs train.py + BRIDGE_RNA_DATA_DIR (batch Parquets). For identical code path as Minggang, use
# train_single.py + merged expression.parquet (see merge.py); separate Slurm optional.
# Per-GPU memory: batch 4 + 768-wide fits ~11 GiB; OOM → export BRIDGE_RNA_BATCH_SIZE=2 before sbatch, or narrow hidden/ffn.
# Full data: BRIDGE_RNA_USE_ALL_SAMPLES=1.
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
# 3× GPU DDP (nproc_per_node=3). savio2_1080ti: 8 CPUs and 64 GB RAM per node — cpus-per-task ≤ 8.
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
# 0 = fixed train/val counts (for comparable logs). 1 = all samples, 80/20 split.
export BRIDGE_RNA_USE_ALL_SAMPLES="${BRIDGE_RNA_USE_ALL_SAMPLES:-0}"
# Val 2000 keeps ~20% of train (matches old 4k/20k proportion); use 1000 for faster val epochs.
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-10000}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-2000}"
# Per-GPU batch (default 4; on OOM retry with 2 — not 8/16).
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-4}"
export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
export BRIDGE_RNA_HUBER_BETA="${BRIDGE_RNA_HUBER_BETA:-1.0}"
export BRIDGE_RNA_LEARNING_RATE="${BRIDGE_RNA_LEARNING_RATE:-2e-4}"
# Architecture match: 768 / 3072 / 2 layers / 8 heads (train.py defaults overridden).
export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-768}"
export BRIDGE_RNA_FFN_DIM="${BRIDGE_RNA_FFN_DIM:-3072}"
export BRIDGE_RNA_NUM_LAYERS="${BRIDGE_RNA_NUM_LAYERS:-2}"
export BRIDGE_RNA_NUM_HEADS="${BRIDGE_RNA_NUM_HEADS:-8}"

export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_dev_${SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna-smoke}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-dev-1080ti-${SLURM_JOB_ID}}"

echo "BRIDGE_RNA_SMOKE=${BRIDGE_RNA_SMOKE} EPOCHS=${BRIDGE_RNA_EPOCHS} USE_ALL_SAMPLES=${BRIDGE_RNA_USE_ALL_SAMPLES} TRAIN_SUBSET=${BRIDGE_RNA_TRAIN_SUBSET} VAL_SUBSET=${BRIDGE_RNA_VAL_SUBSET} batch=${BRIDGE_RNA_BATCH_SIZE}"
echo "BRIDGE_RNA_HIDDEN_DIM=${BRIDGE_RNA_HIDDEN_DIM} BRIDGE_RNA_FFN_DIM=${BRIDGE_RNA_FFN_DIM} layers=${BRIDGE_RNA_NUM_LAYERS} heads=${BRIDGE_RNA_NUM_HEADS}"
echo "BRIDGE_RNA_LOSS=${BRIDGE_RNA_LOSS} BRIDGE_RNA_HUBER_BETA=${BRIDGE_RNA_HUBER_BETA}"
echo "BRIDGE_RNA_LEARNING_RATE=${BRIDGE_RNA_LEARNING_RATE}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

# Reduce fragmentation OOMs on tight GPUs (PyTorch 2.x+).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "torchrun: 3 processes (3 GPUs); ~ceil(train_subset/batch/world_size) steps per rank per epoch"
torchrun --standalone --nproc_per_node=3 train.py

echo "Done dev job ${SLURM_JOB_ID}"
