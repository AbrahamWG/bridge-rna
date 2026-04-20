#!/bin/bash
# Longer dev run: joint human+mouse, 20k/4k subsets, 10 epochs, up to 4-GPU DDP.
# Safe with data_mode=preload: all ranks load equal batches upfront → no NCCL desync.
# Same partition/account as savio_smoke_train_savio2_1080ti.slurm.
#
# Submit from repo root:
#   sbatch scripts/savio_dev_train_savio2_1080ti.slurm
#
# Tune via exports before sbatch, e.g.:
#   export BRIDGE_RNA_EPOCHS=5
#   sbatch --export=ALL scripts/savio_dev_train_savio2_1080ti.slurm
#
# Joint human+mouse: DATA_DIR=archs4_joint (106 human + 115 mouse), VOCAB_FILE=joint_vocab.csv.
# Sampling is ~50/50 human:mouse by row count across files.
# Best anchor config: hidden=320, ffn=1536, layers=2, favor+, mask=0.15, lr=2e-4, loss=mse.
# Per-GPU batch (default 8; validated in batch_lr sweep 66o78upl on 1080Ti).
# Full data: BRIDGE_RNA_USE_ALL_SAMPLES=1.
#
#SBATCH --job-name=bridge-rna-dev-1080ti
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
# Flaky nodes excluded (hardware/CUDA failures confirmed 2026-04-18).
#SBATCH --exclude=n0227.savio2,n0228.savio2,n0302.savio2,n0298.savio2
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#STRICT: use --mem=60G, not 64G (see project_state.md).
#SBATCH --mem=60G
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

# Joint human+mouse data (106 human + 115 mouse processed parquets via symlinks).
export BRIDGE_RNA_DATA_DIR="${BRIDGE_RNA_DATA_DIR:-/global/scratch/users/abrahamguan/archs4_joint}"
if [[ ! -d "$BRIDGE_RNA_DATA_DIR" ]]; then
  echo "ERROR: BRIDGE_RNA_DATA_DIR is not a directory: $BRIDGE_RNA_DATA_DIR" >&2
  exit 1
fi
# Joint vocab (15,523 ENSMUSG IDs). Required for joint training — fixes gene column list
# across human (~16,734) and mouse (~16,551) parquets which have different schemas.
export BRIDGE_RNA_VOCAB_FILE="${BRIDGE_RNA_VOCAB_FILE:-/global/scratch/users/abrahamguan/joint_vocab.csv}"

export BRIDGE_RNA_SMOKE=0
export BRIDGE_RNA_EPOCHS="${BRIDGE_RNA_EPOCHS:-10}"
# 0 = fixed train/val counts (for comparable logs). 1 = all samples, 80/20 split.
export BRIDGE_RNA_USE_ALL_SAMPLES="${BRIDGE_RNA_USE_ALL_SAMPLES:-0}"
export BRIDGE_RNA_TRAIN_SUBSET="${BRIDGE_RNA_TRAIN_SUBSET:-20000}"
export BRIDGE_RNA_VAL_SUBSET="${BRIDGE_RNA_VAL_SUBSET:-4000}"
export BRIDGE_RNA_BATCH_SIZE="${BRIDGE_RNA_BATCH_SIZE:-8}"
export BRIDGE_RNA_DATA_MODE="${BRIDGE_RNA_DATA_MODE:-preload}"
export BRIDGE_RNA_NUM_WORKERS="${BRIDGE_RNA_NUM_WORKERS:-4}"
# Best anchor hyperparams from walt sweep.
export BRIDGE_RNA_LOSS="${BRIDGE_RNA_LOSS:-mse}"
export BRIDGE_RNA_HUBER_BETA="${BRIDGE_RNA_HUBER_BETA:-1.0}"
export BRIDGE_RNA_LEARNING_RATE="${BRIDGE_RNA_LEARNING_RATE:-2e-4}"
export BRIDGE_RNA_WEIGHT_DECAY="${BRIDGE_RNA_WEIGHT_DECAY:-0}"
export BRIDGE_RNA_MASK_RATIO="${BRIDGE_RNA_MASK_RATIO:-0.15}"
export BRIDGE_RNA_REE_BASE="${BRIDGE_RNA_REE_BASE:-100}"
export BRIDGE_RNA_FEATURE_TYPE="${BRIDGE_RNA_FEATURE_TYPE:-favor+}"
export BRIDGE_RNA_HIDDEN_DIM="${BRIDGE_RNA_HIDDEN_DIM:-320}"
export BRIDGE_RNA_FFN_DIM="${BRIDGE_RNA_FFN_DIM:-1536}"
export BRIDGE_RNA_NUM_LAYERS="${BRIDGE_RNA_NUM_LAYERS:-2}"
export BRIDGE_RNA_NUM_HEADS="${BRIDGE_RNA_NUM_HEADS:-8}"

export BRIDGE_RNA_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/checkpoints_dev_${SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-bridge-rna}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-dev-joint-1080ti-${SLURM_JOB_ID}}"

echo "BRIDGE_RNA_SMOKE=${BRIDGE_RNA_SMOKE} EPOCHS=${BRIDGE_RNA_EPOCHS} USE_ALL_SAMPLES=${BRIDGE_RNA_USE_ALL_SAMPLES} TRAIN_SUBSET=${BRIDGE_RNA_TRAIN_SUBSET} VAL_SUBSET=${BRIDGE_RNA_VAL_SUBSET} batch=${BRIDGE_RNA_BATCH_SIZE}"
echo "BRIDGE_RNA_HIDDEN_DIM=${BRIDGE_RNA_HIDDEN_DIM} BRIDGE_RNA_FFN_DIM=${BRIDGE_RNA_FFN_DIM} layers=${BRIDGE_RNA_NUM_LAYERS} heads=${BRIDGE_RNA_NUM_HEADS}"
echo "BRIDGE_RNA_FEATURE_TYPE=${BRIDGE_RNA_FEATURE_TYPE} mask=${BRIDGE_RNA_MASK_RATIO} ree_base=${BRIDGE_RNA_REE_BASE} weight_decay=${BRIDGE_RNA_WEIGHT_DECAY}"
echo "BRIDGE_RNA_LOSS=${BRIDGE_RNA_LOSS} lr=${BRIDGE_RNA_LEARNING_RATE}"
echo "BRIDGE_RNA_DATA_DIR=${BRIDGE_RNA_DATA_DIR}"
echo "BRIDGE_RNA_VOCAB_FILE=${BRIDGE_RNA_VOCAB_FILE}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR}"

# Reduce fragmentation OOMs on tight GPUs (PyTorch 2.x+).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Dynamic GPU count — use all visible GPUs up to 4 (preload mode ensures equal batches).
VISIBLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
echo "PyTorch sees ${VISIBLE_GPUS} CUDA device(s) on this node."
if [[ "${VISIBLE_GPUS}" -lt 1 ]]; then
  echo "ERROR: No CUDA devices visible to PyTorch; aborting." >&2
  exit 1
fi
NPROC=$(( VISIBLE_GPUS < 4 ? VISIBLE_GPUS : 4 ))
echo "Launching torchrun with nproc_per_node=${NPROC}"
torchrun --standalone --nproc_per_node="${NPROC}" train.py

echo "Done dev job ${SLURM_JOB_ID}"
