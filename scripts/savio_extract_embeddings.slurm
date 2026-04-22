#!/bin/bash
#SBATCH --job-name=extract-embeddings
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0227.savio2,n0228.savio2,n0302.savio2,n0298.savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Submit: sbatch scripts/savio_extract_embeddings.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

echo "Running on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python extract_embeddings.py \
  --checkpoint checkpoints_sweep_33524015/best_model.pt \
  --osdr_all   /global/scratch/users/kmeng/osdr_data/osdr_processed_all.csv \
  --osdr_space /global/scratch/users/kmeng/osdr_data/osdr_processed_spaceflight.csv \
  --vocab_file /global/scratch/users/abrahamguan/joint_vocab.csv \
  --output embeddings.npz \
  --device cuda

echo "Done. embeddings.npz saved."
