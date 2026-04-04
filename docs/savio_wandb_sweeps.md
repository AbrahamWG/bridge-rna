# W&B sweeps on Savio (mentor-style ranges)

This repo includes `sweeps/mentor_style_sweep.yaml`: random search over architecture and optimization knobs similar to your mentor’s W&B sweep (batch size, LR, weight decay, mask ratio, `ffn_dim`, `num_layers`, `hidden_dim`, `ree_base`, `feature_type`). **`val_mse`** is the sweep metric (logged every epoch in `train.py`) for fair comparison across runs.

## Prerequisites

- `wandb login` on the machine where you create the sweep (laptop or Savio login node).
- Parquet training data: set `BRIDGE_RNA_DATA_DIR` to a directory of preprocessed `*.parquet` batches (see `docs/savio_smoke_training.md`).
- Large models (`hidden_dim` × `ffn_dim` × `batch_size`) can **OOM** on 11 GB GPUs — if needed, cap ranges in the YAML or set a smaller max `batch_size`.

## 1. Create the sweep

From the repo root (anywhere with the CLI and network):

```bash
wandb sweep --project bridge-rna sweeps/mentor_style_sweep.yaml
```

Copy the **`wandb agent ...`** line. The sweep ID looks like `your-entity/bridge-rna/sweep_abc123`.

## 2. Run agents on Savio (one trial per job)

Each Slurm job runs **`wandb agent --count 1`**, which pulls the **next** hyperparameter set from the sweep queue.

```bash
cd /global/scratch/users/<USER>/bridge-rna
git pull
module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna

export WANDB_SWEEP_ID="your-entity/bridge-rna/sweep_xxxxxxxx"
export BRIDGE_RNA_DATA_DIR="/global/scratch/users/<USER>/your-preprocessed-parquet-dir"
# Optional: more data / epochs than script defaults
export BRIDGE_RNA_TRAIN_SUBSET=8000
export BRIDGE_RNA_VAL_SUBSET=800
export BRIDGE_RNA_EPOCHS=5

mkdir -p logs
sbatch --export=ALL scripts/savio_wandb_sweep_agent.slurm
```

Submit **several** jobs with the **same** `WANDB_SWEEP_ID` to run trials in parallel (up to your fair-share / queue limits).

## 3. Notes

- **`train.py`** no longer forces `ffn_dim = 4 × hidden_dim` after W&B config merge, so sweeps can vary **`ffn_dim` and `hidden_dim` independently** (mentor-style).
- Non-sweep runs can still set architecture via env, e.g. `BRIDGE_RNA_HIDDEN_DIM`, `BRIDGE_RNA_FFN_DIM`, `BRIDGE_RNA_NUM_LAYERS`, `BRIDGE_RNA_LEARNING_RATE`, etc. (see `_apply_runtime_env_config` in `train.py`).
- For **Bayesian** optimization instead of `random`, change `method: bayes` in the YAML and re-create the sweep (see [W&B sweep docs](https://docs.wandb.ai/guides/sweeps)).
