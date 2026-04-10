# W&B sweeps on Savio (mentor-style ranges)

This repo includes:

- **`sweeps/walt_sweep.yaml`** — grid **`learning_rate`** ∈ `{1.8e-4, 2e-4, 2.2e-4}` × **`num_layers`** ∈ `{2, 4}`, **`loss` locked to `smooth_l1`** (6 trials). Other knobs fixed (see YAML).
- **`sweeps/walt_mse_anchor.yaml`** — **one** trial: **`mse`** at **`lr=2e-4`**, **`num_layers=2`**, to keep a single MSE baseline for **`val_mse`** comparison without a full MSE grid.

Both use **`torchrun --nproc_per_node=1`** (single GPU). **`val_mse`** is the sweep metric (logged every epoch in `train.py`).

## Prerequisites

- `wandb login` on the machine where you create the sweep (laptop or Savio login node).
- Parquet training data: set `BRIDGE_RNA_DATA_DIR` to a directory of preprocessed `*.parquet` batches (see `docs/savio_smoke_training.md`).
- Large models (`hidden_dim` × `ffn_dim` × `batch_size`) can **OOM** on 11 GB GPUs — if needed, cap ranges in the YAML or set a smaller max `batch_size`.

## 1. Create the sweep

From the repo root (anywhere with the CLI and network):

```bash
wandb sweep --project bridge-rna sweeps/walt_sweep.yaml
```

Copy the **`wandb agent ...`** line. The sweep ID looks like `your-entity/bridge-rna/sweep_abc123`.

**Important:** The command block (e.g. `torchrun --nproc_per_node=...`) is **fixed when you create the sweep**. If you edit `walt_sweep.yaml` later, run **`wandb sweep ...` again** and use the **new** sweep ID — old sweeps on W&B still use the previous command.

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

Submit **several** jobs with the **same** `WANDB_SWEEP_ID` to run trials in parallel (up to your fair-share / queue limits). **`walt_sweep.yaml`** has **6** trials (`3 × 2`); **`walt_mse_anchor.yaml`** has **1** trial (create a separate sweep and run one agent).

## 3. Notes

- Non-sweep runs can still set architecture via env, e.g. `BRIDGE_RNA_HIDDEN_DIM`, `BRIDGE_RNA_FFN_DIM`, `BRIDGE_RNA_NUM_LAYERS`, `BRIDGE_RNA_LEARNING_RATE`, etc. (see `_apply_runtime_env_config` in `train.py`).
- To use **random** or **Bayesian** search instead, change `method` in the YAML and re-create the sweep (see [W&B sweep docs](https://docs.wandb.ai/guides/sweeps)).
