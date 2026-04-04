# Running smoke training on Berkeley Savio

This document describes how to run a **short GPU smoke test** of `train.py` on [Savio](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/). Adjust **account**, **partition**, and **paths** to match your project and allocation.

## What the smoke job does

- Uses `BRIDGE_RNA_SMOKE=1`: small train/val subsets (default 32 / 8), 1 epoch, streaming Parquet reads, smaller model dim.
- Writes Slurm logs under `logs/slurm-<JOBID>.out` and `.err`.
- Optionally logs to [Weights & Biases](https://wandb.ai) if `WANDB_API_KEY` or `wandb login` is set.

## One-time setup

1. **SSH to Savio** (Campus VPN if required).

2. **Conda environment** (login node example; use your team’s Python/PyTorch/CUDA choices):

   ```bash
   module load anaconda3/2024.10-1-11.4
   eval "$(conda shell.bash hook)"
   conda activate bridge-rna   # or create env from repo environment.yml / requirements.txt
   ```

   Use **`module load anaconda3/...`** inside batch scripts too — the login node’s system Python is too old for this project.

3. **Clone the repo** somewhere on scratch, e.g. `/global/scratch/users/<USER>/bridge-rna`.

4. **Slurm account and partition** — your **account** is not your username. Check with:

   ```bash
   sshare -U
   ```

   Edit `#SBATCH --account=` and partition/QoS in the Slurm script you use if your group differs.

5. **Weights & Biases** (optional):

   ```bash
   wandb login
   ```

## Data layout

- Training reads **preprocessed** sample-major Parquet batches (e.g. `processed_mouse_*.parquet`), not raw `input_chunk_*` files, unless you are re-running preprocessing.
- Set **`BRIDGE_RNA_DATA_DIR`** to a directory that contains **only** those batch Parquets (or a `batch_files/` subdirectory). The repo includes `scripts/setup_smoke_data_symlinks.sh` to create a directory of symlinks so mixed folders do not confuse the loader.

Example:

```bash
bash scripts/setup_smoke_data_symlinks.sh
export BRIDGE_RNA_DATA_DIR=/global/scratch/users/<USER>/bridge-rna-smoke-data
```

## Submit the smoke job (savio2 1080 Ti, CDSS-friendly)

For accounts that work with **`ic_cdss170`** + **`savio2_1080ti`** + **`savio_normal`**, use:

```bash
cd /global/scratch/users/<USER>/bridge-rna
mkdir -p logs
sbatch scripts/savio_smoke_train_savio2_1080ti.slurm
```

The script sets `BRIDGE_RNA_DATA_DIR`, `BRIDGE_RNA_SMOKE`, checkpoint dir, and `torchrun` for a single GPU.

**Alternative:** `scripts/savio_smoke_train.slurm` targets **savio3** GPUs (`savio3_gpu` / `savio_lowprio`). Use whichever matches your allocation and passes `sbatch` validation.

## Monitor jobs and logs

```bash
squeue -u <YOUR_USERNAME>
sacct -j <JOBID> --format=JobID,NodeList,State,Elapsed,ExitCode
tail -f logs/slurm-<JOBID>.out
tail -80 logs/slurm-<JOBID>.err
```

Success: **`State=COMPLETED`**, **`ExitCode=0:0`**, and training/validation loss lines in `.out`.

Cancel: `scancel <JOBID>`.

## GPU nodes: hangs and excludes

If **`nvidia-smi`** or **`nvidia-smi -L`** **hangs** on a compute node, or PyTorch reports **`CUDA unknown error`** / **`no GPUs found`**, that **node’s GPU/driver may be unhealthy**.

1. Check node state:

   ```bash
   sinfo -N -p savio2_1080ti -o "%N %t"
   ```

2. For **interactive** debugging, request a **specific idle** node:

   ```bash
   srun -A <ACCOUNT> -p savio2_1080ti -q savio_normal --gres=gpu:1 -t 0:30:00 -w n0229.savio2 --pty bash -l
   ```

   Use **`hostname`**, **`nvidia-smi -L`**, then conda + `python -c "import torch; print(torch.cuda.is_available())"`.

3. For **batch** jobs, add an exclude for bad nodes (full hostname as shown by `sinfo`):

   ```bash
   #SBATCH --exclude=n0228.savio2
   ```

   `scripts/savio_smoke_train_savio2_1080ti.slurm` may already include excludes; **remove or update** them after admins repair a node.

Slurm **node names** must be fully qualified (e.g. `n0228.savio2`), not short names, in `#SBATCH --exclude=` and `srun -w`.

## Environment variables (reference)

| Variable | Role |
|----------|------|
| `BRIDGE_RNA_DATA_DIR` | Directory with Parquet batches (set in Slurm or export before submit). |
| `BRIDGE_RNA_SMOKE` | `1` enables tiny subsets / short run. |
| `BRIDGE_RNA_CHECKPOINT_DIR` | Where checkpoints are written (script sets per-job dir). |
| `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_DIR` | Weights & Biases (optional). |

See `train.py` (`_apply_runtime_env_config`) for overrides such as `BRIDGE_RNA_TRAIN_SUBSET`, `BRIDGE_RNA_EPOCHS`, etc.

## Official Savio docs

- [Submitting jobs](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/)
- [Using Python on Savio](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/software/using-software/using-python-savio/)
