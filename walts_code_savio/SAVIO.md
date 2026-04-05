# Training on Savio (short guide)

[Savio](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/) is Berkeley’s cluster. This folder has the Python files for `train.py` plus scripts under `scripts/`.

**What you actually need:** the **Slurm file** `scripts/savio_train.slurm`. You submit it with `sbatch` (after setting your data path and editing `#SBATCH` headers). Everything else in `scripts/` is **optional**—nice shortcuts, not required.

| Main | Optional (skip if you do not need them) |
|------|----------------------------------------|
| `scripts/savio_train.slurm` — GPU training job | `scripts/submit_and_tail.sh` — same job, but waits and prints log tails for you |
| | `scripts/setup_smoke_data_symlinks.sh` — only if your Parquet folder mixes files you must not train on |

---

## Quick start

```bash
cd /path/to/this/folder          # must contain train.py
mkdir -p logs
export BRIDGE_RNA_DATA_DIR=/path/to/your/parquet/folder

# Quick test (small data — default). --export=ALL sends your exports into the job.
sbatch --export=ALL scripts/savio_train.slurm

# Full training: turn off smoke and request a long wall time
export BRIDGE_RNA_SMOKE=0
sbatch --time=24:00:00 --export=ALL scripts/savio_train.slurm
```

Logs: `tail -f logs/slurm-<JOBID>.out` (job id prints when you run `sbatch`).

---

## Before you run (do once)

**1.** SSH to Savio (VPN if your lab says so).

**2.** Clone or copy this folder under **scratch**, e.g. `/global/scratch/users/YOUR_NETID/...`.

**3.** Python env:

```bash
module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda env create -f environment.yml    # first time only
conda activate bridge-rna
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Use the CUDA build that matches Savio — see [pytorch.org](https://pytorch.org/get-started/locally/) if `cu124` is wrong.

**4.** Open `scripts/savio_train.slurm` and set `#SBATCH --account=`, `--partition=`, `--qos=` for your group (`sshare -U` lists accounts).

**5.** `BRIDGE_RNA_DATA_DIR` must point at preprocessed Parquet batches. **Optional:** if that directory also has Parquet files you must **not** load, use `scripts/setup_smoke_data_symlinks.sh` (see the top of that file), then set `BRIDGE_RNA_DATA_DIR` to the clean symlink folder. If your folder only has the right Parquets, skip this.

**6.** Optional: `wandb login` for experiment tracking.

---

## One Slurm file, three behaviors

All are `scripts/savio_train.slurm` (edit `#SBATCH` once).

| What you want | What to do |
|----------------|------------|
| **Smoke test** | `export BRIDGE_RNA_DATA_DIR=...` then `sbatch --export=ALL scripts/savio_train.slurm` (smoke is the default). |
| **Full training** | `export BRIDGE_RNA_DATA_DIR=...` and `BRIDGE_RNA_SMOKE=0`, then `sbatch --time=24:00:00 --export=ALL scripts/savio_train.slurm` so the job does not hit the short default time limit. |
| **W&B sweep** | `export BRIDGE_RNA_DATA_DIR=...` and `WANDB_SWEEP_ID=entity/project/sweep_xxx`, then `sbatch --time=04:00:00 --export=ALL scripts/savio_train.slurm`. |

**Optional:** instead of `sbatch` by hand, you can run `bash scripts/submit_and_tail.sh` after exporting `BRIDGE_RNA_DATA_DIR` — it submits `savio_train.slurm` and tails the logs when the job ends. Same Slurm file; the shell script is only a helper.

---

## Check that it worked

```bash
squeue -u YOUR_NETID
sacct -j JOBID --format=JobID,State,ExitCode
```

`ExitCode 0:0` means success.

---

## If something breaks

- **Import error for `slim_performer_model`** — run from the folder with `train.py`; for interactive runs use `export PYTHONPATH="$(pwd):$PYTHONPATH"`.
- **`sbatch` account/partition errors** — fix `#SBATCH` lines in `savio_train.slurm`.
- **CUDA errors** — reinstall PyTorch for Savio’s CUDA.
- **Permission denied on a shared folder** — the owner must grant your account read access.

---

## Files here

**Python + env**

- `train.py`, `slim_performer_model.py`, `numerator_and_denominator.py` — model + training.
- `environment.yml`, `requirements.txt` — deps (PyTorch installed separately as above).

**Scripts (main vs optional)**

- **`scripts/savio_train.slurm` (main)** — submit with `sbatch`; this is what runs training on the GPU.
- **`scripts/submit_and_tail.sh` (optional)** — convenience wrapper around the same Slurm file.
- **`scripts/setup_smoke_data_symlinks.sh` (optional)** — only for messy Parquet folders; see the table at the top.
