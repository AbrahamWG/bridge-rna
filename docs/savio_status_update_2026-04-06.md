# Savio Status Update (2026-04-06)

## TL;DR
- We identified two distinct failure modes on Savio and resolved both for the current run.
- Current job is running successfully with Minggang-aligned hyperparameters on `savio2_1080ti`.
- We now have a copyable team config in `scripts/savio_dev_train_savio2_1080ti.slurm`.

## What Failed Before (and Why)
- **Failure 1 (multi-GPU):** `invalid device ordinal` on `n0302.savio2` (`local_rank:3`), indicating GPU visibility/device mapping issues on that allocation.
- **Failure 2 (single-GPU):** CUDA OOM when using larger batch on 1080 Ti (~11 GiB).
- These failures were infrastructure/resource related, not a core training-logic bug.

## Current Working Config (Team Baseline)
- Script: `scripts/savio_dev_train_savio2_1080ti.slurm`
- Partition/account: `savio2_1080ti`, `ic_cdss170`, `savio_normal`
- Resources: `gpu:1`, `cpus-per-task=8`, `mem=64G`, `time=24:00:00`
- Data split: job `33210572` used `train_subset=20000`, `val_subset=4000`; **Slurm defaults are now `10000` / `2000`** for lighter sweep runs (override for 20k/4k comparisons).
- Model/training:
  - `hidden_dim=768`
  - `ffn_dim=3072`
  - `num_layers=2`
  - `num_heads=8`
  - `batch_size=4`
  - `learning_rate=2e-4`
  - `normalization=log1p_tpm`
- Runtime: `torchrun --standalone --nproc_per_node=1 train.py`

## Current Run Snapshot
- **Job ID:** `33210572`
- **State:** `RUNNING`
- **Node:** `n0302.savio2`
- **W&B run:** `dev-1080ti-33210572`
- **W&B URL:** https://wandb.ai/abraham_guan-ucb/bridge-rna-smoke/runs/r2rc58tz

### Log highlights
- `[DATA] Total samples available: 46,422`
- `[DATA] Train: 20,000 samples`
- `[DATA] Val: 4,000 samples`
- `[CHECK] num_genes=16551`
- `[MODEL] Parameters: 25,706,497`
- Training is progressing through Epoch 1 with stable loss updates (no OOM so far).

## Comparison to Minggang Run
- We are now aligned on key hyperparameters and subset sizing.
- One remaining difference: this run uses `train.py` (batch Parquet directory), while Minggang's referenced run used `train_single.py` (merged `expression.parquet`).

## Decisions / Asks for Team
- Confirm official comparison path:
  - `train.py` (multi-file Parquet) vs
  - `train_single.py` (merged Parquet)
- Confirm primary reporting metric for meetings:
  - `val_loss` vs `val_mse`
- Confirm whether to permanently exclude problematic nodes (if recurring GPU issues).

## Repro Commands (Current Baseline)
```bash
cd /global/scratch/users/abrahamguan/bridge-rna
mkdir -p logs
sbatch scripts/savio_dev_train_savio2_1080ti.slurm
```

```bash
JOBID=33210572
squeue -j "$JOBID"
tail -f "logs/slurm-${JOBID}.out"
```
