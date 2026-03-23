#!/usr/bin/env bash
# Wrapper for wandb sweep to launch DDP single-parquet training with torchrun
echo "[SWEEP] Starting single-parquet training with torchrun..." >&2
exec torchrun --nproc_per_node=2 train_single.py "$@"
