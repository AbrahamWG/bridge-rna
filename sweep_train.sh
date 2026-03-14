#!/usr/bin/env bash
# Wrapper for wandb sweep to launch DDP training with torchrun
exec torchrun --nproc_per_node=2 train.py "$@"
