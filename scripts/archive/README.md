# Archived Savio / sweep helpers

Not part of the default workflow. Current paths:

- Smoke: `scripts/savio_smoke_train_savio2_1080ti.slurm`
- Dev: `scripts/savio_dev_train_savio2_1080ti.slurm`
- W&B mini-sweep: `sweeps/walt_sweep.yaml` + `scripts/savio_wandb_sweep_agent.slurm`

| File | Notes |
|------|--------|
| `savio_tail_job.sh` | `sacct` + head/tail for `logs/slurm-<JOBID>.*` only (not `slurm-sweep-*`). |
| `savio_sweep_arch1080ti.slurm` | Legacy 3-GPU agent; pair with `sweeps/archive/arch1080ti_grid.yaml`. Uses `--mem=64G` (often fails to schedule on 1080 Ti nodes). |

Submit archived Slurm from **repo root** (same as main scripts):

```bash
sbatch --export=ALL scripts/archive/savio_sweep_arch1080ti.slurm
```
