# Archived sweep configs

- **`arch1080ti_grid.yaml`** — legacy full factorial architecture grid (uses 3 GPUs in its `command:`). Prefer **`sweeps/walt_sweep.yaml`** for current mentor mini-sweeps.

Create a sweep from archive only if you intentionally resurrect that experiment:

```bash
wandb sweep --project bridge-rna sweeps/archive/arch1080ti_grid.yaml
```
