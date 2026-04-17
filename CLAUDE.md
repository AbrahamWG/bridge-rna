# CLAUDE.md — bridge-rna

Working guide for Claude Code (and human collaborators). For current experiment state,
sweep results, and infrastructure status, read `project_state.md` — it is authoritative.
This file covers stable, project-specific "how to work here" rules.

## Purpose

Ultimate goal: pretrained ExpressionBERT → fine-tune or MLP head for NASA OSDR
space/microgravity classification. Pretraining is a means, not the end. OSDR data is
sacred — fine-tuning and evaluation only, never pretraining.

---

## Repo map

```text
bridge-rna/
├── train.py                      # Main trainer: multi-file Parquet, DDP, streaming
├── train_single.py               # Trainer for single merged expression.parquet
├── slim_performer_model.py       # SLiMPerformer layers (linear attention / FAVOR+)
├── numerator_and_denominator.py  # Prefix-sum kernels for Performer attention
├── preprocessing.py              # ARCHS4 H5 → batch Parquets + sidecar metadata
├── merge.py                      # Batch Parquets → single expression.parquet
├── scripts/
│   ├── savio_smoke_train_savio2_1080ti.slurm   # Smoke: 1 GPU, short, proves env
│   ├── savio_dev_train_savio2_1080ti.slurm     # Long dev: 2 GPU, 10k/2k default
│   ├── savio_wandb_sweep_agent.slurm           # W&B sweep worker: 1 GPU
│   ├── verify_slurm.sh                         # Pre-submit guardrail (--test-only)
│   ├── setup_smoke_data_symlinks.sh            # Prepare smoke Parquet dir on scratch
│   └── archive/                                # Legacy scripts (not canonical)
├── sweeps/
│   ├── walt_sweep.yaml           # Current sweep: LR × num_layers, smooth_l1
│   ├── walt_mse_anchor.yaml      # 1-trial MSE baseline
│   └── archive/                  # Old arch grid (not current)
├── docs/                         # Savio howto, sweep notes
├── tests/                        # preprocessing_check.py, preprocessing_single_check.py
└── project_state.md              # Authoritative state: experiments, infra, open bugs
```

---

## Workflow rules

**Plan → execute → verify.** For non-trivial changes, produce a bullet-point plan first
and wait for approval. Execute in small reviewable chunks. Verify (tests, syntax, or
explicit Savio commands) before claiming done.

**Never claim GPU code works without running it.** GPU training cannot be tested locally.
For Slurm / training changes, produce the diff and state the exact Savio commands to
verify (git pull, sbatch, how to read logs). Never run heavy compute on the login node.

**Local-vs-scratch drift is real.** Local edits must be `git push`ed and `git pull`ed on
Savio before `sbatch`. Remind the user if this step seems forgotten.

**Session hygiene.** Use `/clear` between unrelated tasks. At end of substantive
sessions, update `project_state.md` and append to the decision log if applicable.

---

## Savio constraints (hard rules)

| Constraint | Value |
|------------|-------|
| Partition | `savio2_1080ti` (no `savio3_gpu` access on `ic_cdss170`) |
| Account | `ic_cdss170` |
| Memory | `--mem=60G` (not 64G — nodes report ~64317 MB, 64G fails) |
| CPUs | `--cpus-per-task ≤ 8` |
| Excluded nodes | `n0227.savio2`, `n0228.savio2`, `n0302.savio2` |
| GPU default for sweeps | `--gres=gpu:1`, `torchrun --nproc_per_node=1` (DDP desync issues) |
| GPU for dev | `--gres=gpu:2` with dynamic `nproc_per_node` (fall back to 1 if NCCL flakes) |

Pre-submit: `bash scripts/verify_slurm.sh <script.slurm>` on the machine where you'll
call `sbatch`. Session setup (every new login):

```bash
cd /global/scratch/users/abrahamguan/bridge-rna
module load anaconda3/2024.10-1-11.4
eval "$(conda shell.bash hook)"
conda activate bridge-rna
```

---

## Metric discipline (critical)

- **`val_mse`** — always masked MSE on log(TPM), logged every epoch regardless of
  training loss. This is the **primary cross-run metric**. Use this for all comparisons.
- **`epoch_val_loss`** — training objective (MSE or Smooth L1). When using MSE loss,
  `epoch_val_loss` equals `val_mse` and is **directly comparable to mentor's val_loss**
  (mentor uses MSE on log(TPM)). When using Smooth L1, `epoch_val_loss` differs in
  magnitude — use `val_mse` for cross-run comparison.
- **Scale gap is expected and explained.** Our `val_mse ~0.09` on 10k/2k mouse smoke vs
  mentor's `~0.6` on 1M joint is a data-scale and cohort difference, not a metric
  definition mismatch. Numbers will increase (worse) when we scale to full joint cohort,
  and that is expected — it means we are solving a harder, more realistic problem.

---

## Data rules

- `train.py` globs all `*.parquet` under `BRIDGE_RNA_DATA_DIR`. Point it at a
  **processed-only** directory — not the raw `archs4_mouse` dir that contains both
  `processed_*` and `input_chunk_*`.
- **Do not merge `archs4_human` and `archs4_mouse` into one `BRIDGE_RNA_DATA_DIR`**
  without harmonizing gene schemas first. Mouse has ~16,551 genes, human ~16,734 —
  different column counts. The loader assumes all files share the schema of the first.
- Joint human+mouse training requires either a combined `preprocessing.py` run
  (`species=both`) or Brian-provided combined batches with a shared gene schema.

---

## Current best-known config (2026-04)

```
num_layers=2, hidden_dim=320, ffn_dim=1536
lr=2e-4, loss=smooth_l1, batch_size=4, mask_ratio=0.15
val_mse ≈ 0.09 on 10k/2k mouse smoke
```

The 2L ≫ 4L finding is likely a data-scale artifact. Do not re-sweep depth until
training on ≥100k samples.

---

## Open bugs (as of 2026-04-15)

1. **FAVOR+ singular matrix** — `_sample_orth_matrix` in `slim_performer_model.py`.
   Resample-on-failure retry landed 2026-04-15. If it recurs, switch to QR-based
   orthogonal construction instead of `torch.inverse`.
2. **PyTorch serialization errors** (`inline_container.cc / unexpected pos`) on some
   sweep trials — likely NFS/scratch glitch or truncated checkpoint. Unresolved.

---

## What not to do

- Do not run training or preprocessing on the Savio login node.
- Do not use `--mem=64G` on `savio2_1080ti`.
- Do not compare `val_loss` across runs with different loss functions.
- Do not symlink both `archs4_human` and `archs4_mouse` into one data dir without
  schema harmonization.
- Do not force-push to main.
- Do not sweep `hidden_dim` or `num_layers` — mentor established that bigger = better
  in the pretraining regime. Further architecture sweeps are wasted compute until
  downstream OSDR classification shows architecture-dependent transfer behavior.
- Do not sweep `mask_ratio` — frozen at 0.15 per mentor guidance.
- Do not use OSDR data for pretraining. OSDR is reserved for downstream classification
  fine-tuning and evaluation only.
