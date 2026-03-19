# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ExpressionBERT: Masked gene expression prediction using Transformer.

Adapted from Google's SLiMPerformer for continuous gene expression data.
Architecture:
  - Gene identity embedding (learned, like BERT token IDs)
  - Rotary Expression Embedding (REE) for value-based positional encoding
  - N Transformer layers (multi-head attention + FFN + LayerNorm)
  - Output projection for per-gene expression reconstruction

Training objective: MLM-style masking (mask 15% of genes, predict their expression)

Usage:
  torchrun --nproc_per_node=2 train.py
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
import pandas as pd

from slim_performer_model import SLiMPerformerLayer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'hidden_dim': 256,
    'ffn_dim': 1024,
    'num_heads': 8,
    'num_layers': 2,
    'ree_base': 100.0,
    'feature_type': 'sqr',       # Linear attention kernel: 'relu', 'elu+1', 'sqr', 'favor+'
    'compute_type': 'iter',      # Prefix sum method: 'iter', 'ps', 'parallel_ps'
    'normalization': 'tpm',      # 'tpm' or 'log1p_tpm' applied before REE/model input
    'mask_ratio': 0.15,
    'mask_token': -10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 4,
    'epochs': 5,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,
    # Data subset sizes (set to None for full data)
    'train_subset': 10000,
    'val_subset': 2000,
    'balanced_sampling': True,
    'data_dir': './data/archs4/train_orthologs',
    'checkpoint_dir': './checkpoints_performer',
}


# ============================================================
# ROTARY EXPRESSION EMBEDDING (REE)
# ============================================================
class RotaryExpressionEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE): Converts continuous gene expression
    values into sinusoidal rotation features.

    Modulates rotary positional encodings using expression magnitude.
    Includes masking support for special tokens (e.g., masked expression = -10).
    Original base=100 (from Google SLiMPerformer research).
    """

    def __init__(self, dim, base=100.0, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.mask_token_id = mask_token_id

        # inv_freq for sinusoidal encoding
        # base=100 (from original code) vs 10000 (standard Transformer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values

        Returns:
            [batch_size, num_genes, dim] sinusoidal encodings
        """
        # Identify masked positions
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)

        # Multiply expression values by frequencies: [B, G] x [D/2] → [B, G, D/2]
        freqs = torch.einsum("bi,j->bij", x, self.inv_freq)

        # Apply sin and cos, then concatenate: [B, G, D/2] → [B, G, D]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        # Mask out special token positions (set to 0)
        if len(x_mask_idx) > 0:
            emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0

        return emb


# ============================================================
# EXPRESSION PERFORMER MODEL
# ============================================================
class ExpressionPerformer(nn.Module):
    """
    ExpressionBERT: Transformer for continuous gene expression data.
    Uses SLiMPerformer's linear attention (O(n) memory) from Google Research.

    Input:  [batch, num_genes] expression values (with masked positions = -10)
    Output: [batch, num_genes] predicted expression values

    Embeddings (summed, like BERT):
      1. Gene identity embedding — learned per-gene vector (like BERT token IDs)
      2. REE — sinusoidal encoding driven by expression magnitude
    """

    def __init__(self, num_genes, hidden_dim=256, n_heads=8, n_layers=4,
                 ffn_dim=1024, ree_base=100.0, mask_token_id=-10,
                 feature_type='sqr', compute_type='iter'):
        super().__init__()
        self.num_genes = num_genes
        self._hidden_dim = hidden_dim

        # Gene identity embedding (like BERT's token embedding)
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

        # Rotary Expression Embedding
        self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base,
                                              mask_token_id=mask_token_id)

        # SLiMPerformer layers (linear O(n) attention via prefix sums)
        self.layers = nn.ModuleList([
            SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads,
                               feature_type, compute_type, on_gptln=True)
            for _ in range(n_layers)
        ])

        # Output: predict single expression value per gene
        self.output_map = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, num_genes] expression values
        Returns:
            [batch, num_genes] predicted expression
        """
        B, G = x.shape
        device = x.device

        # Gene identity embeddings: [G, hidden_dim] → broadcast to [B, G, hidden_dim]
        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids)

        # REE from expression values: [B, G, hidden_dim]
        ree_emb = self.ree(x)

        # Sum embeddings (like BERT: token + position)
        h = gene_emb.unsqueeze(0) + ree_emb

        # Pass through SLiMPerformer layers (linear attention)
        for layer in self.layers:
            rfs = layer.attention.sample_rfs(device)
            h = layer.full_forward(h, rfs)

        # Project to scalar per gene
        out = self.output_map(h).squeeze(-1)  # [B, G]

        return out


# ============================================================
# DATASET
# ============================================================
class ExpressionMLMDataset(Dataset):
    """Expression dataset with MLM-style masking."""

    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10):
        """
        Args:
            expr_array: [samples, genes] numpy array
            mask_ratio: fraction of genes to mask per sample
            mask_token: value for masked positions
        """
        self.X = expr_array.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]

        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)

        x_masked = x.copy()
        x_masked[mask_indices] = self.mask_token

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask_indices, dtype=torch.long),
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_from_batch_files(batch_dir, train_frac=0.8, subset=None, balanced_sampling=True, seed=42, verbose=True):
    """
    Load expression data from batch parquet files.
    Splits at the batch-file level (not individual samples) for efficiency.
    
    Args:
        batch_dir: Path to directory containing batch*.parquet files
        train_frac: Fraction of batch files for training (rest goes to val)
        subset: Max total samples to load (None = all available, but respects balanced_sampling)
        balanced_sampling: If True, balance human/mouse species (50/50)
        seed: Random seed for reproducible splitting
        verbose: Print progress
    
    Returns:
        (X_train, X_val): Both [samples, genes] arrays
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    if not batch_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    
    # Load metadata to determine species per batch
    metadata_file = batch_dir.parent / "samples.json"
    sample_to_species = {}
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            samples_meta = json.load(f)
        # Build species mapping: sample_id → species
        sample_to_species = {s["id"]: s["species"] for s in samples_meta if "species" in s}
    
    # Deterministic shuffle and split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(batch_files))
    rng.shuffle(indices)
    
    split_idx = int(len(batch_files) * train_frac)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    if verbose:
        print(f"[DATA] Batch files: {len(batch_files)} total")
        print(f"       Train: {len(train_indices)} files, Val: {len(val_indices)} files")
    
    # Load and concatenate batch files
    def load_batches(batch_indices, max_samples=None):
        parts_by_species = {}
        total_loaded = 0
        
        # First pass: load all data and count by species
        for idx in batch_indices:
            df = pd.read_parquet(batch_files[idx])
            data = df.values.T.astype(np.float32)  # [samples, genes]
            
            if sample_to_species:
                # Separate by species
                sample_ids = df.columns.tolist()
                for i, s in enumerate(sample_ids):
                    species = sample_to_species.get(s, "unknown")
                    if species not in parts_by_species:
                        parts_by_species[species] = []
                    parts_by_species[species].append(data[i:i+1])
            else:
                # No species info, treat as single group
                if "all" not in parts_by_species:
                    parts_by_species["all"] = []
                parts_by_species["all"].append(data)
            
            total_loaded += data.shape[0]
        
        # Second pass: apply subsampling/balancing
        result_parts = []
        
        if balanced_sampling and len(parts_by_species) > 1:
            # Determine per-species limit
            if max_samples:
                per_species = max_samples // len(parts_by_species)
            else:
                # Use min species count (balance naturally across all data)
                species_counts = [np.vstack(parts).shape[0] for parts in parts_by_species.values()]
                per_species = min(species_counts)
            
            if verbose:
                species_counts = {sp: np.vstack(parts).shape[0] for sp, parts in parts_by_species.items()}
                print(f"       Species counts (before balancing): {species_counts}")
                print(f"       Balanced to {per_species} per species")
            
            for species, parts in parts_by_species.items():
                concatenated = np.vstack(parts)
                if concatenated.shape[0] > per_species:
                    indices = rng.choice(concatenated.shape[0], per_species, replace=False)
                    result_parts.append(concatenated[indices])
                else:
                    result_parts.append(concatenated)
        
        else:
            # No balancing, just subsample if needed
            all_data = np.vstack([np.vstack(parts) for parts in parts_by_species.values()])
            if max_samples and all_data.shape[0] > max_samples:
                indices = rng.choice(all_data.shape[0], max_samples, replace=False)
                result_parts.append(all_data[indices])
            else:
                result_parts.append(all_data)
        
        return np.vstack(result_parts) if result_parts else np.array([])
    
    X_train = load_batches(train_indices, max_samples=subset)
    X_val = load_batches(val_indices, max_samples=subset)
    
    if verbose:
        print(f"       Final: train {X_train.shape}, val {X_val.shape}")
    
    return X_train, X_val


def apply_input_normalization(x: np.ndarray, normalization: str) -> np.ndarray:
    """Apply input normalization mode expected by the model."""
    if normalization == 'tpm':
        return x.astype(np.float32, copy=False)
    if normalization == 'log1p_tpm':
        # Clamp negatives to 0 before log1p for numerical stability.
        return np.log1p(np.maximum(x, 0.0)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown normalization: {normalization!r}")


def format_float_for_tag(v: float) -> str:
    s = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:.6f}"
    return s.replace('.', 'p').replace('+', '').replace('-', 'm')


def build_run_tag(cfg: dict) -> str:
    return (
        f"norm-{cfg['normalization']}"
        f"_lr-{format_float_for_tag(cfg['learning_rate'])}"
        f"_wd-{format_float_for_tag(cfg['weight_decay'])}"
        f"_mask-{format_float_for_tag(cfg['mask_ratio'])}"
        f"_ree-{format_float_for_tag(cfg['ree_base'])}"
    )


# ============================================================
# TRAINING (DDP)
# ============================================================
def main():
    script_start = time.time()

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0

    if is_main:
        print("\n" + "=" * 70)
        print(f"ExpressionPerformer Training — DDP ({world_size} GPUs)")
        print("=" * 70)
        print(f"\n[SETUP] Rank: {rank}, Device: {device}")

    # ─────────────────────────────────────────────────────────
    # WANDB (init early so sweep can override CONFIG)
    # ─────────────────────────────────────────────────────────
    if is_main and HAS_WANDB:
        wandb.init(
            project="expression-performer",
            config=CONFIG,
        )
        # When running a sweep, wandb.config overrides CONFIG values
        for key in CONFIG:
            if key in wandb.config:
                CONFIG[key] = wandb.config[key]
        # Always derive ffn_dim from hidden_dim (4x multiplier)
        CONFIG['ffn_dim'] = CONFIG['hidden_dim'] * 4
        wandb.config.update(CONFIG, allow_val_change=True)

    # Broadcast CONFIG from rank 0 so all ranks use the same hyperparams
    config_list = [CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    CONFIG.update(config_list[0])

    # ─────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    data_dir = Path(CONFIG['data_dir'])
    
    if is_main:
        print("\n[DATA] Loading from batch files...")
    
    # Check if batch files exist
    batch_dir = data_dir.parent / "train_orthologs" / "batch_files"
    if not batch_dir.exists():
        # Fallback to merged expression.parquet
        batch_dir = data_dir.parent / "train_orthologs"
    
    if (batch_dir / "batch_files").exists():
        batch_dir = batch_dir / "batch_files"
    
    t0 = time.time()
    X_train, X_val = load_from_batch_files(
        batch_dir,
        train_frac=0.8,
        subset=CONFIG.get('train_subset', None),
        balanced_sampling=CONFIG.get('balanced_sampling', True),
        seed=CONFIG['seed'],
        verbose=is_main,
    )
    num_samples, num_genes = X_train.shape
    if is_main:
        print(f"  ✓ Time: {time.time()-t0:.1f}s")
    
    # Apply selected normalization before sending values into REE/model.
    X_train = apply_input_normalization(X_train, CONFIG['normalization'])
    X_val = apply_input_normalization(X_val, CONFIG['normalization'])
    if is_main:
        print(
            f"[DATA] Input normalization='{CONFIG['normalization']}' | "
            f"train range=({X_train.min():.4f}, {X_train.max():.4f}) | "
            f"val range=({X_val.min():.4f}, {X_val.max():.4f})"
        )

    # Sanity checks
    if is_main:
        print(f"\n[CHECK] num_genes={num_genes}, "
              f"train_samples={X_train.shape[0]}, val_samples={X_val.shape[0]}")
        assert num_genes > 10000, f"Expected ~16K genes, got {num_genes}"

    # ─────────────────────────────────────────────────────────
    # DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    train_ds = ExpressionMLMDataset(X_train, CONFIG['mask_ratio'], CONFIG['mask_token'])
    val_ds = ExpressionMLMDataset(X_val, CONFIG['mask_ratio'], CONFIG['mask_token'])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                      rank=rank, shuffle=False, seed=42)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=train_sampler, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            sampler=val_sampler, num_workers=0,
                            pin_memory=True)

    if is_main:
        print(f"\n[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches")
        print(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # ─────────────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[MODEL] Building ExpressionPerformer...")

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['num_heads'],
        n_layers=CONFIG['num_layers'],
        ffn_dim=CONFIG['ffn_dim'],
        ree_base=CONFIG['ree_base'],
        mask_token_id=CONFIG['mask_token'],
        feature_type=CONFIG['feature_type'],
        compute_type=CONFIG['compute_type'],
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  ✓ Parameters: {total_params:,}")

    # ─────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'])

    if is_main:
        print(f"  ✓ AdamW (lr={CONFIG['learning_rate']})")

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 70)
        print("[TRAIN] Starting training...")
        print("=" * 70 + "\n")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    ckpt_base = Path(CONFIG['checkpoint_dir'])
    # Per-run subdir (wandb run ID if available, else timestamp)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    if is_main and HAS_WANDB and wandb.run is not None:
        run_id = wandb.run.id
    else:
        run_id = run_timestamp
    run_tag = build_run_tag(CONFIG)
    ckpt_dir = ckpt_base / run_id
    if is_main:
        ckpt_base.mkdir(exist_ok=True, parents=True)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Load global best val loss (across all runs)
    global_best_path = ckpt_base / 'global_best_val_loss.json'
    if global_best_path.exists():
        with open(global_best_path) as f:
            global_best_val_loss = json.load(f)['val_loss']
    else:
        global_best_val_loss = float('inf')

    run_metadata = {
        'run_id': run_id,
        'timestamp': run_timestamp,
        'run_tag': run_tag,
        'normalization': CONFIG['normalization'],
        'sweep_parameters': {
            'learning_rate': CONFIG['learning_rate'],
            'weight_decay': CONFIG['weight_decay'],
            'mask_ratio': CONFIG['mask_ratio'],
            'ree_base': CONFIG['ree_base'],
            'early_stopping': CONFIG['early_stopping'],
        },
        'architecture': {
            'hidden_dim': CONFIG['hidden_dim'],
            'ffn_dim': CONFIG['ffn_dim'],
            'num_heads': CONFIG['num_heads'],
            'num_layers': CONFIG['num_layers'],
        },
        'dataset': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'num_genes': int(num_genes),
            'train_used_counts': train_used_counts,
            'val_used_counts': val_used_counts,
            'train_raw_counts': train_raw_counts,
            'val_raw_counts': val_raw_counts,
            'balanced_sampling': CONFIG['balanced_sampling'],
            'train_subset': CONFIG['train_subset'],
            'val_subset': CONFIG['val_subset'],
        },
    }

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        # --- Train ---
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(train_loader):
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)

            pred = model(x_masked)  # [B, G]

            # MSE loss on masked positions only
            loss_parts = []
            for i in range(len(x_masked)):
                idxs = mask_idx[i]
                if len(idxs) > 0:
                    loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

            loss = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Progress every 25%
            if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | Avg: {avg:.6f}")

        epoch_train_loss = running_loss / max(1, num_batches)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                pred = model(x_masked)

                loss_parts = []
                for i in range(len(x_masked)):
                    idxs = mask_idx[i]
                    if len(idxs) > 0:
                        loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

                if loss_parts:
                    val_loss += torch.stack(loss_parts).mean().item()
                    val_batches += 1

        # Sync validation across ranks
        vl = torch.tensor(val_loss, device=device)
        vb = torch.tensor(float(val_batches), device=device)
        dist.all_reduce(vl, op=dist.ReduceOp.SUM)
        dist.all_reduce(vb, op=dist.ReduceOp.SUM)
        epoch_val_loss = (vl / vb.clamp(min=1)).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        # Log to wandb
        if is_main and HAS_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': scheduler.get_last_lr()[0],
            })

        epoch_time = time.time() - epoch_start

        # --- Checkpoint ---
        if is_main:
            model_sd = model.module.state_dict()

            print(f"\n  ╔════════════════════════════════════════════╗")
            print(f"  ║ Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  ║ Train Loss: {epoch_train_loss:.6f}")
            print(f"  ║ Val Loss:   {epoch_val_loss:.6f}")
            print(f"  ║ Time: {epoch_time:.1f}s")

            checkpoint_payload = {
                'model_state_dict': model_sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'config': dict(CONFIG),
                'run_metadata': run_metadata,
                'total_params': total_params,
            }

            torch.save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:02d}.pt")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                best_named = ckpt_dir / f"best_{run_tag}_run-{run_id}.pt"
                torch.save(checkpoint_payload, best_named)
                print(f"  ║ ✓ New best (run)! Saved best_model.pt")

                # Update global best across all runs
                if epoch_val_loss < global_best_val_loss:
                    global_best_val_loss = epoch_val_loss
                    torch.save(checkpoint_payload, ckpt_base / "best_model.pt")
                    with open(global_best_path, 'w') as f:
                        json.dump({'val_loss': global_best_val_loss,
                                   'run_id': run_id,
                                   'epoch': epoch + 1,
                                   'run_tag': run_tag,
                                   'normalization': CONFIG['normalization']}, f, indent=2)
                    print(f"  ║ ★ New global best! {epoch_val_loss:.6f}")
            else:
                if CONFIG['early_stopping']:
                    patience_counter += 1
                    print(f"  ║ ✗ No improvement ({patience_counter}/{CONFIG['patience']})")
                    if patience_counter >= CONFIG['patience']:
                        print(f"  ║ ⚠ Early stopping!")
                        print(f"  ╚════════════════════════════════════════════╝\n")
                        break
                else:
                    print("  ║ ✗ No improvement (early_stopping=False; continuing)")

            print(f"  ╚════════════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────
    if is_main:
        # Config
        cfg = {
            **CONFIG,
            'num_genes': num_genes,
            'total_params': total_params,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'run_id': run_id,
            'timestamp': run_timestamp,
            'run_tag': run_tag,
            'dataset': run_metadata['dataset'],
            'architecture': run_metadata['architecture'],
            'sweep_parameters': run_metadata['sweep_parameters'],
        }
        with open(ckpt_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

        with open(ckpt_dir / "run_metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)

        # Loss CSV
        pd.DataFrame({'epoch': range(len(train_losses)),
                       'train_loss': train_losses,
                       'val_loss': val_losses}).to_csv(
            ckpt_dir / "loss_history.csv", index=False)

        # Loss plot
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker='o', label='Train Loss', linewidth=2)
            plt.plot(val_losses, marker='s', label='Val Loss', linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("ExpressionPerformer Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
            plt.close()

        total_time = time.time() - script_start
        print("=" * 70)
        print(f"Training complete! {total_time:.0f}s ({total_time/60:.1f}m)")
        print(f"  Run best val loss:    {best_val_loss:.6f}")
        print(f"  Global best val loss: {global_best_val_loss:.6f}")
        print(f"  Run checkpoints:      {ckpt_dir}/")
        print(f"  Global best model:    {ckpt_base / 'best_model.pt'}")
        print("=" * 70 + "\n")

    if is_main and HAS_WANDB:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
