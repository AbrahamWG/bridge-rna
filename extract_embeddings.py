"""
Extract mean-pooled ExpressionPerformer embeddings for OSDR samples.

OSDR mode (two CSV files from /global/scratch/users/kmeng/osdr_data/):
    python extract_embeddings.py \
        --checkpoint /path/to/best_model.pt \
        --osdr_all    /global/scratch/users/kmeng/osdr_data/osdr_processed_all.csv \
        --osdr_space  /global/scratch/users/kmeng/osdr_data/osdr_processed_spaceflight.csv \
        --vocab_file  /global/scratch/users/abrahamguan/joint_vocab.csv \
        --output embeddings.npz

    Labels: samples in osdr_space → 1 (spaceflight), rest → 0 (ground).

Parquet mode (custom data dir):
    python extract_embeddings.py \
        --checkpoint /path/to/best_model.pt \
        --data_dir /path/to/parquets \
        --vocab_file /path/to/joint_vocab.csv \
        --labels_csv /path/to/labels.csv --label_col space \
        --output embeddings.npz

Output embeddings.npz keys:
    embeddings  — float32 [N, hidden_dim]
    sample_ids  — str [N]
    labels      — int [N]  (1=spaceflight, 0=ground)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import ExpressionPerformer, _parquet_numeric_gene_columns
import pyarrow.parquet as pq


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    num_genes = (cfg.get('num_genes')
                 or ckpt.get('run_metadata', {}).get('num_genes')
                 or ckpt.get('run_metadata', {}).get('dataset', {}).get('num_genes'))
    if num_genes is None:
        raise ValueError("Cannot determine num_genes from checkpoint.")

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=cfg.get('hidden_dim', 320),
        n_heads=cfg.get('num_heads', 8),
        n_layers=cfg.get('num_layers', 2),
        ffn_dim=cfg.get('ffn_dim', 1536),
        ree_base=cfg.get('ree_base', 100.0),
        feature_type=cfg.get('feature_type', 'favor+'),
        compute_type=cfg.get('compute_type', 'iter'),
    )
    state = {k.removeprefix('module.'): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg, num_genes


def load_gene_list(vocab_file, first_file: str) -> list:
    if vocab_file:
        df = pd.read_csv(vocab_file)
        col = 'gene_id' if 'gene_id' in df.columns else df.columns[0]
        return df[col].tolist()
    if first_file.endswith('.parquet'):
        schema = pq.read_schema(first_file)
        return _parquet_numeric_gene_columns(schema)
    # CSV: infer gene columns (numeric, non-index)
    df = pd.read_csv(first_file, nrows=1, index_col=0)
    return df.columns.tolist()


def _csv_to_array(path: str, gene_cols: list) -> tuple:
    df = pd.read_csv(path, index_col=0)
    sample_ids = df.index.astype(str).tolist()
    for c in gene_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[gene_cols].values.astype(np.float32), sample_ids


def load_osdr(osdr_all: str, osdr_space: str, gene_cols: list) -> tuple:
    """Build (expr, sample_ids, labels) from the two OSDR CSVs."""
    expr_all, ids_all = _csv_to_array(osdr_all, gene_cols)
    _, ids_space = _csv_to_array(osdr_space, gene_cols)
    space_set = set(ids_space)
    labels = np.array([1 if sid in space_set else 0 for sid in ids_all], dtype=np.int32)
    print(f"  OSDR: {len(ids_all)} total  |  spaceflight={labels.sum()}  ground={(labels==0).sum()}")
    return expr_all, ids_all, labels


def load_parquets(data_dir: str, gene_cols: list) -> tuple:
    frames = []
    for p in sorted(Path(data_dir).glob('*.parquet')):
        df = pq.read_table(str(p)).to_pandas()
        ids = df['geo_accession'].tolist() if 'geo_accession' in df.columns else df.index.astype(str).tolist()
        for c in gene_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[gene_cols]
        df.index = ids
        frames.append(df)
    combined = pd.concat(frames)
    return combined.values.astype(np.float32), combined.index.tolist()


@torch.no_grad()
def extract(model, expr: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    embeddings = []
    for i in range(0, len(expr), batch_size):
        batch = torch.tensor(expr[i:i + batch_size], device=device)
        emb = model.encode(batch).cpu().float().numpy()
        embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(emb)}/{len(expr)} samples encoded")
    return np.concatenate(embeddings, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', default='embeddings.npz')
    ap.add_argument('--vocab_file', default=None)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # OSDR mode
    ap.add_argument('--osdr_all', default=None,
                    help='Path to osdr_processed_all.csv')
    ap.add_argument('--osdr_space', default=None,
                    help='Path to osdr_processed_spaceflight.csv')
    # Parquet mode
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--labels_csv', default=None)
    ap.add_argument('--label_col', default='space')
    ap.add_argument('--sample_id_col', default='sample_id')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading model...")
    model, cfg, num_genes = load_model(args.checkpoint, device)
    print(f"  num_genes={num_genes}  hidden_dim={cfg.get('hidden_dim')}  layers={cfg.get('num_layers')}")

    if args.osdr_all and args.osdr_space:
        gene_cols = load_gene_list(args.vocab_file, args.osdr_all)
        print(f"  Gene columns: {len(gene_cols)}")
        expr, sample_ids, labels = load_osdr(args.osdr_all, args.osdr_space, gene_cols)
    elif args.data_dir:
        parquet_paths = sorted(Path(args.data_dir).glob('*.parquet'))
        gene_cols = load_gene_list(args.vocab_file, str(parquet_paths[0]))
        print(f"  Gene columns: {len(gene_cols)}")
        expr, sample_ids = load_parquets(args.data_dir, gene_cols)
        labels = None
        if args.labels_csv:
            ldf = pd.read_csv(args.labels_csv).set_index(args.sample_id_col)[args.label_col]
            labels = np.array([int(ldf.get(s, -1)) for s in sample_ids], dtype=np.int32)
    else:
        ap.error("Provide either --osdr_all + --osdr_space, or --data_dir")

    print(f"Extracting embeddings for {len(sample_ids)} samples...")
    embeddings = extract(model, expr, args.batch_size, device)
    print(f"  Shape: {embeddings.shape}")

    save_kwargs = {'embeddings': embeddings, 'sample_ids': np.array(sample_ids)}
    if labels is not None:
        save_kwargs['labels'] = labels
    np.savez_compressed(args.output, **save_kwargs)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
