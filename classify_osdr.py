"""
Train XGBoost (+ Random Forest baseline) on ExpressionPerformer embeddings
for OSDR space vs ground classification, with PCA visualizations.

Usage:
    # Classification only:
    python classify_osdr.py --embeddings embeddings.npz

    # With metadata coloring (strain, sex, tissue, etc.):
    python classify_osdr.py --embeddings embeddings.npz \
        --metadata_csv /path/to/metadata.csv \
        --metadata_id_col sample_id

    Metadata CSV: one row per sample, any categorical columns (strain, sex,
    tissue, organism, ...) will each get their own PCA plot automatically.
    Columns with >30 unique values are skipped.

Requires: xgboost, scikit-learn, matplotlib
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ── data loading ──────────────────────────────────────────────────────────────

def load(path):
    d = np.load(path, allow_pickle=True)
    X = d['embeddings']
    labels = d['labels']
    sample_ids = d['sample_ids']
    mask = labels >= 0
    return X[mask], labels[mask], sample_ids[mask]


def load_metadata(csv_path, id_col, sample_ids):
    """Load metadata CSV and align rows to sample_ids order."""
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"--metadata_id_col '{id_col}' not found in {csv_path}. "
                         f"Available columns: {df.columns.tolist()}")
    df = df.set_index(id_col)
    # reindex to match sample order; missing rows → NaN
    return df.reindex(sample_ids.astype(str))


# ── classification ────────────────────────────────────────────────────────────

def evaluate(name, clf, X, y, n_splits, seed):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'accuracy': 'accuracy',
        'mcc': make_scorer(matthews_corrcoef),
    }
    results = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n{name}  ({n_splits}-fold CV)")
    for k in ['roc_auc', 'f1', 'accuracy', 'mcc']:
        vals = results[f'test_{k}']
        print(f"  {k:12s}: {vals.mean():.4f} ± {vals.std():.4f}")
    return results


# ── PCA plots ─────────────────────────────────────────────────────────────────

_LABEL_PALETTE = {'spaceflight': '#e74c3c', 'ground': '#3498db'}

# Colorblind-friendly qualitative palette for metadata
_META_COLORS = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
]


def _scatter(ax, X_pca, values, title, palette=None):
    categories = sorted(np.unique(values))
    if palette is None:
        palette = {c: _META_COLORS[i % len(_META_COLORS)]
                   for i, c in enumerate(categories)}
    for cat in categories:
        mask = values == cat
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=str(cat), alpha=0.55, s=18,
                   color=palette.get(str(cat), '#999999'), linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(markerscale=2, fontsize=8,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


def save_pca_plot(X_pca, values, title, out_path, palette=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, X_pca, values, title, palette)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def run_pca(X, sample_ids, labels, meta_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(scaler.fit_transform(X))
    var = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")

    subtitle = f'PC1={var[0]:.1%}, PC2={var[1]:.1%}'

    # --- spaceflight vs ground (always) ---
    label_names = np.where(labels == 1, 'spaceflight', 'ground')
    save_pca_plot(
        X_pca, label_names,
        f'PCA — Spaceflight vs Ground\n({subtitle})',
        out_dir / 'pca_label.png',
        palette=_LABEL_PALETTE,
    )

    # --- metadata columns ---
    if meta_df is not None:
        for col in meta_df.columns:
            values = meta_df[col].fillna('unknown').astype(str).values
            n_unique = len(np.unique(values))
            if n_unique < 2:
                print(f"  Skipping '{col}': only 1 unique value")
                continue
            if n_unique > 30:
                print(f"  Skipping '{col}': {n_unique} unique values (too many)")
                continue
            save_pca_plot(
                X_pca, values,
                f'PCA — {col}\n({subtitle})',
                out_dir / f'pca_{col}.png',
            )

    # --- variance explained bar chart ---
    pca_full = PCA(n_components=min(50, X.shape[1]), random_state=42)
    pca_full.fit(scaler.transform(X))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
           pca_full.explained_variance_ratio_ * 100, color='#4e79a7')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Scree Plot (top 50 PCs)')
    plt.tight_layout()
    plt.savefig(out_dir / 'pca_scree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir / 'pca_scree.png'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pca_dir', default='pca_plots',
                    help='Output directory for PCA figures')
    # optional metadata for extra PCA coloring
    ap.add_argument('--metadata_csv', default=None,
                    help='CSV with per-sample metadata (strain, sex, tissue, ...)')
    ap.add_argument('--metadata_id_col', default='sample_id',
                    help='Column in metadata_csv that matches sample IDs in embeddings')
    args = ap.parse_args()

    X, y, sample_ids = load(args.embeddings)
    print(f"Loaded {len(y)} samples  (space={y.sum()}, ground={(y==0).sum()})")
    print(f"Embedding dim: {X.shape[1]}")

    meta_df = None
    if args.metadata_csv:
        meta_df = load_metadata(args.metadata_csv, args.metadata_id_col, sample_ids)
        print(f"Metadata loaded: {meta_df.shape[1]} columns — {meta_df.columns.tolist()}")

    # PCA plots
    print("\nGenerating PCA plots...")
    run_pca(X, sample_ids, y, meta_df, args.pca_dir)

    # Classification
    xgb_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    rf_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    evaluate('XGBoost', xgb_clf, X, y, args.n_splits, args.seed)
    evaluate('RandomForest', rf_clf, X, y, args.n_splits, args.seed)


if __name__ == '__main__':
    main()
