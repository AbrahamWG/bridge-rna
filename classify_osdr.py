"""
Train XGBoost (+ Random Forest baseline) on ExpressionPerformer embeddings
for OSDR space vs ground classification.

Usage:
    python classify_osdr.py --embeddings embeddings.npz [--n_splits 5]

Requires: xgboost, scikit-learn
"""

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef
import xgboost as xgb


def load(path):
    d = np.load(path, allow_pickle=True)
    X = d['embeddings']
    labels = d['labels']
    sample_ids = d['sample_ids']
    mask = labels >= 0
    return X[mask], labels[mask], sample_ids[mask]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    X, y, sample_ids = load(args.embeddings)
    print(f"Loaded {len(y)} samples  (space={y.sum()}, ground={(y==0).sum()})")
    print(f"Embedding dim: {X.shape[1]}")

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
            use_label_encoder=False,
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
