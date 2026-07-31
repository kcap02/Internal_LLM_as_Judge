"""Statistics: grouped, paired bootstrap on out-of-fold predictions.

Fold-to-fold std of a 5-fold CV is NOT a significance test. Instead:
  1. produce out-of-fold (OOF) scores once per feature set on IDENTICAL folds;
  2. bootstrap-resample GROUPS (question_ids, so pos/neg pairs move together),
     recompute AUROC for both feature sets on each resample, and read the CI
     of the paired difference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap_auroc(y: np.ndarray, groups: np.ndarray,
                           scores_a: np.ndarray, scores_b: np.ndarray,
                           n_boot: int = 2000, seed: int = 42) -> dict:
    """CI and one-sided p-value for AUROC(b) - AUROC(a) on OOF scores."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}

    def auc_pair(idx):
        if len(np.unique(y[idx])) < 2:
            return None
        return (roc_auc_score(y[idx], scores_a[idx]),
                roc_auc_score(y[idx], scores_b[idx]))

    base = auc_pair(np.arange(len(y)))
    deltas = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in gs])
        pair = auc_pair(idx)
        if pair is not None:
            deltas.append(pair[1] - pair[0])
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "auroc_a": base[0] if base else float("nan"),
        "auroc_b": base[1] if base else float("nan"),
        "delta": (base[1] - base[0]) if base else float("nan"),
        "ci95": (float(lo), float(hi)),
        "p_one_sided": float(np.mean(deltas <= 0)),  # H1: b > a
        "n_boot_effective": int(len(deltas)),
    }
