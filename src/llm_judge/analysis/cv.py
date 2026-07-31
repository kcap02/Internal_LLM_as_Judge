"""Nested model comparison via grouped out-of-fold predictions.

Ladder (each rung must beat the previous one to claim a contribution):
  M0    base rate            (no information; AUROC = 0.5 by definition)
  Mn    nuisance             (prompt length only)
  M1    margin               (Yes/No logprob margin — the behavioral signal)
  M1n   margin + nuisance
  M2    M1n + spectral       (attention-graph profile + velocity)
  M3    M1n + activations    (last-token hidden state, PCA inside the fold)
  M4    M1n + spectral + activations

CRITICAL: GroupKFold(groups=question_id). The _pos and _neg items of one
question share nearly all their text; if they land in different folds the
model has already seen the essentials and the score is spuriously good.
PCA for the activation block is fit inside each training fold (leakage-free)
via a ColumnTransformer in the pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F
from .stats import paired_bootstrap_auroc


def _pipeline(n_dense: int, n_act: int, pca_dims: int,
              activation_probe: str) -> Pipeline:
    """activation_probe:
      "full" (default) — standardized full-dim activations, stronger L2
          (C=0.1). Unsupervised PCA keeps high-VARIANCE directions and can
          destroy a correctness signal that has no variance advantage; with
          thousands of items a regularized full-dim probe is the standard
          linear-probe baseline and remains leakage-free (fit per fold).
      "pca" — StandardScaler+PCA(pca_dims) inside the fold; use when item
          counts are small (hundreds), where full-dim would overfit.
    """
    if n_act == 0:
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    if activation_probe == "full":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.1))
    ct = ColumnTransformer([
        ("dense", StandardScaler(), slice(0, n_dense)),
        ("act", make_pipeline(StandardScaler(),
                              PCA(n_components=pca_dims, random_state=0)),
         slice(n_dense, n_dense + n_act)),
    ])
    return Pipeline([("features", ct),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int, n_act: int = 0, pca_dims: int = 24,
               activation_probe: str = "full") -> np.ndarray:
    """Out-of-fold probability scores; NaN where a fold was degenerate."""
    scores = np.full(len(y), np.nan)
    n_dense = X.shape[1] - n_act
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = _pipeline(n_dense, n_act, min(pca_dims, len(tr) - 1),
                         activation_probe)
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def build_feature_sets(rows: list[dict], activations_npz=None,
                       act_layer_frac: str = "1") -> dict:
    """Return {name: (X, n_activation_cols)} for the model ladder."""
    Xm = F.margin_features(rows)
    Xn = F.nuisance_features(rows)
    sets = {
        "Mn   nuisance (length)": (Xn, 0),
        "M1   margin": (Xm, 0),
        "M1n  margin+nuisance": (np.c_[Xm, Xn], 0),
    }
    Xs = None
    if all((r.get("spectral") or {}).get("layers") for r in rows):
        Xs = F.spectral_features(rows)
        sets["M2   margin+nuis+spectral"] = (np.c_[Xm, Xn, Xs], 0)
    if activations_npz is not None:
        Xa = F.activation_features(rows, activations_npz, act_layer_frac)
        sets["M3   margin+nuis+activations"] = (np.c_[Xm, Xn, Xa], Xa.shape[1])
        if Xs is not None:
            sets["M4   all"] = (np.c_[Xm, Xn, Xs, Xa], Xa.shape[1])
    return sets


def compare(rows: list[dict], n_splits: int = 5, n_boot: int = 2000,
            pca_dims: int = 24, activations_npz=None, seed: int = 42,
            activation_probe: str = "full", log=print) -> dict:
    """Full ladder with paired bootstrap deltas between adjacent rungs."""
    y = np.array([float(r["is_correct"]) for r in rows])
    groups = np.array([r["question_id"] for r in rows])
    log(f"  n={len(rows)} items, {len(set(groups))} question groups, "
        f"verdict accuracy {y.mean():.1%} (balanced base rate = 50%)")

    sets = build_feature_sets(rows, activations_npz)
    oof = {name: oof_scores(X, y, groups, n_splits, n_act, pca_dims,
                            activation_probe)
           for name, (X, n_act) in sets.items()}

    # Keep only items scored by every model (degenerate folds excluded).
    valid = np.all([~np.isnan(s) for s in oof.values()], axis=0)
    y_v, g_v = y[valid], groups[valid]

    from sklearn.metrics import roc_auc_score
    report = {"n_items": int(valid.sum()), "aurocs": {}, "contrasts": {}}
    for name, s in oof.items():
        report["aurocs"][name] = float(roc_auc_score(y_v, s[valid]))

    # Adjacent contrasts that carry the paper's claims.
    contrasts = [
        ("M1   margin", "M1n  margin+nuisance"),
        ("M1n  margin+nuisance", "M2   margin+nuis+spectral"),
        ("M1n  margin+nuisance", "M3   margin+nuis+activations"),
        ("M3   margin+nuis+activations", "M4   all"),
    ]
    for a, b in contrasts:
        if a in oof and b in oof:
            report["contrasts"][f"{b.split()[0]} - {a.split()[0]}"] = \
                paired_bootstrap_auroc(y_v, g_v, oof[a][valid], oof[b][valid],
                                       n_boot=n_boot, seed=seed)
    return report


def difficulty_strata(rows: list[dict]) -> dict[str, list[dict]]:
    """Split MCQ rows by panel difficulty (gt_mean_prob terciles).

    Low judge accuracy on hard items means noisy labels; strata show whether
    an absent signal is a power problem rather than a negative result.
    """
    probs = [r.get("gt_mean_prob") for r in rows]
    if any(p is None for p in probs):
        return {"all": rows}
    qs = np.percentile([p for p in probs], [33.3, 66.7])
    out = {"hard": [], "medium": [], "easy": []}
    for r, p in zip(rows, probs):
        out["hard" if p <= qs[0] else "medium" if p <= qs[1] else "easy"].append(r)
    return out
