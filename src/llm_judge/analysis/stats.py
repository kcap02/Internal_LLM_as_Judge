"""Statistics: conditional AUROC, grouped paired bootstrap, FDR control.

Three things this module exists to prevent:

1. **The item-identity confound.** The target `is_correct` equals
   `pred_verdict == gt_verdict`. Whenever a judge has any verdict bias (and
   they all do), `is_correct` is largely determined by `gt_verdict` alone —
   so ANY feature that merely tells pos items from neg items scores well
   without carrying one bit about self-knowledge. Spectral and activation
   features do tell them apart trivially: the prompts contain different text.
   The fix is `stratified_auroc`, which only ever compares items that share
   the same `gt_verdict`. Within a stratum, item identity is constant and the
   only thing left to predict is the judge's own behaviour.

2. **"Fold std" masquerading as inference.** Fold-to-fold spread is not a
   significance test. `paired_bootstrap` resamples GROUPS (question_ids, so
   pos/neg partners move together) and recomputes both metrics on identical
   resamples, giving a CI on the paired difference.

3. **Multiplicity.** Datasets x models x formats x contrasts produces dozens
   of p-values; `benjamini_hochberg` controls the false discovery rate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc_or_none(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


# A stratum needs at least this many items of EACH class before its AUROC
# means anything. Below it, one or two minority items drive the estimate to
# an arbitrary value — and the bootstrap, resampling the same one or two
# items, reports a narrow CI around that arbitrary value. That combination
# manufactures false discoveries that survive FDR, so such strata are
# excluded rather than down-weighted.
MIN_CLASS_PER_STRATUM = 5


def stratified_auroc(y: np.ndarray, scores: np.ndarray, strata: np.ndarray,
                     min_class_n: int = MIN_CLASS_PER_STRATUM) -> float | None:
    """AUROC computed WITHIN strata, pooled by discordant-pair weight.

    Equivalent to the probability that a randomly chosen correct item scores
    above a randomly chosen incorrect item **drawn from the same stratum**.
    With strata = gt_verdict this is the item-identity-free metric.

    Returns None when no stratum has enough of both classes to estimate.
    """
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n_pos = int((y[m] == 1).sum())
        n_neg = int((y[m] == 0).sum())
        if min(n_pos, n_neg) < min_class_n:
            continue
        auc = _auc_or_none(y[m], scores[m])
        if auc is None:
            continue
        w = float(n_pos * n_neg)   # number of comparable pairs in the stratum
        num += auc * w
        den += w
    return (num / den) if den > 0 else None


def pooled_auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    return _auc_or_none(y, scores)


def paired_bootstrap(y: np.ndarray, groups: np.ndarray,
                     scores_a: np.ndarray, scores_b: np.ndarray,
                     strata: np.ndarray | None = None,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """CI and one-sided p-value for metric(b) - metric(a).

    The metric is stratified AUROC when `strata` is given, pooled otherwise.
    Groups are resampled with replacement so correlated items (the pos/neg
    partners of one question, or the two orders of one pair) always move
    together.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}

    def metric(s: np.ndarray, idx: np.ndarray) -> float | None:
        if strata is None:
            return _auc_or_none(y[idx], s[idx])
        return stratified_auroc(y[idx], s[idx], strata[idx])

    all_idx = np.arange(len(y))
    base_a, base_b = metric(scores_a, all_idx), metric(scores_b, all_idx)

    deltas = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in gs])
        ma, mb = metric(scores_a, idx), metric(scores_b, idx)
        if ma is not None and mb is not None:
            deltas.append(mb - ma)

    if not deltas or base_a is None or base_b is None:
        return {"metric_a": base_a, "metric_b": base_b, "delta": None,
                "ci95": (None, None), "p_one_sided": None,
                "n_boot_effective": len(deltas)}

    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric_a": base_a,
        "metric_b": base_b,
        "delta": base_b - base_a,
        "ci95": (float(lo), float(hi)),
        # H1: b > a. Bootstrap-fraction p, floored at 1/(n+1) — a p of exactly
        # 0 is not a claim the resampling can support.
        "p_one_sided": float(max(np.mean(deltas <= 0), 1.0 / (len(deltas) + 1))),
        "n_boot_effective": int(len(deltas)),
    }


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """BH-FDR: return the per-test reject/keep decisions at level `alpha`."""
    p = np.asarray([1.0 if v is None else v for v in pvals], float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out.tolist()
