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
                     n_boot: int = 2000, seed: int = 42,
                     equivalence_band: float = 0.02) -> dict:
    """CI, one-sided p-value and TOST equivalence for metric(b) - metric(a).

    The metric is stratified AUROC when `strata` is given, pooled otherwise.
    Groups are resampled with replacement so correlated items (the pos/neg
    partners of one question, or the two orders of one pair) always move
    together.

    `equivalence_band` is the conditional-AUROC lift treated as negligible;
    the returned `equivalence` block says whether the data rule out an effect
    that large, which is what turns a non-significant contrast into a claim.
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
                "n_boot_effective": len(deltas),
                "equivalence": tost_from_deltas(np.array([]),
                                                band=equivalence_band)}

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
        # A non-significant contrast is only informative if it also says how
        # large an effect the data rule out (see tost_from_deltas).
        "equivalence": tost_from_deltas(deltas, band=equivalence_band),
    }


# ── Equivalence testing ──────────────────────────────────────────────────────
# "We failed to reject" is not a finding: a reviewer cannot distinguish it from
# an underpowered study. TOST inverts the burden of proof — the null becomes
# "the effect is at least as large as `band`", and rejecting it licenses the
# positive claim *"we rule out lifts of `band` or more"*.
#
# The two one-sided tests are equivalent to asking whether the (1 - 2*alpha)
# CI lies entirely inside (-band, +band), so this reads straight off the same
# grouped bootstrap that produced the contrast: no refitting, and the grouping
# (pos/neg partners, both orders of one pairwise item) is respected, which an
# analytic TOST on an independence assumption would not be.
DEFAULT_EQUIVALENCE_BAND = 0.02   # conditional-AUROC lift deemed negligible


def tost_from_deltas(deltas, band: float = DEFAULT_EQUIVALENCE_BAND,
                     alpha: float = 0.05) -> dict:
    """Two one-sided tests for |effect| < band, from bootstrap deltas.

    `p_tost` is the larger of the two one-sided p-values, floored at
    1/(n_boot+1). `equivalent=True` means the data rule out an effect of
    `band` or more in either direction — the publishable form of a null.
    """
    d = np.asarray(deltas, float)
    if d.size == 0:
        return {"band": band, "p_tost": None, "equivalent": None,
                "ci90": (None, None)}
    lo, hi = np.percentile(d, [100 * alpha, 100 * (1 - alpha)])
    floor = 1.0 / (d.size + 1)
    # H0_upper: true effect >= +band. Evidence against it is bootstrap mass
    # at or above +band; likewise, mirrored, for H0_lower.
    p_upper = max(float(np.mean(d >= band)), floor)
    p_lower = max(float(np.mean(d <= -band)), floor)
    return {
        "band": band,
        "p_tost": max(p_upper, p_lower),
        "equivalent": bool(lo > -band and hi < band),
        "ci90": (float(lo), float(hi)),
    }


# ── Paired DeLong (analytic cross-check on the bootstrap) ────────────────────
# The two ROCs being compared are computed on the same items and are therefore
# highly correlated; DeLong's covariance estimator exploits that analytically.
#
# IMPORTANT — it is a cross-check, not the headline test. DeLong assumes
# INDEPENDENT items, which this design violates by construction: the pos/neg
# partners of one question and the two orders of one pairwise item share
# nearly all their text and live in one CV group. Ignoring that clustering
# makes DeLong ANTI-CONSERVATIVE here. The grouped bootstrap is already paired
# (both metrics are recomputed on identical resamples), so it captures the
# same correlation *and* the clustering; the value of running DeLong alongside
# is to see how much of the bootstrap's width is clustering rather than noise.


def _delong_components(y: np.ndarray, scores: np.ndarray):
    """Midrank-based V10/V01 structural components for one score vector."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return None
    # V10[i] = P(pos_i > neg) + 0.5 P(pos_i == neg), and mirrored for V01.
    neg_sorted = np.sort(neg)
    gt = np.searchsorted(neg_sorted, pos, side="left")
    ge = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (gt + 0.5 * (ge - gt)) / n

    pos_sorted = np.sort(pos)
    lt = np.searchsorted(pos_sorted, neg, side="right")
    le = np.searchsorted(pos_sorted, neg, side="left")
    v01 = ((m - lt) + 0.5 * (lt - le)) / m
    return v10, v01


def delong_paired_stratified(y: np.ndarray, scores_a: np.ndarray,
                             scores_b: np.ndarray,
                             strata: np.ndarray | None = None,
                             min_class_n: int = MIN_CLASS_PER_STRATUM) -> dict:
    """Variance of the paired AUROC difference, combined across strata.

    Strata are disjoint sets of items, so their contributions are independent
    and the variance of the discordant-pair-weighted combination is
    sum(w_s^2 var_s) / (sum w_s)^2.

    Returns delta, its analytic SE, a 95% CI and a one-sided p-value — all
    subject to the independence caveat above.
    """
    if strata is None:
        strata = np.zeros(len(y))

    num = den = 0.0
    var_num = 0.0
    for s in np.unique(strata):
        m_s = strata == s
        ys, a_s, b_s = y[m_s], scores_a[m_s], scores_b[m_s]
        n_pos, n_neg = int((ys == 1).sum()), int((ys == 0).sum())
        if min(n_pos, n_neg) < min_class_n:
            continue
        ca, cb = _delong_components(ys, a_s), _delong_components(ys, b_s)
        if ca is None or cb is None:
            continue
        (a10, a01), (b10, b01) = ca, cb
        auc_a, auc_b = float(a10.mean()), float(b10.mean())

        # 2x2 covariance of (AUC_a, AUC_b) from the structural components.
        s10 = np.cov(np.vstack([a10, b10]))
        s01 = np.cov(np.vstack([a01, b01]))
        cov = s10 / n_pos + s01 / n_neg
        var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

        w = float(n_pos * n_neg)
        num += (auc_b - auc_a) * w
        den += w
        var_num += (w ** 2) * var_delta

    if den <= 0:
        return {"delta": None, "se": None, "ci95": (None, None),
                "p_one_sided": None}

    delta = num / den
    se = float(np.sqrt(var_num) / den)
    if not np.isfinite(se) or se <= 0:
        return {"delta": delta, "se": None, "ci95": (None, None),
                "p_one_sided": None}
    z = delta / se
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1 + erf(z / sqrt(2)))     # H1: b > a
    return {"delta": delta, "se": se,
            "ci95": (delta - 1.959963985 * se, delta + 1.959963985 * se),
            "p_one_sided": float(p),
            "caveat": "assumes independent items; ignores CV grouping"}


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
