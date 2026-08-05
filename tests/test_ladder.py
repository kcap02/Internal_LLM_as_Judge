"""Tests for the ladder estimator.

The ladder's whole purpose is to measure what a feature block ADDS to a
baseline. These tests encode the two ways that can silently fail.

Run:  python tests/test_ladder.py      (or: python -m pytest tests/ -q)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from llm_judge.analysis.cv import oof_scores, oof_scores_offset  # noqa: E402
from llm_judge.analysis.stats import stratified_auroc  # noqa: E402


def _planted(n_q=200, strength=1.6, d_add=4096, seed=0):
    """A baseline carrying real signal, plus an added block of PURE NOISE."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_q), 2)
    n = len(groups)
    strata = np.tile(["A", "B"], n_q)
    latent = rng.normal(size=n)
    y = (latent + rng.normal(0, 0.7, n) > 0).astype(float)
    X_base = np.c_[strength * latent + rng.normal(0, .5, n),
                   rng.normal(size=(n, 5))]
    X_add = rng.normal(size=(n, d_add))
    return X_base, X_add, y, groups, strata


def test_offset_ladder_is_monotone_under_a_wide_noise_block():
    """C-LADDER: adding an uninformative block must cost ~nothing.

    Concatenating a hidden_size-wide block onto a 6-column baseline and
    fitting one penalised model does NOT do this: the shared penalty lets the
    wide block swamp the baseline, so the rung scores far BELOW the baseline
    it was supposed to extend. The offset estimator makes addition monotone
    by construction.
    """
    X_base, X_add, y, groups, strata = _planted(d_add=4096)
    base = stratified_auroc(y, oof_scores(X_base, y, groups, 5), strata)

    concat = stratified_auroc(
        y, oof_scores(np.c_[X_base, X_add], y, groups, 5, n_act=X_add.shape[1]),
        strata)
    offset = stratified_auroc(
        y, oof_scores_offset(X_base, X_add, y, groups, 5), strata)

    # The failure mode being guarded against is real and large.
    assert base - concat > 0.10, (
        "concatenation was expected to destroy the baseline here; if this "
        "fails the planted setup no longer reproduces the pathology")
    # The fix: pure noise costs essentially nothing.
    assert abs(offset - base) < 0.02


def test_swamping_damage_grows_with_baseline_strength():
    """The signature that identified this on real data: the stronger the
    baseline, the more concatenation appears to 'lose' — a property of the
    estimator, not of the representation."""
    losses = {}
    for strength in (0.4, 1.6):
        X_base, X_add, y, groups, strata = _planted(strength=strength,
                                                    d_add=4096)
        base = stratified_auroc(y, oof_scores(X_base, y, groups, 5), strata)
        cat = stratified_auroc(
            y, oof_scores(np.c_[X_base, X_add], y, groups, 5,
                          n_act=X_add.shape[1]), strata)
        losses[strength] = base - cat
    assert losses[1.6] > losses[0.4]


def test_pooled_fit_is_what_blocks_the_verdict_channel():
    """C-VERDICT: the ladder MUST be fitted pooled, never within stratum.

    Within a gt_verdict stratum, `is_correct` is exactly
    `(pred_verdict == stratum)`. So a feature that merely decodes the judge's
    OWN verdict is a perfect within-stratum correctness predictor while
    carrying zero self-knowledge — and the last-token hidden state decodes
    pred essentially perfectly, since it is the state the verdict logit is
    read from. Conditioning on gt_verdict does NOT remove this: the thing
    decoded is `pred`, not `gt`.

    What protects the analysis is the POOLED fit: a pure verdict decoder must
    score 1.0 in one stratum and 0.0 in the other, so a single pooled model
    can never select that direction. Fitting per stratum — an innocuous-looking
    refactor — removes the protection and yields a near-perfect, entirely
    meaningless result. This test exists so that refactor fails loudly.
    """
    rng = np.random.default_rng(0)
    n_q = 200
    groups = np.repeat(np.arange(n_q), 2)
    n = len(groups)
    strata = np.tile(["A", "B"], n_q)
    pred = rng.integers(0, 2, n)                  # the judge's own verdict
    y = (pred == (strata == "A")).astype(float)   # correct iff pred == stratum
    # A feature block that decodes pred perfectly and knows nothing else.
    X = np.c_[pred + rng.normal(0, .01, n), rng.normal(size=(n, 5))]

    pooled = stratified_auroc(y, oof_scores(X, y, groups, 5), strata)

    # The forbidden refactor: fit inside each stratum separately.
    per_stratum = np.full(n, np.nan)
    for s in np.unique(strata):
        m = strata == s
        per_stratum[m] = oof_scores(X[m], y[m], groups[m], 5)
    within = stratified_auroc(y, per_stratum, strata)

    assert abs(pooled - 0.5) < 0.10, (
        f"pooled fit should be near chance here, got {pooled:.3f}")
    assert within > 0.95, (
        f"within-stratum fit should expose the verdict channel, got "
        f"{within:.3f}")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
