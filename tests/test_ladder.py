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


def test_offset_ladder_recovers_a_planted_effect_across_magnitudes():
    """The acceptance criterion for ANY ladder estimator: a recovery curve.

    A planted test at a single signal strength cannot distinguish an estimator
    that works from one that has collapsed. Both of this project's ladder
    failures passed such a test:

      * concatenation (C-LADDER) passed at a 32-column block and destroys
        0.315 AUROC of real signal at 4096 columns;
      * offset-with-RidgeCV passed with a STRONG planted latent and, on real
        data at n=200 with 1536 columns, selected maximum shrinkage on every
        fold and returned a delta of 0.000 for every block on every slice --
        a zero manufactured by the estimator.

    So the criterion is recovery across a RANGE: as the planted signal grows,
    the measured delta must grow with it. Zero signal must give ~zero delta
    (that is `test_offset_ladder_is_monotone_under_a_wide_noise_block`), and a
    weak-but-real signal must give a delta that is detectably positive.

    The planted signal is spread over a DENSE random direction rather than a
    single column, because that is how an activation encodes anything; a
    one-column plant in a 512-column block is close to worst case for a
    penalised probe and understates what the estimator can do.
    """
    rng = np.random.default_rng(1)
    n_q, d_add = 250, 512
    groups = np.repeat(np.arange(n_q), 2)
    n = len(groups)
    strata = np.tile(["A", "B"], n_q)

    latent = rng.normal(size=n)
    extra = rng.normal(size=n)
    # A baseline that already explains a lot, which is the regime where the
    # collapse happened: the residual left for the block is small.
    y = ((latent + 0.9 * extra + rng.normal(0, 0.7, n)) > 0).astype(float)
    X_base = np.c_[1.2 * latent + rng.normal(0, .6, n), rng.normal(size=(n, 5))]
    base = stratified_auroc(y, oof_scores(X_base, y, groups, 5), strata)

    direction = rng.normal(size=d_add)
    direction /= np.linalg.norm(direction)

    deltas = {}
    for strength in (0.0, 0.5, 1.0):
        X_add = rng.normal(size=(n, d_add)) + strength * np.outer(extra,
                                                                  direction)
        got = stratified_auroc(
            y, oof_scores_offset(X_base, X_add, y, groups, 5), strata)
        deltas[strength] = got - base

    # Zero planted signal costs ~nothing.
    assert abs(deltas[0.0]) < 0.03, (
        f"noise block moved the score: {deltas[0.0]:+.3f}")
    # A clearly-recoverable planted signal must show up. This is the assertion
    # the RidgeCV form failed: it returned ~0.000 at every strength because it
    # selected maximum shrinkage regardless.
    assert deltas[1.0] > 0.03, (
        f"no recovery: {[(s, round(d, 4)) for s, d in deltas.items()]} — the "
        f"estimator is not expressing a planted effect it should detect")
    assert deltas[1.0] > deltas[0.0], "recovery is not monotone in the plant"


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


def _load_recovery_module():
    """Import stage 22's helpers without executing its CLI."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "22_recovery_curve.py").read_text(encoding="utf-8")
    src = src.split("def main()")[0].replace(
        "import _bootstrap  # noqa: F401", "")
    ns: dict = {}
    exec(compile(src, "recovery_curve", "exec"), ns)
    return ns


def test_recovery_guard_permits_a_trend_only_when_iqrs_separate():
    """An acceptance criterion that has only ever returned PASS is not yet
    known to be one. This exercises the monotonicity guard in BOTH directions.

    The guard exists because of the fifth error in this project: an apparent
    monotone rise in block-only was reported as resolving an anomaly, when it
    was 10-seed noise around a flat value. It must stay silent on overlapping
    IQRs and must speak up when they genuinely separate.
    """
    sepf = _load_recovery_module()["iqr_separations"]

    # Overlapping IQRs -> no claim permitted anywhere.
    overlapping = {
        "200": {"block_only_iqr": [0.543, 0.636]},
        "400": {"block_only_iqr": [0.614, 0.642]},
        "800": {"block_only_iqr": [0.617, 0.657]},
    }
    assert not any(s for *_, s in sepf(overlapping, [200, 400, 800]))

    # Cleanly separated IQRs -> the claim IS permitted, and on the right pair.
    separated = {
        "200": {"block_only_iqr": [0.50, 0.55]},
        "400": {"block_only_iqr": [0.70, 0.75]},   # disjoint from 200
        "800": {"block_only_iqr": [0.72, 0.78]},   # overlaps 400
    }
    got = sepf(separated, [200, 400, 800])
    assert got[0][2] is True, "disjoint IQRs must permit a claim"
    assert got[1][2] is False, "overlapping IQRs must not"

    # Separation in the DOWNWARD direction is also a separation.
    falling = {"200": {"block_only_iqr": [0.70, 0.75]},
               "400": {"block_only_iqr": [0.50, 0.55]}}
    assert sepf(falling, [200, 400])[0][2] is True


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
