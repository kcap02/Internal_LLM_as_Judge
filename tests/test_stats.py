"""Tests for the statistical core.

These four functions decide what the paper is allowed to claim, so each one
is tested against a case where a naive implementation gives the wrong answer.

Run:  python -m pytest tests/ -q      (or: python tests/test_stats.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from llm_judge.analysis.stats import (benjamini_hochberg,  # noqa: E402
                                      delong_paired_stratified,
                                      paired_bootstrap, pooled_auroc,
                                      stratified_auroc, tost_from_deltas)
from llm_judge.grouping import group_key, normalize_text  # noqa: E402


def test_stratified_equals_pooled_when_strata_are_interchangeable():
    """With identical class balance and score distribution per stratum, the
    conditional and pooled AUROCs should agree."""
    rng = np.random.default_rng(0)
    y = np.tile([1, 0], 100).astype(float)
    scores = np.where(y == 1, rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    # Stratum must be INDEPENDENT of the label, otherwise each stratum has no
    # outcome variance and nothing is estimable (that is the next test).
    strata = np.array(["A"] * 100 + ["B"] * 100)
    cond, pool = stratified_auroc(y, scores, strata), pooled_auroc(y, scores)
    assert cond is not None and abs(cond - pool) < 0.05


def test_stratified_auroc_kills_the_item_identity_shortcut():
    """A feature that only encodes the STRATUM must score ~0.5 conditionally
    even though it looks excellent pooled.

    This is C-ID in miniature: the judge always says the first label, so
    is_correct is determined by gt_verdict, and a score that merely tracks
    gt_verdict appears to predict correctness perfectly.
    """
    strata = np.array(["Yes"] * 100 + ["No"] * 100)
    y = np.array([1.0] * 100 + [0.0] * 100)          # always-Yes judge
    rng = np.random.default_rng(1)
    scores = np.where(strata == "Yes", 1.0, 0.0) + rng.normal(0, 1e-3, 200)

    assert pooled_auroc(y, scores) > 0.99, "pooled should be fooled"
    # Every stratum has zero outcome variance -> nothing is estimable.
    assert stratified_auroc(y, scores, strata) is None


def test_stratified_auroc_ignores_strata_with_too_few_minority_items():
    """C-DEGEN: a stratum with one minority item must not drive the estimate.

    Stratum A is informative but tiny on one side (1 negative); stratum B is
    well populated and uninformative. Only B may count.
    """
    y = np.concatenate([np.ones(50), np.zeros(1), np.ones(25), np.zeros(25)])
    strata = np.array(["A"] * 51 + ["B"] * 50)
    # In A the single negative scores lowest => perfect separation.
    # In B scores are pure noise => ~0.5.
    rng = np.random.default_rng(2)
    scores = np.concatenate([np.full(50, 1.0), [-10.0], rng.normal(0, 1, 50)])

    auc = stratified_auroc(y, scores, strata, min_class_n=5)
    assert auc is not None
    assert abs(auc - 0.5) < 0.2, f"stratum A leaked in: {auc}"
    # With the floor removed, the 1-item stratum dominates and inflates it.
    naive = stratified_auroc(y, scores, strata, min_class_n=1)
    assert naive > auc


def test_paired_bootstrap_detects_a_planted_difference_and_respects_groups():
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n).astype(float)
    groups = np.repeat(np.arange(n // 2), 2)        # pairs move together
    strata = np.tile(["Yes", "No"], n // 2)
    weak = y * 0.3 + rng.normal(0, 1, n)
    strong = y * 1.5 + rng.normal(0, 1, n)

    res = paired_bootstrap(y, groups, weak, strong, strata=strata, n_boot=300)
    assert res["delta"] > 0
    assert res["ci95"][0] > 0, "CI should exclude zero for a real difference"
    assert res["p_one_sided"] < 0.05

    null = paired_bootstrap(y, groups, weak, weak.copy(), strata=strata, n_boot=300)
    assert abs(null["delta"]) < 1e-9
    assert null["p_one_sided"] >= 1.0 / 301, "p must be floored, never 0"


def test_benjamini_hochberg_known_case():
    # Classic BH example: with n=4 and alpha=0.05, thresholds are
    # .0125/.025/.0375/.05 -> the two smallest pass.
    assert benjamini_hochberg([0.01, 0.02, 0.30, 0.70], alpha=0.05) == \
        [True, True, False, False]
    assert benjamini_hochberg([0.9, 0.8], alpha=0.05) == [False, False]
    assert benjamini_hochberg([]) == []
    # None (an unestimable contrast) must never be reported as a discovery.
    assert benjamini_hochberg([None, 0.001], alpha=0.05) == [False, True]


def test_group_key_collapses_only_genuine_duplicates():
    """C-DUP: differing whitespace/case is the same question; different text
    is not."""
    assert group_key("What is 2+2?") == group_key("  what   is 2+2?  ")
    assert group_key("What is 2+2?") != group_key("What is 2+3?")
    assert normalize_text("  A  B ") == "a b"


def test_tost_separates_a_tight_null_from_an_underpowered_one():
    """The point of equivalence testing: 'no significant effect' must NOT be
    reported the same way when the CI is tight and when it is merely wide."""
    rng = np.random.default_rng(0)
    tight_null = rng.normal(0.0, 0.004, 2000)     # effect ruled out
    underpowered = rng.normal(0.0, 0.05, 2000)    # nothing ruled out
    real_effect = rng.normal(0.05, 0.004, 2000)

    assert tost_from_deltas(tight_null, band=0.02)["equivalent"] is True
    # An underpowered null is the failure mode this exists to expose: same
    # point estimate, but it cannot rule out an effect of the stated size.
    assert tost_from_deltas(underpowered, band=0.02)["equivalent"] is False
    assert tost_from_deltas(real_effect, band=0.02)["equivalent"] is False
    # No bootstrap samples -> no claim either way.
    assert tost_from_deltas(np.array([]), band=0.02)["equivalent"] is None


def test_delong_matches_the_bootstrap_only_without_clustering():
    """Why the grouped bootstrap stays the headline test.

    DeLong assumes independent items. This design violates that (pos/neg
    partners and the two orders of one pairwise item share a CV group), so
    DeLong understates the SE there and its p-values are anti-conservative.
    Adopting it as the primary test would manufacture significance.
    """
    rng = np.random.default_rng(0)

    # (a) One item per group: DeLong's assumption holds, SEs must agree.
    n = 4000
    y = rng.integers(0, 2, n).astype(float)
    noise = rng.normal(size=n)
    a, b = 0.5 * y + noise, 1.0 * y + noise
    st = np.zeros(n)
    d = delong_paired_stratified(y, a, b, st)
    bs = paired_bootstrap(y, np.arange(n), a, b, strata=st, n_boot=500, seed=1)
    se_boot = (bs["ci95"][1] - bs["ci95"][0]) / 3.92
    assert abs(d["se"] - se_boot) / se_boot < 0.25

    # (b) Four near-duplicate rows per question: the bootstrap must be wider.
    n_groups = 500
    g = np.repeat(np.arange(n_groups), 4)
    y2 = np.repeat(rng.integers(0, 2, n_groups).astype(float), 4)
    noise2 = np.repeat(rng.normal(size=n_groups), 4) \
        + 0.05 * rng.normal(size=len(g))
    a2, b2 = 0.5 * y2 + noise2, 1.0 * y2 + noise2
    st2 = np.zeros(len(g))
    d2 = delong_paired_stratified(y2, a2, b2, st2)
    bs2 = paired_bootstrap(y2, g, a2, b2, strata=st2, n_boot=500, seed=1)
    se_boot2 = (bs2["ci95"][1] - bs2["ci95"][0]) / 3.92
    assert se_boot2 > 1.5 * d2["se"]


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
