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
                                      paired_bootstrap, pooled_auroc,
                                      stratified_auroc)
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


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _main()
