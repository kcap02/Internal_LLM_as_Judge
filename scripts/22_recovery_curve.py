"""Stage 22 (CPU) — the ladder estimator's detection threshold, as a function of n.

Why this stage exists. A delta of zero from the ladder is uninterpretable on
its own. This project produced three estimators that each passed a planted test
and each failed outside the region the plant covered:

  * concatenation passed at a 32-column block and destroys 0.315 AUROC of real
    signal at 4096 columns (C-LADDER);
  * offset with cross-validated ridge passed with a strong planted latent and,
    at n=200 with 1536 columns, returned exactly zero for every block on every
    slice — a zero manufactured by the estimator;
  * the first version of the recovery-curve test itself passed on jitter,
    because its plant sat below what was recoverable at that n.

The generalisation: **a passing test on planted data certifies an estimator
only over the region the plant covers.** So the acceptance criterion is a
curve, not a point, and the curve has to be measured at the n and the block
width actually in use.

What this measures. For a given n and block width, plant a signal of varying
strength along a dense random direction (how an activation encodes anything;
a single-column plant is near worst case and understates the estimator), run
the real `oof_scores_offset`, and record both the measured delta and the
block-only AUROC. The **detection threshold** is the smallest block-only AUROC
at which the delta becomes reliably positive.

What it is for. `N_FREEZE` (prereg §8) is the smallest n whose detection
threshold falls *below* the largest block-only AUROC observed in the pilot
(`M3only`, currently 0.633). Below that n, the ladder cannot resolve primary
test 1 no matter what the bootstrap SE says, and the result is unresolvable
rather than null.

Usage:
    python scripts/22_recovery_curve.py                       # default sweep
    python scripts/22_recovery_curve.py --n 200 400 800 1600 --width 1536
    python scripts/22_recovery_curve.py --target 0.633        # pilot M3only
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from llm_judge.analysis.cv import oof_scores, oof_scores_offset
from llm_judge.analysis.stats import stratified_auroc
from llm_judge.config import RESULTS_DIR, tagged
from llm_judge.io_utils import atomic_write_json
from llm_judge.log_utils import setup_logging

# Delta below this is treated as no detection. Matches the preregistered
# equivalence band, so "detected" and "not equivalent to zero" agree.
DETECT_DELTA = 0.02


def one_setting(n_items: int, width: int, strength: float, seed: int,
                n_splits: int) -> tuple[float, float, float]:
    """Return (baseline, delta, block_only) for one planted magnitude."""
    rng = np.random.default_rng(seed)
    n_q = n_items // 2
    groups = np.repeat(np.arange(n_q), 2)
    n = len(groups)
    strata = np.tile(["A", "B"], n_q)

    latent = rng.normal(size=n)
    extra = rng.normal(size=n)
    y = ((latent + 0.9 * extra + rng.normal(0, 0.7, n)) > 0).astype(float)
    # A baseline that already explains a lot — the regime where both ladder
    # failures happened, since it leaves the block only a small residual.
    X_base = np.c_[1.2 * latent + rng.normal(0, .6, n), rng.normal(size=(n, 5))]

    direction = rng.normal(size=width)
    direction /= np.linalg.norm(direction)
    X_add = rng.normal(size=(n, width)) + strength * np.outer(extra, direction)

    base = stratified_auroc(y, oof_scores(X_base, y, groups, n_splits), strata)
    got = stratified_auroc(
        y, oof_scores_offset(X_base, X_add, y, groups, n_splits), strata)
    alone = stratified_auroc(
        y, oof_scores(X_add, y, groups, n_splits, n_act=width), strata)
    return base, (got - base), alone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="*", type=int, default=[200, 400, 800, 1600])
    ap.add_argument("--width", type=int, default=1536,
                    help="block width; 1536 = Qwen2.5-1.5B hidden size")
    ap.add_argument("--strengths", nargs="*", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--target", type=float, default=0.633,
                    help="largest block-only AUROC observed in the pilot")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    log = setup_logging("22_recovery_curve").info
    log(f"block width {args.width}, target block-only AUROC {args.target:.3f}, "
        f"detection at |delta| >= {DETECT_DELTA}")

    out: dict = {"width": args.width, "target_block_only": args.target,
                 "detect_delta": DETECT_DELTA, "by_n": {}}
    thresholds: dict[int, float | None] = {}

    for n_items in args.n:
        log(f"\n=== n = {n_items} ===")
        log(f"{'plant':>7} {'baseline':>9} {'delta':>9} {'block-only':>11}"
            f" {'detected':>9}")
        rows, thr = [], None
        for s in args.strengths:
            base, delta, alone = one_setting(n_items, args.width, s,
                                             args.seed, args.n_splits)
            det = delta >= DETECT_DELTA
            rows.append({"strength": s, "baseline": base, "delta": delta,
                         "block_only": alone, "detected": bool(det)})
            log(f"{s:>7.2f} {base:>9.3f} {delta:>+9.4f} {alone:>11.3f}"
                f" {'yes' if det else 'no':>9}")
            if det and thr is None:
                thr = alone
        thresholds[n_items] = thr
        out["by_n"][str(n_items)] = {"rows": rows, "detection_threshold": thr}
        if thr is None:
            log(f"  no detection at any planted strength -> threshold above "
                f"{max(r['block_only'] for r in rows):.3f}")
        else:
            log(f"  detection threshold: block-only AUROC ~{thr:.3f}")

    ok = [n for n, t in thresholds.items() if t is not None and t <= args.target]
    out["n_freeze_candidate"] = min(ok) if ok else None
    log("\n" + "=" * 62)
    if ok:
        log(f"N_FREEZE candidate = {min(ok)} items per stratum — the smallest "
            f"n whose detection threshold falls below the pilot's observed "
            f"block-only AUROC of {args.target:.3f}.")
    else:
        log(f"NO tested n reaches a detection threshold below {args.target:.3f}. "
            f"If no affordable n does, the ladder result is preregistered as "
            f"UNRESOLVABLE and the paper says so in the abstract (prereg §8).")

    p = RESULTS_DIR / f"{tagged('recovery_curve', args.tag)}.json"
    atomic_write_json(p, out)
    log(f"wrote {p.name}")


if __name__ == "__main__":
    main()
