"""Stage 22 (CPU) — the ladder estimator's detection threshold, as a function of n.

Why this stage exists. A delta of zero from the ladder is uninterpretable on
its own, and so is a small standard error. This project produced four
validations that each returned a healthy-looking number about a degenerate
object:

  * concatenation passed a planted test at a 32-column block and destroys
    0.315 AUROC of real signal at 4096 columns (C-LADDER);
  * offset with cross-validated ridge passed with a strong planted latent and,
    at n=200 with 1536 columns, returned exactly zero for every block on every
    slice — a zero manufactured by the estimator;
  * the concat block diagnostic reported healthy coefficient ratios (0.68-1.64)
    while the estimator the claims used was returning nothing;
  * the bootstrap MDE reported a median SE of 0.006 and "72 items per slice"
    *because* the rungs sat on top of the baseline.

Two generalisations follow, and both are in prereg §4.1: a passing test on
planted data certifies an estimator only over the region the plant covers, and
a **precision** estimate is uninterpretable without a **recovery** estimate
beside it.

--------------------------------------------------------------------------
Design of the measurement, and why it is not the obvious one
--------------------------------------------------------------------------

The naive version plants a signal, reads off the block-only AUROC that plant
produced at that n, and calls the smallest detected one the threshold. That is
ill-posed: block-only AUROC is itself estimated at that n, and at n=200 with
1536 columns it is biased *downward* by the same shrinkage that suppresses the
delta. A threshold read at n=800 and the pilot's `M3only` read at n=200 are
then two different estimation regimes being compared as if they were one.

So the curve is indexed on the **planted** strength, which is a fixed property
of the simulation, and the pilot's observation is translated into that index
by a separate calibration:

  Phase 1 (calibrate).  At the pilot's own n, measure block-only AUROC across
      plant strengths. Interpolate the strength whose block-only reading at
      that n equals the pilot's observed value. This is the plant that "looks
      like" the real data *through the same biased lens*.
  Phase 2 (sweep).      Hold that plant strength fixed and vary n. Report the
      fraction of seeds at which the delta clears the detection band.

`N_FREEZE` is the smallest n at which detection occurs in at least
`--detect-frac` of seeds. Every cell is averaged over `--seeds` draws, because
the threshold is by definition the quantity most sensitive to single-draw
noise, and it is about to determine how much GPU gets bought.

Usage:
    python scripts/22_recovery_curve.py
    python scripts/22_recovery_curve.py --n 200 400 800 1600 3200 --seeds 15
    python scripts/22_recovery_curve.py --target 0.633 --width 1536
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from llm_judge.analysis.cv import oof_scores, oof_scores_offset
from llm_judge.analysis.stats import stratified_auroc
from llm_judge.config import RESULTS_DIR, tagged
from llm_judge.io_utils import atomic_write_json
from llm_judge.log_utils import setup_logging

# Delta at or above this counts as a detection. Matches the preregistered
# equivalence band, so "detected" and "not equivalent to zero" agree.
DETECT_DELTA = 0.02


def simulate(n_items: int, width: int, strength: float, seed: int,
             n_splits: int, want_delta: bool = True):
    """One draw. Returns (delta or None, block_only)."""
    rng = np.random.default_rng(seed)
    n_q = max(2, n_items // 2)
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

    alone = stratified_auroc(
        y, oof_scores(X_add, y, groups, n_splits, n_act=width), strata)
    if not want_delta:
        return None, alone
    base = stratified_auroc(y, oof_scores(X_base, y, groups, n_splits), strata)
    got = stratified_auroc(
        y, oof_scores_offset(X_base, X_add, y, groups, n_splits), strata)
    if base is None or got is None:
        return None, alone
    return got - base, alone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="*", type=int,
                    default=[200, 400, 800, 1600, 3200])
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--strengths", nargs="*", type=float,
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--target", type=float, default=0.633,
                    help="largest block-only AUROC observed in the pilot")
    ap.add_argument("--pilot-n", type=int, default=200,
                    help="n at which the pilot value was measured")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--detect-frac", type=float, default=0.80)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    log = setup_logging("22_recovery_curve").info
    log(f"width {args.width} | {args.seeds} seeds/cell | detection at "
        f"delta >= {DETECT_DELTA} in >= {args.detect_frac:.0%} of seeds")

    # ── Phase 1: calibrate plant strength against the pilot's observation ──
    log(f"\n=== calibration: block-only AUROC at the pilot's n="
        f"{args.pilot_n} ===")
    log(f"{'plant':>7} {'median block-only':>19} {'IQR':>16}")
    cal = []
    for s in args.strengths:
        vals = [simulate(args.pilot_n, args.width, s, 1000 + k,
                         args.n_splits, want_delta=False)[1]
                for k in range(args.seeds)]
        vals = [v for v in vals if v is not None]
        med = float(np.median(vals))
        lo, hi = np.percentile(vals, [25, 75])
        cal.append({"strength": s, "median_block_only": med,
                    "iqr": [float(lo), float(hi)]})
        log(f"{s:>7.2f} {med:>19.3f} {f'[{lo:.3f}, {hi:.3f}]':>16}")

    xs = [c["strength"] for c in cal]
    ys = [c["median_block_only"] for c in cal]
    if args.target <= min(ys):
        s_star = min(xs)
        note = "target is at or below the zero-plant reading"
    elif args.target >= max(ys):
        s_star = max(xs)
        note = ("target EXCEEDS the largest plant tested — extend --strengths; "
                "s* is a lower bound")
    else:
        s_star = float(np.interp(args.target, ys, xs))
        note = "interpolated"
    log(f"\ncalibrated plant strength for a pilot reading of {args.target:.3f}"
        f": s* = {s_star:.3f}  ({note})")

    # ── Phase 2: sweep n at the calibrated plant ──────────────────────────
    log(f"\n=== sweep at s* = {s_star:.3f} ===")
    log(f"{'n':>6} {'median delta':>14} {'detect frac':>12} {'median block-only':>19}")
    by_n, threshold = {}, None
    for n_items in args.n:
        ds, alones = [], []
        for k in range(args.seeds):
            d, a = simulate(n_items, args.width, s_star, 2000 + k,
                            args.n_splits)
            if d is not None:
                ds.append(d)
            if a is not None:
                alones.append(a)
        if not ds:
            continue
        med_d = float(np.median(ds))
        frac = float(np.mean([d >= DETECT_DELTA for d in ds]))
        med_a = float(np.median(alones)) if alones else float("nan")
        by_n[str(n_items)] = {"median_delta": med_d, "detect_fraction": frac,
                              "median_block_only": med_a, "n_seeds": len(ds),
                              "deltas": [float(d) for d in ds]}
        log(f"{n_items:>6} {med_d:>+14.4f} {frac:>12.0%} {med_a:>19.3f}")
        if threshold is None and frac >= args.detect_frac:
            threshold = n_items

    out = {"width": args.width, "target_block_only": args.target,
           "pilot_n": args.pilot_n, "detect_delta": DETECT_DELTA,
           "detect_fraction_required": args.detect_frac,
           "seeds_per_cell": args.seeds,
           "calibration": cal, "calibrated_strength": s_star,
           "calibration_note": note, "by_n": by_n,
           "n_freeze_candidate": threshold}

    log("\n" + "=" * 66)
    if threshold is not None:
        log(f"N_FREEZE candidate = {threshold} items per stratum — the smallest "
            f"n detecting a plant calibrated to the pilot's own block-only "
            f"reading, in >= {args.detect_frac:.0%} of {args.seeds} seeds.")
    else:
        log(f"NO tested n detects at >= {args.detect_frac:.0%} of seeds. If no "
            f"affordable n does, primary test 1 is preregistered as "
            f"UNRESOLVABLE and the abstract says so (prereg §8.2).")

    p = RESULTS_DIR / f"{tagged('recovery_curve', args.tag)}.json"
    atomic_write_json(p, out)
    log(f"wrote {p.name}")


if __name__ == "__main__":
    main()
