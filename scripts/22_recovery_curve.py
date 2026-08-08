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

**The canonical output path is protected structurally, not by convention.**
`results/recovery_curve.json` is what `paper/fill_numbers.py` resolves the
paper's detection-threshold macros from. A rigged failure run used to verify
the PASS/FAIL logic (`--n 50 100 --target 0.999`) once overwrote it, and the
only reason this was noticed is that an unrelated macro downstream happened to
be flagged; had it not been, the paper would have printed thresholds from a run
designed to detect nothing, with every check in the repo passing.

The verification path and the thing verified must not share a write target. So
this script writes to a **tagged** path by default, and will only write the
canonical one when `--canonical` is given *and* every parameter matches
`PREREG_CONFIG` below. Any exploratory sweep therefore cannot reach it.
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

# The preregistered sweep. `--canonical` writes results/recovery_curve.json,
# which the paper reads, and is refused unless every value below matches. This
# is the structural form of "exploratory runs must be tagged": a rigged or
# reduced sweep cannot reach the canonical path even if the operator forgets.
PREREG_CONFIG = {
    "n": [200, 400, 800, 1600],
    "width": 2048,   # Qwen2.5-3B hidden size: the spectral arm ceiling (prereg §3)
    "strengths": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "target": 0.633,
    "pilot_n": 200,
    "detect_frac": 0.80,
    "n_splits": 5,
    "seeds": 12,
}


def canonical_mismatches(args) -> list[str]:
    """Which arguments differ from the preregistered sweep."""
    got = {"n": list(args.n), "width": args.width,
           "strengths": list(args.strengths), "target": args.target,
           "pilot_n": args.pilot_n, "detect_frac": args.detect_frac,
           "n_splits": args.n_splits, "seeds": args.seeds}
    return [f"{k}: {got[k]!r} != {v!r}"
            for k, v in PREREG_CONFIG.items() if got[k] != v]


def iqr_separations(by_n: dict, ns: list, key: str = "block_only_iqr"):
    """Which consecutive n-pairs have DISJOINT IQRs.

    A directional claim about `key` is permitted only where a pair separates.
    Extracted so the guard itself is unit-testable: an acceptance criterion
    that has only ever returned PASS is not yet known to be one
    (`tests/test_ladder.py:test_recovery_guard_*`).
    """
    out = []
    for a, b in zip(ns, ns[1:]):
        la, ha = by_n[str(a)][key]
        lb, hb = by_n[str(b)][key]
        out.append((a, b, bool(lb > ha or la > hb)))
    return out


def make_world(n_max: int, width: int, strength: float, seed: int) -> dict:
    """One realisation of the whole problem, generated ONCE at the largest n.

    Cells at different n are then **nested subsets of this same realisation**,
    which is what makes the monotonicity check meaningful. Regenerating per n
    from `seed` alone does not do that: the generator consumes a different
    number of draws at each n, so the cells are unrelated realisations and
    cannot be compared as though they tracked one signal.
    """
    rng = np.random.default_rng(seed)
    n_q = max(2, n_max // 2)
    n = 2 * n_q
    latent = rng.normal(size=n)
    extra = rng.normal(size=n)
    y = ((latent + 0.9 * extra + rng.normal(0, 0.7, n)) > 0).astype(float)
    # A baseline that already explains a lot — the regime where both ladder
    # failures happened, since it leaves the block only a small residual.
    X_base = np.c_[1.2 * latent + rng.normal(0, .6, n), rng.normal(size=(n, 5))]
    direction = rng.normal(size=width)
    direction /= np.linalg.norm(direction)
    X_add = rng.normal(size=(n, width)) + strength * np.outer(extra, direction)
    return {"y": y, "X_base": X_base, "X_add": X_add,
            "groups": np.repeat(np.arange(n_q), 2),
            "strata": np.tile(["A", "B"], n_q)}


def evaluate(world: dict, n_items: int, n_splits: int, width: int,
             want_delta: bool = True):
    """Score a nested prefix of `world`. Returns (delta or None, block_only)."""
    n = min(2 * (n_items // 2), len(world["y"]))
    y, groups = world["y"][:n], world["groups"][:n]
    strata, X_add = world["strata"][:n], world["X_add"][:n]
    X_base = world["X_base"][:n]

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
    ap.add_argument("--width", type=int, default=2048)
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
    ap.add_argument("--canonical", action="store_true",
                    help="write results/recovery_curve.json, which the paper "
                         "reads. Refused unless every parameter matches "
                         "PREREG_CONFIG.")
    args = ap.parse_args()

    # Resolve the write target BEFORE spending the compute, so a refusal is
    # immediate rather than discovered after the sweep.
    if args.canonical:
        bad = canonical_mismatches(args)
        if bad:
            raise SystemExit(
                "refusing --canonical: parameters differ from the "
                "preregistered sweep:\n  " + "\n  ".join(bad)
                + "\nRun without --canonical (results are written to a tagged "
                  "path) or match PREREG_CONFIG.")
        if args.tag:
            raise SystemExit("--canonical and --tag are mutually exclusive")
        write_tag = None
    else:
        write_tag = args.tag or "exploratory"

    log = setup_logging("22_recovery_curve").info
    log(f"width {args.width} | {args.seeds} seeds/cell | detection at "
        f"delta >= {DETECT_DELTA} in >= {args.detect_frac:.0%} of seeds")

    # ── Phase 1: calibrate plant strength against the pilot's observation ──
    log(f"\n=== calibration: block-only AUROC at the pilot's n="
        f"{args.pilot_n} ===")
    log(f"{'plant':>7} {'median block-only':>19} {'IQR':>16}")
    cal = []
    for s in args.strengths:
        vals = [evaluate(make_world(args.pilot_n, args.width, s, 1000 + k),
                         args.pilot_n, args.n_splits, args.width,
                         want_delta=False)[1]
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
    log("  cells are PAIRED: one realisation per seed, evaluated at every n as "
        "a nested prefix,\n  so the columns track one underlying signal rather "
        "than unrelated draws.")
    log(f"{'n':>6} {'median delta':>14} {'delta IQR':>18} {'detect':>8}"
        f" {'block-only':>12} {'block IQR':>18}")

    n_max = max(args.n)
    worlds = [make_world(n_max, args.width, s_star, 2000 + k)
              for k in range(args.seeds)]
    per_seed: dict[int, list] = {n: [] for n in args.n}
    per_seed_alone: dict[int, list] = {n: [] for n in args.n}
    for w in worlds:
        for n_items in args.n:
            d, a = evaluate(w, n_items, args.n_splits, args.width)
            if d is not None:
                per_seed[n_items].append(d)
            if a is not None:
                per_seed_alone[n_items].append(a)

    by_n, threshold = {}, None
    for n_items in args.n:
        ds, alones = per_seed[n_items], per_seed_alone[n_items]
        if not ds:
            continue
        med_d = float(np.median(ds))
        d_lo, d_hi = np.percentile(ds, [25, 75])
        a_lo, a_hi = (np.percentile(alones, [25, 75]) if alones else (0., 0.))
        frac = float(np.mean([d >= DETECT_DELTA for d in ds]))
        med_a = float(np.median(alones)) if alones else float("nan")
        by_n[str(n_items)] = {"median_delta": med_d, "detect_fraction": frac,
                              "median_block_only": med_a, "n_seeds": len(ds),
                              "delta_iqr": [float(d_lo), float(d_hi)],
                              "block_only_iqr": [float(a_lo), float(a_hi)],
                              "deltas": [float(d) for d in ds]}
        log(f"{n_items:>6} {med_d:>+14.4f} "
            f"{f'[{d_lo:+.3f}, {d_hi:+.3f}]':>18} {frac:>8.0%}"
            f" {med_a:>12.3f} {f'[{a_lo:.3f}, {a_hi:.3f}]':>18}")
        if threshold is None and frac >= args.detect_frac:
            threshold = n_items

    out = {"width": args.width, "target_block_only": args.target,
           "pilot_n": args.pilot_n, "detect_delta": DETECT_DELTA,
           "detect_fraction_required": args.detect_frac,
           "seeds_per_cell": args.seeds,
           "calibration": cal, "calibrated_strength": s_star,
           "calibration_note": note, "by_n": by_n,
           "n_freeze_candidate": threshold}

    # ── Pass condition, evaluated by the script, not by the reader ────────
    # The third half of prereg §4.1. Every estimator in this pipeline has a
    # pre-stated acceptance criterion; the tool that validates them did not,
    # and a table read afterwards for "did the fix work" is the same act as
    # reading one for "did the hypothesis hold". Both of this project's
    # misreadings of THIS table — a non-monotone block-only treated as a
    # defect, then a monotone rise treated as its resolution — appeared
    # immediately after an argument that predicted them.
    log("\n" + "=" * 66)
    verdict = {}

    verdict["detection"] = threshold is not None
    if threshold is not None:
        log(f"[PASS] detection: n={threshold} reaches "
            f">= {args.detect_frac:.0%} of {args.seeds} seeds. "
            f"N_FREEZE candidate = {threshold} items per stratum.")
    else:
        log(f"[FAIL] detection: NO tested n reaches "
            f">= {args.detect_frac:.0%} of seeds. If no affordable n does, "
            f"primary test 1 is preregistered as UNRESOLVABLE and the abstract "
            f"says so (prereg §8.2).")

    # Monotonicity in block-only may be CLAIMED only where consecutive IQRs
    # are disjoint. Otherwise the script says so and no trend is reported.
    ns = [n for n in args.n if str(n) in by_n]
    sep = iqr_separations(by_n, ns)
    verdict["block_only_trend_claimable"] = any(s for *_, s in sep)
    if verdict["block_only_trend_claimable"]:
        pairs = ", ".join(f"{a}->{b}" for a, b, s in sep if s)
        log(f"[NOTE] block-only: IQRs separate at {pairs}; a trend may be "
            f"claimed there and only there.")
    else:
        log("[NOTE] block-only: NO consecutive IQRs separate — "
            "NO MONOTONICITY CLAIM IS PERMITTED from this table. Any apparent "
            "rise or fall is within sampling noise.")

    # The calibration must reproduce its target on INDEPENDENT draws. Checking
    # it against the sweep's own n=200 cell is not a check: that cell shares
    # its seeds with the sweep, so a low draw there produces a "confirmed" low
    # reading. (This bit us: two sweeps both reported block-only 0.574 at
    # n=200, which read as replication and was the same seed set twice.)
    cal_seeds = 9000
    fresh = [evaluate(make_world(args.pilot_n, args.width, s_star,
                                 cal_seeds + k),
                      args.pilot_n, args.n_splits, args.width,
                      want_delta=False)[1]
             for k in range(args.seeds)]
    fresh = [v for v in fresh if v is not None]
    cal_ok = None
    if fresh:
        med = float(np.median(fresh))
        lo, hi = np.percentile(fresh, [25, 75])
        cal_ok = bool(lo <= args.target <= hi)
        out["calibration_self_check"] = {
            "seed_offset": cal_seeds, "median": med,
            "iqr": [float(lo), float(hi)], "n_seeds": len(fresh)}
        log(f"[{'PASS' if cal_ok else 'WARN'}] calibration self-check on "
            f"{len(fresh)} INDEPENDENT worlds at n={args.pilot_n}: "
            f"block-only {med:.3f} IQR [{lo:.3f}, {hi:.3f}] against target "
            f"{args.target:.3f}"
            + ("" if cal_ok else " — target OUTSIDE the IQR; s* is not "
               "delivering the intended plant"))
    verdict["calibration_self_check"] = cal_ok

    out["verdict"] = verdict
    out["canonical"] = bool(args.canonical)
    p = RESULTS_DIR / f"{tagged('recovery_curve', write_tag)}.json"
    atomic_write_json(p, out, tag=write_tag)
    log(f"wrote {p.name}"
        + ("" if args.canonical else "  (NOT canonical: the paper reads "
                                     "recovery_curve.json, written only "
                                     "under --canonical)"))

    if not verdict["detection"] or cal_ok is False:
        log("\nOVERALL: FAIL — do not use this sweep to set N_FREEZE.")
        raise SystemExit(1)
    log("\nOVERALL: PASS")


if __name__ == "__main__":
    main()
