"""Stage 21 (CPU) — minimum detectable effect from the pilot's own bootstrap.

Answers the question a reviewer asks first about a null result: *was this
study able to detect the effect it was looking for?*

The pilot already paid for the hard part. Every contrast in
`results/analysis_<name>.json` carries a paired grouped-bootstrap CI on the
delta, and the width of that CI **is** the standard error of the estimator at
the pilot's sample size — no analytic AUROC-variance approximation needed, and
it automatically accounts for the grouping, the stratification and the fact
that the two ROCs share items.

From SE at n_pilot we get, for a target n:

    SE(n)  ~  SE(n_pilot) * sqrt(n_pilot / n)          (root-n scaling)
    MDE    =  (z_{1-alpha} + z_{1-beta}) * SE(n)       (one-sided)

`alpha` is the *effective* level after multiplicity. Under BH-FDR the
threshold for the smallest p-value in a family of m is alpha/m, so a study
that must detect a single effect among m contrasts is powered at alpha/m —
which is why cutting the contrast family buys power for free, and why adding
judges (which multiplies contrasts) costs it.

Usage:
    python scripts/21_power.py                       # all analyses found
    python scripts/21_power.py --only llmbar
    python scripts/21_power.py --target-effect 0.04 --power 0.8
"""

import _bootstrap  # noqa: F401

import argparse
import math

from llm_judge.config import RESULTS_DIR, tagged
from llm_judge.io_utils import atomic_write_json, read_json
from llm_judge.log_utils import setup_logging

# Contrasts that carry the paper's primary claims. Everything else in the
# ladder is descriptive and does not need to be powered.
PRIMARY = ("M2 - M1nd", "M3 - M1nd", "M2 - M1n", "M3 - M1n", "M4 - M3")


def _z(p: float) -> float:
    """Normal quantile (Acklam-free: bisection on erf, plenty accurate here)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def se_from_ci(ci: list) -> float | None:
    """SE implied by a 95% bootstrap CI. Width = 2 * 1.96 * SE."""
    if not ci or ci[0] is None or ci[1] is None:
        return None
    return (float(ci[1]) - float(ci[0])) / (2 * 1.959963985)


def mde(se: float, alpha: float, power: float) -> float:
    return (_z(1 - alpha) + _z(power)) * se


def n_required(se_pilot: float, n_pilot: int, effect: float,
               alpha: float, power: float) -> int:
    """Items needed for `effect` to be detectable, by root-n scaling."""
    if effect <= 0:
        return -1
    se_needed = effect / (_z(1 - alpha) + _z(power))
    return int(math.ceil(n_pilot * (se_pilot / se_needed) ** 2))


def analyse(name: str, tag: str | None, target_effect: float, power: float,
            alpha_raw: float, log) -> dict:
    path = RESULTS_DIR / f"{tagged('analysis_' + name, tag)}.json"
    if not path.exists():
        log(f"[{name}] no analysis file at {path} — skipped")
        return {}
    report = read_json(path)

    fdr_path = RESULTS_DIR / f"{tagged('analysis_fdr', tag)}.json"
    fdr = read_json(fdr_path) if fdr_path.exists() else {}
    m = int(fdr.get("n_contrasts") or 0)
    alpha_eff = alpha_raw / m if m else alpha_raw

    log(f"\n=== {name} ===")
    log(f"  FDR family size m={m}; effective one-sided alpha for the smallest "
        f"p-value = {alpha_raw}/{m or 1} = {alpha_eff:.5f}")
    log(f"  target effect = {target_effect:.3f} conditional AUROC, "
        f"power = {power:.0%}")

    out: dict = {"family_size": m, "alpha_raw": alpha_raw,
                 "alpha_effective": alpha_eff, "target_effect": target_effect,
                 "power": power, "slices": {}}

    ses: list[float] = []
    for slice_name, rep in report.items():
        if not isinstance(rep, dict) or "contrasts" not in rep:
            continue
        n = int(rep.get("n_items") or 0)
        if rep.get("unreliable"):
            log(f"  {slice_name}: UNRELIABLE (contrasts suppressed) — "
                f"carries no power")
            out["slices"][slice_name] = {"n_items": n, "unreliable": True}
            continue

        rows = {}
        for cname, c in (rep.get("contrasts") or {}).items():
            se = se_from_ci(c.get("ci95"))
            if se is None:
                continue
            entry = {
                "n_items": n,
                "observed_delta": c.get("delta"),
                "se": se,
                "mde_at_pilot_n_uncorrected": mde(se, alpha_raw, power),
                "mde_at_pilot_n_fdr": mde(se, alpha_eff, power),
                "n_needed_uncorrected": n_required(se, n, target_effect,
                                                   alpha_raw, power),
                "n_needed_fdr": n_required(se, n, target_effect,
                                           alpha_eff, power),
            }
            rows[cname] = entry
            if cname in PRIMARY:
                ses.append(se)
        out["slices"][slice_name] = {"n_items": n, "unreliable": False,
                                     "contrasts": rows}

        prim = {k: v for k, v in rows.items() if k in PRIMARY}
        if prim:
            log(f"  {slice_name} (n={n}):")
            for cname, e in prim.items():
                log(f"    {cname:<14} delta={e['observed_delta']:+.3f}  "
                    f"SE={e['se']:.3f}  "
                    f"MDE(a={alpha_raw})={e['mde_at_pilot_n_uncorrected']:.3f}  "
                    f"MDE(FDR)={e['mde_at_pilot_n_fdr']:.3f}  "
                    f"n for {target_effect:.02f}: "
                    f"{e['n_needed_fdr']:,}")

    if ses:
        med = sorted(ses)[len(ses) // 2]
        n_med = max(
            (s["n_items"] for s in out["slices"].values()
             if not s.get("unreliable")), default=0)
        summary = {
            "n_primary_contrasts_measured": len(ses),
            "median_se": med,
            "median_slice_n": n_med,
            "mde_uncorrected": mde(med, alpha_raw, power),
            "mde_fdr": mde(med, alpha_eff, power),
            "n_needed_for_target_uncorrected": n_required(
                med, n_med, target_effect, alpha_raw, power),
            "n_needed_for_target_fdr": n_required(
                med, n_med, target_effect, alpha_eff, power),
        }
        out["summary"] = summary
        log(f"\n  SUMMARY across {len(ses)} primary contrasts "
            f"(median SE={med:.3f} at n={n_med}):")
        log(f"    detectable at alpha={alpha_raw} (no correction): "
            f"{summary['mde_uncorrected']:.3f} conditional-AUROC lift")
        log(f"    detectable under BH-FDR (m={m}):                "
            f"{summary['mde_fdr']:.3f}")
        log(f"    to detect {target_effect:.3f} under BH-FDR you need "
            f"~{summary['n_needed_for_target_fdr']:,} items per slice "
            f"({summary['n_needed_for_target_fdr'] / max(n_med, 1):.0f}x the "
            f"pilot)")
        log(f"    to detect {target_effect:.3f} with m=3 primary tests only: "
            f"~{n_required(med, n_med, target_effect, alpha_raw / 3, power):,}"
            f" items per slice")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--target-effect", type=float, default=0.04,
                    help="conditional-AUROC lift the study should detect")
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    log = setup_logging("21_power").info

    names = args.only
    if not names:
        names = sorted({
            p.stem.replace("analysis_", "").split("__")[0]
            for p in RESULTS_DIR.glob("analysis_*.json")
            if "fdr" not in p.stem
        })
    log(f"datasets: {', '.join(names)}")

    out = {}
    for name in names:
        res = analyse(name, args.tag, args.target_effect, args.power,
                      args.alpha, log)
        if res:
            out[name] = res

    path = RESULTS_DIR / f"{tagged('power', args.tag)}.json"
    atomic_write_json(path, out, tag=args.tag)
    log(f"\nwrote {path}")


if __name__ == "__main__":
    main()
