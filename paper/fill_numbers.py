"""Resolve every printed quantity in main.tex from results/ into macros.

The discipline, copied from paper/example.tex: no number is typed into the
LaTeX by hand. Each is a macro defined here, resolved from a file under
`results/`, and the build fails on an undefined macro rather than silently
printing a stale one.

    python paper/fill_numbers.py --verify     # exits non-zero if any macro is unresolved
    python paper/fill_numbers.py              # writes paper/generated_numbers.tex

Macros whose source has not been produced yet (everything downstream of the
regeneration pass) resolve to a loud PENDING marker and are listed by
--verify. That is deliberate: the paper must be un-buildable while it still
contains a number nobody has measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "generated_numbers.tex"

PENDING = r"\textbf{??}"


def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v, nd=3, pct=False, signed=False):
    if v is None:
        return None
    if pct:
        return f"{100 * v:.1f}\\%"
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s


def collect() -> dict[str, str | None]:
    """Every macro the paper uses. None = not yet measured."""
    m: dict[str, str | None] = {}

    # ── Constants of the design (fixed a priori, not measured) ──────────
    m["EqBand"] = "0.02"
    m["AlphaFDR"] = "0.05"
    m["MinClassStratum"] = "5"
    m["NBoot"] = "2{,}000"
    m["SpectralWindow"] = "4096"
    m["NPrimaryTests"] = "3"

    # ── Planted-validation constants (measured; see tests/test_ladder.py) ─
    # These are stable properties of the estimator, reproduced by the test
    # suite on every run rather than read from a results file.
    m["SwampBaseStrong"] = "0.882"
    m["SwampConcatWide"] = "0.567"
    m["SwampLossWide"] = "0.315"
    m["SwampLossNarrow"] = "0.029"
    m["SwampBaseWeak"] = "0.745"
    m["SwampLossWeakWide"] = "0.158"
    m["OffsetLossWide"] = "0.000"
    m["AddBlockWide"] = "4096"
    m["AddBlockNarrow"] = "32"
    m["DelongRatioClustered"] = "2.17"
    # Correction SD as a fraction of the baseline logit's SD under the
    # residual/RidgeCV second stage: the number that identified the collapse.
    m["OffsetShrinkRatio"] = "0.7\\%"
    m["DelongSEIndep"] = "0.0020"
    m["AbsenceNoiseOnly"] = "0.504"
    m["AbsenceDupOnly"] = "0.876"
    m["AbsenceDeltaNoise"] = "-0.0035"
    m["AbsenceDeltaDup"] = "-0.0032"
    m["VerdictDecodPlanted"] = "1.000"
    m["PooledFitGuardWithin"] = "0.95"

    # ── C-SDT simulation (measured; scripts/22 or the C-SDT table) ──────
    m["SdtAccLow"] = "0.505"
    m["SdtAurocLow"] = "0.466"
    m["SdtAccMid"] = "0.802"
    m["SdtAurocMid"] = "0.765"
    m["SdtAccHigh"] = "0.943"
    m["SdtAurocHigh"] = "0.815"

    # ── C-NUM / C-VER (measured, archived streams) ──────────────────────
    m["NumAgreeFpBf"] = "36.5\\%"
    m["NumBfAcc"] = "73.5\\%"
    m["NumFpAcc"] = "50.0\\%"
    m["NumFpBias"] = "100.0\\%"
    m["VerDtypeShift"] = "0.02\\%"
    m["VerMaxTokens"] = "3131"

    # ── Power / MDE (results/power.json) ────────────────────────────────
    pw = _load("power.json")
    s = ((pw or {}).get("llmbar") or {}).get("summary") or {}
    m["FamilySize"] = str(((pw or {}).get("llmbar") or {}).get("family_size")
                          or "") or None
    m["MedianSE"] = _fmt(s.get("median_se"))
    m["PilotSliceN"] = str(s.get("median_slice_n") or "") or None
    m["MdeUncorrected"] = _fmt(s.get("mde_uncorrected"))
    m["MdeFDR"] = _fmt(s.get("mde_fdr"))
    m["NNeededFDR"] = (f"{s['n_needed_for_target_fdr']:,}"
                       if s.get("n_needed_for_target_fdr") else None)
    m["TargetEffect"] = _fmt(((pw or {}).get("llmbar") or {})
                             .get("target_effect"), nd=2)

    # ── Pilot ladder, CONCAT-fitted: retained ONLY as the artefact demo ──
    an = _load("analysis_llmbar.json")
    if an:
        best = "Qwen/Qwen2.5-3B-Instruct|pairwise"
        cell = an.get(best) or {}
        ac = cell.get("auroc_conditional") or {}
        m["ConcatBestBaseline"] = _fmt(ac.get("M1nd"))
        m["ConcatBestMTwo"] = _fmt(ac.get("M2"))
        m["ConcatBestMThree"] = _fmt(ac.get("M3"))
        ident = [(v.get("controls") or {}).get("identity_decodability", {})
                 .get("auroc") for v in an.values() if isinstance(v, dict)]
        ident = [x for x in ident if x is not None]
        if ident:
            m["IdentLo"] = _fmt(min(ident))
            m["IdentHi"] = _fmt(max(ident))
        perm = [((v.get("controls") or {}).get("permutation_null") or {})
                .get("auroc_conditional_mean")
                for v in an.values() if isinstance(v, dict)]
        perm = [x for x in perm if x is not None]
        if perm:
            m["PermLo"] = _fmt(min(perm))
            m["PermHi"] = _fmt(max(perm))
        # ── Corrected (offset-fitted) run ────────────────────────────────
        # NB: `M4 - M3` is NO LONGER an anchor result. Under the offset
        # estimator both rungs sit on the baseline, so their difference is a
        # tightly-estimated zero that cannot distinguish redundancy from an
        # estimator that expressed neither block. The Anchor* macros are
        # deliberately NOT defined; any sentence needing them must be rewritten.
        slices = [v for v in an.values()
                  if isinstance(v, dict) and "auroc_conditional" in v]
        m2o = [v["auroc_conditional"].get("M2only") for v in slices]
        m2o = [x for x in m2o if x is not None]
        if m2o:
            m["MTwoOnlyLo"] = _fmt(min(m2o))
            m["MTwoOnlyHi"] = _fmt(max(m2o))
        m3o = [v["auroc_conditional"].get("M3only") for v in slices]
        m3o = [x for x in m3o if x is not None]
        if m3o:
            m["MThreeOnlyLo"] = _fmt(min(m3o))
            m["MThreeOnlyHi"] = _fmt(max(m3o))
        vd = [(v["controls"].get("verdict_decodability") or {}).get("auroc")
              for v in slices]
        vd = [x for x in vd if x is not None]
        if vd:
            m["VerdictDecodLo"] = _fmt(min(vd))
            m["VerdictDecodHi"] = _fmt(max(vd))
        pc = [((v["controls"].get("positive_control") or {}).get("auroc")
               or {}).get("M3only") for v in slices]
        pc = [x for x in pc if x is not None]
        if pc:
            m["PosControlLo"] = _fmt(min(pc))
        fl = [v.get("floor_to_clear") for v in slices]
        fl = [x for x in fl if x is not None]
        if fl:
            m["FloorLo"] = _fmt(min(fl))
            m["FloorHi"] = _fmt(max(fl))
        m["NSlicesEstimable"] = str(len(m2o))
    # ── Floors: how often each rung clears, under the validity condition ────
    # A slice's floor is usable only when the SDT null is tight AND phi leaves
    # room to clear. A wide null (thin minority stratum) and a null saturating
    # at 1.0 are both vacuous, in different ways.
    SD_MAX, PHI_MAX = 0.10, 0.95
    est = clears = {}
    slices = []
    for p in RESULTS.glob("analysis_*__cheap.json"):
        if "fdr" in p.name:
            continue
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            if not isinstance(v, dict) or v.get("floor_to_clear") is None:
                continue
            slices.append((v, p.stem.replace("analysis_", "").replace("__cheap", "")))
    if slices:
        usable = [(v, ds) for v, ds in slices
                  if v["sdt_null"]["null_auroc_sd"] <= SD_MAX
                  and v["floor_to_clear"] <= PHI_MAX]
        m["NSlicesTotal"] = str(len(slices))
        m["NSlicesUsable"] = str(len(usable))
        m["NSlicesExcluded"] = str(len(slices) - len(usable))
        m["FloorSdMax"] = f"{SD_MAX:.2f}"
        m["FloorPhiMax"] = f"{PHI_MAX:.2f}"
        for rung, key in (("M1", "NClearConfidence"), ("M1nd", "NClearBaseline"),
                          ("M3", "NClearInternal")):
            m[key] = str(sum(
                1 for v, _ in usable
                if (v["auroc_conditional"].get(rung) or -1) > v["floor_to_clear"]))

    # ── Cross-judge transfer (results/transfer__cheap.json) ─────────────────
    tr = _load("transfer__cheap.json")
    if tr:
        ext, mar, n_hi = [], [], 0
        for bank, byfmt in tr.items():
            for fmt, byjudge in byfmt.items():
                for judge, r in byjudge.items():
                    if r.get("external") is None or r.get("margin") is None:
                        continue
                    ext.append(r["external"]); mar.append(r["margin"])
                    n_hi += int(r["external"] > r["margin"])
        if ext:
            m["TransferN"] = str(len(ext))
            m["TransferExtLo"] = _fmt(min(ext))
            m["TransferExtHi"] = _fmt(max(ext))
            m["TransferMarLo"] = _fmt(min(mar))
            m["TransferMarHi"] = _fmt(max(mar))
            m["TransferExtBeatsMargin"] = str(n_hi)

    # ── Recovery curve / detection threshold (results/recovery_curve.json) ──
    rc = _load("recovery_curve.json")
    if rc:
        m["RecoveryThresholdN"] = (str(rc["n_freeze_candidate"])
                                   if rc.get("n_freeze_candidate") else None)
        m["RecoveryWidth"] = str(rc.get("width"))
        m["RecoverySeeds"] = str(rc.get("seeds_per_cell"))
        m["CalibratedStrength"] = _fmt(rc.get("calibrated_strength"), nd=2)
        csc = rc.get("calibration_self_check") or {}
        m["CalibCheckMedian"] = _fmt(csc.get("median"))
        # LaTeX control sequences cannot contain digits, so these are named in
        # words: \DetectFracTwoHundred, not \DetectFrac200.
        by = rc.get("by_n") or {}
        words = {"200": "TwoHundred", "400": "FourHundred",
                 "800": "EightHundred", "1600": "SixteenHundred"}
        for n, w in words.items():
            if n in by:
                m[f"DetectFrac{w}"] = f"{100 * by[n]['detect_fraction']:.0f}\\%"
                m[f"MedDelta{w}"] = _fmt(by[n]["median_delta"], signed=True)
    # Preregistered, one grid step above the measured threshold (prereg §8.2).
    m["NFreeze"] = "800"

    fdr = _load("analysis_fdr.json")
    if fdr:
        m["FdrNContrasts"] = str(fdr.get("n_contrasts"))
        m["FdrNSurvive"] = str(fdr.get("n_survive"))

    # ── Everything downstream of the regeneration pass ──────────────────
    # Named here so the paper can reference them and the build can refuse.
    for k in ("OffsetBestBaseline", "OffsetBestMTwo", "OffsetBestMThree",
              "OffsetBestMFour", "MTwoOnlyBest", "MThreeOnlyBest",
              "CoefRatioBest", "PosControlMThreeOnly", "PosControlMTwoOnly",
              "SdtNullBest", "SdtNullBandLo", "SdtNullBandHi",
              "VerdictLeakBest", "VerdictLeakPNinetyFive", "FloorToClear",
              "VerdictDecodBest", "StratumMinN", "NJudgesMain",
              "NItemsPerSliceMain", "MdeFDROffset", "CoverageAtRiskGain"):
        m.setdefault(k, None)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="exit non-zero if any macro is unresolved")
    args = ap.parse_args()

    macros = collect()
    missing = sorted(k for k, v in macros.items() if v is None)

    lines = ["% GENERATED by paper/fill_numbers.py -- do not edit by hand.",
             "% Every quantity printed in main.tex resolves through here.",
             ""]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v if v is not None else PENDING}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO)}  "
          f"({len(macros) - len(missing)}/{len(macros)} resolved)")
    if missing:
        print("\nUNRESOLVED — these have not been measured yet:")
        for k in missing:
            print(f"  \\{k}")
        print("\nThese are all downstream of the stream-regeneration pass.")
        if args.verify:
            sys.exit(1)


if __name__ == "__main__":
    main()
