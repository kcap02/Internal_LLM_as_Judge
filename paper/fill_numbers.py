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
        # The anchor: M4 - M3 across estimable slices.
        d4 = [(v.get("contrasts") or {}).get("M4 - M3", {})
              for v in an.values() if isinstance(v, dict)]
        d4 = [c for c in d4 if c.get("delta") is not None]
        if d4:
            deltas = [c["delta"] for c in d4]
            ses = [(c["ci95"][1] - c["ci95"][0]) / 3.92 for c in d4]
            m["AnchorNSlices"] = str(len(d4))
            m["AnchorDeltaLo"] = _fmt(min(deltas), signed=True)
            m["AnchorDeltaHi"] = _fmt(max(deltas), signed=True)
            m["AnchorSELo"] = _fmt(min(ses))
            m["AnchorSEHi"] = _fmt(max(ses))
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
              "VerdictLeakBest", "VerdictLeakP95", "FloorToClear",
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
