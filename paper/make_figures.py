"""Generate the paper's figures from results/, one file per figure.

Every figure resolves from a results file, on the same principle as
fill_numbers.py: no number is drawn by hand. Figures are written to
paper/figs/ as PDF for vector output at the ICLR column width.

Run:  python paper/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, ACCENT, MUTED = "#1a1a1a", "#b2182b", "#8c8c8c"


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── Figure 2: the first-order floor as a curve in d' ────────────────────────
def fig_sdt_floor():
    """Simulate the floor across d', and place the observed judges on it."""
    from llm_judge.analysis.cv import oof_scores
    from llm_judge.analysis.stats import stratified_auroc

    rng = np.random.default_rng(0)
    n, n_sim = 400, 8
    ds = np.linspace(0.05, 2.6, 14)
    med, lo, hi, accs, ds_ok = [], [], [], [], []
    for d in ds:
        vals, acc = [], []
        for k in range(n_sim):
            truth = np.tile([1, -1], n // 2)
            s = d * truth + rng.normal(size=n)
            pred = np.where(s > 0, 1, -1)
            corr = (pred == truth).astype(float)
            if len(np.unique(corr)) < 2:
                continue
            X = np.c_[s, np.abs(s), (pred > 0).astype(float)]
            sc = oof_scores(X, corr, np.arange(n), 5)
            ok = ~np.isnan(sc)
            a = stratified_auroc(corr[ok], sc[ok],
                                 np.where(truth == 1, "A", "B")[ok])
            if a is not None:
                vals.append(a)
                acc.append(corr.mean())
        if not vals:          # no estimable stratum at this d'
            continue
        ds_ok.append(d)
        med.append(np.median(vals)); lo.append(np.percentile(vals, 10))
        hi.append(np.percentile(vals, 90)); accs.append(np.mean(acc))
    ds = np.array(ds_ok)

    an = _load("analysis_llmbar.json") or {}
    obs = []
    for k, v in an.items():
        if not isinstance(v, dict) or not v.get("sdt_null"):
            continue
        a = v["auroc_conditional"].get("M1")
        if a is None:
            continue
        obs.append((v["sdt_null"]["d_prime"], a,
                    k.split("/")[-1].split("|")[0].replace("-Instruct", "")))

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.fill_between(ds, lo, hi, color=MUTED, alpha=.25, lw=0)
    ax.plot(ds, med, color=INK, lw=1.4,
            label="first-order observer (no metacognition)")
    ax.axhline(0.5, color=MUTED, ls=":", lw=.8)
    ax.text(0.06, 0.505, "conventional reference", color=MUTED, fontsize=6,
            va="bottom")
    if obs:
        ax.scatter([o[0] for o in obs], [o[1] for o in obs], s=22,
                   color=ACCENT, zorder=5, label="observed judges")
    ax.set_xlabel(r"first-order sensitivity $d'$")
    ax.set_ylabel("conditional AUROC")
    ax.set_ylim(0.42, 1.0)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGS / "fig_sdt_floor.pdf")
    plt.close(fig)
    print("fig_sdt_floor.pdf   ", len(obs), "judges plotted")


# ── Figure 3: both ladder failures, in opposite directions ──────────────────
def fig_ladder_failures():
    """Delta against block width: concatenation falls, ridge-offset pins at
    zero, the corrected estimator tracks the planted truth."""
    from llm_judge.analysis.cv import oof_scores, oof_scores_offset
    from llm_judge.analysis.stats import stratified_auroc

    rng = np.random.default_rng(3)
    n_q = 250
    groups = np.repeat(np.arange(n_q), 2)
    n = len(groups)
    strata = np.tile(["A", "B"], n_q)
    latent = rng.normal(size=n)
    extra = rng.normal(size=n)
    y = ((latent + 0.9 * extra + rng.normal(0, .7, n)) > 0).astype(float)
    X_base = np.c_[1.2 * latent + rng.normal(0, .6, n), rng.normal(size=(n, 5))]
    base = stratified_auroc(y, oof_scores(X_base, y, groups, 5), strata)

    widths = [32, 128, 512, 2048]
    cat, off = [], []
    for w in widths:
        d = rng.normal(size=w); d /= np.linalg.norm(d)
        X_add = rng.normal(size=(n, w)) + 1.0 * np.outer(extra, d)
        c = stratified_auroc(
            y, oof_scores(np.c_[X_base, X_add], y, groups, 5, n_act=w), strata)
        o = stratified_auroc(
            y, oof_scores_offset(X_base, X_add, y, groups, 5), strata)
        cat.append(c - base); off.append(o - base)

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.axhline(0, color=MUTED, lw=.8, ls=":")
    ax.plot(widths, cat, "o-", color=ACCENT, lw=1.4, ms=4,
            label="concatenation, shared penalty")
    ax.plot(widths, off, "s-", color=INK, lw=1.4, ms=4,
            label="offset, penalty by out-of-fold AUROC")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("added block width (columns)")
    ax.set_ylabel(r"$\Delta$ conditional AUROC")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIGS / "fig_ladder_failures.pdf")
    plt.close(fig)
    print("fig_ladder_failures.pdf", [f"{c:+.3f}" for c in cat])


# ── Figure 4: the recovery curve and its detection threshold ────────────────
def fig_recovery():
    rc = _load("recovery_curve.json")
    if not rc:
        print("fig_recovery: results/recovery_curve.json missing")
        return
    by = rc["by_n"]
    ns = sorted(int(k) for k in by)
    med = [by[str(n)]["median_delta"] for n in ns]
    lo = [by[str(n)]["delta_iqr"][0] for n in ns]
    hi = [by[str(n)]["delta_iqr"][1] for n in ns]
    frac = [by[str(n)]["detect_fraction"] for n in ns]

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.fill_between(ns, lo, hi, color=MUTED, alpha=.25, lw=0)
    ax.plot(ns, med, "o-", color=INK, lw=1.4, ms=4, label="median $\\Delta$ (IQR)")
    ax.axhline(rc["detect_delta"], color=ACCENT, ls="--", lw=1,
               label=f"detection band {rc['detect_delta']}")
    thr = rc.get("n_freeze_candidate")
    if thr:
        ax.axvline(thr, color=ACCENT, lw=.8, ls=":")
        ax.text(thr * 1.05, min(lo), f"threshold\n$n={thr}$", color=ACCENT,
                fontsize=6, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("items per stratum")
    ax.set_ylabel(r"$\Delta$ conditional AUROC")
    ax.legend(frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(ns, frac, "^:", color=MUTED, ms=3.5, lw=.9)
    ax2.set_ylabel("detection fraction", color=MUTED, fontsize=7)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=6)
    ax2.set_ylim(0, 1.05); ax2.spines["top"].set_visible(False)
    fig.savefig(FIGS / "fig_recovery.pdf")
    plt.close(fig)
    print("fig_recovery.pdf    ", "threshold n =", thr)


# ── Figure 5: every rung against its own floor, not against 0.5 ─────────────
def fig_floors():
    an = _load("analysis_llmbar.json")
    if not an:
        print("fig_floors: results/analysis_llmbar.json missing")
        return
    rows = []
    for k, v in an.items():
        if not isinstance(v, dict) or v.get("floor_to_clear") is None:
            continue
        ac = v.get("auroc_conditional") or {}
        vals = [ac.get(r) for r in ("M1nd", "M2", "M3", "M4")]
        vals = [x for x in vals if x is not None]
        if not vals:
            continue
        model, fmt = k.split("|")
        rows.append((f"{model.split('/')[-1].replace('-Instruct','')}\n{fmt}",
                     vals, v["floor_to_clear"]))
    if not rows:
        print("fig_floors: no estimable slices")
        return

    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    xs = np.arange(len(rows))
    for i, (_, vals, floor) in enumerate(rows):
        ax.scatter([i] * len(vals), vals, s=18, color=INK, zorder=3,
                   label="rungs" if i == 0 else None)
        ax.plot([i - .3, i + .3], [floor, floor], color=ACCENT, lw=1.6,
                label="floor $\\phi$" if i == 0 else None)
    ax.axhline(0.5, color=MUTED, ls=":", lw=.8)
    ax.text(len(rows) - .4, 0.505, "0.5", color=MUTED, fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6)
    ax.set_ylabel("conditional AUROC")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(FIGS / "fig_floors.pdf")
    plt.close(fig)
    print("fig_floors.pdf      ", len(rows), "slices")


if __name__ == "__main__":
    fig_sdt_floor()
    fig_ladder_failures()
    fig_recovery()
    fig_floors()
    print("\nwrote", len(list(FIGS.glob("*.pdf"))), "figures to paper/figs/")
