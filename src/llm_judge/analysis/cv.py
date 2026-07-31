"""Nested model comparison via grouped out-of-fold predictions.

Ladder (each rung must beat the previous one to claim a contribution):
  Mn    nuisance only        (length + task-start position + subject)
  M1    margin               (signed logprob margin — the behavioural signal)
  M1n   margin + nuisance    (THE baseline every internal claim must clear)
  M2    M1n + spectral       (attention-graph profile + Fiedler velocity)
  M3    M1n + activations    (last-token hidden state, probe fitted per fold)
  M4    M1n + spectral + activations

Three structural safeguards:

* **GroupKFold(groups=question_id).** The _pos and _neg items of one question
  share nearly all their text, and the two orders of one pairwise item share
  all of it; if they land in different folds the model has already seen the
  essentials and the score is spuriously good.

* **Stratified (conditional) AUROC as the primary metric.** See stats.py:
  pooling pos and neg items lets any item-identity feature impersonate
  self-knowledge. The reported headline number only ever compares items
  sharing a `gt_verdict`.

* **One common item subset for every rung.** Rows lacking spectral or
  activation features are dropped BEFORE any model is fitted, so all rungs
  are computed on identical data and their AUROCs are comparable. The number
  dropped is logged, never silent.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F
from .stats import paired_bootstrap, pooled_auroc, stratified_auroc

def contrasts_for(baseline: str) -> list[tuple[str, str]]:
    """Contrasts that carry the paper's claims, against the live baseline."""
    out = [("M1", "M1n")]          # does length/position/subject add to confidence?
    if baseline != "M1n":
        out.append(("M1n", baseline))   # does peer difficulty add beyond that?
    out += [(baseline, "M2"),      # does the attention graph add beyond ALL that?
            (baseline, "M3"),      # does a plain activation probe?
            ("M3", "M4")]          # does spectral add anything activations did not?
    return out


def _pipeline(n_dense: int, n_act: int, pca_dims: int,
              activation_probe: str) -> Pipeline:
    """activation_probe:
      "full" (default) — standardised full-dim activations with stronger L2
          (C=0.1). Unsupervised PCA keeps high-VARIANCE directions and can
          discard a correctness signal that has no variance advantage; with
          thousands of items a regularised full-dim probe is the standard
          linear-probe baseline and stays leakage-free (fitted per fold).
      "pca" — StandardScaler+PCA(pca_dims) inside the fold; for small item
          counts where full-dim would overfit.
    """
    if n_act == 0:
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=1.0))
    if activation_probe == "full":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=0.1))
    ct = ColumnTransformer([
        ("dense", StandardScaler(), slice(0, n_dense)),
        ("act", make_pipeline(StandardScaler(),
                              PCA(n_components=pca_dims, random_state=0)),
         slice(n_dense, n_dense + n_act)),
    ])
    return Pipeline([("features", ct),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0))])


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int, n_act: int = 0, pca_dims: int = 24,
               activation_probe: str = "full") -> np.ndarray:
    """Out-of-fold probability scores; NaN where a fold was degenerate."""
    scores = np.full(len(y), np.nan)
    n_dense = X.shape[1] - n_act
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = _pipeline(n_dense, n_act, min(pca_dims, max(2, len(tr) - 1)),
                         activation_probe)
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]
    return scores


def select_rows(rows: list[dict], activations_npz=None,
                act_layer_frac: str = "1", min_coverage: float = 0.5,
                log=print) -> tuple[list[dict], dict]:
    """Restrict to rows usable by EVERY rung, and report what was dropped.

    All rungs must be scored on identical rows or their AUROCs are not
    comparable. But a partially finished spectral run would then delete most
    of the behavioural data too, so a family covering less than
    `min_coverage` of the rows is DROPPED instead: better to report the
    ladder without M2 on all rows than the full ladder on a tenth of them.
    """
    n0 = len(rows)
    cov_spec = sum(F.has_spectral(r) for r in rows) / max(1, n0)
    use_spectral = cov_spec >= min_coverage
    if 0 < cov_spec < min_coverage:
        log(f"  spectral covers only {cov_spec:.0%} of rows (< {min_coverage:.0%})"
            f" — dropping the spectral family rather than discarding "
            f"{n0 - int(cov_spec * n0)} rows from every other rung")
    keep = [r for r in rows if (not use_spectral or F.has_spectral(r))]
    n_after_spec = len(keep)

    cov_act = 0.0
    if activations_npz is not None and keep:
        cov_act = sum(F.has_activation(r, activations_npz, act_layer_frac)
                      for r in keep) / len(keep)
    use_act = cov_act >= min_coverage
    if 0 < cov_act < min_coverage:
        log(f"  activations cover only {cov_act:.0%} of rows — dropping the "
            f"activation family rather than discarding rows")
    if use_act:
        keep = [r for r in keep
                if F.has_activation(r, activations_npz, act_layer_frac)]

    avail = {"spectral": use_spectral, "activations": use_act,
             "spectral_coverage": cov_spec, "activation_coverage": cov_act,
             "n_input": n0, "n_used": len(keep),
             "dropped_no_spectral": n0 - n_after_spec,
             "dropped_no_activation": n_after_spec - len(keep)}
    if avail["dropped_no_spectral"] or avail["dropped_no_activation"]:
        log(f"  common subset: {len(keep)}/{n0} rows "
            f"(dropped {avail['dropped_no_spectral']} without spectral, "
            f"{avail['dropped_no_activation']} without activations) — all "
            f"rungs are scored on these same rows")
    return keep, avail


def build_feature_sets(rows: list[dict], activations_npz=None,
                       act_layer_frac: str = "1") -> tuple[dict, list, str]:
    """Return ({name: (X, n_act_cols)}, nuisance names, baseline rung name).

    The baseline the internal families must clear is M1nd (margin + nuisance
    + peer difficulty) when peer difficulty is available, otherwise M1n.
    """
    Xm = F.margin_features(rows)
    Xn, nuis_names = F.nuisance_features(rows)
    base = np.c_[Xm, Xn]
    sets = {
        "Mn": (Xn, 0),
        "M1": (Xm, 0),
        "M1n": (base, 0),
    }
    baseline = "M1n"
    Xd = F.peer_difficulty_features(rows)
    if Xd is not None:
        base = np.c_[base, Xd]
        sets["M1nd"] = (base, 0)
        baseline = "M1nd"

    Xs = None
    if all(F.has_spectral(r) for r in rows):
        Xs = F.spectral_features(rows)
        sets["M2"] = (np.c_[base, Xs], 0)
    if activations_npz is not None and all(
            F.has_activation(r, activations_npz, act_layer_frac) for r in rows):
        Xa = F.activation_features(rows, activations_npz, act_layer_frac)
        sets["M3"] = (np.c_[base, Xa], Xa.shape[1])
        if Xs is not None:
            sets["M4"] = (np.c_[base, Xs, Xa], Xa.shape[1])
    return sets, nuis_names, baseline


def health_checks(rows: list[dict], log=print) -> dict:
    """Verdict-behaviour sanity before any modelling.

    A judge with no verdict variance inside a gt_verdict stratum carries no
    detectable self-knowledge signal by construction — that is a real (and
    reportable) outcome, not a bug, but it must be surfaced rather than
    silently producing an undefined or meaningless AUROC.
    """
    y = np.array([float(r["is_correct"]) for r in rows])
    gt = np.array([str(r["gt_verdict"]) for r in rows])
    labels = sorted(set(gt))
    first = labels[0]
    pred_first = np.mean([r["pred_verdict"] == first for r in rows])
    per_stratum = {}
    degenerate = False
    for s in labels:
        m = gt == s
        acc = float(y[m].mean()) if m.any() else float("nan")
        per_stratum[s] = {"n": int(m.sum()), "accuracy": acc}
        if m.sum() >= 2 and len(np.unique(y[m])) < 2:
            degenerate = True
    # Near-degenerate: the judge almost always answers one way. Not literally
    # constant, so conditional AUROC is defined, but it rests on a handful of
    # minority-class items and its bootstrap CI will be uselessly wide.
    extreme_bias = bool(pred_first < 0.05 or pred_first > 0.95)
    out = {"accuracy": float(y.mean()),
           f"pred_rate_{first}": float(pred_first),
           "per_stratum": per_stratum, "degenerate": degenerate,
           "extreme_bias": extreme_bias}
    log(f"  verdict behaviour: accuracy {y.mean():.1%}, "
        f"P(pred={first}) {pred_first:.1%}, "
        + ", ".join(f"acc|gt={s} {v['accuracy']:.1%} (n={v['n']})"
                    for s, v in per_stratum.items()))
    if degenerate:
        log("  DEGENERATE: a gt_verdict stratum has zero outcome variance "
            "(the judge answers one way regardless) — conditional AUROC is "
            "undefined there; this model carries no measurable signal.")
    elif extreme_bias:
        log(f"  EXTREME VERDICT BIAS: P(pred={first})={pred_first:.1%}. "
            f"Conditional AUROC rests on very few minority-class items; treat "
            f"its CI as uninformative and report the bias itself instead.")
    return out


def compare(rows: list[dict], n_splits: int = 5, n_boot: int = 2000,
            pca_dims: int = 24, activations_npz=None, seed: int = 42,
            activation_probe: str = "full", act_layer_frac: str = "1",
            run_permutation_null: bool = True, n_permutations: int = 5,
            min_coverage: float = 0.5, log=print) -> dict:
    """Full ladder, conditional-AUROC primary metric, bootstrap contrasts."""
    rows, avail = select_rows(rows, activations_npz, act_layer_frac,
                              min_coverage, log)
    if len(rows) < 40:
        return {"skipped": f"only {len(rows)} usable rows"}
    if not avail["spectral"]:
        activations_npz = activations_npz if avail["activations"] else None

    y = np.array([float(r["is_correct"]) for r in rows])
    # Group on the text-derived group_id when the bank provides one: some
    # datasets reuse the same question under several question_ids (JudgeBench
    # across its claude/gpt splits, MMLU across subjects), and grouping on
    # ids would let those straddle folds.
    groups = np.array([r.get("group_id") or r["question_id"] for r in rows])
    strata = np.array([str(r["gt_verdict"]) for r in rows])
    n_qid = len({r["question_id"] for r in rows})
    log(f"  n={len(rows)} items, {len(set(groups))} CV groups "
        f"({n_qid} question_ids)")
    health = health_checks(rows, log)

    sets, nuis_names, baseline = build_feature_sets(rows, activations_npz,
                                                    act_layer_frac)
    log(f"  families: {', '.join(sets)} | nuisance cols: {len(nuis_names)} | "
        f"baseline to beat: {baseline}")
    if baseline == "M1n":
        log("  (no peer_difficulty available — the 'is it just item "
            "difficulty?' control needs >=2 judges scoring the same items)")

    oof = {name: oof_scores(X, y, groups, n_splits, n_act, pca_dims,
                            activation_probe)
           for name, (X, n_act) in sets.items()}

    valid = np.all([~np.isnan(s) for s in oof.values()], axis=0)
    y_v, g_v, st_v = y[valid], groups[valid], strata[valid]
    oof_v = {k: v[valid] for k, v in oof.items()}

    report = {"n_items": int(valid.sum()), "availability": avail,
              "health": health, "baseline": baseline,
              "auroc_conditional": {}, "auroc_pooled": {},
              "contrasts": {}, "controls": {}}

    for name, s in oof_v.items():
        report["auroc_conditional"][name] = stratified_auroc(y_v, s, st_v)
        report["auroc_pooled"][name] = pooled_auroc(y_v, s)

    # A judge whose verdict is (near-)constant cannot support any contrast:
    # its conditional AUROC rests on a handful of minority-class items, and a
    # bootstrap that keeps resampling those same items returns a narrow CI
    # around an arbitrary value. Suppress the contrasts outright and mark the
    # slice unreliable so it is also excluded from the FDR family.
    report["unreliable"] = bool(health["degenerate"] or health["extreme_bias"])
    if report["unreliable"]:
        log("  contrasts SUPPRESSED: verdict behaviour is (near-)constant, so "
            "no conditional contrast here is estimable — report the verdict "
            "bias itself, not an AUROC.")
    else:
        for a, b in contrasts_for(baseline):
            if a in oof_v and b in oof_v:
                report["contrasts"][f"{b} - {a}"] = paired_bootstrap(
                    y_v, g_v, oof_v[a], oof_v[b], strata=st_v,
                    n_boot=n_boot, seed=seed)

    # ── Control 1: item-identity decodability ────────────────────────────
    # How well do the SAME features predict gt_verdict (i.e. merely tell a
    # pos item from a neg one)? High values here are expected and harmless —
    # they are precisely why the pooled AUROC must not be the headline.
    best = next(m for m in ("M4", "M2", "M3", baseline) if m in sets)
    X_best, n_act_best = sets[best]
    y_id = (strata == sorted(set(strata))[0]).astype(float)
    s_id = oof_scores(X_best, y_id, groups, n_splits, n_act_best, pca_dims,
                      activation_probe)
    ok = ~np.isnan(s_id)
    report["controls"]["identity_decodability"] = {
        "features": best, "auroc": pooled_auroc(y_id[ok], s_id[ok])}

    # ── Control 2: permutation null ──────────────────────────────────────
    # Labels shuffled within gt_verdict strata; the full pipeline is refitted
    # each time. Conditional AUROC must average ~0.5 — anything else means
    # leakage in the CV itself. Several permutations are run because a single
    # one is very noisy at small n, and a noisy null would either cry wolf or
    # hide a real leak.
    if run_permutation_null:
        rng = np.random.default_rng(seed + 1)
        nulls = []
        for _ in range(n_permutations):
            y_perm = y.copy()
            for s in np.unique(strata):
                m = np.where(strata == s)[0]
                y_perm[m] = rng.permutation(y[m])
            s_perm = oof_scores(X_best, y_perm, groups, n_splits, n_act_best,
                                pca_dims, activation_probe)
            ok = ~np.isnan(s_perm)
            a = stratified_auroc(y_perm[ok], s_perm[ok], strata[ok])
            if a is not None:
                nulls.append(a)
        mean_null = float(np.mean(nulls)) if nulls else None
        report["controls"]["permutation_null"] = {
            "features": best, "n_permutations": len(nulls),
            "auroc_conditional_mean": mean_null,
            "auroc_conditional_sd": float(np.std(nulls)) if nulls else None}
        # Only escalate when the sample is large enough for the null to be
        # meaningfully estimated; below that the spread is the story.
        if (mean_null is not None and abs(mean_null - 0.5) > 0.08
                and len(rows) >= 200):
            log(f"  LEAKAGE WARNING: permutation null conditional AUROC "
                f"{mean_null:.3f} over {len(nulls)} permutations "
                f"(expected ~0.500)")

    ctrl = report["controls"]
    log(f"  control identity-decodability ({best}): "
        f"{_fmt(ctrl['identity_decodability']['auroc'])} "
        f"(high is expected; it is why pooled AUROC is not the headline)")
    if "permutation_null" in ctrl:
        pn = ctrl["permutation_null"]
        log(f"  control permutation-null: {_fmt(pn['auroc_conditional_mean'])}"
            f" +/- {_fmt(pn['auroc_conditional_sd'])} over "
            f"{pn['n_permutations']} permutations (must be ~0.500)")
    if len(rows) < 200:
        log(f"  LOW POWER: {len(rows)} items — contrasts here are indicative "
            f"only; CIs will be too wide to support a claim.")
    return report


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def difficulty_strata(rows: list[dict]) -> dict[str, list[dict]]:
    """Split MCQ rows by panel difficulty (gt_mean_prob terciles).

    Low judge accuracy on hard items means noisy labels; strata show whether
    an absent signal is a power problem rather than a negative result.
    """
    probs = [r.get("gt_mean_prob") for r in rows]
    if any(p is None for p in probs):
        return {"all": rows}
    qs = np.percentile(probs, [33.3, 66.7])
    out: dict[str, list[dict]] = {"hard": [], "medium": [], "easy": []}
    for r, p in zip(rows, probs):
        out["hard" if p <= qs[0] else "medium" if p <= qs[1] else "easy"].append(r)
    return out
