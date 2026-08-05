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
from .stats import (_auc_or_none, paired_bootstrap, pooled_auroc,
                    stratified_auroc)

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


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit_offset_logistic(X: np.ndarray, y: np.ndarray, z: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Penalised logistic regression with `z` as an UNPENALISED offset.

    Minimises  sum log(1 + exp(-(2y-1)(z + Xw)))  +  alpha * ||w||^2  over w.

    Why not RidgeCV on the working residual (the first attempt). That fits
    squared error on `y - sigmoid(z)`, an objective on a different scale from
    the logit and only loosely related to the AUROC we report. At n=160 per
    training fold with 1536 columns its CV curve is nearly flat and its
    minimum sits at maximum shrinkage: RidgeCV selected the top of the grid on
    every fold and the correction it produced had 0.7% of the offset's scale,
    so every internal rung came back numerically identical to the baseline.
    That is a zero manufactured by the estimator, the mirror image of the
    swamping C-LADDER describes. Capping the grid would only hide it — the
    penalty would then be chosen by hand.

    Here the second stage optimises the same likelihood the first stage did,
    on the same scale, and `alpha` is selected by out-of-fold AUROC (the
    reported metric) rather than by squared error.
    """
    from scipy.optimize import minimize

    s = 2.0 * y - 1.0                      # +/-1 labels

    def obj(w):
        m = s * (z + X @ w)
        # log(1+exp(-m)), computed stably
        loss = np.logaddexp(0.0, -m).sum()
        p = 1.0 / (1.0 + np.exp(np.clip(m, -30, 30)))   # sigmoid(-m)
        grad = -(X.T @ (s * p))
        return loss + alpha * w @ w, grad + 2.0 * alpha * w

    w0 = np.zeros(X.shape[1])
    res = minimize(obj, w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300})
    return res.x


def oof_scores_offset(X_base: np.ndarray, X_add: np.ndarray, y: np.ndarray,
                      groups: np.ndarray, n_splits: int) -> np.ndarray:
    """Out-of-fold scores for 'baseline PLUS a block', made monotone by
    construction.

    The problem this solves. Concatenating a `hidden_size`-wide activation
    block onto a 6-column baseline and fitting one penalised model does not
    test *incremental* information: the penalty is shared, so the wide block
    swamps the baseline and the fitted model is approximately 'activations',
    not 'baseline + activations'. The added block then has to reconstruct the
    margin from scratch, and when it cannot, the rung scores BELOW the
    baseline it was supposed to extend. Measured on the LLMBar pilot, the
    delta correlates r = -0.85 with baseline strength for M3 — the better the
    baseline, the more the rung "loses", which is a property of the estimator
    and not of the representation.

    The fix. Fit the baseline first, carry its out-of-fold logit as a fixed
    OFFSET, and let the added block fit only the residual:

        score = z_baseline + f(X_add),   f fitted on the working residual

    If `X_add` is uninformative the ridge shrinks f to ~0 and the score
    reduces to the baseline, so a negative delta now means overfitting rather
    than destruction of information already in hand. This is one offset-GLM /
    boosting step, which is easier to defend in the paper than group-wise
    penalties and behaves identically when the block is small.

    Nesting matters. The offset for TRAINING rows is itself computed by an
    inner CV inside the training fold; using the outer OOF logit there would
    leak the outer test fold into the baseline the second stage sees.
    """
    scores = np.full(len(y), np.nan)
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    outer = GroupKFold(n_splits=n_splits)

    for tr, te in outer.split(X_base, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        g_tr = groups[tr]
        # (a) offset for the TEST rows: baseline fitted on the whole train fold
        base_pipe = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
        base_pipe.fit(X_base[tr], y[tr])
        z_te = _logit(base_pipe.predict_proba(X_base[te])[:, 1])

        # (b) offset for the TRAINING rows: inner CV, so the second stage
        #     never sees a baseline that was fitted on the row it is fitting.
        z_tr = np.full(len(tr), np.nan)
        n_inner = max(2, min(n_splits, len(np.unique(g_tr))))
        for itr, ite in GroupKFold(n_splits=n_inner).split(
                X_base[tr], y[tr], g_tr):
            if len(np.unique(y[tr][itr])) < 2:
                continue
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=1.0))
            p.fit(X_base[tr][itr], y[tr][itr])
            z_tr[ite] = _logit(p.predict_proba(X_base[tr][ite])[:, 1])
        ok = ~np.isnan(z_tr)
        if ok.sum() < 10:
            scores[te] = z_te
            continue

        # (c) the added block extends the offset model, with its penalty
        #     selected by out-of-fold AUROC — the metric we report — rather
        #     than by squared error on a working residual. See
        #     `_fit_offset_logistic` for why the residual/RidgeCV form
        #     manufactured a zero.
        Xtr, ytr, ztr = X_add[tr][ok], y[tr][ok], z_tr[ok]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(X_add[te])
        g_sel = g_tr[ok]

        best_alpha, best_auc = ALPHA_GRID[-1], -np.inf
        n_sel = max(2, min(3, len(np.unique(g_sel))))
        for a in ALPHA_GRID:
            oof = np.full(len(ytr), np.nan)
            for str_, ste in GroupKFold(n_splits=n_sel).split(Xtr_s, ytr, g_sel):
                if len(np.unique(ytr[str_])) < 2:
                    continue
                w = _fit_offset_logistic(Xtr_s[str_], ytr[str_], ztr[str_], a)
                oof[ste] = ztr[ste] + Xtr_s[ste] @ w
            m = ~np.isnan(oof)
            auc = _auc_or_none(ytr[m], oof[m]) if m.any() else None
            if auc is not None and auc > best_auc:
                best_auc, best_alpha = auc, a

        w = _fit_offset_logistic(Xtr_s, ytr, ztr, best_alpha)
        scores[te] = z_te + Xte_s @ w
    return scores


# Penalty grid for the second stage. Spans "essentially unpenalised" to
# "essentially zero correction", so the selected value is informative: if the
# top of the grid always wins, the block genuinely carries nothing at this n,
# and `offset_shrinkage_report` says so rather than leaving it to be inferred
# from a delta of 0.000.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


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
                       act_layer_frac: str = "1"
                       ) -> tuple[dict, list, str, dict]:
    """Return ({name: (X, n_act_cols)}, nuisance names, baseline, ladder).

    The baseline the internal families must clear is M1nd (margin + nuisance
    + peer difficulty) when peer difficulty is available, otherwise M1n.

    `ladder` carries the baseline matrix and each internal block SEPARATELY
    ({"base": X, "add": {"M2": Xs, ...}}), which is what the offset estimator
    needs: the concatenated matrices in `sets` are kept only so the legacy
    `ladder_mode="concat"` path stays reproducible.
    """
    Xm = F.margin_features(rows)
    Xn, nuis_names = F.nuisance_features(rows)
    # pred_verdict joins the baseline: an internal block must beat "we already
    # know what the judge said", not merely "we know how confident it was".
    # See F.verdict_features and the verdict_decodability control.
    Xv = F.verdict_features(rows)
    base = np.c_[Xm, Xv, Xn]
    sets = {
        "Mn": (Xn, 0),
        "M1": (np.c_[Xm, Xv], 0),
        "M1n": (base, 0),
    }
    baseline = "M1n"
    Xd = F.peer_difficulty_features(rows)
    if Xd is not None:
        base = np.c_[base, Xd]
        sets["M1nd"] = (base, 0)
        baseline = "M1nd"

    Xs = Xa = None
    if all(F.has_spectral(r) for r in rows):
        Xs = F.spectral_features(rows)
        sets["M2"] = (np.c_[base, Xs], 0)
        # Spectral ALONE, with no baseline features. A tight M4-M3 equivalence
        # ("spectral adds nothing beyond activations") has two very different
        # explanations, and the delta cannot tell them apart: either spectral
        # carries no correctness information at all, or it carries information
        # that activations already contain. This rung separates them — at
        # ~0.500 the claim is absence, above it the claim is redundancy, which
        # is a different and more interesting sentence.
        sets["M2only"] = (Xs, 0)
    if activations_npz is not None and all(
            F.has_activation(r, activations_npz, act_layer_frac) for r in rows):
        Xa = F.activation_features(rows, activations_npz, act_layer_frac)
        sets["M3"] = (np.c_[base, Xa], Xa.shape[1])
        sets["M3only"] = (Xa, Xa.shape[1])
        if Xs is not None:
            sets["M4"] = (np.c_[base, Xs, Xa], Xa.shape[1])

    add: dict[str, np.ndarray] = {}
    if Xs is not None:
        add["M2"] = Xs
    if Xa is not None:
        add["M3"] = Xa
        if Xs is not None:
            add["M4"] = np.c_[Xs, Xa]
    ladder = {"base": base, "add": add}
    return sets, nuis_names, baseline, ladder


# Rungs that exist to diagnose the ladder rather than to carry a claim. They
# are reported but never entered into the contrast family, so they cost no
# multiplicity budget.
DIAGNOSTIC_RUNGS = ("M2only", "M3only")


def spectral_block_diagnostic(y: np.ndarray,
                              groups: np.ndarray, sets: dict, n_splits: int,
                              activation_probe: str, pca_dims: int,
                              log=print) -> dict | None:
    """Is the spectral block shrunk to zero inside M4?

    The second explanation for a tight `M4 - M3` equivalence: the regulariser
    drove the spectral coefficients to ~0, so M4 is numerically M3 with dead
    columns. That is indistinguishable from a genuine null in the delta alone,
    but it is directly visible in the fitted coefficients.

    Reports the per-feature mean |coefficient| on the spectral block against
    the same quantity on the activation block, both on standardised inputs so
    the magnitudes are comparable. A spectral/activation ratio near zero means
    the equivalence is an artefact of shrinkage, not evidence about spectral.
    """
    if "M4" not in sets or "M2" not in sets:
        return None
    X4, n_act = sets["M4"]
    n_spec = sets["M2"][0].shape[1] - (sets.get("M1nd") or sets["M1n"])[0].shape[1]
    if n_spec <= 0 or n_act <= 0:
        return None
    n_base = X4.shape[1] - n_spec - n_act

    coefs = []
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    for tr, _ in GroupKFold(n_splits=n_splits).split(X4, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = _pipeline(X4.shape[1] - n_act, n_act,
                         min(pca_dims, max(2, len(tr) - 1)), activation_probe)
        pipe.fit(X4[tr], y[tr])
        clf = pipe[-1]
        c = np.ravel(clf.coef_)
        if c.size != X4.shape[1]:
            return None      # PCA path reshapes the design; not comparable
        coefs.append(np.abs(c))
    if not coefs:
        return None

    m = np.mean(coefs, axis=0)
    base_m = float(m[:n_base].mean()) if n_base else float("nan")
    spec_m = float(m[n_base:n_base + n_spec].mean())
    act_m = float(m[n_base + n_spec:].mean())
    ratio = spec_m / act_m if act_m > 0 else float("inf")
    out = {"n_baseline_cols": int(n_base), "n_spectral_cols": int(n_spec),
           "n_activation_cols": int(n_act),
           "mean_abs_coef_baseline": base_m,
           "mean_abs_coef_spectral": spec_m,
           "mean_abs_coef_activation": act_m,
           "spectral_to_activation_ratio": ratio,
           "shrunk_to_zero": bool(ratio < 0.05)}
    log(f"  spectral-block coefficients: mean|w| spectral={spec_m:.4g} vs "
        f"activation={act_m:.4g} (ratio {ratio:.2f})"
        + ("  <-- SHRUNK: an M4-M3 null here is about the regulariser, "
           "not about spectral" if out["shrunk_to_zero"] else ""))
    return out


def positive_control(rows: list[dict], groups: np.ndarray, sets: dict,
                     n_splits: int, activation_probe: str, pca_dims: int,
                     log=print) -> dict | None:
    """Can the same pipeline recover a target that is certainly encoded?

    Required before any null is defensible. C-NUM showed this pipeline can
    lose 100% of a model's spectral data and still emit healthy-looking
    numbers, so "the features predict nothing" must be distinguished from
    "the features never arrived". Prompt length is unambiguously recoverable
    from a hidden state, so the probe must find it.

    Runs at matched n, with the same estimator and the same CV, on a balanced
    long-vs-short target (outer terciles of prompt length). Failure here
    invalidates the null; success localises it to the signal.
    """
    lens = np.array([r.get("n_tokens_prompt") or 0 for r in rows], float)
    if len(np.unique(lens)) < 3:
        return None
    lo, hi = np.percentile(lens, [33.3, 66.7])
    keep = (lens <= lo) | (lens >= hi)
    if keep.sum() < 40:
        return None
    y_len = (lens[keep] >= hi).astype(float)
    if len(np.unique(y_len)) < 2:
        return None

    out: dict = {"target": "prompt length (outer terciles)",
                 "n": int(keep.sum()), "auroc": {}}
    for name in ("M2only", "M3only", "M2", "M3", "M4"):
        if name not in sets:
            continue
        X, n_act = sets[name]
        s = oof_scores(X[keep], y_len, groups[keep], n_splits, n_act,
                       pca_dims, activation_probe)
        ok = ~np.isnan(s)
        out["auroc"][name] = pooled_auroc(y_len[ok], s[ok])

    shown = ", ".join(f"{k}={_fmt(v)}" for k, v in out["auroc"].items())
    log(f"  POSITIVE CONTROL (recover prompt length, n={out['n']}): {shown}")
    weak = [k for k, v in out["auroc"].items()
            if v is not None and v < 0.70 and k in ("M2only", "M3only")]
    if weak:
        log(f"    WARNING: {', '.join(weak)} cannot recover a target that is "
            f"certainly encoded — a null on the real target is a pipeline "
            f"problem, not a result.")
    return out


def verdict_only_null(rows: list[dict], n_splits: int = 5, n_sim: int = 40,
                      seed: int = 0, log=print) -> dict | None:
    """The LEAKAGE floor: what a pure verdict decoder scores at THIS slice's
    stratum imbalance.

    C-VERDICT is blocked by sign cancellation — a feature decoding only
    `pred_verdict` scores 1.0 in one `gt_verdict` stratum and 0.0 in the
    other, so a pooled fit cannot select it. **That cancellation is exact only
    when the strata are balanced.** The pilot panel's verdict bias runs 0.14
    to 0.64, so cancellation is partial and the residual leakage differs per
    slice — a quantity no other control here measures.

    So measure it directly: build a feature that decodes `pred_verdict` and
    nothing else, push it through the identical pipeline at the slice's own
    observed imbalance, and report the conditional AUROC it reaches. At 0.500
    the slice is clean. At 0.56 or 0.58, that is the floor every rung in that
    slice must clear — and it is a *different* floor from the SDT null.

    Two nulls with distinct meanings, both of which a rung must beat:
      * `sdt_null_reference`  — the first-order floor: what confidence
        arithmetic alone yields for a judge with no metacognition.
      * `verdict_only_null`   — the leakage floor: what decoding the judge's
        own answer yields through incomplete cancellation.
    """
    gt = np.array([str(r["gt_verdict"]) for r in rows])
    pred = np.array([str(r["pred_verdict"]) for r in rows])
    y = np.array([float(r["is_correct"]) for r in rows])
    groups = np.array([r.get("group_id") or r["question_id"] for r in rows])
    if len(set(gt)) != 2 or len(np.unique(y)) < 2:
        return None

    first = sorted(set(pred))[0]
    v = (pred == first).astype(float)
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_sim):
        # Decodes pred_verdict essentially perfectly, and carries nothing
        # else; the noise columns only stop the scaler from degenerating.
        X = np.c_[v + rng.normal(0, 0.01, len(v)),
                  rng.normal(size=(len(v), 5))]
        s = oof_scores(X, y, groups, n_splits)
        ok = ~np.isnan(s)
        a = stratified_auroc(y[ok], s[ok], gt[ok])
        if a is not None:
            vals.append(a)
    if not vals:
        return None

    arr = np.array(vals)
    out = {"n_sim": int(len(arr)),
           "leak_auroc_mean": float(arr.mean()),
           "leak_auroc_p95": float(np.percentile(arr, 95)),
           "clean": bool(np.percentile(arr, 95) < 0.53)}
    log(f"  verdict-leakage floor (pure pred decoder at this slice's "
        f"imbalance): {arr.mean():.3f} (p95 {out['leak_auroc_p95']:.3f})"
        + ("" if out["clean"] else "  <-- NOT clean: cancellation is partial "
           "here; every rung must clear this too"))
    return out


def sdt_null_reference(rows: list[dict], n_splits: int = 5, n_sim: int = 40,
                       seed: int = 0, log=print) -> dict | None:
    """The conditional AUROC a judge with NO metacognition already achieves.

    0.5 is the wrong reference point for a type-2 (correctness) ROC. A pure
    first-order signal-detection observer — one latent decision variable `s`,
    verdict `= sign(s - c)`, margin `= s - c`, and *nothing* that knows about
    its own errors — already scores well above 0.5, and scores higher as its
    discriminability rises. Reporting a conditional AUROC against 0.5
    therefore credits first-order accuracy as if it were self-knowledge.

    This is the Monte-Carlo form of meta-d'/d' (the M-ratio) from the
    metacognition literature (Maniscalco & Lau 2012; Fleming & Lau 2014),
    where 1.0 means no metacognitive sensitivity beyond first-order
    performance. Here: simulate the first-order observer at the judge's OWN
    d' and criterion, push it through the identical M1 features, the identical
    estimator and the identical stratified metric, and report the observed
    value against that null.

    Any rung exceeding this null carries information beyond first-order
    performance; any rung at or below it does not, however far above 0.5 it
    reads.
    """
    from math import erf, sqrt

    def _z(p: float) -> float:                     # normal quantile
        p = min(max(p, 1e-4), 1 - 1e-4)
        lo, hi = -6.0, 6.0
        for _ in range(120):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    gt = np.array([str(r["gt_verdict"]) for r in rows])
    pred = np.array([str(r["pred_verdict"]) for r in rows])
    labels = sorted(set(gt) | set(pred))
    if len(labels) != 2:
        return None
    a, b = labels
    n_a, n_b = int((gt == a).sum()), int((gt == b).sum())
    if min(n_a, n_b) < 20:
        return None

    hit = float((pred[gt == a] == a).mean())       # P(say a | gt a)
    fa = float((pred[gt == b] == a).mean())        # P(say a | gt b)
    d_prime = _z(hit) - _z(fa)
    crit = -0.5 * (_z(hit) + _z(fa))

    rng = np.random.default_rng(seed)
    n = len(rows)
    truth = np.where(gt == a, 1.0, -1.0)
    strata_sim = np.where(truth == 1, a, b)
    groups_sim = np.arange(n)
    idx_a = np.where(gt == a)[0]
    idx_b = np.where(gt == b)[0]

    nulls = []
    for _ in range(n_sim):
        # Propagate PARAMETER uncertainty, not just simulation noise. d' and c
        # are themselves estimated from the observed hit/false-alarm rates, so
        # simulating at the point estimate gives a null that is too narrow and
        # makes "above null" anticonservative. Resample items, re-estimate.
        ra = rng.choice(idx_a, size=len(idx_a), replace=True)
        rb = rng.choice(idx_b, size=len(idx_b), replace=True)
        h_b = float((pred[ra] == a).mean())
        f_b = float((pred[rb] == a).mean())
        d_b = _z(h_b) - _z(f_b)
        c_b = -0.5 * (_z(h_b) + _z(f_b))

        s = (d_b / 2.0) * truth + rng.normal(size=n)
        pred_sim = np.where(s > c_b, 1.0, -1.0)
        corr = (pred_sim == truth).astype(float)
        if len(np.unique(corr)) < 2:
            continue
        m = s - c_b
        X = np.c_[m, np.abs(m), (pred_sim > 0).astype(float)]   # = M1
        sc = oof_scores(X, corr, groups_sim, n_splits)
        ok = ~np.isnan(sc)
        val = stratified_auroc(corr[ok], sc[ok], strata_sim[ok])
        if val is not None:
            nulls.append(val)
    if not nulls:
        return None

    arr = np.array(nulls)
    out = {"d_prime": float(d_prime), "criterion": float(crit),
           "hit_rate": hit, "false_alarm_rate": fa,
           "n_sim": int(len(arr)),
           "null_auroc_mean": float(arr.mean()),
           "null_auroc_sd": float(arr.std()),
           "null_auroc_p05": float(np.percentile(arr, 5)),
           "null_auroc_p95": float(np.percentile(arr, 95)),
           "_draws": arr}
    width = out["null_auroc_p95"] - out["null_auroc_p05"]
    log(f"  SDT null (first-order observer, d'={d_prime:.2f}, c={crit:+.2f}, "
        f"parameter uncertainty propagated): conditional AUROC "
        f"{arr.mean():.3f}, 90% band [{out['null_auroc_p05']:.3f}, "
        f"{out['null_auroc_p95']:.3f}] — this, not 0.500, is the reference")
    if width > 0.15:
        log(f"    NOTE: null band is {width:.2f} wide at n={n}; per-slice "
            f"comparison can only detect very large metacognitive effects. "
            f"The pooled hierarchical test is the primary one.")
    return out


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
            min_coverage: float = 0.5, ladder_mode: str = "offset",
            log=print) -> dict:
    """Full ladder, conditional-AUROC primary metric, bootstrap contrasts."""
    rows, avail = select_rows(rows, activations_npz, act_layer_frac,
                              min_coverage, log)
    if len(rows) < 40:
        return {"skipped": f"only {len(rows)} usable rows"}
    if not avail["activations"]:
        # Coverage was too thin to keep the family; make sure the rung is not
        # rebuilt downstream from a partially-populated archive.
        activations_npz = None

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

    sets, nuis_names, baseline, ladder = build_feature_sets(
        rows, activations_npz, act_layer_frac)
    log(f"  families: {', '.join(sets)} | nuisance cols: {len(nuis_names)} | "
        f"baseline to beat: {baseline}")
    if baseline == "M1n":
        log("  (no peer_difficulty available — the 'is it just item "
            "difficulty?' control needs >=2 judges scoring the same items)")

    # The internal rungs are fitted as baseline-plus-block via the offset
    # estimator, so adding a block can only help up to noise. Fitting them by
    # concatenation instead lets a hidden_size-wide block swamp a 6-column
    # baseline, which makes the rung score BELOW the baseline it extends —
    # measured r = -0.85 against baseline strength on the pilot. See
    # `oof_scores_offset` and C-LADDER.
    oof = {}
    for name, (X, n_act) in sets.items():
        if ladder_mode == "offset" and name in ladder["add"]:
            oof[name] = oof_scores_offset(ladder["base"], ladder["add"][name],
                                          y, groups, n_splits)
        else:
            oof[name] = oof_scores(X, y, groups, n_splits, n_act, pca_dims,
                                   activation_probe)

    valid = np.all([~np.isnan(s) for s in oof.values()], axis=0)
    y_v, g_v, st_v = y[valid], groups[valid], strata[valid]
    oof_v = {k: v[valid] for k, v in oof.items()}

    # Per-stratum counts belong NEXT TO every conditional AUROC, not in a
    # separate health block: a conditional number resting on a handful of
    # minority-class items looks exactly like one resting on hundreds.
    strata_n = {}
    for s in np.unique(st_v):
        m = st_v == s
        strata_n[str(s)] = {
            "n": int(m.sum()),
            "n_correct": int((y_v[m] == 1).sum()),
            "n_incorrect": int((y_v[m] == 0).sum()),
            "estimable": bool(min((y_v[m] == 1).sum(),
                                  (y_v[m] == 0).sum()) >= 5),
        }
    n_min = min(min(v["n_correct"], v["n_incorrect"])
                for v in strata_n.values())
    log("  conditional-AUROC support: " + ", ".join(
        f"{s}: {v['n_correct']}+/{v['n_incorrect']}-"
        f"{'' if v['estimable'] else ' (NOT estimable)'}"
        for s, v in strata_n.items()))
    if n_min < 25:
        log(f"  THIN STRATUM: the conditional estimate rests on as few as "
            f"{n_min} minority-class items — treat every conditional "
            f"number in this slice as indicative only.")

    report = {"n_items": int(valid.sum()), "availability": avail,
              "health": health, "baseline": baseline,
              "ladder_mode": ladder_mode, "strata_support": strata_n,
              "auroc_conditional": {}, "auroc_pooled": {},
              "contrasts": {}, "controls": {}}

    for name, s in oof_v.items():
        report["auroc_conditional"][name] = stratified_auroc(y_v, s, st_v)
        report["auroc_pooled"][name] = pooled_auroc(y_v, s)

    # Every conditional AUROC gets its matched first-order reference. Without
    # it, 0.716 reads as strong self-knowledge when a judge with none at all
    # scores 0.765 at the same accuracy (C-SDT).
    sdt = sdt_null_reference(rows, n_splits, seed=seed, log=log)
    leak = verdict_only_null(rows, n_splits, seed=seed, log=log)
    if leak:
        report["verdict_only_null"] = leak
    if sdt:
        draws = sdt.pop("_draws")
        report["sdt_null"] = sdt
        # A percentile against the null distribution IS a test; a mean and an
        # SD is not. p = P(null >= observed), floored at 1/(n+1), and these
        # go into the same BH-FDR family as every other contrast.
        floor_leak = (leak or {}).get("leak_auroc_p95", 0.5)
        vs_null, p_null = {}, {}
        for k, v in report["auroc_conditional"].items():
            if v is None:
                vs_null[k] = p_null[k] = None
                continue
            vs_null[k] = v - sdt["null_auroc_mean"]
            p_null[k] = float(max(np.mean(draws >= v), 1.0 / (len(draws) + 1)))
        report["auroc_conditional_vs_null"] = vs_null
        report["p_vs_sdt_null"] = p_null
        # A rung must clear BOTH floors: the first-order arithmetic floor and
        # the verdict-leakage floor from incomplete cancellation.
        report["floor_to_clear"] = float(max(sdt["null_auroc_p95"], floor_leak))
        clears = [k for k, v in report["auroc_conditional"].items()
                  if v is not None and v > report["floor_to_clear"]]
        log(f"  floor to clear = max(SDT p95 {sdt['null_auroc_p95']:.3f}, "
            f"verdict-leak p95 {floor_leak:.3f}) = "
            f"{report['floor_to_clear']:.3f}")
        log("  rungs clearing BOTH floors: "
            + (", ".join(clears) if clears else "NONE — no rung carries "
               "information beyond first-order performance and verdict "
               "leakage"))

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

    # ── Diagnostics for the M4-M3 equivalence ────────────────────────────
    # Neither enters the contrast family: they explain a null, they do not
    # test one, so they cost no multiplicity budget.
    diag = spectral_block_diagnostic(y_v, g_v, sets, n_splits,
                                     activation_probe, pca_dims, log)
    if diag:
        report["controls"]["spectral_block"] = diag

    pc = positive_control(rows, groups, sets, n_splits, activation_probe,
                          pca_dims, log)
    if pc:
        report["controls"]["positive_control"] = pc

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

    # ── Control 1b: SELF-VERDICT decodability (the twin channel) ─────────
    # identity_decodability covers leakage through gt_verdict, which the
    # conditional metric removes. This is its twin, and the conditional
    # metric does NOT remove it: within a stratum, is_correct is exactly
    # (pred == stratum), so a representation that decodes the judge's own
    # verdict is a perfect within-stratum predictor with zero self-knowledge.
    # The last-token hidden state decodes pred essentially perfectly — it is
    # the state the verdict logit is read from — so expect ~1.0 here.
    #
    # We are protected, but by the POOLED fit rather than by the metric: a
    # pure verdict decoder scores 1.0 in one stratum and 0.0 in the other, so
    # the pooled objective never selects that direction. That protection
    # degrades with verdict bias and stratum imbalance, and disappears
    # entirely under a within-stratum fit — see
    # tests/test_ladder.py:test_pooled_fit_is_what_blocks_the_verdict_channel.
    first_pred = sorted({str(r["pred_verdict"]) for r in rows})[0]
    y_pred = np.array([1.0 if str(r["pred_verdict"]) == first_pred else 0.0
                       for r in rows])
    s_pv = oof_scores(X_best, y_pred, groups, n_splits, n_act_best,
                      pca_dims, activation_probe)
    ok_pv = ~np.isnan(s_pv)
    report["controls"]["verdict_decodability"] = {
        "features": best, "auroc": pooled_auroc(y_pred[ok_pv], s_pv[ok_pv])}
    log(f"  control verdict-decodability ({best}): "
        f"{_fmt(report['controls']['verdict_decodability']['auroc'])} "
        f"(near 1.0 is EXPECTED and is why the ladder must be fitted pooled, "
        f"never within stratum)")

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
