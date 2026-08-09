"""Stage 23 (CPU) — cross-judge transfer: can we predict judge C's errors
without ever observing judge C?

This is the test that turns "peer difficulty correlates with correctness" into
a method. Peer difficulty surviving BH-FDR on its own slice is a within-judge
statement and is close to tautological: it is built from other models'
correctness on the same items, so of course it tracks difficulty.

The method claim is stronger and falsifiable: fit a predictor of correctness on
judges A and B, then apply it to a HELD-OUT judge C whose rows were never seen
in training. If that transfers, a practitioner can predict a new judge's errors
from signals obtainable without access to it. If it does not, peer difficulty is
a per-judge correlate and there is no method.

Three feature sets, all evaluated under the same conditional (within-stratum)
metric and the same grouping as the main analysis:

  external   peer difficulty from the TRAINING judges only + nuisance
             (length, position, subject). Nothing from the held-out judge.
  margin     the held-out judge's own confidence margin alone.
  both       external + margin.

`external` is the deployable one: it needs other judges' verdicts on the same
items, which is black-box and cheap, and nothing from the judge under test.

Usage:  python scripts/23_transfer.py --only mmlu mmlu_pro --tag cheap
"""

import _bootstrap  # noqa: F401

import argparse
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_judge.analysis import features as F
from llm_judge.analysis.stats import stratified_auroc
from llm_judge.config import RESULTS_DIR, Config, tagged
from llm_judge.io_utils import atomic_write_json, read_rows
from llm_judge.log_utils import setup_logging


def peer_difficulty_from(rows_by_model: dict, train_models: list,
                         item_id: str) -> float | None:
    """Fraction of the TRAINING judges that got this item right.

    Deliberately excludes the held-out judge, so the feature is computable
    before that judge exists.
    """
    vals = [rows_by_model[m][item_id]["is_correct"]
            for m in train_models if item_id in rows_by_model[m]]
    return float(np.mean(vals)) if vals else None


def build(rows: list, peer: dict):
    """(X_external, X_margin, y, groups, strata) for one held-out judge."""
    keep = [r for r in rows if peer.get(r["item_id"]) is not None]
    if not keep:
        return None
    Xn, _ = F.nuisance_features(keep)
    pd_ = np.array([peer[r["item_id"]] for r in keep], float)[:, None]
    X_ext = np.c_[pd_, pd_ ** 2, Xn]
    X_mar = F.margin_features(keep)
    y = np.array([float(r["is_correct"]) for r in keep])
    g = np.array([r.get("group_id") or r["question_id"] for r in keep])
    s = np.array([str(r["gt_verdict"]) for r in keep])
    return X_ext, X_mar, y, g, s


def fit_predict(X_tr, y_tr, X_te):
    if len(np.unique(y_tr)) < 2:
        return None
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, C=1.0))
    p.fit(X_tr, y_tr)
    return p.predict_proba(X_te)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("23_transfer", cfg.dump()).info
    out: dict = {}

    for name in (args.only or cfg.datasets):
        rows = read_rows(
            RESULTS_DIR / f"{tagged('judge_' + name, args.tag)}.jsonl")
        rows = [r for r in rows if not r.get("skipped")
                and r.get("margin") is not None]
        if not rows:
            log(f"{name}: no rows")
            continue

        by_fmt = defaultdict(list)
        for r in rows:
            by_fmt[r.get("format", "mcq")].append(r)

        log(f"\n=== {name} ===")
        out[name] = {}
        for fmt, frows in sorted(by_fmt.items()):
            models = sorted({r["model"] for r in frows})
            if len(models) < 3:
                log(f"  {fmt}: needs >=3 judges, has {len(models)}")
                continue
            idx = defaultdict(dict)
            for r in frows:
                idx[r["model"]][r["item_id"]] = r

            log(f"  {fmt}: {len(models)} judges, holding each out in turn")
            log(f"    {'held-out judge':30}{'external':>10}{'margin':>9}"
                f"{'both':>8}{'n':>7}")
            res = {}
            for held in models:
                train = [m for m in models if m != held]
                peer = {i: peer_difficulty_from(idx, train, i)
                        for i in idx[held]}
                b_te = build(list(idx[held].values()), peer)
                tr_rows, tr_peer = [], {}
                for m in train:
                    others = [x for x in train if x != m]
                    for i, r in idx[m].items():
                        pv = peer_difficulty_from(idx, others, i)
                        if pv is not None:
                            tr_rows.append(r)
                            tr_peer[(m, i)] = pv
                if not b_te or not tr_rows:
                    continue
                Xe_te, Xm_te, y_te, g_te, s_te = b_te
                te_rows = [r for r in idx[held].values()
                           if peer.get(r["item_id"]) is not None]

                # Nuisance columns must be built on train and test TOGETHER.
                # The subject dummies are those above a frequency threshold, so
                # a 2400-row training set and an 800-row test set otherwise
                # yield different column counts and the design matrices do not
                # align.
                Xn_all, _ = F.nuisance_features(tr_rows + te_rows)
                Xn_tr, Xn_te = Xn_all[:len(tr_rows)], Xn_all[len(tr_rows):]

                pd_tr = np.array([tr_peer[(r["model"], r["item_id"])]
                                  for r in tr_rows], float)[:, None]
                Xe_tr = np.c_[pd_tr, pd_tr ** 2, Xn_tr]
                pd_te = np.array([peer[r["item_id"]] for r in te_rows],
                                 float)[:, None]
                Xe_te = np.c_[pd_te, pd_te ** 2, Xn_te]
                Xm_tr = F.margin_features(tr_rows)
                y_tr = np.array([float(r["is_correct"]) for r in tr_rows])
                scores = {}
                for lbl, (A, B) in (("external", (Xe_tr, Xe_te)),
                                    ("margin", (Xm_tr, Xm_te)),
                                    ("both", (np.c_[Xe_tr, Xm_tr],
                                              np.c_[Xe_te, Xm_te]))):
                    sc = fit_predict(A, y_tr, B)
                    scores[lbl] = (stratified_auroc(y_te, sc, s_te)
                                   if sc is not None else None)
                res[held] = {**scores, "n": int(len(y_te))}
                f = lambda v: "  n/a" if v is None else f"{v:.3f}"
                log(f"    {held.split('/')[-1][:29]:30}"
                    f"{f(scores['external']):>10}{f(scores['margin']):>9}"
                    f"{f(scores['both']):>8}{len(y_te):>7}")
            out[name][fmt] = res

    p = RESULTS_DIR / f"{tagged('transfer', args.tag)}.json"
    atomic_write_json(p, out, tag=args.tag)
    log(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
