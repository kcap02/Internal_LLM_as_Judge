"""Stage 20 (CPU) — nested-model analysis with honest baselines.

Per (dataset x model x format): the AUROC ladder Mn/M1/M1n/M2/M3/M4 on
grouped out-of-fold predictions, with the CONDITIONAL (within-gt_verdict)
AUROC as the headline metric, paired grouped-bootstrap CIs on the contrasts
that carry the claims, BH-FDR across every contrast in the run, and
difficulty strata for MCQ.

Inputs are merged so no model is lost:
  results/judge_<name>.json           margins + activations coverage
  results/judge_spectral_<name>.json  margins + spectral profiles
  results/activations/<name>_<model>.npz
A model present in only one of the two files still gets every rung its
features allow.

Output:  results/analysis_<name>.json + full log in logs/.

Usage:  python scripts/20_analyse.py [--only mmlu ...] [--no-permutation]
"""

import _bootstrap  # noqa: F401

import argparse
from collections import defaultdict

import numpy as np

from llm_judge.analysis.cv import compare, difficulty_strata
from llm_judge.analysis.stats import benjamini_hochberg
from llm_judge.config import RESULTS_DIR, Config, tagged
from llm_judge.diagnostics import audit_spectral_coverage
from llm_judge.io_utils import atomic_write_json, read_json, read_rows
from llm_judge.log_utils import setup_logging


def merge_sources(judge_rows, spectral_rows, log) -> list[dict]:
    """Union of both result files, keyed by (model, item_id).

    Spectral profiles are grafted onto the behavioural rows. Where both files
    scored the same item, the verdicts must agree — the two stages run the
    same prompt through the same model, so a mismatch means a prompt or
    tokenisation drift between stages and is reported loudly.
    """
    merged: dict[tuple, dict] = {}
    for r in (judge_rows or []):
        merged[(r["model"], r["item_id"])] = dict(r)

    n_overlap = n_disagree = 0
    for r in (spectral_rows or []):
        key = (r["model"], r["item_id"])
        if key in merged:
            n_overlap += 1
            if (merged[key].get("pred_verdict") is not None
                    and r.get("pred_verdict") is not None
                    and merged[key]["pred_verdict"] != r["pred_verdict"]):
                n_disagree += 1
            merged[key]["spectral"] = r.get("spectral")
            merged[key].setdefault("task_start_idx", r.get("task_start_idx"))
        else:
            merged[key] = dict(r)

    if n_overlap:
        rate = n_disagree / n_overlap
        msg = (f"  stage11/stage12 overlap {n_overlap} items, verdict "
               f"disagreement {n_disagree} ({rate:.2%})")
        log.warning(msg + " — investigate prompt drift between stages"
                    ) if rate > 0.01 else log.info(msg)
    return list(merged.values())


def usable(rows):
    return [r for r in rows
            if not r.get("skipped") and r.get("margin") is not None]


def add_peer_difficulty(rows: list[dict]) -> int:
    """Annotate each row with leave-one-model-out peer correctness.

    `peer_difficulty` = fraction of the OTHER judges that got this same item
    right. It becomes a baseline regressor so that spectral/activation
    families must beat "how hard is this item for models in general" before
    the result can be called self-knowledge rather than difficulty tracking.

    Needs >= 2 models on the item; rows without enough peers are left
    unannotated, which disables the control for that slice.
    """
    by_item = defaultdict(list)
    for r in rows:
        by_item[(r["dataset"], r["item_id"])].append(r)
    n = 0
    for group in by_item.values():
        if len(group) < 2:
            continue
        total = sum(bool(r["is_correct"]) for r in group)
        for r in group:
            r["peer_difficulty"] = (total - bool(r["is_correct"])) / (len(group) - 1)
            n += 1
    return n


def load_activations(dataset: str, model: str, tag: str | None = None):
    short = model.split("/")[-1]
    path = RESULTS_DIR / "activations" / f"{tagged(dataset, tag)}_{short}.npz"
    return np.load(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-permutation", action="store_true",
                    help="skip the permutation-null control (faster)")
    ap.add_argument("--tag", default=None,
                    help="analyse a tagged variant run instead of the main one")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("20_analyse", cfg.dump())

    all_contrasts = []          # (dataset, key, contrast_name, result dict)
    reports = {}

    for name in (args.only or cfg.datasets):
        rows = merge_sources(
            read_rows(RESULTS_DIR / f"{tagged('judge_' + name, args.tag)}.jsonl"),
            read_rows(RESULTS_DIR
                      / f"{tagged('judge_spectral_' + name, args.tag)}.jsonl"),
            log)
        if not rows:
            log.info("%s: no judge results yet — skipping", name)
            continue

        # Spectral coverage per model BEFORE modelling: a model whose spectral
        # rows all failed would otherwise vanish from M2/M4 while every
        # summary still looked healthy (C-NUM).
        cov = audit_spectral_coverage(rows)
        for m, d in sorted(cov["per_model"].items()):
            if d["with_layers"] or d["errors"]:
                log.info("  spectral coverage %-28s %-5s %.0f%% (%d/%d)%s",
                         m.split("/")[-1], d["status"], d["coverage"] * 100,
                         d["with_layers"], d["rows"] - d["skipped"],
                         f" errors={d['errors']}" if d["errors"] else "")
        if cov["status"] == "FAIL":
            log.warning("  a model produced NO valid spectral rows — check "
                        "model_dtype (fp16 overflow) before trusting any "
                        "spectral comparison")

        rows = usable(rows)
        n_peer = add_peer_difficulty(rows)
        log.info("%s: %d rows, peer-difficulty available for %d",
                 name, len(rows), n_peer)

        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["model"], r.get("format", "mcq"))].append(r)

        report = {}
        for (model, fmt), rs in sorted(by_key.items()):
            if len(rs) < 40:
                log.info("%s | %s | %s: only %d usable items — skipped",
                         name, model, fmt, len(rs))
                continue
            log.info("### %s | %s | format=%s", name, model.split("/")[-1], fmt)
            npz = load_activations(name, model, args.tag)
            res = compare(rs, n_splits=cfg.n_splits, n_boot=cfg.n_bootstrap,
                          pca_dims=cfg.activation_pca_dims,
                          activations_npz=npz, seed=cfg.seed,
                          activation_probe=cfg.activation_probe,
                          run_permutation_null=not args.no_permutation,
                          log=log.info)
            if res.get("skipped"):
                log.info("  skipped: %s", res["skipped"])
                continue

            log.info("  %-5s %-12s %-12s", "rung", "conditional", "pooled")
            for rung in res["auroc_conditional"]:
                c = res["auroc_conditional"][rung]
                p = res["auroc_pooled"][rung]
                log.info("  %-5s %-12s %-12s", rung,
                         "n/a" if c is None else f"{c:.3f}",
                         "n/a" if p is None else f"{p:.3f}")
            for cname, c in res["contrasts"].items():
                if c["delta"] is None:
                    log.info("  %-10s n/a (degenerate stratum)", cname)
                    continue
                log.info("  %-10s delta=%+.3f  CI95=[%+.3f, %+.3f]  p=%.4f",
                         cname, c["delta"], c["ci95"][0], c["ci95"][1],
                         c["p_one_sided"])
                # A non-significant contrast means nothing on its own: report
                # what size of effect the data actually rule out, so a tight
                # null is distinguishable from an underpowered one.
                # A delta is unreadable without the block-only AUROC beside
                # it: zero with block-only at 0.63 (below the estimator's own
                # detection threshold) means something entirely different from
                # zero with block-only at 0.85.
                only = {"M2": "M2only", "M3": "M3only",
                        "M4": "M3only"}.get(cname.split(" - ")[0])
                alone = (res["auroc_conditional"].get(only)
                         if only else None)
                if alone is not None:
                    log.info("             block alone (%s) = %.3f  "
                             "[a delta near zero is only informative if this "
                             "clears the estimator's detection threshold]",
                             only, alone)

                eq = c.get("equivalence") or {}
                if eq.get("equivalent") is not None:
                    log.info("             equivalence: %s |delta| < %.3f "
                             "(TOST p=%.4f, CI90=[%+.3f, %+.3f])",
                             "RULED OUT effects >=" if eq["equivalent"]
                             else "CANNOT rule out effects >=",
                             eq["band"], eq["p_tost"],
                             eq["ci90"][0], eq["ci90"][1])
                # Unreliable slices never enter the FDR family: including a
                # degenerate judge's contrasts would let an artefact consume
                # the error budget and be reported as a discovery.
                if not res.get("unreliable"):
                    all_contrasts.append((name, f"{model}|{fmt}", cname, c))

            # Clearing the first-order SDT null is a claim like any other and
            # consumes the same error budget. Only the rungs that carry a
            # claim enter; the diagnostic rungs (M2only/M3only) do not.
            if not res.get("unreliable"):
                for rung, p in (res.get("p_vs_sdt_null") or {}).items():
                    if p is None or rung in ("Mn", "M2only", "M3only"):
                        continue
                    all_contrasts.append(
                        (name, f"{model}|{fmt}", f"{rung} > SDT-null",
                         {"delta": (res.get("auroc_conditional_vs_null")
                                    or {}).get(rung),
                          "ci95": (None, None), "p_one_sided": p}))

            strata = {}
            if fmt == "mcq":
                for stratum, srs in difficulty_strata(rs).items():
                    if stratum != "all" and len(srs) >= 60:
                        acc = float(np.mean([r["is_correct"] for r in srs]))
                        strata[stratum] = {"n": len(srs), "judge_acc": acc}
                        log.info("  stratum %-6s n=%4d judge_acc=%.1f%%",
                                 stratum, len(srs), acc * 100)

            report[f"{model}|{fmt}"] = {**res, "strata": strata}

        reports[name] = report
        out = RESULTS_DIR / f"{tagged('analysis_' + name, args.tag)}.json"
        atomic_write_json(out, report)
        log.info("%s: analysis written -> %s", name, out.name)

    # ── Multiplicity control across every contrast in this run ───────────
    if all_contrasts:
        pvals = [c["p_one_sided"] for *_, c in all_contrasts]
        rejected = benjamini_hochberg(pvals, alpha=0.05)
        log.info("=" * 72)
        log.info("BH-FDR over %d contrasts (alpha=0.05): %d survive",
                 len(pvals), sum(rejected))
        for (ds, key, cname, c), rej in zip(all_contrasts, rejected):
            if rej:
                log.info("  SURVIVES  %-12s %-28s %-10s delta=%+.3f p=%.4f",
                         ds, key.split("/")[-1], cname, c["delta"],
                         c["p_one_sided"])
        fdr = {"n_contrasts": len(pvals), "n_survive": int(sum(rejected)),
               "survivors": [{"dataset": ds, "key": key, "contrast": cname,
                              "delta": c["delta"], "p": c["p_one_sided"]}
                             for (ds, key, cname, c), rej
                             in zip(all_contrasts, rejected) if rej]}
        atomic_write_json(RESULTS_DIR / "analysis_fdr.json", fdr)

    log.info("Reading guide: the CONDITIONAL AUROC is the headline — it only "
             "compares items sharing a gt_verdict, so a feature cannot score "
             "by merely telling a pos item from a neg one. Pooled AUROC is "
             "reported for completeness and is inflated by exactly that. A "
             "family contributes only if its contrast over M1n is positive, "
             "its CI clears zero, and it survives BH-FDR.")


if __name__ == "__main__":
    main()
