"""Stage 20 (CPU) — nested-model analysis with honest baselines.

Per (dataset x model x format): the AUROC ladder Mn/M1/M1n/M2/M3/M4 on
grouped out-of-fold predictions, paired grouped-bootstrap CIs on the deltas
that carry the claims, and difficulty strata for MCQ.

Inputs:  results/judge_spectral_<name>.json  (preferred: margins + spectral)
         results/judge_<name>.json           (fallback: margins only)
         results/activations/<name>_<model>.npz (optional: probe baseline)
Output:  results/analysis_<name>.json + full log in logs/.

Usage:  python scripts/20_analyse.py [--only mmlu ...]
"""

import _bootstrap  # noqa: F401

import argparse
from collections import defaultdict

import numpy as np

from llm_judge.analysis.cv import compare, difficulty_strata
from llm_judge.config import RESULTS_DIR, Config
from llm_judge.io_utils import atomic_write_json, read_json
from llm_judge.log_utils import setup_logging


def usable(rows):
    return [r for r in rows
            if not r.get("skipped") and r.get("margin") is not None]


def load_activations(dataset: str, model: str):
    short = model.split("/")[-1]
    path = RESULTS_DIR / "activations" / f"{dataset}_{short}.npz"
    if path.exists():
        return np.load(path)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("20_analyse", cfg.dump())

    for name in (args.only or cfg.datasets):
        rows = (read_json(RESULTS_DIR / f"judge_spectral_{name}.json")
                or read_json(RESULTS_DIR / f"judge_{name}.json"))
        if not rows:
            log.info("%s: no judge results yet — skipping", name)
            continue

        by_key = defaultdict(list)
        for r in usable(rows):
            by_key[(r["model"], r.get("format", "mcq"))].append(r)

        report = {}
        for (model, fmt), rs in sorted(by_key.items()):
            if len(rs) < 40:
                log.info("%s | %s | %s: only %d usable items — skipped",
                         name, model, fmt, len(rs))
                continue
            log.info("### %s | %s | format=%s", name, model.split("/")[-1], fmt)
            npz = load_activations(name, model)
            res = compare(rs, n_splits=cfg.n_splits, n_boot=cfg.n_bootstrap,
                          pca_dims=cfg.activation_pca_dims,
                          activations_npz=npz, seed=cfg.seed,
                          activation_probe=cfg.activation_probe, log=log.info)
            for label, auc in res["aurocs"].items():
                log.info("  %-34s AUROC %.3f", label, auc)
            for label, c in res["contrasts"].items():
                log.info("  %-12s delta=%+.3f  CI95=[%+.3f, %+.3f]  p=%.4f",
                         label, c["delta"], c["ci95"][0], c["ci95"][1],
                         c["p_one_sided"])

            # Difficulty strata (MCQ only; needs panel logprob metadata)
            strata = {}
            if fmt == "mcq":
                for stratum, srs in difficulty_strata(rs).items():
                    if stratum != "all" and len(srs) >= 60:
                        acc = float(np.mean([r["is_correct"] for r in srs]))
                        strata[stratum] = {"n": len(srs), "judge_acc": acc}
                        log.info("  stratum %-6s n=%4d judge_acc=%.1f%%",
                                 stratum, len(srs), acc * 100)

            report[f"{model}|{fmt}"] = {"n": len(rs), **res, "strata": strata}

        out = RESULTS_DIR / f"analysis_{name}.json"
        atomic_write_json(out, report)
        log.info("%s: analysis written -> %s", name, out.name)

    log.info("Reading guide: 0.50 = no information. A family contributes only "
             "if its delta over M1n is positive with a CI clear of zero. If "
             "M3 (activations) already matches M2 (spectral), the spectral "
             "story needs the selectivity/velocity angle, not raw AUROC.")


if __name__ == "__main__":
    main()
