"""Stage 01 (CPU) — build the judge banks every judge scores.

For MCQ datasets: pos/neg pairs with distractors from the solver panel's
logprobs (results/solver_<name>.json, produced by stage 10) when available,
otherwise synthetic seeded distractors (bootstrap mode — rebuild with --force
after the solver runs; the bank file records which mode produced it).

For free-text datasets: items are already ground-truthed; the bank is the
normalized item file passed through unchanged (single + pairwise).

All downstream stages read ONLY data/bank_<name>.json — this is what
guarantees the behavioral and spectral runs score exactly the same items.

Usage:  python scripts/01_build_judge_banks.py [--only mmlu ...] [--force]
"""

import _bootstrap  # noqa: F401

import argparse

from llm_judge.config import DATA_DIR, RESULTS_DIR, Config
from llm_judge.grouping import assign_group_ids
from llm_judge.io_utils import atomic_write_json, read_json, read_rows
from llm_judge.items import make_items_logprob, make_items_synthetic
from llm_judge.log_utils import setup_logging
from llm_judge.registry import KIND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("01_build_judge_banks", cfg.dump())

    for name in (args.only or cfg.datasets):
        out = DATA_DIR / f"bank_{name}.json"
        if out.exists() and not args.force:
            log.info("%s: %s exists — skipping (--force to rebuild)", name,
                     out.name)
            continue

        if KIND.get(name) == "questions":
            questions = read_json(DATA_DIR / f"{name}_questions.json")
            if questions is None:
                log.error("%s: run stage 00 first", name)
                continue
            solver = read_rows(RESULTS_DIR / f"solver_{name}.jsonl")
            if solver:
                items = make_items_logprob(questions, solver, name,
                                           panel_models=cfg.distractor_panel,
                                           log=log.info)
                mode = "logprob"
            else:
                items = make_items_synthetic(questions, name, cfg.seed)
                mode = "synthetic"
                log.warning("%s: no solver results yet — SYNTHETIC distractors "
                            "(random wrong letters). Rebuild with --force after "
                            "stage 10 for ecologically valid distractors.", name)
        else:
            items = read_json(DATA_DIR / f"{name}_items.json")
            if items is None:
                log.error("%s: run stage 00 first", name)
                continue
            mode = "native"

        # CV group = normalised question TEXT, not question_id: JudgeBench
        # reuses questions across its claude/gpt splits and the MCQ sets have
        # a few cross-subject duplicates. Grouping on ids would leak those.
        grp = assign_group_ids(items)
        if grp["n_merged_by_text"]:
            log.info("%s: %d question_ids collapsed into shared text groups "
                     "(duplicate questions that would otherwise leak across "
                     "folds)", name, grp["n_merged_by_text"])

        atomic_write_json(out, {"dataset": name, "mode": mode, "seed": cfg.seed,
                                "grouping": grp, "items": items})
        log.info("%s: bank written (%d items, %d question_ids, %d CV groups, "
                 "mode=%s) -> %s", name, len(items), grp["n_question_ids"],
                 grp["n_groups"], mode, out.name)

    log.info("done.")


if __name__ == "__main__":
    main()
