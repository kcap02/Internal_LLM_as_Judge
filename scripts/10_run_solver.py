"""Stage 10 (GPU) — MCQ solver runs (canonical Hendrycks logprob scoring).

One forward pass per question; letter logprobs are stored so stage 01 can
build ecologically valid distractors. Resumable per (model, question_id).

Usage:  python scripts/10_run_solver.py [--only mmlu mmlu_pro]
                                        [--models Qwen/Qwen2.5-7B-Instruct ...]
"""

import _bootstrap  # noqa: F401

import argparse

import torch

from llm_judge.config import DATA_DIR, RESULTS_DIR, Config
from llm_judge.io_utils import ResumableResults, read_json
from llm_judge.registry import KIND
from llm_judge.log_utils import setup_logging
from llm_judge.model_loading import free_vram, load_model_safe, unload, vram_free_gb
from llm_judge.prompts import format_mcq_example, gen_solver_prompt, letters_for
from llm_judge.scoring import score_targets
from llm_judge.token_ids import resolve_target_token_ids


def fit_prompt(tokenizer, dev_examples, subject, k_max, q, max_len):
    """Drop few-shot exemplars until the prompt fits (k=0 always returns)."""
    prompt = ""
    for k in range(k_max, -1, -1):
        prompt = (gen_solver_prompt(dev_examples, subject, k)
                  + format_mcq_example(q["question"], q["choices"]))
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= max_len or k == 0:
            return prompt, k
    return prompt, 0


def run_model(model_name, questions, dev_by_subject, store, cfg, log, limit=None):
    short = model_name.split("/")[-1]
    todo = [q for q in questions if not store.is_done(model_name, q["question_id"])]
    if limit:
        todo = todo[:limit]
    if not todo:
        log.info("%s: already complete", short)
        return
    log.info("%s: %d questions to score", short, len(todo))
    model = tokenizer = None
    try:
        model, tokenizer, _info = load_model_safe(model_name)
        probe = todo[0]
        probe_prompt = (gen_solver_prompt(dev_by_subject.get(probe["subject"], []),
                                          probe["subject"], 0)
                        + format_mcq_example(probe["question"], probe["choices"]))
        letters = letters_for(max(len(q["choices"]) for q in todo))
        try:
            letter_ids = resolve_target_token_ids(tokenizer, probe_prompt, letters)
        except ValueError as e:
            log.warning("%s skipped: %s", short, e)
            return
        max_len = min(getattr(tokenizer, "model_max_length", 4096) or 4096,
                      cfg.max_prompt_tokens) - 8

        for i, q in enumerate(todo):
            try:
                dev_ex = dev_by_subject.get(q["subject"], [])
                prompt, k_used = fit_prompt(tokenizer, dev_ex, q["subject"],
                                            cfg.ntrain, q, max_len)
                ls = letters_for(len(q["choices"]))
                ids = {l: letter_ids[l] for l in ls}
                pred, letter_lp, _, n_tok = score_targets(model, tokenizer,
                                                          prompt, ids)
                store.append({
                    "model": model_name,
                    "question_id": q["question_id"],
                    "subject": q["subject"],
                    "gt_letter": q["gt_letter"],
                    "pred_letter": pred,
                    "is_correct": pred == q["gt_letter"],
                    "letter_logprobs": letter_lp,
                    "k_shot_used": k_used,
                    "n_tokens_prompt": n_tok,
                })
                if (i + 1) % 50 == 0:
                    log.info("  [%d/%d] running acc so far this model: n/a",
                             i + 1, len(todo))
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM — %s skipped", q["question_id"])
                free_vram()
            except Exception as e:
                log.error("%s: %s", q["question_id"], e)
    finally:
        unload(model)
        del tokenizer
        free_vram()
        log.info("%s: unloaded, %.1f GB VRAM free", short, vram_free_gb())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="use the <4B pilot panel instead of the main panel")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions per model (pilot runs)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("10_run_solver", cfg.dump())
    models = args.models or (cfg.judge_models_pilot if args.pilot
                             else cfg.judge_models)

    mcq_sets = [n for n in (args.only or cfg.datasets)
                if KIND.get(n) == "questions"]
    for name in mcq_sets:
        questions = read_json(DATA_DIR / f"{name}_questions.json")
        dev = read_json(DATA_DIR / f"{name}_dev.json", default=[]) or []
        if questions is None:
            log.error("%s: run stage 00 first", name)
            continue
        dev_by_subject = {}
        for r in dev:
            dev_by_subject.setdefault(r["subject"], []).append(r)
        store = ResumableResults(RESULTS_DIR / f"solver_{name}.json",
                                 key_fields=("model", "question_id"))
        log.info("=== dataset %s: %d questions, resume=%d rows ===",
                 name, len(questions), len(store))
        for model_name in models:
            run_model(model_name, questions, dev_by_subject, store, cfg, log,
                      limit=args.limit)

    log.info("done. Rebuild banks now: python scripts/01_build_judge_banks.py --force")


if __name__ == "__main__":
    main()
