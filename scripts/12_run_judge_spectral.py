"""Stage 12 (GPU) — judge run WITH spectral_trust diagnostics per layer.

The expensive stage: eigendecomposition is O(N^3) per layer per item
(~0.4 s/layer at 512 tokens, ~3 s/layer at 1024). Reads the SAME bank as
stage 11 — that identity is what makes behavioral and spectral numbers
comparable. Two forwards per item: (a) plain forward for the verdict (no
attentions allocated), (b) instrumented forward for the spectral profile.

Corrections vs. the legacy script, all verified against spectral_trust 0.2.2:
  - calc_velocity=True      -> Fiedler velocity features
  - subgraph_indices per item -> graph restricted to task (body) tokens,
    controlling for header length (config: spectral_task_subgraph)
  - empty layer_diagnostics is treated as an ERROR, never silently kept

Usage:  python scripts/12_run_judge_spectral.py [--only mmlu ...]
                                                [--models ...] [--dry-run 5]
"""

import _bootstrap  # noqa: F401

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # one tqdm bar per item otherwise

import numpy as np
import torch

from llm_judge.config import DATA_DIR, RESULTS_DIR, Config, tagged
from llm_judge.io_utils import ResumableResults, read_json
from llm_judge.log_utils import setup_logging
from llm_judge.model_loading import (estimate_params_billions, free_vram,
                                     load_model_safe, unload, vram_free_gb)
from llm_judge.prompts import (build_judge_prompt, render, task_token_start,
                               verdict_labels)
from llm_judge.scoring import margin, score_targets
from llm_judge.spectral import (HAS_SPECTRAL_TRUST, GSPDiagnosticsFramework,
                                analyze_prompt, attach_model, build_gsp_config)
from llm_judge.token_ids import resolve_target_token_ids


def run_model(model_name, items, dataset, store, cfg, log, dry_run=None):
    short = model_name.split("/")[-1]
    todo = [it for it in items if not store.is_done(model_name, it["item_id"])]
    if not todo:
        log.info("%s: already complete", short)
        return
    log.info("%s: %d items", short, len(todo))

    model = tokenizer = framework = None
    try:
        model, tokenizer, info = load_model_safe(
            model_name, allow_cpu_offload=(model_name not in cfg.no_cpu_offload))

        ids_by_format = {}
        for fmt in sorted({it["format"] for it in todo}):
            probe = next(it for it in todo if it["format"] == fmt)
            h, b = build_judge_prompt(probe)
            probe_prompt = render(h, b, tokenizer, cfg.use_chat_template)
            try:
                ids_by_format[fmt] = resolve_target_token_ids(
                    tokenizer, probe_prompt, verdict_labels(fmt))
            except ValueError as e:
                log.warning("%s: format %s skipped (%s)", short, fmt, e)

        if HAS_SPECTRAL_TRUST:
            framework = GSPDiagnosticsFramework(build_gsp_config(model_name, cfg))
            framework.__enter__()
            attach_model(framework, model, tokenizer, model_name)
        else:
            log.warning("spectral_trust not installed — verdicts only.")

        model_max = min(getattr(tokenizer, "model_max_length", 4096) or 4096,
                        cfg.max_prompt_tokens)
        max_len = min(model_max, cfg.spectral_max_len) - 8
        log.info("  prompt capped at %d tokens | formats=%s", max_len,
                 sorted(ids_by_format))

        durations = []
        limit = dry_run or len(todo)
        for i, it in enumerate(todo[:limit]):
            fmt = it["format"]
            if fmt not in ids_by_format:
                continue
            try:
                t0 = time.time()
                h, b = build_judge_prompt(it)
                prompt = render(h, b, tokenizer, cfg.use_chat_template)
                n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if n_tok > max_len:
                    store.append({"model": model_name, "item_id": it["item_id"],
                                  "question_id": it["question_id"],
                                  "group_id": it.get("group_id"),
                                  "dataset": dataset, "skipped": "too_long",
                                  "n_tokens_prompt": n_tok})
                    continue

                labels = verdict_labels(fmt)
                pred, lp, _, n_tok = score_targets(model, tokenizer, prompt,
                                                   ids_by_format[fmt])

                start = task_token_start(h, b, tokenizer, prompt)
                spectral = None
                if framework is not None:
                    total = len(tokenizer(prompt)["input_ids"])
                    sub = (list(range(start, total))
                           if cfg.spectral_task_subgraph else None)
                    spectral = analyze_prompt(framework, prompt,
                                              subgraph_indices=sub,
                                              expected_n_tokens=total)
                    if "error" in (spectral or {}):
                        log.warning("%s spectral: %s", it["item_id"],
                                    spectral["error"])

                dt = time.time() - t0
                durations.append(dt)
                store.append({
                    "model": model_name,
                    "item_id": it["item_id"],
                    "question_id": it["question_id"],
                    "group_id": it.get("group_id"),
                    "dataset": dataset,
                    "subject": it.get("subject"),
                    "format": fmt,
                    "gt_verdict": it["gt_verdict"],
                    "pred_verdict": pred,
                    "is_correct": pred == it["gt_verdict"],
                    "verdict_logprobs": lp,
                    "margin": margin(lp, labels),
                    "n_tokens_prompt": n_tok,
                    "task_start_idx": start,
                    "neg_source": it.get("neg_source"),
                    "gt_mean_prob": it.get("gt_mean_prob"),
                    "load_mode": info["mode"],
                    "spectral": spectral,
                })
                if (i + 1) % 20 == 0:
                    free_vram()
                    log.info("  [%d/%d] median %.1f s/item", i + 1, limit,
                             float(np.median(durations)))
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM — %s skipped", it["item_id"])
                free_vram()
            except Exception as e:
                log.error("%s: %s: %s", it["item_id"], type(e).__name__, e)

        if durations:
            med = float(np.median(durations))
            log.info("  median %.1f s/item -> ~%.0f min for %d items",
                     med, med * len(todo) / 60, len(todo))
            if dry_run:
                log.info("  DRY RUN (%d items) — drop --dry-run for the "
                         "full campaign.", dry_run)
    except RuntimeError as e:
        log.warning("%s skipped — %s", short, e)
    finally:
        if framework is not None:
            try:
                framework.__exit__(None, None, None)
            except Exception:
                pass
            try:
                framework.instrumenter.model = None
            except Exception:
                pass
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
    ap.add_argument("--dry-run", type=int, default=None,
                    help="stop after N items per model to estimate duration")
    ap.add_argument("--tag", default=None,
                    help="variant tag; keeps ablation runs in their own "
                         "result stream instead of colliding with the main one")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("12_run_judge_spectral", cfg.dump())
    models = args.models or (cfg.judge_models_pilot if args.pilot
                             else cfg.judge_models)
    # Smallest first: if the biggest fails, the others are already done.
    try:
        models = sorted(models, key=estimate_params_billions)
        for m in models:
            log.info("order: %5.1f B params  %s", estimate_params_billions(m), m)
    except Exception:
        pass

    for name in (args.only or cfg.datasets):
        bank = read_json(DATA_DIR / f"bank_{name}.json")
        if bank is None:
            log.error("%s: run stages 00+01 first", name)
            continue
        items = bank["items"]
        store = ResumableResults(
            RESULTS_DIR / f"{tagged('judge_spectral_' + name, args.tag)}.jsonl")
        log.info("=== dataset %s: %d items (bank mode=%s), resume=%d ===",
                 name, len(items), bank.get("mode"), len(store))
        for model_name in models:
            run_model(model_name, items, name, store, cfg, log,
                      dry_run=args.dry_run)
        store.close()

    log.info("done. Next: python scripts/20_analyse.py")


if __name__ == "__main__":
    main()
