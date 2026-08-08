"""Stage 11 (GPU) — behavioral judge run + activation capture.

For every bank item: one forward pass, verdict from target-token logprobs
(Yes/No or A/B), margin saved, and the last-token hidden state at the
configured depth fractions stored to results/activations/<dataset>_<model>.npz
— the activation-probe baseline any spectral claim must beat.

Resumable per (model, item_id). Activations are re-saved after each model.

Usage:  python scripts/11_run_judge.py [--only judgebench ...] [--models ...]
"""

import _bootstrap  # noqa: F401

import argparse

import torch

from llm_judge.config import DATA_DIR, RESULTS_DIR, Config, tagged
from llm_judge.io_utils import ResumableResults, read_json
from llm_judge.log_utils import setup_logging
from llm_judge.model_loading import (free_vram, load_model_safe,
                                     throughput_gate, unload, vram_free_gb)
from llm_judge.prompts import (build_judge_prompt, render, task_token_start,
                               verdict_labels)
from llm_judge.scoring import ActivationStore, margin, score_targets
from llm_judge.token_ids import resolve_target_token_ids


def run_model(model_name, items, dataset, store, cfg, log, limit=None,
              tag=None, allow_slow=False):
    short = model_name.split("/")[-1]
    todo = [it for it in items if not store.is_done(model_name, it["item_id"])]
    if limit:
        todo = todo[:limit]
    if not todo:
        log.info("%s: already complete", short)
        return
    log.info("%s: %d items to judge", short, len(todo))

    acts = ActivationStore(RESULTS_DIR / "activations"
                           / f"{tagged(dataset, tag)}_{short}.npz")
    model = tokenizer = None
    try:
        model, tokenizer, info = load_model_safe(
            model_name, allow_cpu_offload=(model_name not in cfg.no_cpu_offload),
            dtype=cfg.model_dtype)

        # Resolve verdict token ids once per format present in the bank.
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

        max_len = min(getattr(tokenizer, "model_max_length", 4096) or 4096,
                      cfg.max_prompt_tokens) - 8

        # Throughput gate. Under CPU offload each forward transfers weights and
        # throughput can collapse, so time a few real items and REFUSE a
        # campaign that cannot finish, rather than warning and starting one.
        if info.get("mode") == "cpu_offload" and ids_by_format:
            _fmt0 = next(iter(ids_by_format))
            _probe_item = next(it for it in todo if it["format"] == _fmt0)
            _h, _b = build_judge_prompt(_probe_item)
            _p = render(_h, _b, tokenizer, cfg.use_chat_template)

            def _one():
                score_targets(model, tokenizer, _p, ids_by_format[_fmt0],
                              activation_layers=cfg.activation_layers)

            throughput_gate(_one, len(todo), info["mode"],
                            override=allow_slow, log=log.info)

        n_done = 0
        for it in todo:
            fmt = it["format"]
            if fmt not in ids_by_format:
                continue
            try:
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
                pred, lp, activations, n_tok = score_targets(
                    model, tokenizer, prompt, ids_by_format[fmt],
                    activation_layers=cfg.activation_layers)
                acts.add(it["item_id"], activations)
                store.append({
                    "model": model_name,
                    "item_id": it["item_id"],
                    "question_id": it["question_id"],
                    # CV group key (normalised question text) — the analysis
                    # groups on this, not on question_id, so duplicated
                    # questions cannot straddle folds.
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
                    # Position-artefact alarm: if this alone separates
                    # classes, the "signal" is a RoPE/position effect.
                    "task_start_idx": task_token_start(h, b, tokenizer, prompt),
                    "neg_source": it.get("neg_source"),
                    "gt_mean_prob": it.get("gt_mean_prob"),
                    # Provenance for the execution path, not just its name.
                    # `plan_loading` is a function of MACHINE STATE: the same
                    # model and config offloads to CPU with ample free RAM and
                    # to disk without it, with very different throughput and
                    # failure modes. Recording only `load_mode` says what
                    # happened without recording why, so these pin it.
                    "load_mode": info["mode"],
                    "load_devices": info.get("devices"),
                    "free_vram_gb_at_load": info.get("free_vram_gb"),
                    "free_ram_gb_at_load": info.get("free_ram_gb"),
                })
                n_done += 1
                if n_done % 25 == 0:
                    acts.save()
                    free_vram()
                    log.info("  [%d/%d] done", n_done, len(todo))
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM — %s skipped", it["item_id"])
                free_vram()
            except Exception as e:
                log.error("%s: %s: %s", it["item_id"], type(e).__name__, e)
    finally:
        acts.save()
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
                    help="cap items per model (pilot runs)")
    ap.add_argument("--allow-slow", action="store_true",
                    help="override the throughput gate for a campaign the "
                         "probe projects cannot finish")
    ap.add_argument("--tag", default=None,
                    help="variant tag; keeps ablation runs in their own "
                         "result stream instead of colliding with the main one")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("11_run_judge", cfg.dump())
    models = args.models or (cfg.judge_models_pilot if args.pilot
                             else cfg.judge_models)

    for name in (args.only or cfg.datasets):
        bank = read_json(DATA_DIR / f"bank_{name}.json")
        if bank is None:
            log.error("%s: run stages 00+01 first", name)
            continue
        items = bank["items"]
        store = ResumableResults(
            RESULTS_DIR / f"{tagged('judge_' + name, args.tag)}.jsonl")
        log.info("=== dataset %s: %d items (bank mode=%s), resume=%d rows ===",
                 name, len(items), bank.get("mode"), len(store))
        if bank.get("mode") == "synthetic":
            log.warning("%s: bank uses SYNTHETIC distractors — fine for a "
                        "pilot, rebuild from solver logprobs for the paper.",
                        name)
        for model_name in models:
            run_model(model_name, items, name, store, cfg, log,
                      limit=args.limit, tag=args.tag,
                      allow_slow=args.allow_slow)
        store.close()

    log.info("done.")


if __name__ == "__main__":
    main()
