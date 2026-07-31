"""Stage 02 (CPU) — confound audit of the judge banks.

Runs every bank-level check in llm_judge.diagnostics and prints a PASS /
WARN / FAIL table. Run this BEFORE spending GPU time: a bank that fails here
would produce results whose "signal" is an artefact.

Uses a tokenizer only for length-based checks; any tokenizer gives the right
qualitative answer, so a small ungated one is the default.

Usage:  python scripts/02_audit_confounds.py [--only mmlu ...]
                                             [--tokenizer Qwen/Qwen2.5-0.5B-Instruct]
"""

import _bootstrap  # noqa: F401

import argparse
import json

from llm_judge.config import DATA_DIR, RESULTS_DIR, Config
from llm_judge.diagnostics import run_bank_audit
from llm_judge.io_utils import atomic_write_json, read_json
from llm_judge.log_utils import setup_logging

SYMBOL = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip length/window/token checks (offline)")
    ap.add_argument("--pilot", action="store_true",
                    help="audit against the pilot judge panel instead of the "
                         "main one (changes only the C-SELF overlap check)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("02_audit_confounds", cfg.dump())

    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            log.info("tokenizer for length checks: %s", args.tokenizer)
        except Exception as e:
            log.warning("tokenizer unavailable (%s) — skipping length checks", e)

    summary, worst_overall = {}, "PASS"
    for name in (args.only or cfg.datasets):
        bank = read_json(DATA_DIR / f"bank_{name}.json")
        if bank is None:
            log.error("%s: no bank — run stages 00+01 first", name)
            continue
        log.info("=" * 72)
        log.info("BANK %s  (%d items, mode=%s)", name, len(bank["items"]),
                 bank.get("mode"))
        # C-SELF asks whether the models that will JUDGE also helped pick the
        # distractors, so it must be checked against the panel actually used
        # for this campaign — not against every model in the config.
        judges = cfg.judge_models_pilot if args.pilot else cfg.judge_models
        checks = run_bank_audit(name, bank, tokenizer, cfg.spectral_max_len,
                                judge_models=judges,
                                distractor_panel=cfg.distractor_panel)
        summary[name] = checks
        for label, res in checks.items():
            status = res.get("status", "?")
            if status == "FAIL":
                worst_overall = "FAIL"
            elif status == "WARN" and worst_overall == "PASS":
                worst_overall = "WARN"
            detail = {k: v for k, v in res.items() if k != "status"}
            log.info("  [%-4s] %-32s %s", SYMBOL.get(status, status), label,
                     json.dumps(detail, default=str)[:400])

    out = RESULTS_DIR / "audit_banks.json"
    atomic_write_json(out, summary)
    log.info("=" * 72)
    log.info("OVERALL BANK AUDIT: %s   -> %s", worst_overall, out.name)
    log.info("WARN entries are documented limitations, not blockers; FAIL "
             "entries must be fixed before GPU stages.")


if __name__ == "__main__":
    main()
