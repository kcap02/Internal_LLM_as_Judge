"""Stage 00 (CPU) — download and normalize all datasets.

Writes, per dataset:
  data/<name>_questions.json + data/<name>_dev.json   (MCQ datasets)
  data/<name>_items.json                              (free-text datasets)

Idempotent: existing files are kept unless --force. Seeded: same config ->
byte-identical sampling.

Usage:  python scripts/00_download_datasets.py [--config configs/foo.json]
                                               [--only mmlu judgebench]
                                               [--force]
"""

import _bootstrap  # noqa: F401

import argparse

from llm_judge.config import DATA_DIR, Config
from llm_judge.datasets import DATASET_NAMES, get_loader
from llm_judge.io_utils import atomic_write_json
from llm_judge.log_utils import setup_logging
from llm_judge.registry import KIND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    log = setup_logging("00_download_datasets", cfg.dump())
    names = args.only or cfg.datasets

    for name in names:
        if name not in DATASET_NAMES:
            log.error("unknown dataset %r (known: %s)", name, DATASET_NAMES)
            continue
        kind = KIND[name]
        out_main = DATA_DIR / (f"{name}_questions.json" if kind == "questions"
                               else f"{name}_items.json")
        if out_main.exists() and not args.force:
            log.info("%s: %s exists — skipping (--force to redo)", name,
                     out_main.name)
            continue

        log.info("%s: downloading & normalizing ...", name)
        n = cfg.n_questions.get(name)
        payload = get_loader(name)(n, cfg.seed)

        if kind == "questions":
            atomic_write_json(out_main, payload["questions"], tag=None)
            atomic_write_json(DATA_DIR / f"{name}_dev.json", payload["dev"], tag=None)
            log.info("%s: %d questions, %d dev examples -> %s", name,
                     len(payload["questions"]), len(payload["dev"]),
                     out_main.name)
        else:
            atomic_write_json(out_main, payload["items"], tag=None)
            n_q = len({it["question_id"] for it in payload["items"]})
            by_fmt = {}
            for it in payload["items"]:
                by_fmt[it["format"]] = by_fmt.get(it["format"], 0) + 1
            log.info("%s: %d items (%s) over %d questions -> %s", name,
                     len(payload["items"]), by_fmt, n_q, out_main.name)

    log.info("done.")


if __name__ == "__main__":
    main()
