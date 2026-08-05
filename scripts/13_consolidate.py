"""Stage 13 (CPU) — one archival per-item feature file per dataset.

Why this stage exists. The result streams are `judge_<d>.jsonl`,
`judge_spectral_<d>.jsonl` and a loose `.npz` of activations per model, all
gitignored as "regenerable from (commit, config, bank)". That is the right
reproducibility stance for a pipeline whose output is a claim, and the wrong
one for a project whose Phase 2 — transfer, risk-coverage, every diagnostic a
reviewer asks for during rebuttal — is pure re-analysis. Regenerating costs
GPU time that will not be available in November.

So this stage writes ONE self-describing archive per dataset:

    results/features_<dataset>.npz
        rows           JSON array of every merged per-item row (no activations)
        act__<model>   float16 [n_items, hidden] activation matrix
        act_keys__<model>  the item_id::layer_frac key for each row of the above
        provenance     JSON: git commit, package versions INCLUDING the
                       spectral-trust VCS SHA, config, bank digest, timestamp

Downstream stages read this file and need no GPU. It is also the artefact to
archive with a DOI: a reviewable object, which for a measurement paper is part
of the argument rather than a courtesy.

Usage:  python scripts/13_consolidate.py [--only llmbar ...] [--tag TAG]
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from llm_judge.config import RESULTS_DIR, DATA_DIR, Config, tagged
from llm_judge.io_utils import read_rows
from llm_judge.log_utils import setup_logging, _git_commit, _versions


def bank_digest(name: str) -> str:
    """SHA-256 of the bank file: what the judges actually scored."""
    p = DATA_DIR / f"bank_{name}.json"
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def merge(judge_rows, spectral_rows, log) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for r in (judge_rows or []):
        merged[(r["model"], r["item_id"])] = dict(r)
    grafted = 0
    for r in (spectral_rows or []):
        key = (r["model"], r["item_id"])
        if key in merged:
            merged[key]["spectral"] = r.get("spectral")
            merged[key].setdefault("task_start_idx", r.get("task_start_idx"))
            grafted += 1
        else:
            merged[key] = dict(r)
    log("  merged %d behavioural + %d spectral rows (%d grafted)",
        len(judge_rows or []), len(spectral_rows or []), grafted)
    return list(merged.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    logger = setup_logging("13_consolidate", cfg.dump())
    log = logger.info

    for name in (args.only or cfg.datasets):
        jr = read_rows(RESULTS_DIR / f"{tagged('judge_' + name, args.tag)}.jsonl")
        sr = read_rows(RESULTS_DIR
                       / f"{tagged('judge_spectral_' + name, args.tag)}.jsonl")
        if not jr and not sr:
            log("%s: no streams — skipped", name)
            continue
        log("### %s", name)
        rows = merge(jr, sr, log)

        payload: dict[str, np.ndarray] = {}
        act_dir = RESULTS_DIR / "activations"
        n_act_models = 0
        for model in sorted({r["model"] for r in rows}):
            short = model.split("/")[-1]
            p = act_dir / f"{tagged(name, args.tag)}_{short}.npz"
            if not p.exists():
                log("  %s: no activation archive at %s", short, p.name)
                continue
            with np.load(p) as z:
                keys = sorted(z.files)
                if not keys:
                    continue
                mat = np.vstack([z[k] for k in keys]).astype(np.float16)
            payload[f"act__{model}"] = mat
            payload[f"act_keys__{model}"] = np.array(keys, dtype=object)
            n_act_models += 1
            log("  %s: activations %s", short, mat.shape)

        prov = {
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "versions": _versions(),          # carries the spectral-trust SHA
            "config": cfg.dump(),
            "dataset": name,
            "tag": args.tag,
            "bank_sha256_16": bank_digest(name),
            "n_rows": len(rows),
            "n_models": len({r["model"] for r in rows}),
            "n_activation_models": n_act_models,
            "n_with_spectral": sum(
                1 for r in rows if (r.get("spectral") or {}).get("layers")),
        }
        payload["rows"] = np.array(json.dumps(rows), dtype=object)
        payload["provenance"] = np.array(json.dumps(prov, default=str),
                                         dtype=object)

        out = RESULTS_DIR / f"{tagged('features_' + name, args.tag)}.npz"
        np.savez_compressed(out, **payload)
        mb = out.stat().st_size / 1e6
        log("  -> %s (%.1f MB): %d rows, %d models, %d with spectral, "
            "bank %s, spectral_trust %s",
            out.name, mb, prov["n_rows"], prov["n_models"],
            prov["n_with_spectral"], prov["bank_sha256_16"],
            prov["versions"].get("spectral_trust"))


if __name__ == "__main__":
    main()
