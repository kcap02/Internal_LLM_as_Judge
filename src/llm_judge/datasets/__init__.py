"""Dataset loaders (CPU-only; requires the HuggingFace `datasets` library).

Loaders are imported LAZILY: importing this package must not pull in
`datasets`, so that GPU stages which merely touch `llm_judge.registry` can
run in an environment where `datasets` is unavailable. Use `get_loader(name)`.

Each loader downloads its source and normalizes it into the repo's unified
schema (written by stage 00 into data/):

MCQ datasets (mmlu, mmlu_pro) produce *questions*:
    {question_id, subject, question, choices: [str], gt_letter}
Judge banks (pos/neg pairs) are derived from questions later by items.py —
ideally from panel solver logprobs (ecologically valid distractors), or
synthetically when no solver run exists yet.

Free-text datasets (judgebench, llmbar, rewardbench2) produce *judge items*
directly, since ground truth about response quality is built in:
    {item_id, question_id, dataset, subject, format, gt_verdict, ...}
with format "single" (Yes/No on one response) and "pairwise" (A/B), both
balanced by construction (every question contributes one Yes and one No item;
every pair is shown in both A/B orders).
"""

from __future__ import annotations

import importlib

from ..registry import KIND  # noqa: F401  (re-exported for convenience)

_MODULES = {
    "mmlu": "mmlu",
    "mmlu_pro": "mmlu_pro",
    "judgebench": "judgebench",
    "llmbar": "llmbar",
    "rewardbench2": "rewardbench2",
}


def get_loader(name: str):
    """Return the `load(n_questions, seed) -> dict` callable for `name`."""
    if name not in _MODULES:
        raise KeyError(f"unknown dataset {name!r} (known: {sorted(_MODULES)})")
    mod = importlib.import_module(f".{_MODULES[name]}", __package__)
    return mod.load


DATASET_NAMES = sorted(_MODULES)
