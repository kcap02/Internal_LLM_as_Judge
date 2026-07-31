"""Dataset registry.

Each loader downloads its source from the HF hub and normalizes it into the
repo's unified schema (data/<name>_raw.json):

MCQ datasets (mmlu, mmlu_pro) produce *questions*:
    {question_id, subject, question, choices: [str], gt_letter, dev: bool}
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

from . import judgebench, llmbar, mmlu, mmlu_pro, rewardbench2

LOADERS = {
    "mmlu": mmlu.load,
    "mmlu_pro": mmlu_pro.load,
    "judgebench": judgebench.load,
    "llmbar": llmbar.load,
    "rewardbench2": rewardbench2.load,
}

# Which kind of artifact each loader emits.
KIND = {
    "mmlu": "questions",
    "mmlu_pro": "questions",
    "judgebench": "items",
    "llmbar": "items",
    "rewardbench2": "items",
}
