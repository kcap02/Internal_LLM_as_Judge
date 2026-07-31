"""MMLU-Pro (TIGER-Lab/MMLU-Pro) — up to 10 options, harder distractors.

Harder than MMLU and with 10 choices, so the panel's distractor logprobs are
much richer; the main MCQ arm should run here once the pilot validates.
"""

from __future__ import annotations

import random

from datasets import load_dataset

from ..prompts import letters_for


def load(n_questions: int | None, seed: int) -> dict:
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    def to_question(r, i):
        choices = [c for c in r["options"] if c is not None and c != "N/A"]
        ls = letters_for(len(choices))
        gt_idx = int(r["answer_index"])
        if gt_idx >= len(choices):
            return None
        return {"question_id": f"mmlupro-{r['category']}-{r.get('question_id', i)}",
                "subject": r["category"],
                "question": r["question"],
                "choices": choices,
                "gt_letter": ls[gt_idx]}

    questions = [q for q in (to_question(r, i) for i, r in enumerate(test)) if q]
    if n_questions is not None and n_questions < len(questions):
        rng = random.Random(f"{seed}-mmlupro-sample")
        questions = rng.sample(questions, n_questions)

    dev_examples = []
    for r in val:
        choices = [c for c in r["options"] if c is not None and c != "N/A"]
        gt_idx = int(r["answer_index"])
        if gt_idx < len(choices):
            dev_examples.append({"subject": r["category"], "question": r["question"],
                                 "choices": choices, "answer": gt_idx})
    return {"questions": questions, "dev": dev_examples}
