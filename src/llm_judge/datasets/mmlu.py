"""MMLU (cais/mmlu) — 4-choice MCQ, the controlled pilot dataset.

Sampling is seeded and stratified-by-availability: a fixed random subsample
of the test split, plus the full dev split for few-shot examples.
"""

from __future__ import annotations

import random

from datasets import load_dataset

from ..prompts import letters_for

LETTERS = letters_for(4)


def load(n_questions: int | None, seed: int) -> dict:
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")

    questions = [
        {"question_id": f"mmlu-{r['subject']}-{i}",
         "subject": r["subject"],
         "question": r["question"],
         "choices": list(r["choices"]),
         "gt_letter": LETTERS[r["answer"]]}
        for i, r in enumerate(test)
    ]
    if n_questions is not None and n_questions < len(questions):
        rng = random.Random(f"{seed}-mmlu-sample")
        questions = rng.sample(questions, n_questions)

    dev_examples = [
        {"subject": r["subject"], "question": r["question"],
         "choices": list(r["choices"]), "answer": r["answer"]}
        for r in dev
    ]
    return {"questions": questions, "dev": dev_examples}
