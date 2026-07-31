"""RewardBench 2 (allenai/reward-bench-2) — best-of-N reward evaluation.

Rows carry a prompt, chosen completion(s) and rejected completion(s). We
normalize to the same two item families as the other free-text sets:

  single   — one chosen (Yes) and one rejected (No) per prompt, seeded pick
             when several are available: balanced 50/50.
  pairwise — chosen vs one rejected, both A/B orders.

The `subset` column (factuality, focus, math, safety, precise-if, ties) is
kept as `subject` for stratified analysis. Ties rows (no strict winner) are
skipped. Safety subset is standard benchmark content used for defensive
evaluation of judges.
"""

from __future__ import annotations

import random

from datasets import load_dataset


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [s for s in x if isinstance(s, str) and s.strip()]


def load(n_questions: int | None, seed: int) -> dict:
    ds = load_dataset("allenai/reward-bench-2", split="test")
    items = []
    for i, r in enumerate(ds):
        subset = str(r.get("subset", ""))
        if subset.lower() == "ties":
            continue  # no strict winner -> not a Yes/No or A/B ground truth
        chosen = _as_list(r.get("chosen"))
        rejected = _as_list(r.get("rejected"))
        if not chosen or not rejected:
            continue
        qid = f"rewardbench2-{subset}-{r.get('id', i)}"
        rng = random.Random(f"{seed}-{qid}")
        pos = rng.choice(chosen)
        neg = rng.choice(rejected)
        base = {"dataset": "rewardbench2", "question_id": qid,
                "subject": subset, "question": r["prompt"]}
        items.append({**base, "format": "single", "item_id": f"{qid}_pos",
                      "response": pos, "gt_verdict": "Yes"})
        items.append({**base, "format": "single", "item_id": f"{qid}_neg",
                      "response": neg, "gt_verdict": "No"})
        items.append({**base, "format": "pairwise", "item_id": f"{qid}_ab",
                      "response_a": pos, "response_b": neg, "gt_verdict": "A"})
        items.append({**base, "format": "pairwise", "item_id": f"{qid}_ba",
                      "response_a": neg, "response_b": pos, "gt_verdict": "B"})

    if n_questions is not None:
        qids = sorted({it["question_id"] for it in items})
        rng = random.Random(f"{seed}-rewardbench2-sample")
        keep = set(rng.sample(qids, min(n_questions, len(qids))))
        items = [it for it in items if it["question_id"] in keep]
    return {"items": items}
