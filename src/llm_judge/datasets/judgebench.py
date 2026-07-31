"""JudgeBench (ScalerLab/JudgeBench) — response pairs with objective labels.

Each row: a question, two responses (one objectively correct, one flawed),
label "A>B" or "B>A". Two item families are emitted per pair:

  single   — each response shown alone, verdict Yes (better one) / No (worse
             one): balanced 50/50 by construction, same machinery as MCQ.
  pairwise — both responses shown, in BOTH orders (counterbalanced), verdict
             A/B: position bias is measurable and cancels in aggregate.
"""

from __future__ import annotations

import random

from datasets import load_dataset


def load(n_questions: int | None, seed: int) -> dict:
    items = []
    for split in ("gpt", "claude"):
        try:
            ds = load_dataset("ScalerLab/JudgeBench", split=split)
        except Exception:
            continue
        for i, r in enumerate(ds):
            label = r["label"].strip()
            if label not in ("A>B", "B>A"):
                continue  # skip ties/unlabeled
            better, worse = (("A", "B") if label == "A>B" else ("B", "A"))
            resp = {"A": r["response_A"], "B": r["response_B"]}
            qid = f"judgebench-{split}-{r.get('pair_id', i)}"
            base = {"dataset": "judgebench", "question_id": qid,
                    "subject": r.get("source", split),
                    "question": r["question"]}
            # single: Yes for the better response, No for the worse one
            items.append({**base, "format": "single",
                          "item_id": f"{qid}_pos", "response": resp[better],
                          "gt_verdict": "Yes"})
            items.append({**base, "format": "single",
                          "item_id": f"{qid}_neg", "response": resp[worse],
                          "gt_verdict": "No"})
            # pairwise, both orders
            items.append({**base, "format": "pairwise",
                          "item_id": f"{qid}_ab",
                          "response_a": resp["A"], "response_b": resp["B"],
                          "gt_verdict": better})
            items.append({**base, "format": "pairwise",
                          "item_id": f"{qid}_ba",
                          "response_a": resp["B"], "response_b": resp["A"],
                          "gt_verdict": {"A": "B", "B": "A"}[better]})

    if n_questions is not None:
        qids = sorted({it["question_id"] for it in items})
        rng = random.Random(f"{seed}-judgebench-sample")
        keep = set(rng.sample(qids, min(n_questions, len(qids))))
        items = [it for it in items if it["question_id"] in keep]
    return {"items": items}
