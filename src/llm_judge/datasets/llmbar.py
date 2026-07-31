"""LLMBar (princeton-nlp/LLMBar) — adversarial pairwise judging.

The Adversarial subsets are built so the worse output *looks* better
(superficial cues vs. instruction-following). This is where judges fail most
— the strongest testbed for "internal signals predict judge errors".

The HF dataset repo only hosts a legacy loading script (unsupported by
datasets >= 3), so the JSON files are fetched straight from the official
GitHub repo (same source the script used) and cached in data/raw/llmbar/.
Rows: input, output_1, output_2, label (1 or 2 = the better output).
Subset name is kept as `subject` so analysis can stratify Natural vs
Adversarial_*.
"""

from __future__ import annotations

import json
import random
import urllib.request

from ..config import DATA_DIR

_BASE = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/"
SUBSETS = {
    "Natural": "Natural/dataset.json",
    "Adversarial_Neighbor": "Adversarial/Neighbor/dataset.json",
    "Adversarial_GPTInst": "Adversarial/GPTInst/dataset.json",
    "Adversarial_GPTOut": "Adversarial/GPTOut/dataset.json",
    "Adversarial_Manual": "Adversarial/Manual/dataset.json",
}


def _fetch(subset: str, rel_path: str) -> list[dict]:
    cache = DATA_DIR / "raw" / "llmbar" / f"{subset}.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_BASE + rel_path, timeout=60) as r:
            cache.write_bytes(r.read())
    with open(cache, encoding="utf-8") as f:
        return json.load(f)


def load(n_questions: int | None, seed: int) -> dict:
    items = []
    for subset, rel_path in SUBSETS.items():
        rows = _fetch(subset, rel_path)
        for i, r in enumerate(rows):
            label = int(r["label"])
            if label not in (1, 2):
                continue
            resp = {"A": r["output_1"], "B": r["output_2"]}
            better = "A" if label == 1 else "B"
            worse = "B" if better == "A" else "A"
            qid = f"llmbar-{subset}-{i}"
            base = {"dataset": "llmbar", "question_id": qid,
                    "subject": subset, "question": r["input"]}
            items.append({**base, "format": "single",
                          "item_id": f"{qid}_pos", "response": resp[better],
                          "gt_verdict": "Yes"})
            items.append({**base, "format": "single",
                          "item_id": f"{qid}_neg", "response": resp[worse],
                          "gt_verdict": "No"})
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
        rng = random.Random(f"{seed}-llmbar-sample")
        keep = set(rng.sample(qids, min(n_questions, len(qids))))
        items = [it for it in items if it["question_id"] in keep]
    return {"items": items}
