"""MCQ judge-item construction (pos/neg pairs, balanced 50/50).

Three distractor sources, best first:

  logprob   — RECOMMENDED. neg = the wrong letter the solver panel leans
              toward most (mean of per-model renormalized letter probs).
              Exists for EVERY question -> one pair per question, and the
              distractor is ecologically valid: the most tempting wrong
              option, not a random letter.
  bank      — historical. neg = the most *predicted* wrong letter; requires
              >=1 model right and >=1 wrong -> keeps only a fraction.
  synthetic — neg drawn at random (seeded). Distractors often trivial; only
              for bootstrapping before any solver run exists (CPU-only phase).

Every question yields two items: pos (proposed = gt, verdict Yes) and neg
(proposed = distractor, verdict No). The set is deterministic given the seed,
identical for all judges, and no self-judgment is possible.

Difficulty metadata (gt_mean_prob, neg_mean_prob, n_models_correct) is kept
on each item so the analysis can stratify by difficulty.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from .prompts import letters_for


def _base(q: dict, dataset: str) -> dict:
    return {"dataset": dataset, "format": "mcq",
            "subject": q["subject"], "question": q["question"],
            "choices": q["choices"], "gt_letter": q["gt_letter"],
            "question_id": q["question_id"]}


def _pair(q: dict, dataset: str, neg_letter: str, extra: dict) -> list[dict]:
    qid = q["question_id"]
    return [
        {**_base(q, dataset), **extra, "item_id": f"{qid}_pos",
         "proposed_letter": q["gt_letter"], "gt_verdict": "Yes"},
        {**_base(q, dataset), **extra, "item_id": f"{qid}_neg",
         "proposed_letter": neg_letter, "gt_verdict": "No"},
    ]


def make_items_synthetic(questions: list, dataset: str, seed: int) -> list:
    """neg = random wrong letter (seeded)."""
    items = []
    for q in questions:
        letters = letters_for(len(q["choices"]))
        rng = random.Random(f"{seed}-{q['question_id']}")
        wrong = rng.choice([l for l in letters if l != q["gt_letter"]])
        items += _pair(q, dataset, wrong, {"neg_source": "synthetic"})
    return items


def _softmax(lp: dict, letters: list[str]) -> dict:
    """Renormalize letter logprobs to a proper distribution.

    Without this, a very "confident" model (large-amplitude logits) would
    dominate the panel mean; after renormalization each model contributes a
    sum-1 distribution and weighs equally.
    """
    vals = [lp[l] for l in letters]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    tot = sum(exps)
    return {l: e / tot for l, e in zip(letters, exps)}


def make_items_logprob(questions: list, solver_rows: list, dataset: str,
                       min_panel: int = 3, log=print) -> list:
    """neg = wrong letter with the highest panel-mean probability."""
    by_q = {q["question_id"]: q for q in questions}
    probs = defaultdict(lambda: defaultdict(list))   # qid -> letter -> [p]
    preds = defaultdict(list)                        # qid -> [pred_letter]
    models = set()
    for r in solver_rows:
        q = by_q.get(r["question_id"])
        lp = r.get("letter_logprobs")
        if q is None or not lp:
            continue
        letters = letters_for(len(q["choices"]))
        if any(l not in lp for l in letters):
            continue
        p = _softmax(lp, letters)
        for l in letters:
            probs[r["question_id"]][l].append(p[l])
        preds[r["question_id"]].append(r["pred_letter"])
        models.add(r["model"])

    if len(models) < min_panel:
        log(f"  WARNING: only {len(models)} model(s) in the solver results — "
            f"the 'panel' is not one. Distractors will mirror that single "
            f"model, which is then advantaged when judging its own traps. "
            f"Run the solver on the full panel first.")

    items, n_pred, n_lp_only, n_skip = [], 0, 0, 0
    for q in questions:
        qid, gt = q["question_id"], q["gt_letter"]
        if qid not in probs:
            n_skip += 1
            continue
        letters = letters_for(len(q["choices"]))
        mean_p = {l: sum(v) / len(v) for l, v in probs[qid].items()}
        wrong = [l for l in letters if l != gt]
        neg_letter = max(wrong, key=lambda l: mean_p[l])

        was_predicted = neg_letter in preds.get(qid, [])
        n_pred += was_predicted
        n_lp_only += (not was_predicted)
        extra = {"neg_source": "panel_pred" if was_predicted else "panel_logprob",
                 "neg_mean_prob": round(mean_p[neg_letter], 4),
                 "gt_mean_prob": round(mean_p[gt], 4),
                 "n_models_correct": sum(1 for p in preds.get(qid, []) if p == gt)}
        items += _pair(q, dataset, neg_letter, extra)

    log(f"  logprob bank: {len(items) // 2}/{len(questions)} questions kept "
        f"({n_pred} distractors actually predicted, {n_lp_only} from logprobs "
        f"only" + (f", {n_skip} without logprobs" if n_skip else "") + ")")
    return items


def make_items_bank(questions: list, solver_rows: list, dataset: str,
                    seed: int, log=print) -> list:
    """neg = most-predicted wrong letter; requires disagreement in the panel."""
    preds = defaultdict(list)
    for r in solver_rows:
        preds[r["question_id"]].append(r["pred_letter"])
    items, drop_all_ok, drop_all_ko, drop_nopred = [], 0, 0, 0
    for q in questions:
        gt = q["gt_letter"]
        letters = letters_for(len(q["choices"]))
        p = preds.get(q["question_id"], [])
        if not p:
            drop_nopred += 1
            continue
        wrong = [l for l in p if l != gt and l in letters]
        if gt not in p:
            drop_all_ko += 1
            continue
        if not wrong:
            drop_all_ok += 1
            continue
        counts = defaultdict(int)
        for l in wrong:
            counts[l] += 1
        top = max(counts.values())
        cands = sorted(l for l, c in counts.items() if c == top)
        rng = random.Random(f"{seed}-{q['question_id']}")
        items += _pair(q, dataset, rng.choice(cands), {"neg_source": "bank"})
    log(f"  common bank: {len(items) // 2}/{len(questions)} questions kept "
        f"(dropped: {drop_all_ok} all-panel-correct, {drop_all_ko} "
        f"none-correct, {drop_nopred} unpredicted)")
    return items
