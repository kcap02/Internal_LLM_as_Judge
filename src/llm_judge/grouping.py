"""Cross-validation group assignment.

GroupKFold must isolate every item that shares text, not merely every item
that shares a `question_id`. Two ways that breaks if ids are trusted blindly:

* JudgeBench ships its `claude` and `gpt` splits over the SAME questions with
  different response pairs — ~17% of its question texts appear under two ids.
* MMLU / MMLU-Pro contain a handful of questions duplicated across subjects.

In both cases an id-grouped split puts the same question in train and test,
so the model has already seen the essentials and the score is inflated. The
group key is therefore derived from the normalised question TEXT, and the
per-item ids are kept only for joining and reporting.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def group_key(text: str) -> str:
    h = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
    return f"g{h[:16]}"


def assign_group_ids(items: list[dict]) -> dict:
    """Add `group_id` to every item in place; return a small summary."""
    for it in items:
        it["group_id"] = group_key(it["question"])
    n_groups = len({it["group_id"] for it in items})
    n_qids = len({it["question_id"] for it in items})
    return {"n_items": len(items), "n_question_ids": n_qids,
            "n_groups": n_groups, "n_merged_by_text": n_qids - n_groups}
