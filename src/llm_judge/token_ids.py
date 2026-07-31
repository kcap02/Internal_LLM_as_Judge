"""Per-model resolution of target token IDs (letters or verdicts).

The trap: " A" does not necessarily tokenize to the same single token across
tokenizers. We measure by encoding-diff on a *real* context instead of
assuming a fixed token, and explicitly detect collisions between targets —
a collision makes the single-position argmax unreliable for that tokenizer,
so the model is skipped cleanly rather than producing silently wrong scores.
"""

from __future__ import annotations


def resolve_target_token_ids(tokenizer, sample_context: str,
                             targets: list[str]) -> dict[str, int]:
    base_ids = tokenizer(sample_context, add_special_tokens=False)["input_ids"]
    ids: dict[str, int] = {}
    for t in targets:
        full_ids = tokenizer(sample_context + " " + t,
                             add_special_tokens=False)["input_ids"]
        new_ids = full_ids[len(base_ids):]
        if not new_ids:
            raise ValueError(f"no new token for target {t!r}")
        ids[t] = new_ids[0]  # first new token = the one read at position -1
    if len(set(ids.values())) < len(targets):
        raise ValueError(
            f"token collision between targets: {ids} — single-position argmax "
            f"unreliable for this tokenizer")
    return ids
