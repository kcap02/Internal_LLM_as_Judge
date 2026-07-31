"""Confound diagnostics on judge banks and result files.

Each check returns a dict with a `status` in {PASS, WARN, FAIL} and the
numbers behind it. The point is to find every way a "signal" could be
something other than judge self-knowledge, and to either neutralise it in the
analysis or state it as a documented limitation — before any GPU time is
spent, and again after results exist.

Checks are keyed by the confound IDs used in docs/CONFOUNDS.md.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .prompts import build_judge_prompt


def _auc_from_scores(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC of `scores` separating labels 1 vs 0 (ties averaged)."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(l for _, l in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ── C-BAL: balance of the bank ────────────────────────────────────────────────
def check_balance(items: list[dict]) -> dict:
    out = {}
    worst = "PASS"
    for fmt in sorted({it["format"] for it in items}):
        sub = [it for it in items if it["format"] == fmt]
        counts = Counter(it["gt_verdict"] for it in sub)
        total = sum(counts.values())
        rates = {k: v / total for k, v in counts.items()}
        skew = max(abs(v - 0.5) for v in rates.values()) if len(rates) == 2 else 1.0
        status = "PASS" if skew < 0.02 else ("WARN" if skew < 0.1 else "FAIL")
        out[fmt] = {"counts": dict(counts), "rates": rates, "status": status}
        if status == "FAIL" or (status == "WARN" and worst == "PASS"):
            worst = status
    return {"status": worst, "per_format": out}


# ── C-LEN: does response/prompt length give the answer away? ──────────────────
def check_length_verdict_coupling(items: list[dict], tokenizer=None) -> dict:
    """AUC of predicting gt_verdict from prompt length alone.

    This is the classic chosen-is-longer artefact of preference datasets. An
    AUC far from 0.5 means length alone identifies which item is the "good"
    one, so any length-sensitive representation (all spectral metrics are)
    can impersonate self-knowledge in a POOLED analysis.
    """
    per_format = {}
    worst = "PASS"
    for fmt in sorted({it["format"] for it in items}):
        sub = [it for it in items if it["format"] == fmt]
        labels_seen = sorted({it["gt_verdict"] for it in sub})
        if len(labels_seen) != 2:
            continue
        lens, labs = [], []
        for it in sub:
            h, b = build_judge_prompt(it)
            n = (len(tokenizer(h + b, add_special_tokens=False)["input_ids"])
                 if tokenizer is not None else len(h) + len(b))
            lens.append(float(n))
            labs.append(1 if it["gt_verdict"] == labels_seen[0] else 0)
        auc = _auc_from_scores(lens, labs)
        dev = abs(auc - 0.5)
        status = "PASS" if dev < 0.05 else ("WARN" if dev < 0.15 else "FAIL")
        per_format[fmt] = {"auc_length_predicts_verdict": auc,
                           "positive_label": labels_seen[0],
                           "median_len": sorted(lens)[len(lens) // 2],
                           "status": status}
        if status == "FAIL" or (status == "WARN" and worst == "PASS"):
            worst = status
    return {"status": worst, "per_format": per_format}


# ── C-WIN: spectral window coverage ───────────────────────────────────────────
def check_window_coverage(items: list[dict], tokenizer, window: int) -> dict:
    """Fraction of items exceeding the spectral window (they get skipped).

    Skipping is length-biased, so the retained subset is not the bank; with
    window=4096 this should be zero everywhere.
    """
    per_format = {}
    worst = "PASS"
    for fmt in sorted({it["format"] for it in items}):
        sub = [it for it in items if it["format"] == fmt]
        lens = []
        for it in sub:
            h, b = build_judge_prompt(it)
            lens.append(len(tokenizer(h + b, add_special_tokens=False)["input_ids"]))
        lens.sort()
        over = sum(1 for n in lens if n > window - 8) / len(lens)
        status = "PASS" if over == 0 else ("WARN" if over < 0.02 else "FAIL")
        per_format[fmt] = {
            "median": lens[len(lens) // 2], "p90": lens[int(0.9 * len(lens))],
            "max": lens[-1], "frac_over_window": over, "status": status}
        if status == "FAIL" or (status == "WARN" and worst == "PASS"):
            worst = status
    return {"status": worst, "window": window, "per_format": per_format}


# ── C-DUP: duplicate / near-duplicate questions across groups ─────────────────
def check_duplicate_questions(items: list[dict]) -> dict:
    """Identical question text under different question_ids leaks across folds
    — UNLESS the CV group key is derived from the text, which is what
    `group_id` is for. This check reports the duplication and then verifies
    the grouping actually neutralises it.
    """
    from .grouping import group_key, normalize_text

    by_text = defaultdict(set)
    for it in items:
        by_text[normalize_text(it["question"])].add(it["question_id"])
    dup = {t: sorted(q) for t, q in by_text.items() if len(q) > 1}
    frac = len(dup) / max(1, len(by_text))

    have_gid = all("group_id" in it for it in items)
    neutralised = False
    if have_gid:
        # every duplicated text must map to exactly one group_id
        neutralised = all(
            len({it["group_id"] for it in items
                 if normalize_text(it["question"]) == t}) == 1
            for t in list(dup)[:200])
        # and group_id must be the text-derived key
        neutralised = neutralised and all(
            it["group_id"] == group_key(it["question"]) for it in items[:200])

    if frac == 0:
        status = "PASS"
    elif neutralised:
        status = "PASS"
    else:
        status = "WARN" if frac < 0.02 else "FAIL"
    return {"status": status, "n_duplicate_texts": len(dup),
            "n_unique_texts": len(by_text), "fraction": frac,
            "group_id_present": have_gid,
            "neutralised_by_text_grouping": neutralised,
            "examples": list(dup.values())[:2]}


# ── C-SUBJ: subject concentration ─────────────────────────────────────────────
def check_subject_concentration(items: list[dict]) -> dict:
    """A bank dominated by a few subjects lets 'subject difficulty' pose as
    an internal signal. Subject is a nuisance regressor in the ladder; this
    reports how much work it has to do."""
    counts = Counter(str(it.get("subject") or "") for it in items)
    total = sum(counts.values())
    top = counts.most_common(1)[0] if counts else ("", 0)
    share = top[1] / total if total else 0.0
    ent = -sum((c / total) * math.log(c / total) for c in counts.values() if c)
    max_ent = math.log(len(counts)) if len(counts) > 1 else 1.0
    status = "PASS" if share < 0.35 else ("WARN" if share < 0.6 else "FAIL")
    return {"status": status, "n_subjects": len(counts),
            "top_subject": top[0], "top_share": share,
            "normalized_entropy": ent / max_ent if max_ent else 0.0}


# ── C-PAIR: pairwise counterbalancing ─────────────────────────────────────────
def check_pairwise_counterbalance(items: list[dict]) -> dict:
    """Every pairwise question must appear in both A/B orders, with opposite
    gt_verdicts, so position bias cancels and both orders share a group."""
    pw = [it for it in items if it["format"] == "pairwise"]
    if not pw:
        return {"status": "PASS", "note": "no pairwise items"}
    by_q = defaultdict(list)
    for it in pw:
        by_q[it["question_id"]].append(it)
    bad = [q for q, v in by_q.items()
           if len(v) != 2 or {x["gt_verdict"] for x in v} != {"A", "B"}]
    verdict_counts = Counter(it["gt_verdict"] for it in pw)
    status = "PASS" if not bad else "FAIL"
    return {"status": status, "n_questions": len(by_q),
            "n_unbalanced": len(bad), "verdict_counts": dict(verdict_counts)}


# ── C-SELF: self-preference exposure (MCQ distractors) ────────────────────────
def check_distractor_provenance(bank: dict, judge_models: list[str] | None = None,
                                distractor_panel: list[str] | None = None) -> dict:
    """MCQ distractors derived from a panel that includes the judge itself
    give that judge privileged familiarity with the trap it is shown."""
    items = bank["items"]
    mode = bank.get("mode")
    if mode == "native":
        return {"status": "PASS", "mode": mode,
                "note": "free-text bank; ground truth is dataset-native"}
    srcs = Counter(it.get("neg_source") for it in items if it["format"] == "mcq")
    if mode == "synthetic":
        return {"status": "WARN", "mode": mode, "sources": dict(srcs),
                "note": "random distractors: no self-preference exposure, but "
                        "distractors are often trivial (ceiling effect). Run "
                        "the solver and rebuild for ecological validity."}
    overlap = sorted(set(judge_models or []) & set(distractor_panel or judge_models or []))
    if distractor_panel and not overlap:
        return {"status": "PASS", "mode": mode, "sources": dict(srcs),
                "distractor_panel": distractor_panel,
                "note": "distractor panel is disjoint from the judge panel"}
    return {"status": "WARN", "mode": mode, "sources": dict(srcs),
            "overlap_with_judges": overlap,
            "note": "the judging model helped select its own distractors; set "
                    "config.distractor_panel to a disjoint model set"}


# ── C-TOK: verdict-token stability across contexts ────────────────────────────
def check_verdict_token_stability(tokenizer, items: list[dict],
                                  n_probe: int = 32) -> dict:
    """The verdict token ids are resolved once on a probe prompt and reused.

    If " Yes" tokenises differently after a different context, the readout
    position would be reading the wrong token for those items.
    """
    from .prompts import render, verdict_labels
    from .token_ids import resolve_target_token_ids

    per_format = {}
    worst = "PASS"
    for fmt in sorted({it["format"] for it in items}):
        sub = [it for it in items if it["format"] == fmt][:n_probe]
        labels = verdict_labels(fmt)
        seen = set()
        for it in sub:
            h, b = build_judge_prompt(it)
            prompt = render(h, b, tokenizer, False)
            try:
                ids = resolve_target_token_ids(tokenizer, prompt, labels)
            except ValueError as e:
                per_format[fmt] = {"status": "FAIL", "error": str(e)}
                worst = "FAIL"
                break
            seen.add(tuple(sorted(ids.items())))
        else:
            status = "PASS" if len(seen) == 1 else "FAIL"
            per_format[fmt] = {"status": status, "n_distinct_id_maps": len(seen),
                               "n_probed": len(sub)}
            if status == "FAIL":
                worst = "FAIL"
    return {"status": worst, "per_format": per_format}


# ── C-LABEL: label validity of the `single` format ────────────────────────────
def check_single_label_validity(dataset: str, items: list[dict]) -> dict:
    """`single` turns a RELATIVE preference into an absolute Yes/No.

    Valid for JudgeBench (objectively correct vs incorrect responses); noisy
    for preference sets, where the "rejected" response may still be a fine
    answer, so the No label is partly wrong and caps achievable accuracy.
    """
    has_single = any(it["format"] == "single" for it in items)
    if not has_single:
        return {"status": "PASS", "note": "no single-format items"}
    objective = {"judgebench", "mmlu", "mmlu_pro"}
    if dataset in objective:
        return {"status": "PASS", "dataset": dataset,
                "note": "labels are objective (correct vs incorrect)"}
    return {"status": "WARN", "dataset": dataset,
            "note": "single-format labels come from a relative preference; "
                    "the 'No' class is noisy. Prefer the pairwise arm here, "
                    "and report single-format results as secondary."}


# ── C-NUM: per-model spectral coverage in the RESULTS ─────────────────────────
def audit_spectral_coverage(rows: list[dict]) -> dict:
    """Spectral success/failure per model, with the failure reasons.

    A model whose spectral rows all carry an `error` contributes nothing to
    M2/M4 while every summary still looks healthy — this is how fp16 overflow
    silently removed a quarter of the pilot panel (C-NUM). Coverage is
    therefore reported per model, and zero coverage is a FAIL, not a note.
    """
    per_model: dict = {}
    for r in rows:
        m = r["model"]
        d = per_model.setdefault(m, {"rows": 0, "with_layers": 0, "skipped": 0,
                                     "errors": Counter()})
        d["rows"] += 1
        if r.get("skipped"):
            d["skipped"] += 1
            continue
        sp = r.get("spectral") or {}
        if sp.get("layers"):
            d["with_layers"] += 1
        elif sp.get("error"):
            d["errors"][str(sp["error"])[:120]] += 1

    worst = "PASS"
    out = {}
    for m, d in per_model.items():
        scored = d["rows"] - d["skipped"]
        cov = d["with_layers"] / scored if scored else 0.0
        attempted = d["with_layers"] + sum(d["errors"].values())
        if attempted == 0:
            status = "PASS"          # behavioural-only run: nothing to judge
        elif cov == 0:
            status = "FAIL"
        elif cov < 0.98:
            status = "WARN"
        else:
            status = "PASS"
        if status == "FAIL":
            worst = "FAIL"
        elif status == "WARN" and worst == "PASS":
            worst = "WARN"
        out[m] = {"status": status, "coverage": cov, "rows": d["rows"],
                  "with_layers": d["with_layers"], "skipped": d["skipped"],
                  "errors": dict(d["errors"])}
    return {"status": worst, "per_model": out}


def run_bank_audit(dataset: str, bank: dict, tokenizer=None,
                   window: int = 4096, judge_models: list[str] | None = None,
                   distractor_panel: list[str] | None = None) -> dict:
    """All bank-level checks for one dataset."""
    items = bank["items"]
    checks = {
        "C-BAL  class balance": check_balance(items),
        "C-DUP  duplicate questions": check_duplicate_questions(items),
        "C-SUBJ subject concentration": check_subject_concentration(items),
        "C-PAIR pairwise counterbalance": check_pairwise_counterbalance(items),
        "C-SELF distractor provenance": check_distractor_provenance(
            bank, judge_models, distractor_panel),
        "C-LABEL single-format validity": check_single_label_validity(dataset, items),
    }
    if tokenizer is not None:
        checks["C-LEN  length->verdict leak"] = check_length_verdict_coupling(items, tokenizer)
        checks["C-WIN  spectral window"] = check_window_coverage(items, tokenizer, window)
        checks["C-TOK  verdict-token stability"] = check_verdict_token_stability(tokenizer, items)
    return checks
