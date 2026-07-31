# Confound register

Every way a result in this repo could be something other than *"the judge's
internal state predicts when its verdict is wrong"*, what was done about it,
and how to re-check it.

Status legend: **NEUTRALISED** — the design or analysis removes it.
**MEASURED** — it exists, is quantified, and is reported as a limitation.
**OPEN** — not yet addressed.

Bank-level checks run in `scripts/02_audit_confounds.py` (CPU, before GPU
time). Analysis-level controls run inside `scripts/20_analyse.py`.

---

## C-ID — Item identity impersonating self-knowledge  *(the critical one)*

**Status: NEUTRALISED**

The target is `is_correct = (pred_verdict == gt_verdict)`. Whenever a judge
has any verdict bias — and every judge does — `is_correct` is largely
determined by `gt_verdict` alone. So **any** feature that merely tells a pos
item from a neg item scores well without carrying a single bit about
self-knowledge, and spectral/activation features do that trivially: the two
prompts contain different text.

Left alone, this alone could produce the entire M2/M3 "lift".

*Fix.* The headline metric is the **conditional (stratified) AUROC**
(`analysis/stats.py:stratified_auroc`): items are only ever compared with
other items sharing the same `gt_verdict`. Within a stratum the item identity
is constant, so the only thing left to predict is the judge's own behaviour.
Pooled AUROC is still reported, clearly labelled as the inflated number.

*Evidence it matters.* The `identity_decodability` control reports how well
the same features predict `gt_verdict` itself. A high value there with a
flat conditional AUROC is exactly the failure mode this prevents.

---

## C-DUP — Duplicate questions straddling CV folds

**Status: NEUTRALISED** (was FAIL)

The audit found **JudgeBench reuses the same questions across its `claude`
and `gpt` splits**: 92 of 528 question texts (17.4%) appear under two
`question_id`s with different response pairs. MMLU and MMLU-Pro have a
handful of cross-subject duplicates. Grouping folds on `question_id` would
put the same question in train and test.

*Fix.* `llm_judge/grouping.py` derives `group_id` from the **normalised
question text**; banks store it and `GroupKFold` uses it. JudgeBench
collapses 620 ids into 528 real groups.

---

## C-LEN — Length giving the answer away

**Status: MEASURED, mitigated**

Preference datasets have the classic chosen-is-longer artefact, and every
spectral metric is length-sensitive. Measured AUC of *predicting the verdict
from prompt length alone*:

| bank | pairwise | single |
|---|---|---|
| mmlu / mmlu_pro | — | 0.50 (clean) |
| judgebench | 0.500 | 0.504 |
| llmbar | 0.500 | **0.626** |
| rewardbench2 | 0.500 | **0.565** |

**Counterbalancing makes the pairwise arm immune**: both A/B orders contain
the same two responses, so length is identical across the pair and the AUC is
exactly 0.500. This is a strong reason to treat **pairwise as the primary
free-text format**.

*Mitigation for `single`*: length and log-length are nuisance regressors in
`M1n`, so spectral/activation families must beat them; and the conditional
AUROC compares only within a verdict class, where the length gap is not the
thing being predicted.

---

## C-POS — Position / RoPE artefacts

**Status: NEUTRALISED**

Header length varies with dataset and subject name, so the judged content
starts at different token indices. If that index alone separates classes, any
"signal" is a position artefact.

*Fix.* `task_start_idx` is recorded per item and is a nuisance regressor in
`M1n`. Spectral graphs are additionally restricted to task tokens via
`subgraph_indices`, so the header is not part of the analysed graph at all.

---

## C-WIN — Length-biased spectral coverage

**Status: NEUTRALISED** (was a 47% skip rate)

At a 1024-token window, ~47% of JudgeBench and ~24% of RewardBench 2 items
exceeded it and were skipped — a length-biased retained subset.

*Fix.* `spectral_max_len = 4096`. Measured maximum over all banks is 3131
tokens, so **0% of items are skipped anywhere**. Cost scales with each item's
actual length, not with the cap (dense eigh ≈ 0.3 s/layer at 1024 tokens,
≈ 6 s/layer at 4096), so raising the cap is free for short items and simply
pays the true price for long ones instead of silently dropping them.

---

## C-SELF — Self-preference in MCQ distractors

**Status: MEASURED, fix available**

MCQ distractors are the wrong answer the solver panel finds most tempting. If
the judging model is in that panel, it is being shown a trap it helped
select — its own inclinations, not a neutral distractor.

*Fix available.* `config.distractor_panel` restricts distractor construction
to a named model set. Set it **disjoint from the judge panel** (e.g. build
distractors with the <4B pilot panel, judge with the 7–27B panel) and the
audit flips to PASS. Currently the MCQ banks are `synthetic` (no solver run
yet), which has no self-preference exposure but produces easier distractors.

---

## C-LABEL — `single` format turns preference into absolute truth

**Status: MEASURED (inherent limitation)**

Showing one response and asking "is this correct?" is valid where labels are
objective (JudgeBench: correct vs incorrect; MCQ). For LLMBar and
RewardBench 2 the label is a *relative* preference — the rejected response
may still be a perfectly good answer, so the "No" class is partly mislabelled
and caps achievable accuracy.

*Handling.* Report the **pairwise arm as primary** for those two datasets and
`single` as secondary; state the noise ceiling.

---

## C-DEGEN — Judges with no verdict variance

**Status: NEUTRALISED**

A judge that always answers "Yes" has `is_correct` exactly equal to
`gt_verdict == Yes`; within-stratum outcome variance is zero and conditional
AUROC is undefined. Pooled AUROC would still print a confident-looking
number.

*Fix.* `health_checks()` reports accuracy, verdict rate and per-stratum
accuracy, and flags `degenerate: true`. Undefined contrasts are reported as
`n/a`, never silently filled.

---

## C-LEAK — Leakage inside the CV pipeline itself

**Status: NEUTRALISED, actively tested**

*Fix.* Scalers and PCA are fitted **inside each training fold**
(`ColumnTransformer` in the pipeline, never pre-fitted on all data). A
**permutation-null control** shuffles labels within `gt_verdict` strata and
refits the entire pipeline; conditional AUROC must land near 0.500, and a
deviation > 0.08 logs a `LEAKAGE WARNING`.

---

## C-SUBSET — Rungs compared on different items

**Status: NEUTRALISED**

If M2 were computed on spectral-covered items and M3 on activation-covered
items, their AUROCs would not be comparable.

*Fix.* `select_rows()` intersects to the rows usable by **every** rung before
any model is fitted, and logs how many were dropped and why.

---

## C-MULT — Multiple comparisons

**Status: NEUTRALISED**

Datasets × models × formats × contrasts yields dozens of p-values.

*Fix.* Benjamini–Hochberg FDR at α = 0.05 across every contrast in a run,
written to `results/analysis_fdr.json`. Only survivors may be called results.

---

## C-INFER — "Fold std" mistaken for a significance test

**Status: NEUTRALISED**

*Fix.* Paired **grouped bootstrap** on out-of-fold scores: resample whole
question groups, recompute both metrics on identical resamples, report the CI
of the difference. Bootstrap p-values are floored at `1/(n+1)` — a p of
exactly 0 is not something resampling can support.

*Documented caveat.* The bootstrap resamples OOF predictions without refitting
per resample, so it slightly understates variance from model refitting. This
is standard practice and is stated in the paper.

---

## C-TOK — Verdict-token instability

**Status: NEUTRALISED, tested**

Verdict token ids are resolved once per model on a probe prompt and reused.
If " Yes" tokenised differently after another context, the readout would read
the wrong token.

*Fix.* `check_verdict_token_stability()` re-resolves the ids on 32 real
prompts per format and asserts a single id map. All banks and all pilot
tokenizers: PASS.

---

## C-STAGE — Stage 11 / stage 12 drift

**Status: NEUTRALISED, monitored**

Both GPU stages score the same prompt with the same model, so verdicts must
agree; a mismatch would mean prompt or tokenisation drift between stages.

*Fix.* `merge_sources()` reports the overlap and the disagreement rate, and
escalates to a warning above 1%.

---

## C-FORMAT — Raw prompts on instruct models

**Status: OPEN (deliberate, revisit after pilot)**

`use_chat_template=False` applies raw Hendrycks-style prompts uniformly to all
models, which is comparable across the panel but off-distribution for
instruct models. If pilot judges land near chance, verdict labels are mostly
noise and no feature can show a lift — a *power* problem, not a negative
result.

*Plan.* Compare both settings on the pilot panel (`use_chat_template` is a
config key) and adopt whichever gives non-degenerate verdict behaviour;
report the choice.

---

## Engineering defects found and fixed along the way

| defect | consequence if unfixed |
|---|---|
| `Model loading.py` (space in filename) | stages 2–3 crash on import |
| `import torch` before `pyarrow` | interpreter segfault, no traceback |
| provenance logger imported `datasets` to read its version | took down every GPU-env run (broken Windows cert store) |
| per-item full-JSON rewrite | O(n²) writes; **rows silently lost** to Windows file locks — now append-only JSONL |
| `KeyError` on missing activations | crashed the whole analysis instead of dropping rows |
