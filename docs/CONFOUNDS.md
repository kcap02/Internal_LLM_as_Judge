# Confound register

Every way a result in this repo could be something other than *"the judge's
internal state predicts when its verdict is wrong"*, what was done about it,
and how to re-check it.

Status legend: **NEUTRALISED** — the design or analysis removes it.
**MEASURED** — it exists, is quantified, and is reported as a limitation.
**RESOLVED** — tested and shown not to bite. **OPEN** — not yet addressed.

Bank-level checks run in `scripts/02_audit_confounds.py` (CPU, before GPU
time). Analysis-level controls run inside `scripts/20_analyse.py`.

## Scoreboard

| id | confound | status | mechanism |
|---|---|---|---|
| C-ID | item identity ≠ self-knowledge | NEUTRALISED | conditional (within-verdict) AUROC |
| C-DIFF | item difficulty ≠ self-knowledge | NEUTRALISED | peer-difficulty baseline rung M1nd |
| C-DUP | duplicate questions across folds | NEUTRALISED | text-derived `group_id` |
| C-DEGEN | degenerate judges manufacturing discoveries | NEUTRALISED | min-class strata, contrast suppression, FDR exclusion |
| C-LEAK | leakage inside the CV | NEUTRALISED | per-fold fitting + permutation null |
| C-SUBSET | rungs on different items | NEUTRALISED | common subset, coverage floor |
| C-MULT | multiple comparisons | NEUTRALISED | BH-FDR over all contrasts |
| C-INFER | fold std ≠ significance | NEUTRALISED | paired grouped bootstrap |
| C-POS | position / RoPE artefacts | NEUTRALISED | task-token subgraph + `task_start_idx` nuisance |
| C-WIN | length-biased spectral coverage | NEUTRALISED | 4096 window → 0% skipped by the bank; VRAM bound computed, warned, and recorded |
| C-NUM | fp16 overflow deleting a model's spectral data | NEUTRALISED | bfloat16 default, uniform across the panel |
| C-TOK | verdict-token instability | NEUTRALISED | re-resolved on 32 prompts/format |
| C-STAGE | stage 11/12 drift | NEUTRALISED | agreement monitored (0.00% observed) |
| C-FORMAT | raw prompts on instruct models | RESOLVED | ablation: no material effect |
| C-LEN | length gives the verdict away | MEASURED | pairwise immune (0.500); nuisance regressor for `single` |
| C-SELF | self-preference in distractors | MEASURED | fix ready: `configs/main.json` disjoint panel |
| C-LABEL | `single` = preference as absolute truth | MEASURED | inherent; pairwise is primary |

No FAIL-level confound remains, and no analysis-level confound is merely
"noted" — each is either removed by construction or has a control that would
expose it. The three MEASURED items are properties of the source datasets
rather than of this pipeline: they are quantified and reported, and two of
them (C-LEN, C-LABEL) are the reason **pairwise is the primary free-text
format**. C-SELF flips to PASS as soon as the solver run lands and the MCQ
banks are rebuilt with the disjoint panel in `configs/main.json`.

Four of these were found by *running* the pilot, not by reading the code:
C-DUP (JudgeBench reusing 92 questions across splits), C-DEGEN (two artefact
contrasts surviving FDR), C-NUM (fp16 silently deleting a model's spectral
data), and the VRAM half of C-WIN. That is the argument for keeping the
pilot cheap and the audits standing rather than one-off.

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

## C-DIFF — Item difficulty impersonating self-knowledge

**Status: NEUTRALISED** (when ≥2 judges score the same items)

The sharpest objection to a positive result: perhaps the features encode how
*hard the item is*, not anything about this particular judge. Difficulty is a
property of the item shared by every model — a probe that only recovers it
has learned nothing self-referential, even though it would look like a
genuine internal signal on the conditional metric.

*Fix.* `peer_difficulty` — the fraction of the **other** judges that got the
same item right, computed leave-one-model-out in `20_analyse.py` — enters the
baseline rung as **M1nd**. When it is available, the ladder's baseline
becomes M1nd rather than M1n, so spectral (M2) and activation (M3) families
must beat *"how hard is this item for models in general"* before anything may
be called self-knowledge. The active baseline is printed per slice and stored
as `baseline` in the report.

*Limitation.* Needs at least two judges over the same items; single-model
slices fall back to M1n and say so explicitly.

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

| bank | mcq | pairwise | single |
|---|---|---|---|
| mmlu | 0.498 | — | — |
| mmlu_pro | 0.497 | — | — |
| judgebench | — | 0.500 | 0.504 |
| llmbar | — | 0.500 | **0.626** |
| rewardbench2 | — | 0.500 | **0.565** |

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

**Status: NEUTRALISED at the bank level; VRAM-bounded per model**

At a 1024-token window, ~47% of JudgeBench and ~24% of RewardBench 2 items
exceeded it and were skipped — a length-biased retained subset, the worst
kind of missing data for a length-sensitive metric.

*Fix.* `spectral_max_len = 4096`. The measured maximum over all banks is 3131
tokens, so **no item is excluded by the window**. Compute scales with each
item's actual length, not the cap (dense eigh ≈ 0.3 s/layer at 1024 tokens,
≈ 6 s/layer at 4096), so raising it is free for short items and simply pays
the honest price for long ones.

*The remaining bound is VRAM, not the window.* With eager attention and
`output_attentions=True`, a `[heads, N, N]` tensor is retained for **every**
layer, so the requirement is quadratic in length:

| model | 1024 tok | 2048 tok | 4096 tok |
|---|---|---|---|
| Qwen2.5-3B | 1.1 GB | 4.5 GB | 18.0 GB |
| Qwen2.5-7B | 1.5 GB | 6.1 GB | 24.5 GB |

A full-window item therefore does **not** fit a 16 GB card. Left alone, the
per-item OOM handler would skip exactly the longest items and quietly
recreate the bias this entry is about. Three guards:

1. `attention_memory_gb()` computes the requirement and stage 12 reserves it
   as load headroom instead of the flat default.
2. Before the run, a warning fires if a full-window item cannot fit, naming
   the fix (lower `spectral_max_len` for that model, or use a bigger card).
3. OOM skips are **written to the results stream** as `skipped: "oom"` and
   counted, with an explicit "coverage is now length-biased" warning — never
   a silent gap.

*Practical consequence.* On the 16 GB card: LLMBar (max 1119 tok) is safe at
any window; RewardBench 2 (max 2441) fits for ≤3B; JudgeBench pairwise (max
3131) needs a larger card or a reduced window, and that choice must be
reported per dataset.

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

## C-NUM — Numerical precision silently deleting a model, and mixed dtype

**Status: NEUTRALISED** (found by the pilot)

Loading in **float16**, Qwen2.5-1.5B produced attention values that overflow
to `inf`, and `spectral_trust` then raised *"array must not contain infs or
NaNs"* on **every one of its 400 items**. The verdicts still landed (the
spectral failure is caught per item by design), so nothing crashed and
nothing looked wrong — the model simply contributed zero spectral rows, and
the analysis correctly but silently dropped its spectral family. A quarter of
the pilot panel had no spectral data and only the per-item error field said
so.

*Fix.* `config.model_dtype = "bfloat16"` is the default for every stage. Same
memory as fp16, far wider exponent range. Re-running the identical model and
items: **6/6 valid** where fp16 gave 0/400.

*Second-order confound.* Precision must be **uniform across the panel**:
spectral metrics computed at different dtypes are not comparable, so a mixed
run would confound "model" with "numerical precision". The fp16 pilot stream
is archived as `__fp16` and the pilot was re-run end to end under bf16.

*Standing check.* `spectral: {"error": ...}` rows are counted per model in the
audit; a model whose spectral coverage is 0 must be investigated, never
averaged over.

---

## C-DEGEN — Judges with no verdict variance manufacturing discoveries

**Status: NEUTRALISED** (caught by the pilot, three layers of fix)

A judge that always answers "Yes" has `is_correct` exactly equal to
`gt_verdict == Yes`: within-stratum outcome variance is zero and conditional
AUROC is undefined. Pooled AUROC still prints a confident-looking number.

The pilot showed this is not hypothetical and is *worse* than it looks.
Qwen2.5-0.5B answered "Yes" to 99.5% of LLMBar `single` items and
Llama-3.2-1B chose "B" on 98.4% of `pairwise` items. The conditional AUROC
was then estimated from a stratum containing **one** minority-class item; the
bootstrap, resampling that same item, returned a *narrow* CI around an
arbitrary value. Two such contrasts (`delta=+0.980`, `delta=+0.375`)
**survived BH-FDR and were reported as discoveries.**

*Fix, in three layers.*
1. `stratified_auroc` ignores any stratum with fewer than
   `MIN_CLASS_PER_STRATUM = 5` items of either class; if none qualifies it
   returns `None`.
2. `health_checks()` flags `degenerate` (zero variance) and `extreme_bias`
   (verdict rate <5% or >95%); `compare()` then **suppresses every contrast**
   for that slice rather than estimating one.
3. Unreliable slices are excluded from the BH-FDR family, so an artefact
   cannot consume the error budget.

After the fix the same pilot data yields **0 of 16 surviving contrasts** —
the correct answer for judges this small.

*Consequence for the study.* Judges below ~3B are not usable as measurement
subjects; they serve only as pipeline tests and as distractor panels.

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

**Status: RESOLVED — measured, no material effect**

`use_chat_template=False` applies raw Hendrycks-style prompts uniformly to
all models: comparable across the panel, but off-distribution for instruct
models. The worry was that the pilot's verdict degeneracy was a *format*
artefact, which would make every downstream null a power problem rather than
a result.

*Test.* The same 400 LLMBar items, both settings, four pilot judges
(`--tag chat` keeps the variant in its own stream). Bias = P(pred == first
label); 50% is unbiased, 0/100% is degenerate.

| model | format | raw acc | raw bias | chat acc | chat bias |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | pairwise | 54.0% | 40.0% | 53.5% | 33.5% |
| Qwen2.5-0.5B | single | 50.5% | 0.5% | 55.0% | 9.0% |
| Qwen2.5-1.5B | pairwise | 50.0% | 100.0% | 50.0% | 100.0% |
| Qwen2.5-1.5B | single | 50.0% | 0.0% | 50.0% | 0.0% |
| Llama-3.2-1B | pairwise | 49.5% | 1.5% | 53.5% | 19.5% |
| Llama-3.2-1B | single | 56.0% | 51.0% | 49.5% | 96.5% |
| **Qwen2.5-3B** | **pairwise** | **83.5%** | **41.5%** | 82.0% | 44.0% |
| Qwen2.5-3B | single | 64.0% | 65.0% | 61.0% | 74.0% |

*Conclusion.* The chat template does not rescue the small models —
Qwen2.5-1.5B is fully degenerate under **both** formats, so the degeneracy is
a capability limit, not a prompting artefact. For the one competent judge the
two formats agree to within 1.5 points. **Raw prompts are kept** for
uniformity across base and instruct models, and this table is the
justification.

*Two design conclusions fall out.* Judges must be ≳3B to be measurable at
all; and **pairwise is the primary format** — it is both far more accurate
for a competent judge (83.5% vs 64.0%) and structurally immune to the length
confound (C-LEN AUC exactly 0.500 by counterbalancing).

---

## Engineering defects found and fixed along the way

| defect | consequence if unfixed |
|---|---|
| `Model loading.py` (space in filename) | stages 2–3 crash on import |
| `import torch` before `pyarrow` | interpreter segfault, no traceback |
| provenance logger imported `datasets` to read its version | took down every GPU-env run (broken Windows cert store) |
| per-item full-JSON rewrite | O(n²) writes; **rows silently lost** to Windows file locks — now append-only JSONL |
| `KeyError` on missing activations | crashed the whole analysis instead of dropping rows |
