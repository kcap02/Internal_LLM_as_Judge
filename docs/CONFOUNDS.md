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
| C-FORMAT | raw prompts on instruct models | RESOLVED for competent judges | ablation: ≤1.5 pt; small-model half pending a bf16 re-run |
| C-LEN | length gives the verdict away | MEASURED | pairwise immune (0.500); nuisance regressor for `single` |
| C-SELF | self-preference in distractors | NEUTRALISED | distractor panel disjoint from judges (`configs/main.json`) |
| C-LABEL | `single` = preference as absolute truth | MEASURED | inherent; pairwise is primary |
| C-POWER | an underpowered null read as a negative result | MEASURED | MDE from the pilot's own bootstrap (stage 21) + TOST equivalence |
| C-ABSENCE | "adds nothing" vs "was never there" | NEUTRALISED | `M2only`/`M3only` rungs, spectral-block coefficient norms, positive control |
| C-VERDICT | self-verdict decoding ≠ self-knowledge | NEUTRALISED | pooled fit (guarded by test), `pred_verdict` in the baseline, `verdict_decodability` |
| C-LADDER | wide block swamps the baseline it extends | NEUTRALISED | offset/residual ladder (`ladder_mode="offset"`) |
| C-SDT | margin↔competence trend is signal-detection arithmetic | MEASURED | SDT null reproduces it without any self-knowledge |
| C-VER | library version drift mid-campaign | NEUTRALISED | `spectral-trust` pinned; sym-path metrics verified stable across 0.2.1→0.2.3 |

No FAIL-level confound remains, and no analysis-level confound is merely
"noted" — each is either removed by construction or has a control that would
expose it. The MMLU bank audits **PASS on every check**. The two remaining
MEASURED items are properties of the source datasets rather than of this
pipeline: they are quantified, reported, and are exactly why **pairwise is
the primary free-text format**.

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

**Status: NEUTRALISED (demonstrated)**

MCQ distractors are the wrong answer the solver panel finds most tempting. If
the judging model sits in that panel, it is being shown a trap it helped
select — its own inclinations rather than a neutral distractor, which
inflates or deflates its score for reasons that have nothing to do with
judging.

*Fix.* `config.distractor_panel` restricts distractor construction to a named
model set; `configs/main.json` sets it to the **<4B pilot panel, disjoint from
the 7–27B judge panel**. Distractors are then chosen by models that never
judge them.

*Demonstrated.* Solver run over the pilot panel → rebuild → audit:

```
[PASS] C-SELF  {"mode": "logprob",
                "sources": {"panel_pred": 552, "panel_logprob": 248},
                "note": "distractor panel is disjoint from the judge panel"}
OVERALL BANK AUDIT: PASS
```

Two-thirds of the distractors were options a panel model actually chose; the
rest are the wrong option the panel leaned toward most without selecting it.
Both are ecologically valid, and the split is recorded per item in
`neg_source` so the analysis can condition on it.

*Operational note.* The bank only covers questions for which solver logprobs
exist, so the solver must run over the **whole** question set before the final
rebuild — a partial solver pass silently shrinks the bank (it reports the
count, e.g. `400/2000 questions kept`).

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

## C-NUM — Numerical precision corrupting verdicts and deleting spectral data

**Status: NEUTRALISED** (found by the pilot; the most consequential finding)

Loading in **float16**, Qwen2.5-1.5B produced attention values that overflow
to `inf`. Two distinct harms followed, one loud and one silent.

**Harm 1 — spectral data deleted.** `spectral_trust` raised *"array must not
contain infs or NaNs"* on **every one of its 400 items**. Verdicts still
landed, because a spectral failure is caught per item by design, so nothing
crashed and no summary looked wrong: the model simply contributed zero
spectral rows and the analysis silently dropped its spectral family. Only the
per-item `error` field recorded it.

**Harm 2 — the verdicts themselves were wrong.** This is worse, and it was
only visible after fixing Harm 1 and comparing:

| model / format | fp16 acc | fp16 bias | bf16 acc | bf16 bias | verdicts agreeing |
|---|---|---|---|---|---|
| Qwen2.5-1.5B pairwise | 50.0% | **100.0%** | **73.5%** | **36.5%** | **36.5%** |
| Qwen2.5-1.5B single | 50.0% | 0.0% | 58.0% | 14.0% | 86.0% |
| Qwen2.5-3B pairwise | 83.5% | 41.5% | 82.5% | 41.5% | 98.0% |
| Qwen2.5-0.5B pairwise | 54.0% | 40.0% | 52.0% | 42.0% | 95.0% |
| Llama-3.2-1B pairwise | 49.5% | 1.5% | 48.5% | 2.5% | 99.0% |

Under fp16, Qwen2.5-1.5B looked like a **fully degenerate judge** (100% one
verdict, chance accuracy). Under bf16 it is a **competent** one (73.5%,
well-balanced), and the two dtypes agree on only 36.5% of its verdicts. Every
other model was stable to within ~2 points.

*The wrong conclusion this nearly produced.* On the fp16 data the pilot
concluded "judges below ~3B are capability-limited, and a chat-template
ablation confirms it is not a prompting artefact". Both arms of that ablation
were fp16, so the evidence was invalid for this model: 1.5B was not
capability-limited at all, it was numerically broken. **A numerical bug was
about to be written up as a finding about model scale.**

*Fix.* `config.model_dtype = "bfloat16"` for every stage — same memory as
fp16, far wider exponent range — and precision must be **uniform across the
panel**, since neither spectral metrics nor verdicts are comparable across
dtypes. The fp16 streams are archived as `__fp16` (as the evidence above) and
the whole pilot was re-run under bf16.

*Standing checks.* `audit_spectral_coverage()` counts `spectral.error` rows
per model and FAILs at zero coverage; the fp16-vs-bf16 verdict-agreement
comparison above is the template for validating any future dtype change.

*Promoted into the library (0.2.3).* Detecting this downstream was luck. It is
now a hard precondition inside `spectral_trust` itself:
`assert_finite_attention()` runs on every instrumented forward pass and raises
`NonFiniteAttentionError` at the first layer carrying NaN/Inf, naming the
model, the dtype and the offending counts. Anyone doing attention-graph work
gets the check for free, and this pipeline can no longer produce a run where
the failure is visible only in a per-item `error` field. See C-VER.

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

Under **fp16** (archived as `__fp16` / `__chat_fp16`), the format changed
competent judges by ≤1.5 points — Qwen2.5-3B pairwise 83.5% raw vs 82.0%
chat — while both arms showed Qwen2.5-1.5B as fully degenerate. That last
part was an fp16 artefact (C-NUM), so the ablation must be re-run under bf16
before its *small-model* rows mean anything; the ≤1.5-point conclusion for
competent judges is unaffected, since those models were dtype-stable to ~2
points.

Current **bf16** raw baseline, for the re-run to be compared against:

| model | pairwise acc | pairwise bias |
|---|---|---|
| Qwen2.5-0.5B | 52.0% | 42.0% |
| Llama-3.2-1B | 48.5% | 2.5% |
| Qwen2.5-1.5B | 73.5% | 36.5% |
| **Qwen2.5-3B** | **82.5%** | **41.5%** |

*Conclusion.* Prompt format is not what separates a usable judge from a
degenerate one — precision (C-NUM) and scale are. **Raw prompts are kept**
for uniformity across base and instruct models.

*Outstanding.* Re-run `--tag chat` under bf16 to complete the small-model half
of this table. Command in DESIGN.md; it is one stage-11 pass (~5 min) and must
not run concurrently with a spectral job.

*Design conclusion that survives.* **Pairwise is the primary format**: far
more accurate than `single` for a competent judge (82.5% vs 64.0% on
Qwen2.5-3B) and structurally immune to the length confound (C-LEN AUC exactly
0.500 by counterbalancing).

*Revised scale statement.* With fp16 corrected, the floor is lower than the
first pilot suggested: **Qwen2.5-1.5B is usable** (73.5% pairwise, balanced),
while Qwen2.5-0.5B (52.0%) and Llama-3.2-1B (48.5%, 2.5% bias) remain at
chance and degenerate. Judges must be ≳1.5B, not ≳3B — and any claim about
scale must be made on dtype-matched data.

---

## C-POWER — An underpowered null read as a negative result

**Status: MEASURED** (quantified by `scripts/21_power.py`)

The pilot returned **0 of 30 contrasts surviving BH-FDR**. That is only a
finding if the study could have detected the effect it was looking for. It
could not.

The pilot's own paired bootstrap already carries the answer: the width of each
contrast's CI *is* the SE of the estimator at n=200, with the grouping and the
stratification already priced in. Stage 21 converts it to a minimum detectable
effect. Measured over the 18 estimable primary contrasts on LLMBar
(median SE = 0.034 at n = 200 per slice):

| level | MDE (conditional-AUROC lift, 80% power) |
|---|---|
| α = 0.05, no correction | **0.083** |
| BH-FDR, m = 30 contrasts | **0.127** |

The literature-plausible effect is 0.03–0.05. **The pilot null is therefore
uninformative about the primary question** — it rules out only very large
effects. Items needed per slice to detect a 0.04 lift at 80% power:

| contrast family | items per slice |
|---|---|
| m = 30 (current design) | ~2,000 (10× the pilot) |
| m = 3 primary tests only | ~1,240 |

Two consequences, both design decisions rather than caveats:

1. **Buy items, not models.** Each extra judge multiplies the contrast family
   and costs α without adding per-test power. Cutting the family from 30 to 3
   is worth ~40% of the required sample on its own.
2. **`M4 - M3` is already adequately powered** (SE 0.005–0.009, MDE ≈ 0.02–0.03
   under FDR, n needed ≈ 50–150). Its pilot deltas run −0.010 to +0.009. So
   *"the attention-graph features add nothing beyond a linear activation
   probe"* is a real, adequately-powered negative — unlike the M2/M3-vs-
   baseline contrasts, which are simply unresolved.

*Fix for reporting.* `paired_bootstrap` now returns a TOST **equivalence**
block against a pre-specified negligible band (default ±0.02 conditional
AUROC), read off the same bootstrap so the grouping is respected. A contrast
is reported as *"rules out effects ≥ band"* or *"cannot rule out effects ≥
band"* — never as a bare "not significant". Stage 20 logs it per contrast.

*Why not paired DeLong.* The suggestion to replace the bootstrap with DeLong
to recover power rests on the bootstrap being unpaired; it is not — both
metrics are recomputed on identical resamples, so the correlation between the
two ROCs is already captured. Measured directly
(`tests/test_stats.py:test_delong_matches_the_bootstrap_only_without_clustering`):
with one item per group the two SEs agree to 3 decimal places (0.0020 vs
0.0020), and with 4 clustered rows per question **DeLong understates the SE by
2.17×** because it assumes independent items — which this design violates by
construction (pos/neg partners, both orders of a pairwise item). Adopting it
as the primary test would manufacture significance of exactly the C-DEGEN
kind. `delong_paired_stratified` is kept as a labelled cross-check: the ratio
of the two SEs measures how much of the bootstrap's width is clustering.

---

## C-VERDICT — Decoding the judge's own verdict  *(the twin of C-ID)*

**Status: NEUTRALISED** — but by the *pooled fit*, which must be protected

C-ID covers leakage through `gt_verdict`, and the conditional metric removes
it. This is its twin, and **the conditional metric does not remove it**.

Within the stratum `gt = A`, an item is correct if and only if the judge said
A. So any representation that decodes `pred_verdict` is a *perfect*
within-stratum correctness predictor while carrying zero self-knowledge.
Conditioning on `gt_verdict` cannot help: the quantity being decoded is
`pred`, not `gt`. And the last-token hidden state decodes `pred` essentially
perfectly — it is the state the verdict logit is read from. M3 therefore has
a route to conditional AUROC 1.0 that has nothing to do with metacognition.

*What actually protects the analysis.* Not the metric — the **pooled fit**. A
pure verdict decoder must score 1.0 in one stratum and 0.0 in the other, so a
single pooled model can never select that direction. That protection is real
but fragile: it degrades with verdict bias and stratum imbalance, and it
disappears entirely under a within-stratum fit, which is an innocuous-looking
refactor.

*Three guards.*

1. **`verdict_decodability`** reported beside `identity_decodability`, same
   estimator, target `pred_verdict`. For activations it should come back near
   1.0 — observed exactly 1.000 on planted data. That number belongs in the
   paper: it is what makes the hazard legible.
2. **`pred_verdict` is a baseline column** (`F.verdict_features`), so every
   internal block must beat *"we already know what the judge said"*. Cheap
   under the offset ladder. Note it is near-redundant with the signed margin,
   which already encodes the verdict — the point is to make the requirement
   explicit rather than implicit.
3. **A test that fails loudly on the forbidden refactor**
   (`tests/test_ladder.py:test_pooled_fit_is_what_blocks_the_verdict_channel`):
   on data where a feature decodes `pred` and nothing else, the pooled fit
   must land near 0.5 and a within-stratum fit must exceed 0.95. Same class of
   hazard as C-NUM — a silent correctness failure that produces entirely
   plausible numbers.

*Taxonomy.* Leakage into type-2 AUROC is **two-dimensional**: item identity
(removed by conditioning) and self-verdict (removed only by the pooled fit).
Probes controlling for neither — the norm in the correctness-probe literature
— are exposed to both.

---

## C-LADDER — The ladder was not measuring addition  *(invalidates M3/M4 in the pilot)*

**Status: NEUTRALISED** (`oof_scores_offset`; `ladder_mode="offset"` is now the default)

Every rung above the baseline was fitted by **concatenating** the internal
block onto the baseline columns and fitting one penalised logistic model. The
penalty is shared across all columns, so a `hidden_size`-wide activation block
(896–2048 columns, and 4096+ on the main panel) against a ~6-column baseline
does not produce "baseline + activations": it produces approximately
"activations", and the added block has to reconstruct the margin from scratch.
When it cannot, the rung scores *below* the baseline it was supposed to extend.

*The signature that exposed it.* Across the six estimable pilot slices, the
rung delta correlates with baseline strength at **r = −0.85 for M3 − M1nd**
(−0.59 for M2, −0.72 pooled). The better the baseline, the more the rung
"loses" — which is a property of the estimator, not of the representation.

*Planted confirmation at realistic width.* Baseline carrying real signal, plus
an added block of **pure noise**. Correct behaviour is a delta of zero.

| baseline | added block | concat | offset |
|---|---|---|---|
| 0.882 | 32 noise cols | 0.853 (**−0.029**) | 0.882 (−0.000) |
| 0.882 | 4096 noise cols | 0.567 (**−0.315**) | 0.882 (+0.000) |
| 0.745 | 32 noise cols | 0.687 (**−0.057**) | 0.744 (−0.001) |
| 0.745 | 4096 noise cols | 0.587 (**−0.158**) | 0.745 (+0.000) |

Pure noise costs a strong baseline **0.315 AUROC** under concatenation, and
the damage is larger for the stronger baseline — reproducing the real-data
signature exactly. The earlier planted validation missed this because its
added block was 32 columns wide; at that width the effect is only −0.03.

*Fix.* `oof_scores_offset()` fits the baseline first, carries its out-of-fold
logit as a fixed **offset**, and lets the internal block fit only the working
residual under RidgeCV shrinkage:

    score = z_baseline + f(X_internal)

If the block is uninformative, `f` shrinks to ~0 and the score reduces to the
baseline, so addition is monotone by construction and a negative delta now
means overfitting rather than destruction of information already in hand. The
offset for training rows comes from an **inner** CV inside the training fold;
using the outer OOF logit would leak the test fold into the baseline the
second stage sees. This is one offset-GLM/boosting step — easier to defend
than group-wise penalties, and identical in behaviour when the block is small.

*Consequence for the pilot.* **Every M3 and M4 number in the LLMBar pilot is
invalid**, and so is every contrast involving them. M2 (a 14-column block) is
much less affected but is not clean either. The streams must be regenerated
and the analysis re-run under `ladder_mode="offset"` before any rung above the
baseline is interpreted.

**`M4 − M3` survives unchanged**: both sides carry the same baseline and the
same swamping, so the difference is unaffected. It remains the anchor result.

*Reproducing the old numbers.* `ladder_mode="concat"` is retained and recorded
per slice in the report, so the pre-fix analysis stays auditable.

*Known cost of the fix, stated honestly.* The offset is **conservative**: the
internal block only ever fits the baseline's residual, so where a block is far
more informative than the baseline the rung will sit below what a jointly
fitted model could reach. On planted data with genuinely informative
activations, M3 reads 0.756 under offset against 0.880 under concat, with
`M3only` at 0.876. The offset therefore answers *"what does this block add to
the baseline?"* — the ladder's actual question — and **not** *"what is the
best achievable score?"* The `M2only` / `M3only` rungs report the latter, which
is why both are kept.

---

## C-ABSENCE — "Adds nothing" vs "was never there"

**Status: NEUTRALISED** (diagnostics in `analysis/cv.py`)

The powered `M4 - M3` equivalence licenses the claim *"spectral adds nothing
beyond a linear activation probe"*. Three different states of the world
produce that same tiny delta, and the delta cannot distinguish them:

1. spectral carries no correctness information — **absence**;
2. spectral carries information that activations already contain —
   **redundancy**, a different and stronger sentence;
3. the L2 penalty shrank the spectral block to ~0, so M4 is numerically M3
   with dead columns — **an artefact of the regulariser**;

and a fourth, which C-NUM proved is not hypothetical: the spectral features
never arrived at all and the pipeline emitted healthy-looking numbers anyway.

*Fixes, none of which cost multiplicity budget — they explain a null rather
than test one, so they never enter the contrast family.*

- **`M2only` / `M3only` rungs** — each family fitted alone, no baseline
  columns. At ~0.500 conditional the claim is absence; well above it, the
  claim is redundancy.
- **`spectral_block_diagnostic()`** — mean |coefficient| on the spectral block
  against the activation block, on standardised inputs, averaged over folds.
  A ratio below 0.05 is flagged `shrunk_to_zero`: the equivalence is then
  about the regulariser and may not be reported as evidence about spectral.
- **`positive_control()`** — the same estimator, same CV, matched n, applied
  to a target that is certainly encoded (prompt length, outer terciles). A
  family that cannot recover *that* has a plumbing problem, and a null on the
  real target means nothing. Warns below 0.70.

*Validated on planted data.* Two synthetic worlds, spectral-as-noise and
spectral-duplicating-activations, produce indistinguishable deltas
(−0.0035 and −0.0032, both `equivalent=True` at band 0.02) — and `M2only`
separates them cleanly at **0.504 vs 0.876**. That is the whole argument for
these rungs in one line.

---

## C-SDT — The margin/competence trend is signal-detection arithmetic

**Status: MEASURED — do not report the trend as self-knowledge**

On pairwise, M1's conditional AUROC tracked judge competence: 0.477, 0.575,
0.716 at accuracies 0.52, 0.73, 0.82. That reads as *"self-knowledge emerges
with competence"*, which would be an attractive headline.

*The null.* Simulate a judge with **no self-knowledge whatsoever**: one latent
decision variable `s`, verdict `= sign(s)`, margin `= s`, discriminability `d`
driving accuracy. Nothing in this judge knows anything about its own errors
beyond what `sign(s)` already determines. Run the identical M1 features
(`[margin, |margin|]`), the identical estimator and the identical conditional
metric:

| d | verdict bias | accuracy | M1 conditional AUROC |
|---|---|---|---|
| 0.05 | 0.0 | 0.505 | 0.466 |
| 0.90 | 0.0 | 0.802 | **0.765** |
| 1.40 | 0.0 | 0.943 | 0.815 |
| 0.90 | 0.5 | 0.797 | 0.784 |
| 1.40 | 0.5 | 0.900 | 0.879 |

The null reproduces the whole trend, and the real judges sit **at or below**
it: 0.716 observed at accuracy 0.82, against 0.765 simulated at accuracy 0.80.
There is no self-knowledge residual to claim.

*Why it is structural, not incidental.* Given the stratum, `is_correct` is a
deterministic function of `sign(margin)` — correct iff `margin > 0` in one
`gt_verdict` stratum and iff `margin < 0` in the other. So M1's conditional
AUROC is largely determined by the geometry of the two strata and the verdict
bias, and is not measuring "does expressed confidence predict error" in the
sense the rung name implies. The two formats disagreeing (`single` runs 0.667
at accuracy 0.58 and 0.619 at 0.64, the wrong direction) is consistent with
this rather than with a competence effect.

*Consequence.* The behavioural trend may not be a headline. If it is reported
at all, it must be reported **against this simulated null at matched accuracy
and matched verdict bias**, with the claim being the residual above the null —
which on current data is zero or negative.

*This is the estimand, not merely a caveat.* The quantity is meta-d′/d′ — the
**M-ratio** of the metacognition literature (Maniscalco & Lau 2012; Fleming &
Lau 2014), where 1.0 means no metacognitive sensitivity beyond first-order
performance. `sdt_null_reference()` is its Monte-Carlo form: estimate the
judge's own d′ and criterion from its hit and false-alarm rates, simulate a
first-order observer at those values, and push it through the identical
features, estimator and metric.

*It applies to every rung, not just M1.* Internals inherit the same arithmetic
floor, so the reference point for the whole ladder is the SDT null at matched
d′, never 0.5. `compare()` now reports `sdt_null` and
`auroc_conditional_vs_null` for every rung, and logs which rungs — if any —
clear it. A conditional AUROC of 0.716 is *below* a no-metacognition observer
at the same accuracy; reported against 0.5 it would read as a strong result.

*Why this makes the negative result publishable.* A null against 0.5 is
uninformative. A null against a matched-d′ first-order observer is a
measurement: **"LLM judges show no metacognitive sensitivity above the
first-order SDT null"** is a locatable, arguable, citable claim.

---

## C-VER — Library version drift mid-campaign

**Status: NEUTRALISED**

The pilot ran `spectral_trust` 0.2.1 (recorded in the stage-12 log header);
the GPU environment has since been upgraded to 0.2.2. Comparing spectral rows
computed by different library versions is the same class of error as mixing
dtypes (C-NUM), and nothing in the results files would have shown it — the
version lives only in the log.

*What actually changed on the path this pipeline uses.*

- **0.2.2 fixed the `rw` eigensolver dispatch.** The random-walk Laplacian is
  non-symmetric, and `scipy.linalg.eigh` reads a single triangle, so it
  silently diagonalised a symmetrized surrogate. **This pipeline is unaffected:
  `spectral_normalization = "sym"` is the default, is asserted at the call
  site in `llm_judge/spectral.py:build_gsp_config`, and is recorded as `"sym"`
  in every pilot log header.** The pilot data never touched the defective path
  and needs no re-run.
- **0.2.2 promotes the adjacency to float32 before building the Laplacian**;
  0.2.1 built it in the model's bfloat16 and upcast afterwards. Measured
  effect on the four metrics in use, at N = 128/512/1024/3131 tokens:
  **≤ 0.02% relative** on every metric at every length — three orders of
  magnitude below between-item variance. Pilot rows remain poolable.
- **0.2.3 (this repo's `spectral-trust` working tree) adds the C-NUM
  assertion**: `assert_finite_attention` raises `NonFiniteAttentionError` on
  the first layer of NaN/Inf attention, naming the model and dtype, instead of
  letting it surface later as an opaque LinAlgError or as a silently absent
  model. This makes the pilot's most consequential finding a permanent
  property of the library rather than a lesson in a document.

*Fix.* `spectral-trust==0.2.3` is pinned in `requirements.txt` and
`pyproject.toml`. The version stays in every log header, so any future drift
is visible in the provenance rather than inferred.

---

## Engineering defects found and fixed along the way

| defect | consequence if unfixed |
|---|---|
| `Model loading.py` (space in filename) | stages 2–3 crash on import |
| `import torch` before `pyarrow` | interpreter segfault, no traceback |
| provenance logger imported `datasets` to read its version | took down every GPU-env run (broken Windows cert store) |
| per-item full-JSON rewrite | O(n²) writes; **rows silently lost** to Windows file locks — now append-only JSONL |
| `KeyError` on missing activations | crashed the whole analysis instead of dropping rows |
