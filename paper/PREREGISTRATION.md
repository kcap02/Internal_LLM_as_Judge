# Preregistration

> **Supersession, 2026-08-08.** §4's description of the offset ladder's second
> stage was **stale at the time of freezing** and has been corrected. It said
> the internal block "fits the working residual under RidgeCV"; the code has
> used a penalised offset logistic with the penalty selected by out-of-fold
> AUROC since commit `9d431d3` (2026-08-05), a day before the freeze. The run
> used the code, not the document. The frozen text also contradicted itself:
> §4.1 and the errata table both list the RidgeCV form as a *failure mode*.
>
> This is a documentation correction, not a specification change, and the test
> is that it **could have been made without seeing any results** — the
> discrepancy is visible by reading §4 against `cv.py`, and nothing about the
> outcome bears on it. A systematic pass over all 14 quantitative claims in
> this document against the code that implements them found this to be the
> only mismatch.
>
> **Second correction, same date.** `N_FREEZE` was written as "800 items per
> *stratum*" in four places. The quantity the recovery curve measures is items
> per **slice** (one model × format); each slice contains two `gt_verdict`
> strata, so 800 per slice is ~400 per stratum. The wording is corrected to
> match what was measured. No number changed.
>
> This one had a consequence. `--limit` in stage 11 capped items per model per
> *bank*, so a two-format bank delivered half the intended count to each slice:
> the three free-text banks ran at 400 per slice rather than 800, while the
> single-format MCQ banks got the full 800. The flag did not implement what
> §8.2 specifies, which is **§9's implementation-bug exception**, declared
> here: the flag now means items per slice and logs the resolved figure, and
> all three free-text banks were re-run — not only the one carrying the
> illustrative slices, since fixing only that bank would select which slices
> get full power after seeing which ones mattered. Before/after item counts are
> reported in the results.

**Status: FROZEN.** All inputs are resolved: the panel and its hardware
constraint (§3), both open numbers from the corrected-estimator re-analysis
(§7), and `N_FREEZE` (§8.2). From here §9's stopping rule applies: the analysis
runs once on this specification, and any confound found afterwards is reported
as a named hazard rather than fixed and re-run.

    Frozen at commit: see FREEZE.txt (recorded in the commit immediately
    following this document's final edit, so that the hash refers to the
    frozen text rather than to a file containing its own hash)

Resolutions, each read off a table by a rule written before the table was
seen: §3 decoupled arms with the spectral panel capped at 3B; §7(a) absence;
§7(b) floors binding on every estimable slice; §8.2 `N_FREEZE` = 800, one grid
step above a measured detection threshold of 400 at the spectral arm's block
width of 2048.

---

## 1. Question

Does an LLM judge's internal state predict its own verdict errors beyond what
its first-order performance and its expressed confidence already imply?

## 2. Primary tests (m = 3)

Exactly three. Everything else is exploratory, reported unstarred, and
labelled as such in the same table.

1. **Do internals beat the behavioural baseline?** Conditional (within
   `gt_verdict`) AUROC, offset ladder, pooled hierarchically over judges and
   banks. TOST against ±0.02.
2. **Does spectral add beyond activations?** `M4 − M3`, equivalence against
   ±0.02. Already powered at pilot n (SE 0.005–0.009).
3. **Does any internal signal improve selective prediction over the margin?**
   Reported as **coverage at fixed risk**, not ΔAUROC — a 0.02 AUROC
   difference has no operational meaning and a coverage change does.

### 2.1 What each outcome licenses

Every primary test has **three** branches, not two. The third is the one that
needs preregistering, because it is the most likely outcome for test 1 and the
easiest to quietly report as a null.

| | test 1 (internals > baseline) | test 2 (spectral > activations) | test 3 (selective prediction) |
|---|---|---|---|
| **effect clears `MDE_FREEZE` and survives BH-FDR** | internals carry correctness information beyond first-order performance, expressed confidence and peer difficulty | spectral adds beyond a linear activation probe | the internal signal buys usable coverage at fixed risk |
| **TOST rejects into ±0.02** | we rule out an internal contribution of 0.02 or more — a **measured** negative | spectral is redundant with activations at the stated band | no operationally meaningful coverage gain |
| **neither** | **UNRESOLVED.** Reported as unresolved, with the interval and the MDE, and explicitly **not** as evidence of absence | same | same |

The third row is a commitment, not a caveat: a contrast that is neither
significant nor equivalent is reported in those words, in the abstract as well
as the results, and never described as "no effect". The pilot's 0/30 was
exactly this case and was nearly written up as a negative result.

## 3. Panel and data

- **Three judges** spanning scale for the spectral arm. Fixed spectral window
  across all three, sized so the eigendecomposition is affordable at the
  **largest** model, not the smallest.

  **DECIDED — the arms are decoupled and the spectral arm is capped at 3B.**
  A decided narrow scope is worth more than an undecided wide one, and the
  per-layer attention reduction that would raise the spectral ceiling is not a
  precondition of this run.

  | arm | panel | why |
  |---|---|---|
  | **cheap** (activations only) | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` | no retained attention, no eigendecomposition, no VRAM ceiling; this is where the scale span and the statistical power live |
  | **spectral** | `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B` | 16 GB is the binding constraint: `output_attentions=True` retains a `[heads, N, N]` tensor per layer, so 7B needs ~6.1 GB at 2048 tokens on top of ~14 GB of weights. 3B is the largest that fits |

  One family throughout, so scale is not confounded with vendor or tokenizer.
  **Two panels are reported separately and never pooled**, and the abstract
  states what the spectral result generalises over: attention-graph
  diagnostics in judges up to 3B. That is a real limit on primary test 2 and it
  is preregistered rather than described afterwards.

  All six checkpoints are named. `Qwen2.5-7B` at `bfloat16` is close to the
  16 GB card's capacity even without retained attention; if it cannot be made
  to complete, that is a **run that fails to complete**, which §9 already
  covers as a declarable exception, and it is reported as such rather than
  resolved by a conditional written here. A specification that names a
  parameter by the condition it must satisfy is exactly where a later choice
  can be made to look preregistered.

  Compute, not VRAM, is why no option reaches 32B: dense eigendecomposition at
  2048 tokens costs on the order of a second per layer per item, and at 64
  layers that is a minute or more per item on CPU. The per-layer reduction
  would raise the VRAM ceiling to roughly 7B and would not change this.

  **Consequence for §8.2:** the spectral arm's block width is 2048
  (Qwen2.5-3B hidden size), so the recovery curve is re-measured there rather
  than at the pilot's 1536.

  **Where the per-layer attention reduction lives, if implemented.** It changes
  what `spectral_trust` does on every forward pass, and `release/0.2.3` is the
  artifact this paper cites. It goes in the **judge repository as a local
  hook**, not into that release: 0.2.3 is currently a clean one-fix patch off
  published 0.2.2, and keeping it minimal keeps the cited artifact reviewable.
  The library absorbs the reduction in 0.3.0 alongside the gini metrics, which
  §6 already excludes from this paper's feature set.
- **Five banks** (MMLU, MMLU-Pro, JudgeBench, LLMBar, RewardBench 2), entering
  the hierarchical model as a **random effect** — pooled, not tested
  separately.
- `pairwise` is the primary free-text format (immune to the length confound by
  counterbalancing; `single` treats preference as absolute truth).

## 4. Estimator (frozen)

- Conditional AUROC within `gt_verdict` strata, minimum 5 of each class per
  stratum; per-stratum support counts reported beside every value.
- **Offset ladder** (`ladder_mode="offset"`): the baseline's out-of-fold logit
  enters as a fixed, unpenalised offset, and the internal block extends it by
  penalised logistic regression on the same likelihood, with the penalty
  selected by **out-of-fold AUROC** and the training-row offset from an inner
  CV. Concatenation is not used for any claim.
  *(Corrected 2026-08-08 — see the supersession note in the header. The frozen
  text described the block as fitting the working residual under RidgeCV,
  which is the superseded second stage that §4.1 and the errata table both
  identify as a failure mode.)*
- Both estimands reported side by side: offset rung (increment) and `*only`
  rung (achievable alone); the gap is redundancy.
- Grouped paired bootstrap, 2,000 resamples, groups = text-derived `group_id`.
  Paired DeLong is a labelled cross-check only — it understates the SE by
  2.17× on clustered data.
- BH-FDR at α = 0.05 across every contrast in the run, including each rung's
  percentile against the SDT null.

### 4.1 Recovery curve — required of any estimator this paper uses

**No estimator enters this specification without a recovery curve**, measured
at the n and the block width actually in use (`scripts/22_recovery_curve.py`).
Not a planted test at one signal strength: a curve.

This project produced three estimators that each passed a single-point planted
test and each failed outside the region the plant covered:

| estimator | passed at | failed at |
|---|---|---|
| concatenation | 32-column block | 4096 columns — destroys 0.315 AUROC of real signal |
| offset + RidgeCV | a strong planted latent | n=200, 1536 columns — returned exactly 0.000 for every block on every slice |
| the recovery test itself, v1 | — | passed on ±0.02 jitter; its plant sat below what was recoverable at that n |

The generalisation has two halves, and both are transferable claims:

1. **A passing test on planted data certifies an estimator only over the
   region the plant covers.** Every failure above lived outside it and was
   invisible from the green result.
2. **A precision estimate is uninterpretable without a recovery estimate
   beside it.** Four of this project's validations returned a healthy-looking
   number *about a degenerate object*: the concat block diagnostic reported
   coefficient ratios of 0.68–1.64 while the estimator the claims used
   returned nothing; the bootstrap MDE reported a median SE of 0.006 and "72
   items per slice" precisely *because* the rungs sat on top of the baseline.
   A tight interval around a collapsed estimator is still tight. Precision and
   recovery must be reported as a pair.
3. **The acceptance criterion must be fixed before the measurement, and
   enforced by the script rather than by the reader.** Reading a table
   afterwards to decide whether a fix worked is the same act as reading one to
   decide whether a hypothesis held, and it fails the same way. Every one of
   the five misreadings in this project was a number that looked like evidence
   *for the thing the reader was already committed to* — and the sharpest was
   the fifth, where an apparent monotone trend appeared in the sweep
   immediately after we argued that pairing would produce it. That one was an
   error in reading, not in an estimator, which is why the rule has to bind
   the reader.

   `scripts/22_recovery_curve.py` therefore prints **PASS/FAIL itself** and
   exits non-zero on failure. It asserts: detection at ≥ 80% of seeds at some
   tested n; a calibration self-check that the target lies inside the IQR at
   the pilot's n; and — the guard against the fifth failure — **no
   monotonicity claim about block-only is permitted unless consecutive IQRs
   are disjoint**, which the script decides and states.

The plant must be a **dense random direction**, not a single column: that is
how an activation encodes anything, and a one-column plant in a wide block is
near worst case for a penalised probe.

**The estimator's detection threshold is reported next to every null**, and
**block-only AUROC is reported next to every ladder delta.** A delta of zero
with block-only at 0.63 means something entirely different from a delta of
zero with block-only at 0.85, and neither is readable alone.

## 5. Two null references (neither is 0.5)

A rung must clear **both**:

- **SDT / first-order floor** — a signal-detection observer at the judge's own
  d′ and criterion, with parameter uncertainty propagated by re-estimating d′
  and c on each resample. The Monte-Carlo form of meta-d′/d′.
- **Verdict-leakage floor** — what a feature decoding only `pred_verdict`
  achieves at that slice's own stratum imbalance, through incomplete sign
  cancellation.

## 6. Feature set (frozen to what the pilot produced)

Margin (+|margin|), `pred_verdict`, nuisance (length, log-length,
`task_start_idx`, subject), peer difficulty, spectral (mean / slope / argmax
over the last third, per metric, plus Fiedler-velocity summary), last-token
activations at 0.5 and 1.0 depth.

**No metric added after seeing a null.** `gini_sparsity` and `attention_gini`
exist in the spectral library's development branch and are **excluded**: they
were not in the pilot. Adding a metric after seeing the result is the one move
that would undo the discipline the rest of this document encodes.

## 7. RESOLVED

The decision rules below were written before the tables were read, and are
numeric, so resolving them was reading a table rather than exercising
judgement. Both resolutions are recorded with the values they came from.

> **(a) resolves to ABSENCE.** Observed `M2only` runs 0.308–0.522 against
> per-slice floors of 0.579–1.000, so `M2only − floor_to_clear` is negative on
> every estimable slice, far below the 0.02 cut. Primary test 2 preregisters as
> a clean absence claim: spectral features carry no correctness information
> detectable at this scale, rather than information redundant with activations.
>
> **(b) resolves to BINDING.** Every estimable slice has
> `verdict_only_null.leak_auroc_p95` ≥ 0.53 (observed 0.566, 0.567, 0.616,
> 0.642, 0.722), so the verdict-leakage floor is not a formality: it excludes
> rungs, and that exclusion is reported per slice. Where the two floors
> disagree the larger is primary, as stated below.

The original text is retained for the record.

**(a) `M2only` under the corrected estimator**, compared against that slice's
own `floor_to_clear` (§5), pooled across slices by the same hierarchical
specification as the primary tests:

| observed | conclusion for primary test 2 |
|---|---|
| `M2only − floor_to_clear` ≤ **0.02** | **absence** — spectral carries no correctness information; test 2 preregisters as a clean equivalence claim |
| `M2only − floor_to_clear` ≥ **0.05** | **redundancy** — spectral carries information activations already contain; test 2's wording changes to the stronger sentence |
| strictly between | **indeterminate** — reported as indeterminate, not rounded to whichever side is convenient |

0.02 is the preregistered equivalence band (§8); 0.05 is the upper end of the
plausible target effect. Both are constants this document already uses.

**(b) Per-slice verdict-leakage floors.** A slice's floor is **binding** when
`verdict_only_null.leak_auroc_p95 ≥ 0.53` — the same threshold the
implementation already uses for its `clean` flag, reused rather than invented
so the document and the code cannot drift. Below it, `floor_to_clear` is a
formality; at or above it, the floor genuinely excludes rungs and that
exclusion is reported per slice.

**When the two floors disagree, the larger is primary.** A rung must clear
`max(SDT p95, verdict-leak p95)`. Written down here before the data decides it.

## 8. Power commitment

Measured on the pilot (`scripts/21_power.py`, from the bootstrap's own CIs):
at **n = 200 per slice**, median SE **0.034**, the minimum detectable effect is
**0.083** uncorrected and **0.127** under BH-FDR over 30 contrasts. The
plausible target effect is 0.03–0.05. **The pilot was not powered for its own
question.**

The honest consequence is a fork, and it is taken *now* rather than after
seeing the data:

> **At n per slice of `N_FREEZE`, we are powered to detect `MDE_FREEZE` at
> 80% power under the preregistered m = 3 family. We will report TOST against
> ±0.02 for every primary test. We will not claim a positive result for any
> effect below `MDE_FREEZE`, regardless of what BH-FDR returns.**

### 8.1 The bootstrap MDE is not sufficient on its own

Recomputed against the offset-fitted contrasts, `21_power.py` reports a median
SE of **0.006** and claims **72 items per slice** suffice to detect 0.04 — a
thirtyfold improvement on the concat-fitted figure. **This number must not be
used.**

The bootstrap SE measures the precision of the *difference between two score
vectors*. When the ladder's rungs sit almost on top of the baseline, that
difference is tightly estimated *because the estimator is barely expressing
the block*, not because the study can detect a real effect. A degenerate
estimator that returned the baseline verbatim would report an SE of exactly
zero and an MDE of zero. The MDE calculation cannot tell precision from
collapse.

### 8.2 `N_FREEZE` is set by the recovery curve, with the MDE as a floor

**The measurement is indexed on the *planted* strength, not on a recovered
one.** Reading the threshold off block-only AUROC at each n is ill-posed:
block-only is itself estimated at that n and is biased downward by the same
shrinkage that suppresses the delta, so a threshold read at n=800 and the
pilot's `M3only` read at n=200 are two different estimation regimes compared
as though they were one. Instead:

*Phase 1 — calibrate.* At the pilot's own n=200, measure block-only AUROC
across plant strengths (10 seeds each) and interpolate the strength whose
median reading equals the pilot's observed value. This is the plant that looks
like the real data *through the same biased lens*.

| plant | median block-only at n=200 | IQR |
|---|---|---|
| 0.0 | 0.479 | [0.443, 0.520] |
| 1.0 | 0.518 | [0.474, 0.557] |
| 2.0 | 0.594 | [0.533, 0.636] |
| 3.0 | 0.648 | [0.590, 0.699] |

Pilot `M3only` = 0.633 (Qwen2.5-3B pairwise) → calibrated **s\* = 2.721**.

*Phase 2 — sweep n at s\*.* Every cell is 10 seeds, and the cells are
**paired**: one realisation per seed, evaluated at every n as a nested prefix,
so the columns track one underlying signal instead of unrelated draws.
Detection is a delta ≥ 0.02 in ≥ 80% of seeds, matching the power convention
used elsewhere in this document.

| n per slice | median delta | delta IQR | detect | block-only | block IQR |
|---|---|---|---|---|---|
| 200 | +0.032 | [+0.018, +0.042] | 60% | 0.574 | [0.537, 0.636] |
| 400 | +0.053 | [+0.046, +0.076] | 100% | 0.626 | [0.608, 0.639] |
| 800 | +0.062 | [+0.059, +0.066] | 100% | 0.647 | [0.626, 0.661] |
| 1600 | +0.082 | [+0.064, +0.084] | 100% | 0.631 | [0.625, 0.637] |

**Block-only is approximately flat in n**, and the script refuses any
monotonicity claim from this table because no consecutive IQRs separate. The
calibration self-check, run on 12 **independent** worlds at the pilot's n,
gives block-only 0.636 IQR [0.598, 0.677] against the 0.633 target — so s\*
delivers the intended plant.

*A prefix is a valid subsample.* We checked, because the sweep's own n=200 cell
reads 0.574 while a fresh measurement reads ≈0.63. Structurally the prefix is
sound (200 items, 100 groups, stratum fraction 0.5). Distributionally the two
are **indistinguishable at this n**: 12 fresh worlds give median 0.627 IQR
[0.587, 0.711], 12 prefixes of n=1600 worlds give 0.648 IQR [0.625, 0.670],
ranges overlapping by 0.154. By this document's own rule (§4.1) no directional
claim is available from those IQRs, and none is made — *consistent* is the
finding and all that is needed. The 0.574 is a low draw of one seed set, not a
property of prefixing, and the n-sweep measures what it claims to.

*A validator wired to its own input.* The calibration self-check originally
compared the target against **the sweep's own n=200 cell**, which shares that
sweep's seeds. It could therefore only ever confirm whatever that cell said: a
low draw produced a "confirmed" low reading. This is not a misreading — it is
the same defect as the concat block diagnostic reporting on a code path no
claim depended on, and it is worth more to a reader than either. The check now
draws **independent worlds at an unrelated seed offset**, and passes with the
target at the median (0.636, IQR [0.598, 0.677]) rather than at its edge.

*Separately:* two sweeps both reported 0.574 and that read as replication. Both
used seed offset `2000+k` and shared ten seeds — the same computation twice.

The consequence is that the downward bias in block-only at n=200 — the
motivation originally given for indexing the curve on planted strength rather
than on a recovered quantity — **is negligible at this width**. The
plant-indexed design is still the right one, because indexing on an estimated
quantity is ill-posed in principle, but the reason it improved this
measurement was the **pairing and the seeding**, not a correction for
regime-mixing.

Two earlier readings of this table were noise. The unpaired sweep's
non-monotone block-only (0.615, 0.680, 0.652, 0.631) and the first paired
sweep's apparent monotone rise (0.574 → 0.626 → 0.647) were both 10-seed
fluctuations around a flat ≈0.65; the 0.574 in particular was a low draw, and
reading a trend into it was an error. **The `n` dependence lives in the
detection fraction and the delta, not in block-only** — and those are what set
`N_FREEZE`.

> **`N_FREEZE` = 800 items per slice** (a slice being one
> model x format; each slice contains two `gt_verdict` strata, so ~400 per
> stratum).
>
> The measured detection threshold is 400 — the smallest n detecting the
> calibrated plant in ≥ 80% of seeds. We preregister **one grid step above
> it**, because the calibration carries uncertainty the sweep does not
> quantify and the caveats compound in one direction:
>
> * s\* is interpolated from the pilot's `M3only` = 0.633, itself a single
>   draw at n=200 where the block-only spread across seeds is wide
>   (IQR ≈ [0.598, 0.677]);
> * 400 sits one grid step above a cell that detects in only 58% of seeds, and
>   at 400 the delta IQR's lower edge (+0.047) is nearer the 0.02 detection
>   band than at 800 (+0.058).
>
> A third reason previously given — that the calibration passed only marginally,
> with the target at the edge of its IQR — **no longer holds**: once the
> self-check was moved onto independent worlds it passes at the median. It is
> struck rather than left standing, so the basis for the number is not
> overstated.
>
> The cost of being wrong is asymmetric: the extra items are inference on the
> cheap arm, which has no VRAM ceiling and no eigendecomposition — hours, not
> days — whereas a run that lands at the edge of the detection cliff produces
> an uninterpretable delta *after* the freeze, when §9's stopping rule forbids
> adjusting. The reason is stated here so the choice cannot be read as a
> number picked after the fact.

The curve must be **re-measured at the scaled block width**; the threshold is
a function of width as well as n and does not carry over from 1536.

The recovery curve is primary because it is calibrated on the estimator
actually in use; the MDE is retained as a floor, not as the criterion.

**If no affordable n reaches a detection threshold below the observed
block-only values, primary test 1 is preregistered as UNRESOLVABLE and the
paper says so in the abstract.** That is a live possibility and it is faced
here rather than discovered later. In that case the empirical contribution is
the two floors, the `M2only` absence, and the estimator failures, and the
ladder becomes a section on why the question is harder than the literature
assumes. That is still a paper — it is a different one, and which one we are
writing must be settled before GPU time is bought.

If `N_FREEZE` is unaffordable, the primary tests become **equivalence tests
with a stated band** rather than superiority tests. Both are defensible.
Choosing between them after seeing the data is not, and it is the most
reviewer-visible decision in a paper about measurement discipline.

## 9. Stopping rule — what the freeze changes

**The analysis is run once, on the specification above.** Any confound
discovered after the freeze is reported in Limitations as a named hazard with a
proposed measurement. It is **not** fixed and re-run.

This is a deliberate change of regime, and it is the thing the freeze marks.
Six of this paper's findings arrived by re-running the analysis after each
discovery — C-DUP, C-DEGEN, C-NUM, C-WIN's VRAM half, C-LADDER, C-SDT. That
loop was correct during development: each iteration removed an artefact, and
the confound register exists because of it. The identical loop after the freeze
is a garden of forking paths, because from here the thing being iterated
against is the result rather than the machinery.

A reader should be able to see that we knew the difference. The register
records which findings came from which regime, and every post-freeze hazard is
labelled as such with its measurement left undone and stated.

Two narrow exceptions, both of which must be declared in the paper if used:
an outright **implementation bug** (a computation not doing what this document
says it does) may be fixed and the analysis re-run, with the bug and the
before/after both reported; and a **run that fails to complete** may be
resumed. Neither licenses a change to the specification.

### 9.1 No pilot-and-scale

**We do not run small experiments and scale the ones that look promising.**
This is selection on the outcome and it is the same forking-paths failure §9
exists to end, arriving under a different name.

The provenance is part of the point: the strategy was **proposed by the
methodological reviewer**, roughly two weeks into building an apparatus whose
purpose is to prevent outcome-selection, and rejected on the grounds below. A
prohibition with anonymous provenance reads as a rule someone thought up. One
that records the auditing party proposing the failure mode is evidence that the
discipline has to be structural rather than maintained as a habit by whoever is
paying attention.

The measurement forbids it specifically. The detection threshold (§8.2) is a
property of $n$: at $n = 200$ the estimator detects a calibrated true effect in
only 58% of seeds. A small-scale run therefore does not produce weak evidence
worth following up — it produces fluctuations at the scale of the effect being
looked for, and scaling whatever looks largest is scaling the loudest noise.
The pilot's 0/30 and 0/66 were uninformative rather than negative for exactly
this reason.

**Legitimate exploration is CPU-side and post hoc**: the transfer matrix,
risk–coverage curves, the full-vector layer contrast and the floor
decomposition all run against the stored features after the scaled run. They
consume no GPU, they consume no $\alpha$ because they are labelled exploratory
and unstarred, and none of them requires a new pass.

**One small run is legitimate before the freeze, and it is not exploration.**
Reproducing a published positive claim (§10) in this harness is a *feasibility*
check: it establishes whether the setup runs here at all, which the budget
depends on. The distinction is what is being tested — that the pipeline
executes, not that a result is interesting — and it is recorded here so the
distinction cannot be claimed retrospectively for something else.

## 10. Sequencing after the freeze

Six weeks to the abstract deadline. This is one scaled run plus writing, not an
exploration budget.

| when | what |
|---|---|
| this week | freeze §3 and §7, record the hash |
| immediately after | **cheap arm** at $n = $ `N_FREEZE` per slice across the panel — hours of inference, no retained attention, no eigendecomposition, no VRAM ceiling |
| behind it | CPU queue against the stored features: transfer matrix, risk–coverage, full-vector layer contrast, floor decomposition |
| in parallel | **published-baseline reimplementation** (§10.1) |
| final three weeks | writing |

The expensive (spectral) arm does not need scale, and is budgeted at a few
hundred items per bank across the named judges, for generality rather than
power. Its detection threshold must be re-measured at its own block width
before any of its deltas are interpreted.

### 10.1 The published-baseline reimplementation is not optional

With the `M4 − M3` claim retracted (§7), this paper's positive empirical
content is one null on a four-judge pilot plus methodological findings
demonstrated largely on planted data. A fair reviewer summary of that alone is
*"the authors show a standard estimator has failure modes, and find nothing
with their own method."*

Applying the two floors and the corrected ladder to a **published positive
claim about internals and metacognition** is the one item that changes this,
and it is the only place where the interesting outcome exists in both
directions: if the published result survives both floors, that is a result; if
it moves, that is a larger one. Neither outcome depends on a number coming out
a particular way, which is what distinguishes it from the pilot-and-scale
strategy §9.1 forbids.

It therefore ranks **above additional judges and above panel breadth**, and
runs in parallel with the cheap arm rather than behind it.

## 11. Minimum viable submission

If something slips, and something will: **five methodological findings +
LLMBar + one published-baseline reimplementation.** Everything beyond that is
upside. Keep this ordering.
