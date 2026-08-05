# Preregistration

**Status: DRAFT — NOT FROZEN.** Three inputs are still open: the two numbers
from the corrected-estimator re-analysis (§7), and the three exact judge
checkpoints plus the hardware decision behind them (§3). Resolve all three,
then commit and record the hash here. After that: no edits.

    Frozen at commit: ____________  (fill on freeze; leave blank until then)

Everything else is decided. The decision rules for the open items are already
written and numeric — resolving them is reading a table, not exercising
judgement after seeing one.

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

  **OPEN — the exact checkpoints must be named before freezing**, because
  `N_FREEZE` (§8) cannot be derived without them: the largest model sets the
  VRAM ceiling, the ceiling sets the window, and the window sets what is
  affordable per item. An undetermined panel leaves §8 with an undetermined
  input.

  Proposed, pending the hardware decision below — **one family, three scales**,
  so that scale is not confounded with vendor or tokenizer:

  | role | checkpoint | note |
  |---|---|---|
  | small | `Qwen/Qwen2.5-1.5B-Instruct` | the pilot's usable-judge floor (73.5% pairwise) |
  | mid | `Qwen/Qwen2.5-7B-Instruct` | already in `config.judge_models` |
  | large | `Qwen/Qwen2.5-32B-Instruct` | already in `config.judge_models_large` |

  **Hardware dependency, stated plainly.** The 16 GB card cannot run the
  spectral arm at 7B, let alone 32B: `output_attentions=True` retains a
  `[heads, N, N]` tensor per layer, so a 7B model needs ~6.1 GB at 2048 tokens
  and ~24.5 GB at 4096 on top of ~14 GB of bf16 weights. On current hardware
  the largest feasible spectral judge is ~3B. So one of the following must be
  chosen and written here before the freeze:

  1. **rent a larger card** for the spectral arm and keep the panel above; or
  2. **keep the spectral arm at ≤3B** and name the three checkpoints
     accordingly (e.g. 0.5B / 1.5B / 3B), accepting that the scale span is
     narrow and saying so; or
  3. **decouple the arms** — run the *cheap* arm (activations only, no
     retained attention, no VRAM ceiling) over the full 1.5B/7B/32B ladder,
     and the spectral arm over whatever the available card permits, reporting
     the two panels separately rather than pretending they are one.

  Option 3 is the cheapest and is consistent with §10's split, but it changes
  what primary test 2 generalises over, so it must be a preregistered choice
  and not a retrospective description of what fitted.

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
- **Offset ladder** (`ladder_mode="offset"`): baseline out-of-fold logit as a
  fixed offset, internal block fits the working residual under RidgeCV, with
  the training-row offset from an inner CV. Concatenation is not used for any
  claim.
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

## 7. OPEN — resolve before freezing

The two numbers below are the only open text. **The decision rules are already
numeric**, so resolving them is reading a table, not exercising judgement:
"near 0.50" and "well above" are the words that let a preregistration be
reinterpreted afterwards, and they do not appear here.

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

> **At n per stratum of `N_FREEZE`, we are powered to detect `MDE_FREEZE` at
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

| n per stratum | median delta | delta IQR | detect | block-only | block IQR |
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
sound (200 items, 100 groups, stratum fraction 0.5), and distributionally the
two agree: 12 fresh worlds give median 0.627 IQR [0.587, 0.711], 12 prefixes of
n=1600 worlds give 0.648 IQR [0.625, 0.670], overlapping by 0.154. The 0.574 is
a low draw of one seed set, not a property of prefixing, and the n-sweep is
measuring what it claims to.

*How we nearly concluded otherwise.* Two sweeps both reported 0.574 and that
read as replication. Both used seed offset `2000+k` and shared ten seeds: it
was the same computation twice. The calibration self-check now runs on an
independent seed offset for exactly this reason — checking a calibration
against a cell that shares its seeds is not a check.

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

> **`N_FREEZE` = 800 items per stratum.**
>
> The measured detection threshold is 400 — the smallest n detecting the
> calibrated plant in ≥ 80% of seeds. We preregister **one grid step above
> it**, because the calibration carries uncertainty the sweep does not
> quantify and the caveats compound in one direction:
>
> * s\* is interpolated from the pilot's `M3only` = 0.633, itself a single
>   draw at n=200 where the calibration IQR spans roughly ±0.05;
> * 400 sits one step above a cell that detects in only 60% of seeds;
> * at 400 the delta IQR is [+0.046, +0.076], whose lower edge is nearer the
>   0.02 detection band than at 800, where it is [+0.059, +0.066].
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

## 10. Sequencing after the freeze

The **cheap arm** (activations only — no retained attention, no
eigendecomposition, no VRAM ceiling) starts the **same day** as the freeze. It
is the only queue with a hardware constraint; the unification experiments, the
published-baseline reimplementation, the transfer matrix and the risk-coverage
curves are all CPU and run against its output.

The expensive (spectral) arm does not need scale — it is already at SE
0.005–0.009 for test 2 — and is budgeted at a few hundred items per bank across
the three judges, for generality rather than power.

## 11. Minimum viable submission

If something slips, and something will: **five methodological findings +
LLMBar + one published-baseline reimplementation.** Everything beyond that is
upside. Keep this ordering.
