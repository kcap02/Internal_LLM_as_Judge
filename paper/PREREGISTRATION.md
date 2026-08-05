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

The generalisation, and it is the transferable claim: **a passing test on
planted data certifies an estimator only over the region the plant covers.**
Every one of these failures lived outside it and was invisible from the green
result.

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

Measured (`results/recovery_curve.json`, block width 1536, dense plant,
detection at |delta| ≥ 0.02):

| n per stratum | detection threshold (block-only AUROC) |
|---|---|
| 200 | none — above 0.518 |
| 400 | none — above 0.589 |
| **800** | **0.602** |

The largest block-only AUROC observed in the pilot is `M3only` = **0.633**
(Qwen2.5-3B pairwise). So:

> **`N_FREEZE` is the smallest n whose measured detection threshold falls
> below the largest block-only AUROC observed in the pilot, and never less
> than the n the MDE calculation requires. On current measurements that is
> `N_FREEZE` ≥ 800 items per stratum.**

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

The curve must be **re-measured at the scaled n and the scaled block width**;
the threshold is a function of both and does not carry over from the pilot.

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
