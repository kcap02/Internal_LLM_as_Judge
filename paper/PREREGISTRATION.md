# Preregistration

**Status: DRAFT — NOT FROZEN.** Two numbers from the corrected-estimator
re-analysis gate the wording (§7). Once they land, resolve §7, commit, and
record the commit hash below. After that: no edits.

    Frozen at commit: ____________  (fill on freeze; leave blank until then)

Everything below is decided. §7 is the only open text.

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

## 3. Panel and data

- **Three judges** spanning scale for the spectral arm; one at ~1.5B, one at
  7–8B, one at 27–32B. Fixed spectral window across all three, sized so the
  eigendecomposition is affordable at the **largest** model, not the smallest.
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

**(a) `M2only` under the corrected estimator.** Near 0.50 → primary test 2
preregisters as a clean **absence** claim. Materially above 0.50 → the claim is
**redundancy**, a different and stronger sentence, and test 2's wording
changes. Do not skip past this once the tables print; the freeze was held for
it.

**(b) Per-slice verdict-leakage floors.** If the floor comes back well above
0.50 on the biased slices, `floor_to_clear` stops being a formality and starts
excluding rungs. State **which floor is primary when the two disagree** — the
default is the larger, but it must be written down before the data decides it.

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

`N_FREEZE` is derived from `21_power.py` re-run against the **offset-fitted**
contrasts, not chosen from what looks affordable. Recomputing power against the
corrected estimator is a prerequisite of the freeze, because the concat-fitted
SEs are not the SEs of the estimator we will use.

If `N_FREEZE` is unaffordable, the primary tests become **equivalence tests
with a stated band** rather than superiority tests. Both are defensible.
Choosing between them after seeing the data is not, and it is the most
reviewer-visible decision in a paper about measurement discipline.

## 9. Sequencing after the freeze

The **cheap arm** (activations only — no retained attention, no
eigendecomposition, no VRAM ceiling) starts the **same day** as the freeze. It
is the only queue with a hardware constraint; the unification experiments, the
published-baseline reimplementation, the transfer matrix and the risk-coverage
curves are all CPU and run against its output.

The expensive (spectral) arm does not need scale — it is already at SE
0.005–0.009 for test 2 — and is budgeted at a few hundred items per bank across
the three judges, for generality rather than power.

## 10. Minimum viable submission

If something slips, and something will: **five methodological findings +
LLMBar + one published-baseline reimplementation.** Everything beyond that is
upside. Keep this ordering.
