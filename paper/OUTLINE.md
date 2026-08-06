# Outline — *How Not to Measure Whether a Judge Knows It Is Wrong*

Target: ICLR 2027. Abstract Sep 18 2026, paper Sep 25 2026 (AoE).

Written first, then `main.tex` written against it. Every printed quantity is a
macro in `generated_numbers.tex`, produced by `fill_numbers.py` from a file in
`results/`; the build fails on an undefined macro. No number is typed by hand.

## What the paper is

A **measurement paper about type-2 (correctness) ROC for LLM judges.** The
spectral and activation families are the demonstration panel, not the point.

The claim: the standard way of asking *"can we predict when a judge is wrong
from its internals?"* — pooled type-2 AUROC against a 0.5 reference, with the
internal block concatenated onto a small baseline — is wrong in five separable
ways, each of which we measure, and four of which inflate the answer.

## The five findings (each with a test that fails if it regresses)

| # | finding | direction | test |
|---|---|---|---|
| 1 | **Item-identity leakage.** `is_correct = (pred == gt)`, so any feature separating pos from neg items scores without self-knowledge. | inflates | `test_stratified_auroc_kills_the_item_identity_shortcut` |
| 2 | **Self-verdict leakage.** Within a stratum, `is_correct` *is* `(pred == stratum)`. Decoding the judge's own verdict is a perfect within-stratum predictor. Conditioning does not remove it; only the pooled fit does, and only when strata are balanced. | inflates | `test_pooled_fit_is_what_blocks_the_verdict_channel` |
| 3 | **Concat swamping.** A `hidden_size`-wide block on a 6-column baseline under a shared penalty is not "baseline + block". Pure noise costs a strong baseline up to 0.315 AUROC. | *deflates* | `test_offset_ladder_is_monotone_under_a_wide_noise_block` |
| 4 | **Clustered DeLong.** The field-default paired test understates the SE by 2.17× on pairwise judge data, which is clustered by construction (both orders of every pair). | inflates | `test_delong_matches_the_bootstrap_only_without_clustering` |
| 5 | **The 0.5 reference.** A first-order SDT observer with no metacognition already scores far above 0.5, and higher as it gets more accurate. Type-2 AUROC against 0.5 credits accuracy as self-knowledge. | inflates | C-SDT simulation, `sdt_null_reference` |

Plus two hygiene case studies: **C-NUM** (fp16 overflow silently deleting a
model's spectral data *and* corrupting its verdicts) and **C-VER** (library
version drift invisible in the results files).

## Structure

1. **Introduction.** The question, the construction, the five ways it fails.
2. **Related work.** Metacognition/meta-d′ (Maniscalco & Lau; Fleming & Lau);
   probing and control tasks (Hewitt & Liang); LLM-as-judge and reward-model
   evaluation; selective prediction.
3. **Setup.** Banks, formats, logprob-readout judging, the panel, the ladder.
4. **The estimand.** Type-2 AUROC, its two floors, and why neither is 0.5.
   §4.1 the SDT (first-order) floor; §4.2 the verdict-leakage floor.
5. **Two leakage channels.** Identity and self-verdict; the two-dimensional
   taxonomy; `identity_decodability` and `verdict_decodability`.
6. **The ladder.** Concat swamping, the offset estimator, the two estimands
   (offset = increment, `*only` = achievable, gap = redundancy).
7. **Inference.** Grouped paired bootstrap vs clustered DeLong; TOST;
   MDE from the pilot's own CIs.
8. **Results.** The demonstration panel. **Anchor result: `M4 − M3`.**
9. **Numerical and provenance hygiene.** C-NUM, C-VER.
10. **Limitations.** Named hazards we did not get to, each with a proposed
    measurement.
11. **Corrections.** Our own analytic errors, grouped by cause.

## Primary tests (preregistered, m = 3)

1. Do internals beat the behavioural baseline under the conditional estimator,
   pooled hierarchically over judges and banks, TOST against ±0.02?
2. Does spectral add beyond activations? (Equivalence; already powered.)
3. Does any internal signal improve selective prediction over the margin at
   matched coverage? Reported as coverage at fixed risk, not ΔAUROC.

Everything else is exploratory and unstarred, in the same table, labelled.

## Status of the numbers

- **Available now:** power/MDE (`results/power.json`), the concat-fitted pilot
  ladder (`results/analysis_llmbar.json`, retained as the §6 artefact
  demonstration), the planted-validation constants, C-NUM/C-VER tables.
- **Pending the regeneration pass:** every offset-fitted rung, both nulls per
  slice, `M2only`/`M3only`, the coefficient ratio, the positive control, the
  per-stratum counts, and the recomputed power.
- `fill_numbers.py --verify` lists every macro that is still unresolved and
  exits non-zero, so the paper cannot be built with a stale or missing number.

## Positioning — Related Work must be rewritten before the abstract freezes

**The SDT/meta-d′ framing is not novel and must not be presented as such.**
meta-d′ and the M-ratio have already been applied to language models, in at
least one case in a logprob-readout setting close to ours. A reviewer from the
metacognition side who sees it presented as new will reject on that alone.

The defensible claim is a **bridge**: the metacognition literature has the tool
and uses it; the interpretability/probing literature reports type-2 AUROC
against 0.5 and does not; this paper connects them and measures what the
omission costs in the setting where it costs most. Bridging contributions are
publishable when both sides are explicitly credited.

**What is actually ours:** the two leakage channels. Identity leakage is
specific to *judge* evaluation — in factual-QA metacognition work, correctness
is not determined by which of two candidates is gold, so the channel does not
arise; here it does, by construction, at ~0.88 on LLMBar. That belongs in the
first two sentences of the abstract. Concat swamping and clustered DeLong are
also ours but are not metacognition-specific.

### Leads to verify (NOT yet citations — none of these has been checked)

Every line below is a pointer to chase, not a claim. Nothing moves into
`refs.bib` until someone resolves it to a real paper with a DOI or arXiv ID.

- Steyvers & Peters — review of LLM metacognition using AUROC / meta-d′.
- Kadavath et al. — models discriminating questions they answer correctly.
- Dai — meta-d′ applied to verbalized confidence ratings.
- Cacioli — Type-2 SDT over four open-weight LLMs, ~224k factual-QA trials,
  token-level logprobs rather than behavioural probes; reports AUROC2 and
  M-ratio rankings **inverted** relative to Type-1 accuracy. Closest to our
  setup, and adjacent to the competence trend we retracted — read first.
- A 2026 *Frontiers* paper computing meta-d′ and type-2 AUROC for ChatGPT.
- Yale-NLP survey repository tracking the area.
- Fleming & Lau 2014 — M-ratio normalisation. **Title and venue unconfirmed.**
- Hewitt & Liang, control tasks — **unconfirmed**, cited from memory.

### Selecting reimplementation targets — by exposure, not by headline number

A prompt-read probe is **not** an exempt target. It cannot decode the output,
so the output-decoding channel is genuinely absent — and that is exactly what
makes it the cleanest test of the **first-order floor**, because the reported
signal must then be either self-knowledge or item difficulty, with no third
explanation available.

The first-order floor is the only channel with **no exemption**, which is a
feature of the taxonomy rather than a limitation: one that exempted nothing
would be suspiciously convenient.

Two targets, one from each category:

1. **A prompt-read probe with a stated difficulty baseline** — floor-only test,
   one channel, cheapest to run. Compute the matched first-order floor and see
   what the headline becomes relative to it. If the floor lands near the
   authors' own difficulty baseline, the entire increment is accounted for by
   first-order accuracy and the result survives only against the wrong
   reference.
2. **A post-generation probe on a task where correctness is agreement with a
   discrete label** — the full three-channel test. Hallucination detection on
   classification or short-answer QA is where these live.

**The selection criterion is readout position and label type, not the headline
AUROC.** Searching for the largest published number selects targets that may be
exempt from the channels being tested.

### Highest-value empirical item remaining

**Li et al., "Language models are capable of metacognitive monitoring and
control of their internal activations", NeurIPS 2026** (unverified). A
published *positive* claim about internals and metacognition. Running it under
both our floors ranks **above adding a third judge and above panel breadth**.
If it survives both floors, we say so and the paper is stronger. If it moves,
that is the result.

## Anchor result, writable today

`M4 − M3`: SE 0.005–0.009 at n=200, deltas −0.010 to +0.009, TOST-equivalent
at ±0.02 in 5 of 6 estimable slices. Both sides carry the same baseline and
the same verdict leakage, so the difference is unaffected by findings 2, 3
and 5. **Spectral adds nothing beyond a linear activation probe**, as a
powered equivalence rather than a failure to reject.
