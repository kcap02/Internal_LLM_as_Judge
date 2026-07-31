# Research design & phased compute plan

Target: ICLR 2027 (submission ≈ late Sept 2026).

## Claim structure

Primary: *a judge's internal state predicts verdict correctness beyond its
expressed confidence.* Secondary: *which representation carries the signal*
(margin < activations vs. spectral), whether it **transfers** across datasets
and models, and whether it buys a practical **selective-judging** gain
(precision at fixed coverage under abstention).

Every positive claim must survive the ladder (see README): a spectral result
that does not beat the fold-fitted activation probe (M3) is reported as such
— the paper's framing does not depend on spectral winning.

The full threat model, with what was done about each item, lives in
**[CONFOUNDS.md](CONFOUNDS.md)**. The load-bearing one is **C-ID**: because
`is_correct = (pred_verdict == gt_verdict)`, a pooled analysis lets any
feature that merely tells a pos item from a neg item impersonate
self-knowledge. The headline metric is therefore the conditional AUROC,
computed only within a `gt_verdict` stratum.

## Phase 0 — CPU (complete)

Datasets downloaded and normalised, banks frozen, confound audit clean of
FAILs:

| bank | items | CV groups | notes |
|---|---|---|---|
| mmlu | 4,000 | 1,993 | synthetic distractors until the solver runs |
| mmlu_pro | 4,000 | 1,991 | idem |
| judgebench | 2,480 | 528 | 620 ids → 528 groups (cross-split duplicates) |
| llmbar | 1,676 | 418 | adversarial |
| rewardbench2 | 4,000 | 999 | ties subset excluded |

Remaining WARNs are documented limitations, not blockers: `single`-format
label noise on the two preference datasets, the length→verdict coupling on
those same `single` items (0.63 / 0.56 AUC), and synthetic MCQ distractors
pending the solver run.

## Phase 1 — local GPU, 16 GB (pilot complete; main panel next)

Environment: `conda run -n gemma_spectral` (torch 2.11+cu128,
transformers 5.5, spectral_trust 0.2.1). CPU stages stay in the base env —
`datasets` is broken in the GPU env and the code enforces the separation.

**Pilot (<4B, done).** Qwen2.5-0.5B/1.5B/3B and Llama-3.2-1B on LLMBar and
JudgeBench validated every stage end to end. Measured throughput:

| stage | rate | note |
|---|---|---|
| 11 (verdict + activations) | ~20–60 items/s·model | negligible |
| 12 (spectral) | ~2–3 s/item at 0.5B on LLMBar | scales with tokens³ and depth |

Pilot findings that shape the main run:

- **Judges below ~3B are unusable as measurement subjects.** Verdict
  behaviour is degenerate: Qwen2.5-1.5B answers one way on 100% of items in
  both formats, Qwen2.5-0.5B says "Yes" to 99.5% of `single` items,
  Llama-3.2-1B picks "B" on 98.4% of `pairwise`. Only **Qwen2.5-3B** is
  competent (83.5% pairwise, 41.5% verdict bias). Small models remain useful
  as pipeline tests and as the distractor panel.
- **The degeneracy is a capability limit, not a prompt artefact.** A
  chat-template ablation (`--tag chat`) moved nothing that mattered: 1.5B
  stayed 100% degenerate under both formats, and the competent judge agreed
  to within 1.5 points. Raw prompts are kept; the table is in CONFOUNDS.md
  under C-FORMAT.
- **`pairwise` is the primary format.** For the competent judge it is far
  more accurate than `single` (83.5% vs 64.0%) and it is structurally immune
  to the length confound, since both orders contain the same two responses
  (C-LEN AUC exactly 0.500).
- **Degenerate judges manufacture false discoveries if unguarded.** Two
  contrasts initially survived BH-FDR purely because a conditional AUROC was
  estimated from a stratum holding one minority-class item. After the
  three-layer fix (C-DEGEN) the same data yields 0 of 16 survivors.
- Permutation nulls landed at 0.46–0.53 → no leakage in the CV.
- Identity-decodability ran 0.55–0.90 → pooled AUROC is inflated exactly as
  predicted; never report it as the headline.

**Main run order.**
1. `10_run_solver.py --only mmlu mmlu_pro` on the pilot panel — these models
   are then the **distractor panel**, disjoint from the judges (fixes C-SELF).
2. Set `config.distractor_panel` to the pilot models and
   `01_build_judge_banks.py --force --only mmlu mmlu_pro`.
3. `11_run_judge.py` over all banks with the 7–27B panel — fast, and alone
   unlocks Mn/M1/M1n/M3.
4. `12_run_judge_spectral.py --dry-run 5` per dataset to price it, then the
   full run, smallest models first. Budget priority when tight:
   **llmbar > judgebench > mmlu > mmlu_pro > rewardbench2** — adversarial
   free-text is where judges fail and where the story lives.
5. `20_analyse.py` after each increment; all stores are resumable.

**Spectral cost note.** `spectral_max_len = 4096` means no item is ever
skipped (measured max across banks: 3131 tokens), and cost tracks each item's
*actual* length: dense eigh is ~0.3 s/layer at 1024 tokens but ~6 s/layer at
4096. The long tail of JudgeBench/RewardBench 2 pairwise items therefore
dominates the bill. If that becomes binding, run `single` first — the choice
must be made by format, never by silently dropping long items, which would
make the retained subset length-biased.

Decision gate: if neither M2 nor M3 clears M1n anywhere on the conditional
metric, that is a publishable negative result — but first check the
difficulty strata and verdict-behaviour report for a power problem (judges
near chance ⇒ labels are mostly noise).

## Phase 2 — larger GPU (>48 GB, later)

1. Add `judge_models_large` (Qwen2.5-32B, Llama-3.3-70B) to the behavioural
   arm for the capability-scaling story: does the internal signal appear
   before behavioural accuracy improves, or with it? The pilot already shows
   the bottom of that curve is degenerate, which makes the scaling question
   sharper.
2. Spectral on the largest models only where Phase-1 effects were found.
3. Transfer experiments (train probe on MMLU, test on JudgeBench and vice
   versa) — pure CPU re-analysis of stored features, no new GPU cost.
4. Abstention experiment: coverage–precision curves using M1n vs M2/M3 scores
   as the abstention signal — again pure re-analysis.

## Paper skeleton

1. Setup: judge banks, balanced construction, three formats, the C-ID problem
   and the conditional metric that solves it.
2. Behavioural results: judge accuracy, verdict bias (including the small-model
   degeneracy), position bias in pairwise, solver↔judge coupling, difficulty
   strata.
3. The ladder: margin < +activations / +spectral, with bootstrap CIs and
   BH-FDR, per dataset × model × format; transfer matrix.
4. Where the signal lives: layer profiles, Fiedler velocity, consistency
   across models.
5. Application: selective judging (abstention curves).
6. Limitations: logprob-readout judging (no CoT), open-weight panel,
   single-token verdicts, `single`-format label noise.

## Compute bookkeeping

Every run's log in `logs/` records commit + config + versions. Results files
are keyed by dataset name only (no config hash), so **never** change `seed`,
sampling sizes, or `spectral_max_len` mid-campaign — start a new `results/`
directory instead.
