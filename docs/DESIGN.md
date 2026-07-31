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

- **The most valuable finding was a numerical one.** Under float16,
  Qwen2.5-1.5B overflowed to `inf` inside attention: it lost 100% of its
  spectral data *and* its verdicts were corrupted, so it presented as a fully
  degenerate judge (100% verdict bias, chance accuracy) where under bfloat16
  it is competent (73.5% pairwise, 36.5% bias). The two dtypes agree on only
  36.5% of its verdicts, while every other model was stable to ~2 points. A
  numerical bug was on its way into the write-up as a finding about model
  scale. `model_dtype = "bfloat16"`, uniform across the panel — see C-NUM.
- **Usable-judge floor is ~1.5B**, on dtype-matched data: Qwen2.5-1.5B
  (73.5%) and Qwen2.5-3B (82.5% pairwise) are measurable; Qwen2.5-0.5B
  (52.0%) and Llama-3.2-1B (48.5%, 2.5% bias) are at chance. Sub-usable
  models remain valuable as the distractor panel.
- **Prompt format is not the discriminator.** The chat-template ablation
  (`--tag chat`) moves a competent judge by ~1.5 points. Raw prompts are kept
  for uniformity across base and instruct models.
- **`pairwise` is the primary format.** For a competent judge it is far more
  accurate than `single` (82.5% vs 64.0%) and it is structurally immune to
  the length confound, since both orders contain the same two responses
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

**Spectral cost and VRAM.** `spectral_max_len = 4096` means the *bank* never
excludes an item (measured max across banks: 3131 tokens), and compute tracks
each item's actual length: dense eigh is ~0.3 s/layer at 1024 tokens but
~6 s/layer at 4096.

The binding constraint is VRAM, not time. `output_attentions=True` retains a
`[heads, N, N]` tensor per layer, so memory is quadratic in length:

| model | 1024 tok | 2048 tok | 4096 tok |
|---|---|---|---|
| Qwen2.5-3B | 1.1 GB | 4.5 GB | 18.0 GB |
| Qwen2.5-7B | 1.5 GB | 6.1 GB | 24.5 GB |

On the 16 GB card: LLMBar (max 1119 tok) is safe at any window; RewardBench 2
(max 2441) fits for ≤3B; **JudgeBench pairwise (max 3131) needs a larger card
or a reduced window.** Stage 12 computes this, reserves it as load headroom,
warns before the run if a full-window item cannot fit, and records any OOM
skip as `skipped: "oom"` with a length-bias warning. Reduce the window per
model and report it — never let long items drop silently, which is exactly
the coverage bias C-WIN exists to prevent.

**Precision must be uniform.** `model_dtype = "bfloat16"` everywhere. Under
fp16, Qwen2.5-1.5B overflowed to `inf` inside attention and spectral analysis
failed on all 400 of its items while the run looked healthy. Mixing dtypes
across a panel would also confound "model" with "precision", so the fp16
pilot stream is archived as `__fp16` and the pilot was re-run under bf16.

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

## Reproducing the pilot

```bash
# CPU (base Python)
python scripts/00_download_datasets.py
python scripts/01_build_judge_banks.py
python scripts/02_audit_confounds.py --pilot        # must show no FAIL

# GPU (gemma_spectral)
CONDA=C:/Users/valno/anaconda3/envs/gemma_spectral/python.exe
$CONDA scripts/11_run_judge.py --only llmbar --pilot --limit 400
$CONDA scripts/12_run_judge_spectral.py --only llmbar --pilot --dry-run 400

# Format ablation (own stream; run AFTER the spectral job, not alongside it)
$CONDA scripts/11_run_judge.py --only llmbar --pilot --limit 400 \
       --config configs/chat_template.json --tag chat
python scripts/20_analyse.py --only llmbar --tag chat

# CPU
python scripts/20_analyse.py --only llmbar
```

Expected on this data: full ladder Mn/M1/M1n/M1nd/M2/M3/M4, permutation nulls
near 0.500, every sub-3B slice reported as degenerate with contrasts
suppressed, and **0 surviving contrasts after BH-FDR** — the correct answer
for judges this small.

## Compute bookkeeping

**Run one GPU stage at a time.** Stage 12 reserves headroom for retained
attention, so a second GPU job (even a light one like stage 10) can push the
card to its limit: observed 15.7 GB of 16 GB with both running, at which
point *neither* made progress. Every stage is resumable, so serialising costs
nothing — stop one, let the other finish, restart. `run_pipeline.ps1`
sequences them deliberately.


Every run's log in `logs/` records commit + config + versions. Results files
are keyed by dataset name only (no config hash), so **never** change `seed`,
sampling sizes, or `spectral_max_len` mid-campaign — start a new `results/`
directory instead.
