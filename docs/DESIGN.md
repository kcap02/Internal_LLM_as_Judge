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

### Known threats and their controls

| threat | control |
|---|---|
| MCQ judging ≈ re-solving | free-text arm (JudgeBench, LLMBar, RewardBench 2) is the headline |
| length confound in spectral metrics | task-token subgraph + length nuisance features (Mn, M1n) |
| pos/neg text leakage across folds | GroupKFold by question_id |
| layer cherry-picking | fixed a-priori profile summaries (mean/slope/argmax) |
| high-dim probe overfitting | PCA-24 fitted inside folds |
| "fold std" as inference | paired grouped bootstrap CIs on AUROC deltas |
| noisy labels on too-hard items | difficulty strata from panel logprobs |
| position bias in pairwise | every pair judged in both A/B orders |
| self-serving distractors | distractors from the *panel mean*, judged by all |

## Phase 0 — CPU only (now, this machine)

No GPU needed; everything is seeded and idempotent.

1. `pip install -e .` and `python scripts/00_download_datasets.py`
   (~5 datasets from the HF hub; gated models come later, datasets are open).
2. `python scripts/01_build_judge_banks.py`
   - free-text banks (judgebench, llmbar, rewardbench2) are final;
   - MCQ banks are provisional (synthetic distractors) until stage 10 runs.
3. Sanity-check the banks: balance (50/50), question counts, token-length
   distributions (items far beyond `spectral_max_len` will be skipped in
   stage 12 — check how many).
4. Dry-run the analysis machinery on the legacy results (if any old
   `mmlu_judge_spectral_results*.json` is copied into results/, stage 20
   exercises the full ladder minus activations).

Deliverable: frozen banks + a logged record of dataset sizes. Everything
downstream scores exactly these items.

## Phase 1 — local GPU (`gemma-spectral` environment)

Small panel (Qwen2.5-7B, Llama-3.1-8B, Phi-4, Mistral-Nemo-12B; gemma-2-27b
only if VRAM allows, never CPU-offloaded).

1. `python scripts/10_run_solver.py --only mmlu mmlu_pro`
2. `python scripts/01_build_judge_banks.py --force --only mmlu mmlu_pro`
   (upgrades distractors from synthetic to panel-logprob; bank mode is
   recorded in the file).
3. `python scripts/11_run_judge.py` — fast behavioral pass over *all* banks,
   captures activations. This alone unlocks M1/M1n/M3 in stage 20.
4. `python scripts/12_run_judge_spectral.py --dry-run 5` per dataset first
   (median s/item × bank size = campaign cost), then the full run, smallest
   models first. If the budget is tight, spectral-run priority:
   **llmbar > judgebench > mmlu > mmlu_pro > rewardbench2** (adversarial
   free-text is where the story lives).

   **Spectral window coverage (measured on the frozen banks, GPT-2 tokenizer
   estimate):** at `spectral_max_len=1024`, items over the window are skipped
   and recorded (`skipped: too_long`) — ~47% of JudgeBench and ~24% of
   RewardBench 2 (pairwise items are the long ones), ~1% of LLMBar, 0% of
   MCQ. Consequences, to state in the paper:
   - ladder comparisons stay internally valid (all rungs use the same
     spectral-covered subset), but spectral results on judgebench/rewardbench2
     generalize to *shorter* items only — report the skip rate;
   - the `single` format has much better coverage than `pairwise`; run
     spectral on `single` first and treat pairwise spectral as optional;
   - raising the window to 2048 costs ~8x per layer (O(N^3)) — only worth it
     on the large-GPU phase, and only if Phase-1 free-text effects are real.
5. `python scripts/20_analyse.py` after each increment — the analysis is
   incremental-friendly since all stores are resumable.

Decision gate after Phase 1: if neither M2 nor M3 clears M1n anywhere, the
result is a (publishable but different) negative finding — stop scaling and
investigate per-stratum power first (are judges near chance on the hard
tercile?).

## Phase 2 — larger GPU (>48 GB, later)

1. Add `judge_models_large` (Qwen2.5-32B, Llama-3.3-70B) to the behavioral
   arm (stages 10–11) for the capability-scaling story: does the internal
   signal appear before behavioral accuracy improves, or with it?
2. Spectral on the largest models only where Phase-1 effects were found
   (targeted, not exhaustive — 70B spectral is expensive; consider
   `spectral_max_len=512` there, the window is a config key).
3. Transfer experiments (train probe on MMLU, test on JudgeBench and
   vice-versa) — pure CPU re-analysis of stored features, no new GPU cost.
4. Abstention experiment: coverage–precision curves using M1n vs M2/M3 scores
   as the abstention signal — again pure re-analysis.

## Paper skeleton

1. Setup: judge banks, balanced construction, three formats.
2. Behavioral results: judge accuracy, Yes-bias/position-bias, solver↔judge
   coupling (good-responder), difficulty strata.
3. The ladder: margin < +activations / +spectral, with CIs, per dataset ×
   model; transfer matrix.
4. Where the signal lives: layer profiles, Fiedler velocity, consistency
   across models.
5. Application: selective judging (abstention curves).
6. Limitations: logprob-readout judging (no CoT), open-weight panel,
   single-token verdicts.

## Compute bookkeeping

Every run's log in `logs/` records commit + config + versions. Results files
are keyed by dataset name only (no config hash), so **never** change
`seed`, sampling sizes, or `spectral_max_len` mid-campaign — bump a new
results/ subdirectory (or clean) instead.
