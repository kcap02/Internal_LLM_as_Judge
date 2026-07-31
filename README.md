# LLM-Judge Internals

**Research question:** can an LLM judge's *internal state* predict when its
verdict is wrong — and which representation carries that signal: the
confidence margin, hidden-state activations, or attention-graph spectral
diagnostics ([spectral-trust](https://pypi.org/project/spectral-trust/))?

The framing is deliberately representation-agnostic: the paper is interesting
whichever family wins, because every claim is tested against the cheaper
alternative (spectral must beat activation probes, activation probes must
beat the margin, everything must beat prompt length).

## Pipeline

```
CPU  00_download_datasets.py   HF hub -> data/*_questions.json / *_items.json
CPU  01_build_judge_banks.py   -> data/bank_<dataset>.json   (what judges score)
GPU  10_run_solver.py          MCQ solver logprobs -> results/solver_<d>.json
CPU  01 (again, --force)       rebuild MCQ banks with panel-derived distractors
GPU  11_run_judge.py           verdicts + margins + ACTIVATIONS (fast)
GPU  12_run_judge_spectral.py  verdicts + spectral profiles (slow, O(N^3)/layer)
CPU  20_analyse.py             AUROC ladder + bootstrap CIs -> results/analysis_<d>.json
```

Stages 11 and 12 read the **same bank file** — that identity is what makes
behavioral and spectral numbers comparable. All GPU stages are resumable
(atomic writes, `(model, item_id)` resume keys) and can be interrupted freely.

## Datasets

| name | task | items | why |
|---|---|---|---|
| `mmlu` | 4-choice MCQ | pos/neg letter pairs | controlled pilot |
| `mmlu_pro` | 10-choice MCQ | pos/neg letter pairs | harder, richer distractors |
| `judgebench` | free-text | single (Yes/No) + pairwise (A/B, both orders) | objective response labels |
| `llmbar` | free-text | single + pairwise | adversarial — where judges fail |
| `rewardbench2` | free-text | single + pairwise | best-of-N reward evaluation |

All judge banks are balanced 50/50 by construction, seeded, and identical
across judges. MCQ distractors come from the solver panel's letter logprobs
(the most *tempting* wrong option, not a random letter).

## Analysis ladder

| model | features | question it answers |
|---|---|---|
| Mn | prompt length | is the "signal" just length? |
| M1 | logprob margin | what does behavior alone predict? |
| M1n | margin + length | the honest behavioral baseline |
| M2 | + spectral profile & Fiedler velocity | does the attention graph add anything? |
| M3 | + activation probe (PCA-24, fold-fitted) | does a cheap linear probe already do it? |
| M4 | everything | headroom |

Out-of-fold predictions with `GroupKFold(groups=question_id)` (pos/neg pairs
never split across folds), paired **grouped bootstrap** CIs on adjacent-rung
AUROC deltas, difficulty strata for MCQ.

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
huggingface-cli login       # or set HF_TOKEN (gated: Llama, Gemma)
python scripts/00_download_datasets.py
python scripts/01_build_judge_banks.py
```

Everything configurable lives in `src/llm_judge/config.py`; override any key
with a JSON file via `--config`. Every stage writes a log to `logs/` whose
header records the git commit, package versions and full config — a results
file is fully traceable to the run that produced it.

## Repo layout

```
src/llm_judge/        library (datasets/, analysis/, model_loading, prompts, ...)
scripts/              numbered pipeline stages (thin wrappers over the library)
data/                 normalized datasets + judge banks
results/              solver / judge / spectral results, activations (.npz), analyses
logs/                 one provenance log per stage run (kept in git)
docs/DESIGN.md        research design + phased compute plan
legacy/               original monolithic scripts (superseded, kept for reference)
```

## Key methodological decisions

- **`normalization="sym"`** in GSPConfig (verified against spectral_trust
  0.2.2: valid values `rw|sym|none`). The `rw` default's eigenvectors are not
  orthonormal, silently invalidating HFER/spectral-entropy.
- **Task-token subgraph** (`subgraph_indices`) so spectral metrics are not
  driven by header/preamble length; prompt length is additionally a nuisance
  feature in the ladder.
- **No layer cherry-picking:** layer profiles are summarized as mean/slope on
  the last third + normalized argmax (3 numbers per metric), fixed a priori.
- **Activation probe = full-dim regularized logistic** (fit per fold,
  C=0.1) by default: unsupervised PCA keeps high-variance directions and can
  destroy a correctness signal with no variance advantage. A fold-fitted
  PCA variant (`activation_probe="pca"`) exists for small-n settings.
- **Spectral window skips are recorded, never silent** (`skipped: too_long`):
  ~47% of JudgeBench and ~24% of RewardBench 2 items exceed the 1024-token
  window (see docs/DESIGN.md for the coverage policy); MCQ banks fit fully.
- **Verdicts survive spectral failures**; empty spectral output is a loud
  error, never a silently missing feature family.
