# LLM-Judge Internals

**Research question:** can an LLM judge's *internal state* predict when its
verdict is wrong — and which representation carries that signal: the
confidence margin, hidden-state activations, or attention-graph spectral
diagnostics ([spectral-trust](https://pypi.org/project/spectral-trust/))?

The framing is deliberately representation-agnostic: the paper is interesting
whichever family wins, because every claim is tested against the cheaper
alternative (spectral must beat activation probes, activation probes must
beat the margin, everything must beat prompt length, position and subject).

Every known way this could produce a spurious result is enumerated, fixed or
quantified in **[docs/CONFOUNDS.md](docs/CONFOUNDS.md)** — read that before
trusting any number here.

## Pipeline

```
CPU  00_download_datasets.py   HF hub -> data/*_questions.json / *_items.json
CPU  01_build_judge_banks.py   -> data/bank_<dataset>.json   (what judges score)
CPU  02_audit_confounds.py     -> results/audit_banks.json   (PASS/WARN/FAIL)
GPU  10_run_solver.py          MCQ solver logprobs -> results/solver_<d>.jsonl
CPU  01 (again, --force)       rebuild MCQ banks with panel-derived distractors
GPU  11_run_judge.py           verdicts + margins + ACTIVATIONS (fast)
GPU  12_run_judge_spectral.py  verdicts + spectral profiles (slow, O(N^3)/layer)
CPU  20_analyse.py             AUROC ladder + controls -> results/analysis_<d>.json
```

Stages 11 and 12 read the **same bank file** — that identity is what makes
behavioural and spectral numbers comparable, and stage 20 merges both so a
model present in only one still gets every rung its features allow. Result
streams are append-only JSONL: O(1) per item, crash-safe, resumable on
`(model, item_id)`.

## Two environments

`datasets` is unusable in the GPU env (Windows cert-store SSL bug) and
`spectral_trust` is absent from the base env, so stages are split — and the
code enforces it: GPU stages import `llm_judge.registry`, never
`llm_judge.datasets`.

| stages | interpreter | needs |
|---|---|---|
| 00, 01, 02 | base Python 3.11 | `datasets`, `transformers` |
| 10, 11, 12 | `conda run -n gemma_spectral` | `torch`, `transformers`, `spectral_trust` |
| 20 | either | `numpy`, `scikit-learn` |

## Datasets

| name | task | items | groups | why |
|---|---|---|---|---|
| `mmlu` | 4-choice MCQ | 4,000 | 1,993 | controlled pilot |
| `mmlu_pro` | 10-choice MCQ | 4,000 | 1,991 | harder, richer distractors |
| `judgebench` | free-text | 2,480 | 528 | objective response labels |
| `llmbar` | free-text | 1,676 | 418 | adversarial — where judges fail |
| `rewardbench2` | free-text | 4,000 | 999 | best-of-N reward evaluation |

Free-text banks carry two item families: `single` (one response, Yes/No) and
`pairwise` (both responses, A/B, **shown in both orders**). All banks are
balanced 50/50 by construction, seeded, and identical across judges. CV
groups come from the normalised question *text*, not the id — JudgeBench
reuses 92 questions across its `claude`/`gpt` splits.

## Analysis ladder

| rung | features | question it answers |
|---|---|---|
| Mn | length + task-start position + subject | is the "signal" just the prompt? |
| M1 | logprob margin | what does behaviour alone predict? |
| M1n | margin + nuisance | is it length, position or topic? |
| M1nd | + peer difficulty (leave-one-model-out) | **the baseline every internal claim must clear** — is it just a hard item? |
| M2 | + spectral profile & Fiedler velocity | does the attention graph add anything? |
| M3 | + activation probe (per-fold) | does a cheap linear probe already do it? |
| M4 | everything | headroom |

The baseline is M1nd wherever ≥2 judges scored the same items, M1n
otherwise; which one was used is printed per slice and stored in the report.

**The headline metric is the conditional AUROC** — computed only between
items sharing a `gt_verdict`. Pooling pos and neg items would let any feature
that merely distinguishes them impersonate self-knowledge (see C-ID); pooled
AUROC is reported alongside, labelled as the inflated number.

Every run also produces:
- an **identity-decodability control** (how much of the pooled number is item identity),
- a **permutation-null control** (labels shuffled within strata, full refit; must land at ~0.500),
- **paired grouped-bootstrap** CIs on each contrast,
- **BH-FDR** across every contrast in the run (`results/analysis_fdr.json`).

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
huggingface-cli login       # or set HF_TOKEN (gated: Llama, Gemma)

python scripts/00_download_datasets.py
python scripts/01_build_judge_banks.py
python scripts/02_audit_confounds.py        # must be free of FAIL

# GPU pilot (<4B models, fast end-to-end validation)
conda run -n gemma_spectral python scripts/11_run_judge.py --pilot --limit 400 --only llmbar
conda run -n gemma_spectral python scripts/12_run_judge_spectral.py --pilot --dry-run 400 --only llmbar
python scripts/20_analyse.py --only llmbar
```

Everything configurable lives in `src/llm_judge/config.py`; override any key
with a JSON file via `--config` (see `configs/`). Every stage writes a log to
`logs/` whose header records the git commit, package versions and full config
— a results file is fully traceable to the run that produced it.

**Ablations** need `--tag`, which gives the variant its own result stream:

```bash
python scripts/11_run_judge.py --config configs/chat_template.json --tag chat ...
python scripts/20_analyse.py --tag chat --only llmbar
```

Without a tag the variant would collide with the main run's `(model,
item_id)` resume keys and be silently skipped as already-done work.

## Repo layout

```
src/llm_judge/        library (datasets/, analysis/, diagnostics, grouping, ...)
scripts/              numbered pipeline stages (thin wrappers over the library)
data/                 normalized datasets + judge banks
results/              *.jsonl result streams, activations (.npz), analyses, audits
logs/                 one provenance log per stage run (kept in git)
docs/CONFOUNDS.md     every confound: neutralised / measured / open
docs/DESIGN.md        research design + phased compute plan
legacy/               original monolithic scripts (superseded, kept for reference)
```

## Key methodological decisions

- **`normalization="sym"`** in GSPConfig (verified against the installed
  spectral_trust: valid values `rw|sym|none`). The `rw` default's
  eigenvectors are not orthonormal, silently invalidating HFER and spectral
  entropy.
- **Task-token subgraph** (`subgraph_indices`) so spectral metrics are not
  driven by header length; `task_start_idx` is additionally a nuisance
  regressor, as a position-artefact alarm.
- **`spectral_max_len = 4096`**: measured max over all banks is 3131 tokens,
  so **no item is ever skipped for length**. Cost tracks each item's actual
  length, not the cap.
- **No layer cherry-picking:** layer profiles are summarised as mean/slope on
  the last third + normalised argmax (3 numbers per metric), fixed a priori.
- **Activation probe = full-dim regularised logistic** fitted per fold
  (C=0.1). Unsupervised PCA keeps high-variance directions and can discard a
  correctness signal with no variance advantage; `activation_probe="pca"`
  remains available for small-n settings.
- **Verdicts survive spectral failures**; empty spectral output is a loud
  error, never a silently missing feature family.
