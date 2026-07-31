"""Central configuration.

Everything that varies between runs lives here (or in a JSON override file
passed via --config). Scripts never hard-code paths, panels, or seeds, so a
run is fully described by (git commit, config file, CLI args) — that triple
goes into the log header of every stage.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"


def tagged(name: str, tag: str | None) -> str:
    """Suffix a results filename with a variant tag.

    Ablations (chat template on/off, a different window, a different
    distractor panel) must never append into the same stream as the main run:
    the resume key is (model, item_id), so a variant would silently be
    treated as already-done work. `--tag` gives each variant its own stream.
    """
    return name if not tag else f"{name}__{tag}"


@dataclass
class Config:
    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42                      # every random draw derives from this

    # ── Model panel ──────────────────────────────────────────────────────────
    # Pilot panel: <4B, ungated, fits a 16 GB card with room for attention
    # matrices. Used to validate the pipeline end to end before spending
    # GPU-days on the real panel.
    judge_models_pilot: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ])
    # Main panel (7-27B; fp16 on a 24-48 GB card).
    judge_models: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/phi-4",
        "google/gemma-2-27b-it",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ])
    # Large panel (behavioral arm only until the big GPU is available).
    judge_models_large: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-32B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ])
    # Models that must not be loaded with CPU offload (forward passes would
    # re-transfer weights every step; spectral analysis becomes intractable).
    no_cpu_offload: list[str] = field(default_factory=lambda: [
        "google/gemma-2-27b-it",
    ])
    # Solver models whose logprobs define MCQ distractors. Keep this DISJOINT
    # from the judge panel: a judge shown a trap it helped select is being
    # tested on its own inclinations (self-preference confound C-SELF).
    # None = use every solver model available (fast, but confounded).
    distractor_panel: list[str] | None = None

    # ── Datasets ─────────────────────────────────────────────────────────────
    # Which datasets to build judge banks for. Keys must match the registry in
    # llm_judge.datasets.
    datasets: list[str] = field(default_factory=lambda: [
        "mmlu", "mmlu_pro", "judgebench", "llmbar", "rewardbench2",
    ])
    # Per-dataset sample sizes (items = pos/neg *pairs* for MCQ, judged
    # responses for free-text). None = take everything available.
    n_questions: dict = field(default_factory=lambda: {
        "mmlu": 2000,
        "mmlu_pro": 2000,
        "judgebench": None,      # ~350 pairs — small, take all
        "llmbar": None,          # ~419 pairs — small, take all
        "rewardbench2": 1000,
    })

    # Weight dtype. bfloat16, not float16: same memory, far wider exponent
    # range. Under fp16 some models overflow to inf inside attention and the
    # spectral analysis then fails on every single item (observed: all 400
    # items of Qwen2.5-1.5B). Never quantized.
    model_dtype: str = "bfloat16"

    # ── Prompting ────────────────────────────────────────────────────────────
    ntrain: int = 0                     # k-shot; 0 = zero-shot everywhere
    use_chat_template: bool = False     # raw Hendrycks-style prompts by default
    max_prompt_tokens: int = 4096       # hard cap before k-reduction

    # ── Spectral (spectral_trust 0.2.x) ─────────────────────────────────────
    # Window = 4096 so NO item is dropped for length (measured max over all
    # banks is ~3.1k tokens). Cost is driven by each item's ACTUAL length,
    # not by the cap: dense eigh is ~0.3 s at 1024 and ~6 s at 4096 tokens
    # PER LAYER, so raising the cap is free for short items and simply pays
    # the true price for the long ones instead of silently skipping them
    # (a length-biased retained subset is a worse problem than compute).
    spectral_max_len: int = 4096
    spectral_normalization: str = "sym"  # valid in 0.2.x: rw | sym | none.
    # "sym" is REQUIRED for basis-dependent metrics (HFER, spectral entropy):
    # the rw Laplacian is non-symmetric -> non-orthonormal eigenvectors ->
    # silently invalid Parseval accounting.
    spectral_calc_velocity: bool = True
    spectral_task_subgraph: bool = True  # restrict graph to task tokens
    hfer_cutoff_ratio: float = 0.1
    eigen_solver: str = "dense"          # at N<=1024 dense eigh beats ARPACK

    # ── Activations (probe baseline) ─────────────────────────────────────────
    activation_layers: list[float] = field(default_factory=lambda: [0.5, 1.0])
    # fractions of depth; 1.0 = last layer. Last-token hidden state is stored.

    # ── Analysis ─────────────────────────────────────────────────────────────
    n_splits: int = 5
    n_bootstrap: int = 2000
    # "full": standardized full-dim activations with stronger L2 (C=0.1) —
    # the standard linear-probe baseline; right choice at thousands of items.
    # "pca": fold-fitted PCA(activation_pca_dims) — for small item counts,
    # where full-dim would overfit. NB: PCA keeps high-VARIANCE directions
    # and can destroy a correctness signal with no variance advantage.
    activation_probe: str = "full"
    activation_pca_dims: int = 24

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Default config, optionally overridden by a JSON file."""
        cfg = cls()
        if path:
            with open(path, encoding="utf-8") as f:
                overrides = json.load(f)
            for k, v in overrides.items():
                if k.startswith("_"):
                    continue          # "_comment" and friends document the file
                if not hasattr(cfg, k):
                    raise KeyError(f"Unknown config key: {k!r}")
                setattr(cfg, k, v)
        return cfg

    def dump(self) -> dict:
        return dataclasses.asdict(self)
