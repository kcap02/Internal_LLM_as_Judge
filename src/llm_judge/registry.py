"""Dataset registry — deliberately dependency-free.

GPU stages (10/11/12) need to know which datasets are MCQ vs free-text, but
must NOT import the HuggingFace `datasets` library: the GPU environment
(gemma_spectral) has a broken `datasets` import (Windows cert-store SSL bug
at aiohttp import time), and downloading is not its job anyway. Keeping the
registry here, with no imports, is what lets the CPU and GPU stages live in
different Python environments.
"""

# What each dataset's loader emits: MCQ "questions" (judge items are derived
# from solver logprobs) vs. ready-made free-text judge "items".
KIND = {
    "mmlu": "questions",
    "mmlu_pro": "questions",
    "judgebench": "items",
    "llmbar": "items",
    "rewardbench2": "items",
}

DATASET_NAMES = sorted(KIND)
