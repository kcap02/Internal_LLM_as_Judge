"""Logprob scoring + last-token hidden-state extraction.

One forward pass per item: read next-token logprobs of the target tokens
(letters or verdicts) at the position following "Answer:". Optionally return
the last-token hidden state at selected layers — the input to the
activation-probe baseline that any spectral claim must beat.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def score_targets(model, tokenizer, prompt: str, target_ids: dict[str, int],
                  activation_layers: list[float] | None = None):
    """Return (pred, target_logprobs, activations, n_tokens).

    activation_layers: fractions of depth (1.0 = last layer). When given, the
    forward runs with output_hidden_states=True and the last-token hidden
    state of each selected layer is returned as float16 numpy arrays keyed by
    the fraction (e.g. {"0.5": ..., "1.0": ...}).
    """
    want_acts = bool(activation_layers)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tokens = int(inputs["input_ids"].shape[1])
    outputs = model(**inputs,
                    output_attentions=False,
                    output_hidden_states=want_acts,
                    use_cache=False)
    logprobs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
    target_lp = {t: logprobs[tid].item() for t, tid in target_ids.items()}
    pred = max(target_lp, key=target_lp.get)

    activations = None
    if want_acts:
        hs = outputs.hidden_states  # tuple: embeddings + one per layer
        n_layers = len(hs) - 1
        activations = {}
        for frac in activation_layers:
            layer_idx = max(1, min(n_layers, round(frac * n_layers)))
            vec = hs[layer_idx][0, -1].detach().float().cpu().numpy()
            activations[f"{frac:g}"] = vec.astype(np.float16)

    del outputs, logprobs, inputs
    return pred, target_lp, activations, n_tokens


def margin(target_lp: dict[str, float], labels: list[str]) -> float:
    """Signed confidence margin: logprob(first label) - logprob(second)."""
    return target_lp[labels[0]] - target_lp[labels[1]]


class ActivationStore:
    """Accumulates last-token activations and saves one .npz per model.

    Activations never go into the results JSON (they would bloat it by GBs);
    they are joined back at analysis time via item_id.
    """

    def __init__(self, path):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}

    def add(self, item_id: str, activations: dict[str, np.ndarray]) -> None:
        for frac, vec in activations.items():
            self.vecs[f"{item_id}::{frac}"] = vec

    def save(self) -> None:
        from pathlib import Path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.vecs)
