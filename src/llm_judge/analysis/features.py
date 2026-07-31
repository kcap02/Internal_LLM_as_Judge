"""Feature families for predicting judge-verdict correctness.

Families (each a matrix aligned with the result rows):
  margin      — signed Yes/No (or A/B) logprob margin + |margin|
  nuisance    — prompt length (tokens, log-tokens): the confound any
                length-sensitive spectral metric must beat
  spectral    — per-metric layer-profile summary: mean and slope over the
                last third of the network + argmax position normalized by
                depth (3 numbers/metric — no post-hoc layer picking), plus
                Fiedler-velocity summary when available
  activation  — last-token hidden states (raw; PCA is applied INSIDE the CV
                pipeline so components are fit on training folds only)
"""

from __future__ import annotations

import numpy as np

SPECTRAL_METRICS = ["fiedler", "hfer", "smoothness", "spectral_entropy"]


def margin_features(rows: list[dict]) -> np.ndarray:
    m = np.array([r["margin"] for r in rows], float)
    return np.c_[m, np.abs(m)]


def nuisance_features(rows: list[dict]) -> np.ndarray:
    n = np.array([r.get("n_tokens_prompt") or 0 for r in rows], float)
    return np.c_[n, np.log1p(n)]


def _profile(spectral_layers: list[dict]) -> np.ndarray:
    n = len(spectral_layers)
    late = slice(int(2 * n / 3), n)
    x = np.arange(n)[late]
    feats = []
    for metric in SPECTRAL_METRICS:
        v = np.array([c.get(metric) if c.get(metric) is not None else np.nan
                      for c in spectral_layers], float)
        fill = np.nanmean(v) if np.isfinite(v).any() else 0.0
        v = np.nan_to_num(v, nan=fill)
        vl = v[late]
        slope = np.polyfit(x, vl, 1)[0] if len(x) > 1 else 0.0
        feats += [vl.mean(), slope, float(np.argmax(v)) / n]
    return np.array(feats)


def _velocity(spectral: dict) -> np.ndarray:
    vel = (spectral or {}).get("velocity") or {}
    fv = np.array(vel.get("fiedler_velocity") or [0.0], float)
    n = max(len(fv), 1)
    return np.array([
        float(vel.get("max_velocity_value") or 0.0),
        float(vel.get("max_velocity_layer_index") or 0) / n,
        float(np.mean(np.abs(fv))),
    ])


def spectral_features(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        sp = r.get("spectral") or {}
        layers = sp.get("layers")
        if not layers:
            raise ValueError(
                f"row {r.get('item_id')} has no spectral layers — filter rows "
                f"upstream; silently empty spectral families are forbidden")
        out.append(np.concatenate([_profile(layers), _velocity(sp)]))
    return np.vstack(out)


def activation_features(rows: list[dict], npz, layer_frac: str = "1") -> np.ndarray:
    """Join activations (saved as item_id::layer_frac keys) back to rows."""
    vecs = []
    for r in rows:
        key = f"{r['item_id']}::{layer_frac}"
        if key not in npz:
            raise KeyError(f"activation missing for {key}")
        vecs.append(np.asarray(npz[key], dtype=np.float32))
    return np.vstack(vecs)
