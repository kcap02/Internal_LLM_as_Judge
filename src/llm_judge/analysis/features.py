"""Feature families for predicting judge-verdict correctness.

Families (each a matrix aligned with the result rows):
  margin      — signed verdict logprob margin + |margin|. The signed margin
                already encodes the predicted verdict (pred = first label iff
                margin > 0), so no separate verdict feature is needed.
  nuisance    — everything a "signal" could trivially be instead of internal
                self-knowledge: prompt length, task-start token index
                (position/RoPE artifact alarm), and subject identity. Any
                spectral or activation claim must clear THIS, not just margin.
  spectral    — per-metric layer-profile summary: mean and slope over the
                last third of the network + argmax position normalized by
                depth (3 numbers/metric, fixed a priori — no post-hoc layer
                picking), plus a Fiedler-velocity summary when available.
  activation  — last-token hidden states (raw; the probe pipeline decides
                whether to PCA them, always fitted inside the training fold).
"""

from __future__ import annotations

import numpy as np

SPECTRAL_METRICS = ["fiedler", "hfer", "smoothness", "spectral_entropy"]

# Subjects rarer than this are lumped into "other" so the design matrix does
# not grow a near-singleton column per rare subject.
MIN_SUBJECT_COUNT = 20


def _finite(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf by column means (0.0 if a whole column is unusable).

    Silent NaNs would propagate into the classifier and be imputed
    arbitrarily by whichever scaler ran first; do it explicitly instead.
    """
    a = np.asarray(a, dtype=float)
    bad = ~np.isfinite(a)
    if bad.any():
        for j in range(a.shape[1]):
            col = a[:, j]
            good = np.isfinite(col)
            col[~good] = col[good].mean() if good.any() else 0.0
    return a


def margin_features(rows: list[dict]) -> np.ndarray:
    m = np.array([r["margin"] for r in rows], float)
    return _finite(np.c_[m, np.abs(m)])


def nuisance_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Length + position + subject. Returns (matrix, column names)."""
    n = np.array([r.get("n_tokens_prompt") or 0 for r in rows], float)
    # Task-start index: where the judged content begins after the header.
    # If this alone separates classes, the "signal" is a position artifact.
    start = np.array([r.get("task_start_idx") or 0 for r in rows], float)
    cols = [n, np.log1p(n), start]
    names = ["n_tokens", "log_n_tokens", "task_start_idx"]

    subjects = [str(r.get("subject") or "") for r in rows]
    counts: dict[str, int] = {}
    for s in subjects:
        counts[s] = counts.get(s, 0) + 1
    keep = sorted(s for s, c in counts.items() if c >= MIN_SUBJECT_COUNT)
    # Drop the first level as reference to keep the design matrix full-rank.
    for s in keep[1:]:
        cols.append(np.array([1.0 if x == s else 0.0 for x in subjects]))
        names.append(f"subject={s}")
    return _finite(np.column_stack(cols)), names


def _profile(spectral_layers: list[dict]) -> np.ndarray:
    n = len(spectral_layers)
    late = slice(int(2 * n / 3), n)
    x = np.arange(n)[late]
    feats = []
    for metric in SPECTRAL_METRICS:
        v = np.array([c.get(metric) if c.get(metric) is not None else np.nan
                      for c in spectral_layers], float)
        finite = np.isfinite(v)
        fill = v[finite].mean() if finite.any() else 0.0
        v = np.where(finite, v, fill)
        vl = v[late]
        slope = np.polyfit(x, vl, 1)[0] if len(x) > 1 else 0.0
        feats += [vl.mean(), slope, float(np.argmax(v)) / n]
    return np.array(feats)


def _velocity(spectral: dict) -> np.ndarray:
    vel = (spectral or {}).get("velocity") or {}
    fv = np.array(vel.get("fiedler_velocity") or [0.0], float)
    fv = fv[np.isfinite(fv)] if np.isfinite(fv).any() else np.array([0.0])
    n = max(len(fv), 1)
    return np.array([
        float(vel.get("max_velocity_value") or 0.0),
        float(vel.get("max_velocity_layer_index") or 0) / n,
        float(np.mean(np.abs(fv))),
    ])


def spectral_feature_names() -> list[str]:
    names = []
    for m in SPECTRAL_METRICS:
        names += [f"{m}_late_mean", f"{m}_late_slope", f"{m}_argmax_frac"]
    return names + ["vel_max", "vel_argmax_frac", "vel_absmean"]


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
    return _finite(np.vstack(out))


def has_spectral(row: dict) -> bool:
    return bool((row.get("spectral") or {}).get("layers"))


def activation_features(rows: list[dict], npz, layer_frac: str = "1") -> np.ndarray:
    """Join activations (saved as `item_id::layer_frac` keys) back to rows."""
    vecs = []
    for r in rows:
        key = f"{r['item_id']}::{layer_frac}"
        if key not in npz:
            raise KeyError(f"activation missing for {key}")
        vecs.append(np.asarray(npz[key], dtype=np.float32))
    return _finite(np.vstack(vecs))


def has_activation(row: dict, npz, layer_frac: str = "1") -> bool:
    return npz is not None and f"{row['item_id']}::{layer_frac}" in npz
