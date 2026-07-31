"""spectral_trust integration (verified against spectral_trust 0.2.2 source).

Key decisions, each verified in the installed library:
  normalization="sym"  — valid values in 0.2.x are {rw, sym, none}. The rw
      default builds a NON-symmetric Laplacian whose eigenvectors are not
      orthonormal: HFER / spectral-entropy / Parseval accounting become
      silently invalid. "sym" is mandatory for basis-dependent metrics.
  calc_velocity=True   — analyze_text() then returns 'velocity_metrics'
      (per-layer Fiedler velocity + max value/layer).
  subgraph_indices     — analyze_text(text, subgraph_indices=[...]) restricts
      the graph AND the signal to the task tokens, controlling for
      header/preamble length (a nuisance for length-sensitive metrics).
  eigen_solver="dense" — at N <= 1024 tokens dense eigh beats ARPACK 'SM'.
  save_plots/display_plots/verbose off — defaults would emit one matplotlib
      figure and one log line per layer per item.
"""

from __future__ import annotations

import numpy as np

try:
    from spectral_trust import GSPConfig, GSPDiagnosticsFramework
    HAS_SPECTRAL_TRUST = True
except ImportError:
    GSPConfig = GSPDiagnosticsFramework = None
    HAS_SPECTRAL_TRUST = False


# Real attribute names of the SpectralDiagnostics dataclass (0.2.2).
METRIC_ATTRS = {
    "fiedler": "fiedler_value",
    "hfer": "hfer",
    "smoothness": "smoothness_index",
    "spectral_entropy": "spectral_entropy",
    "energy": "energy",
    "connectivity": "connectivity",
}


def build_gsp_config(model_name: str, cfg) -> "GSPConfig":
    assert cfg.spectral_normalization == "sym", (
        "normalization must be 'sym' — rw eigenvectors are not orthonormal "
        "and basis-dependent metrics become invalid (no error is raised)")
    return GSPConfig(
        model_name=model_name,
        device="cuda",
        torch_dtype="float16",
        trust_remote_code=True,
        max_length=cfg.spectral_max_len,
        normalization=cfg.spectral_normalization,
        eigen_solver=cfg.eigen_solver,
        hfer_cutoff_ratio=cfg.hfer_cutoff_ratio,
        calc_velocity=cfg.spectral_calc_velocity,
        save_plots=False,
        display_plots=False,
        save_intermediate=False,
        verbose=False,
        output_dir=f"./gsp_out/{model_name.split('/')[-1]}",
    )


def attach_model(framework, model, tokenizer, model_name: str):
    """Inject our already-loaded model instead of letting the instrumenter
    load a second copy of the weights."""
    framework.instrumenter.model = model
    framework.instrumenter.tokenizer = tokenizer
    for attr, val in (("device", model.device), ("model_name", model_name)):
        if hasattr(framework.instrumenter, attr):
            try:
                setattr(framework.instrumenter, attr, val)
            except Exception:
                pass


def analyze_prompt(framework, prompt: str,
                   subgraph_indices: list[int] | None = None,
                   expected_n_tokens: int | None = None) -> dict:
    """Run the instrumented forward and extract JSON-safe scalars.

    Returns {"layers": [...], "velocity": {...} | None} or {"error": ...}.
    The raw analysis dict also holds 'model_outputs' (full attentions +
    hidden states) which must never be serialized, and 'layer_diagnostics'
    holds dataclass objects, not dicts — hence the explicit extraction.

    `expected_n_tokens` guards the subgraph: the indices are computed against
    OUR tokenisation, while the library re-tokenises internally (truncating
    at config.max_length). The two agree today — same tokenizer, same
    add_special_tokens default, prompts capped below the window — but a
    silent divergence would point every subgraph index at the wrong token,
    so the invariant is checked rather than assumed.
    """
    try:
        analysis = framework.analyze_text(prompt, save_results=False,
                                          subgraph_indices=subgraph_indices)
    except Exception as e:  # verdict must survive a spectral failure
        return {"error": f"{type(e).__name__}: {e}"}

    if expected_n_tokens is not None:
        got = len(analysis.get("tokens") or [])
        if got and got != expected_n_tokens:
            del analysis
            return {"error": f"token count mismatch: framework saw {got}, "
                             f"caller indexed {expected_n_tokens} — subgraph "
                             f"indices would be misaligned"}

    layers = []
    for i, d in enumerate(analysis.get("layer_diagnostics") or []):
        row = {"layer": i}
        for key, attr in METRIC_ATTRS.items():
            v = getattr(d, attr, None)
            if v is None:
                row[key] = None
            elif isinstance(v, (bool, np.bool_)):
                row[key] = bool(v)
            else:
                row[key] = float(v)
        layers.append(row)

    velocity = None
    vm = analysis.get("velocity_metrics") or {}
    if vm:
        velocity = {
            "fiedler_velocity": [float(v) for v in vm.get("fiedler_velocity", [])],
            "max_velocity_value": float(vm.get("max_velocity_value", np.nan)),
            "max_velocity_layer_index": int(vm.get("max_velocity_layer_index", -1)),
        }

    del analysis
    if not layers:
        # Empty spectral families must be loud, never silent: a config/string
        # mismatch that filters all rows would otherwise masquerade as
        # "spectral was tested and added nothing".
        return {"error": "empty layer_diagnostics — check GSPConfig"}
    return {"layers": layers, "velocity": velocity}
