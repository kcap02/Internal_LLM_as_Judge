"""Robust model loading — full fp16 precision only, no quantization.

The constraint: gemma-2-27b-it weighs ~57 GB in float16 and simply does not
fit a 44 GB card — that is arithmetic, not fragmentation, and no amount of
cache-clearing changes it. Strategies tried in order, all fp16:

  1. single GPU, if the model fits;
  2. all available GPUs, if the pod has several;
  3. GPU + CPU-RAM offload of some layers (slow but exact);
  4. otherwise: explicit error, and the campaign continues without the model.

Cleanup after an aborted load is handled seriously: without it, the weights
already transferred stay in VRAM and doom the next attempt.

attn_implementation="eager" for ALL models: sdpa does not return the attention
matrices that spectral analysis needs.
"""

from __future__ import annotations

import gc
import os
import shutil

import torch

# Limits fragmentation. Must be set BEFORE the first CUDA call.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Default headroom left free per GPU for activations and cache. Enough for a
# plain forward (stages 10/11). Stage 12 must pass a much larger value: with
# eager attention and output_attentions=True the attention matrices of EVERY
# layer are retained, which is quadratic in prompt length — see
# attention_memory_gb().
VRAM_HEADROOM_GB = 2.0

# Disk offload dir, used only if CPU RAM is insufficient.
OFFLOAD_DIR = "./offload"


# ── Memory ────────────────────────────────────────────────────────────────────
def free_vram(*objs) -> None:
    """Actually release VRAM, including after an aborted load."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def vram_free_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    with torch.cuda.device(device):
        return torch.cuda.mem_get_info()[0] / 1024**3


def vram_total_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return sum(vram_free_gb(i) for i in range(torch.cuda.device_count()))


def cpu_ram_free_gb() -> float:
    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    # Windows / cross-platform fallback
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 0.0


# ── Size estimation without downloading weights ───────────────────────────────
def estimate_params_billions(model_name: str) -> float:
    """Estimate parameter count from the config alone.

      embeddings ~ vocab x hidden      (x2 if weights untied)
      per layer  ~ 4 x hidden^2                 (attention; upper bound if GQA)
                 + 3 x hidden x intermediate    (gated MLP)
    """
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        h = getattr(cfg, "hidden_size", None)
        L = getattr(cfg, "num_hidden_layers", None)
        v = getattr(cfg, "vocab_size", None)
        i = getattr(cfg, "intermediate_size", None) or (4 * h if h else 0)
        if not all([h, L, v]):
            return 0.0
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        emb = v * h * (1 if tied else 2)
        return (emb + L * (4 * h * h + 3 * h * i)) / 1e9
    except Exception:
        return 0.0


def _fp16_gb(model_name: str) -> float:
    return estimate_params_billions(model_name) * 2.0


def attention_memory_gb(model_name: str, n_tokens: int,
                        bytes_per_elem: int = 2) -> float:
    """VRAM the retained attention matrices need for one forward pass.

    `output_attentions=True` keeps a [heads, N, N] tensor for every layer, so
    the cost is quadratic in prompt length and linear in depth. At a 4096
    window this dominates the model weights and is the real reason a spectral
    run OOMs where the behavioural run is comfortable.

    Returns 0.0 when the config cannot be read (caller should not block on it).
    """
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        heads = getattr(cfg, "num_attention_heads", None)
        layers = getattr(cfg, "num_hidden_layers", None)
        if not heads or not layers:
            return 0.0
        return heads * layers * (n_tokens ** 2) * bytes_per_elem / 1024**3
    except Exception:
        return 0.0


# ── Loading plan ──────────────────────────────────────────────────────────────
def plan_loading(model_name: str, allow_cpu_offload: bool = True,
                 headroom_gb: float | None = None) -> dict:
    need = _fp16_gb(model_name)
    head = VRAM_HEADROOM_GB if headroom_gb is None else headroom_gb
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu0 = vram_free_gb(0)
    gpu_all = vram_total_free_gb()
    ram = cpu_ram_free_gb()

    if need == 0.0:
        mode = "single_gpu"                    # estimation failed: just try
    elif need + head <= gpu0:
        mode = "single_gpu"
    elif n_gpu > 1 and need + head * n_gpu <= gpu_all:
        mode = "multi_gpu"
    elif allow_cpu_offload and need + head <= gpu_all + ram:
        mode = "cpu_offload"
    else:
        mode = "impossible"

    return {"model": model_name, "need_gb": need, "headroom_gb": head,
            "n_gpu": n_gpu, "gpu0_gb": gpu0, "gpu_all_gb": gpu_all,
            "ram_gb": ram, "mode": mode}


def _max_memory(mode: str, headroom_gb: float = VRAM_HEADROOM_GB):
    """Per-device memory cap, with the safety headroom subtracted."""
    if mode == "single_gpu":
        return None
    n_gpu = torch.cuda.device_count()
    mm: dict = {i: f"{max(1, int(vram_free_gb(i) - headroom_gb))}GiB"
                for i in range(n_gpu)}
    if mode == "cpu_offload":
        mm["cpu"] = f"{max(1, int(cpu_ram_free_gb() * 0.8))}GiB"
    return mm


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def _kwargs(mode: str, dtype: str = "bfloat16",
            headroom_gb: float = VRAM_HEADROOM_GB) -> dict:
    kw = dict(
        # bfloat16 by default, NOT float16. Same memory, far wider exponent
        # range: several models (Qwen2.5-1.5B here) produce attention values
        # that overflow fp16 to inf, and spectral_trust then dies with
        # "array must not contain infs or NaNs" on EVERY item — a whole model
        # silently contributing no spectral data. Never quantized either way.
        dtype=DTYPES[dtype],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # sdpa would not return attentions
    )
    if mode == "single_gpu":
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = "auto"
        kw["max_memory"] = _max_memory(mode, headroom_gb)
        if mode == "cpu_offload":
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
            kw["offload_folder"] = OFFLOAD_DIR
            kw["offload_state_dict"] = True
    return kw


# ── Loading ───────────────────────────────────────────────────────────────────
def load_model_safe(model_name: str, allow_cpu_offload: bool = True,
                    force: str | None = None, dtype: str = "bfloat16",
                    headroom_gb: float | None = None):
    """Load a model at full precision (bf16 by default), no quantization.

    `headroom_gb` is the VRAM to leave free for activations. Stage 12 must
    raise it well above the default because retained attention matrices are
    quadratic in prompt length (see attention_memory_gb).

    Returns (model, tokenizer, info). Raises RuntimeError if impossible,
    AFTER releasing VRAM.
    """
    short = model_name.split("/")[-1]
    plan = plan_loading(model_name, allow_cpu_offload, headroom_gb)
    head = plan["headroom_gb"]
    if force:
        plan["mode"] = force

    print(f"  [plan] {short}: weights ~{plan['need_gb']:.0f} GB + "
          f"{head:.0f} GB headroom | GPU0 {plan['gpu0_gb']:.0f} GB free"
          + (f", {plan['n_gpu']} GPUs = {plan['gpu_all_gb']:.0f} GB"
             if plan["n_gpu"] > 1 else "")
          + f", RAM {plan['ram_gb']:.0f} GB -> {plan['mode']}")

    if plan["mode"] == "impossible":
        raise RuntimeError(
            f"{short} needs ~{plan['need_gb']:.0f} GB in float16; "
            f"{plan['gpu_all_gb']:.0f} GB VRAM and {plan['ram_gb']:.0f} GB RAM "
            f"available. Without quantization this model cannot run here — "
            f"drop it from the panel or use a bigger pod.")

    if plan["mode"] == "cpu_offload":
        print(f"  [warn] {short}: some layers will stay in CPU RAM. Each "
              f"forward transfers weights — time 3 items before committing "
              f"to the full campaign.")

    tok_kwargs = {}
    if "mistral" in model_name.lower() or "ministral" in model_name.lower():
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"        # preserve the end of the prompt

    # Progressive fallback, never degrading precision.
    chain = {"single_gpu": ["single_gpu", "multi_gpu", "cpu_offload"],
             "multi_gpu": ["multi_gpu", "cpu_offload"],
             "cpu_offload": ["cpu_offload"]}[plan["mode"]]
    if not allow_cpu_offload:
        chain = [m for m in chain if m != "cpu_offload"]
    if torch.cuda.device_count() < 2:
        chain = [m for m in chain if m != "multi_gpu"]

    free_vram()
    last_err = None
    for mode in chain:
        model = None
        try:
            before = vram_total_free_gb()
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **_kwargs(mode, dtype, head))
            model.eval()
            devices = set(str(d) for d in getattr(model, "hf_device_map", {}).values())
            print(f"  [ok] {short} loaded {dtype}/{mode} "
                  f"({before - vram_total_free_gb():.1f} GB VRAM used"
                  + (f", devices: {sorted(devices)}" if devices else "") + ")")
            return model, tokenizer, {"mode": mode, "dtype": dtype,
                                      "need_gb": plan["need_gb"]}
        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, OSError) as e:
            last_err = e
            # CRUCIAL: an aborted load leaves weights in VRAM. Without this
            # cleanup the next attempt fails too.
            _hard_unload(model)
            print(f"  [warn] failed in {mode} ({type(e).__name__}) — VRAM freed, "
                  f"{vram_total_free_gb():.1f} GB available")

    free_vram()
    raise RuntimeError(f"{short}: loading impossible. Last error: {last_err}")


def _hard_unload(model) -> None:
    """Tear down a model, including a partially loaded or sharded one."""
    if model is None:
        free_vram()
        return
    try:
        # detach parameters device by device before dropping the object
        for p in model.parameters(recurse=True):
            if p.data.is_cuda:
                p.data = torch.empty(0, dtype=p.dtype, device="cpu")
            if p.grad is not None:
                p.grad = None
    except Exception:
        pass
    try:
        model.to("meta")
    except Exception:
        pass
    free_vram(model)


def unload(model) -> None:
    """Call between two models of a campaign."""
    _hard_unload(model)
    if os.path.isdir(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
