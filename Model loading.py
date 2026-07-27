# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT ROBUSTE — PLEINE PRÉCISION UNIQUEMENT (aucune quantification)
#
#  Rappel du problème : gemma-2-27b-it pèse ~57 Go en float16. Sur une carte
#  de 44 Go, il ne rentre pas. Aucun nettoyage mémoire ne change ça — c'est de
#  l'arithmétique, pas de la fragmentation.
#
#  Stratégies essayées dans l'ordre, toutes en float16 :
#    1. un seul GPU, si le modèle y tient ;
#    2. tous les GPU disponibles, si le pod en a plusieurs ;
#    3. GPU + déport d'une partie des couches en RAM CPU (lent mais exact) ;
#    4. sinon : erreur explicite, et la campagne continue sans ce modèle.
#
#  Le nettoyage après un chargement avorté est traité sérieusement : sans lui,
#  les poids déjà transférés restent en VRAM et condamnent la tentative
#  suivante.
# ══════════════════════════════════════════════════════════════════════════════

import os, gc, shutil, torch

# Limite la fragmentation. Doit être posé AVANT le premier appel CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Marge laissée libre sur chaque GPU pour les activations, les matrices
# d'attention (output_attentions=True est gourmand) et le cache.
VRAM_HEADROOM_GB = 8.0

# Dossier de déport disque, utilisé seulement si la RAM CPU ne suffit pas.
OFFLOAD_DIR = "./offload"


# ── Mémoire ───────────────────────────────────────────────────────────────────
def free_vram(*objs):
    """Libère réellement la VRAM, y compris après un chargement avorté."""
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
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return 0.0


# ── Estimation de taille, sans télécharger les poids ──────────────────────────
def estimate_params_billions(model_name: str) -> float:
    """Estime le nombre de paramètres depuis la config seule.

      embeddings  ~ vocab x hidden  (x2 si les poids ne sont pas liés)
      par couche  ~ 4 x hidden^2                (attention, majorant si GQA)
                  + 3 x hidden x intermediate   (MLP à porte)
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


# ── Plan de chargement ────────────────────────────────────────────────────────
def plan_loading(model_name: str, allow_cpu_offload: bool = True) -> dict:
    need = _fp16_gb(model_name)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu0 = vram_free_gb(0)
    gpu_all = vram_total_free_gb()
    ram = cpu_ram_free_gb()

    if need == 0.0:
        mode = "single_gpu"                       # estimation impossible : on tente
    elif need + VRAM_HEADROOM_GB <= gpu0:
        mode = "single_gpu"
    elif n_gpu > 1 and need + VRAM_HEADROOM_GB * n_gpu <= gpu_all:
        mode = "multi_gpu"
    elif allow_cpu_offload and need + VRAM_HEADROOM_GB <= gpu_all + ram:
        mode = "cpu_offload"
    else:
        mode = "impossible"

    return {"model": model_name, "need_gb": need, "n_gpu": n_gpu,
            "gpu0_gb": gpu0, "gpu_all_gb": gpu_all, "ram_gb": ram, "mode": mode}


def _max_memory(mode: str) -> dict | None:
    """Plafond mémoire par device, avec la marge de sécurité déduite."""
    if mode == "single_gpu":
        return None
    n_gpu = torch.cuda.device_count()
    mm = {i: f"{max(1, int(vram_free_gb(i) - VRAM_HEADROOM_GB))}GiB"
          for i in range(n_gpu)}
    if mode == "cpu_offload":
        mm["cpu"] = f"{max(1, int(cpu_ram_free_gb() * 0.8))}GiB"
    return mm


def _kwargs(mode: str) -> dict:
    kw = dict(
        dtype=torch.float16,          # pleine précision fp16, jamais quantifié
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # eager pour TOUS les modèles : sdpa ne renvoie pas les matrices
        # d'attention dont l'analyse spectrale a besoin.
        attn_implementation="eager",
    )
    if mode == "single_gpu":
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = "auto"
        kw["max_memory"] = _max_memory(mode)
        if mode == "cpu_offload":
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
            kw["offload_folder"] = OFFLOAD_DIR
            kw["offload_state_dict"] = True
    return kw


# ── Chargement ────────────────────────────────────────────────────────────────
def load_model_safe(model_name: str, allow_cpu_offload: bool = True,
                    force: str | None = None):
    """Charge un modèle en float16, sans quantification.

    Retourne (model, tokenizer, info). Lève RuntimeError si impossible,
    APRÈS avoir libéré la VRAM.
    """
    short = model_name.split("/")[-1]
    plan = plan_loading(model_name, allow_cpu_offload)
    if force:
        plan["mode"] = force

    print(f"  📋 {short} : fp16 ≈ {plan['need_gb']:.0f} Go | "
          f"GPU0 {plan['gpu0_gb']:.0f} Go libres"
          + (f", {plan['n_gpu']} GPU = {plan['gpu_all_gb']:.0f} Go"
             if plan["n_gpu"] > 1 else "")
          + f", RAM {plan['ram_gb']:.0f} Go → {plan['mode']}")

    if plan["mode"] == "impossible":
        raise RuntimeError(
            f"{short} demande ~{plan['need_gb']:.0f} Go en float16 ; "
            f"{plan['gpu_all_gb']:.0f} Go de VRAM et {plan['ram_gb']:.0f} Go de RAM "
            f"disponibles. Sans quantification, ce modèle ne peut pas tourner ici — "
            f"retire-le du panel ou prends un pod plus grand.")

    if plan["mode"] == "cpu_offload":
        print(f"  ⏳ {short} : une partie des couches restera en RAM CPU. "
              f"Le calcul sera nettement plus lent (chaque forward transfère "
              f"des poids). Chronomètre sur 3 items avant de lancer la campagne.")

    tok_kwargs = {}
    if "mistral" in model_name.lower() or "ministral" in model_name.lower():
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"       # préserve la fin du prompt

    # Repli progressif, sans jamais dégrader la précision.
    chain = {"single_gpu": ["single_gpu", "multi_gpu", "cpu_offload"],
             "multi_gpu":  ["multi_gpu", "cpu_offload"],
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
            model = AutoModelForCausalLM.from_pretrained(model_name, **_kwargs(mode))
            model.eval()
            devices = set(str(d) for d in getattr(model, "hf_device_map", {}).values())
            print(f"  ✅ {short} chargé en float16 / {mode} "
                  f"({before - vram_total_free_gb():.1f} Go de VRAM utilisés"
                  + (f", devices: {sorted(devices)}" if devices else "") + ")")
            return model, tokenizer, {"mode": mode, "dtype": "float16",
                                      "need_gb": plan["need_gb"]}
        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, OSError) as e:
            last_err = e
            # CRUCIAL : un chargement avorté laisse des poids en VRAM.
            # Sans ce nettoyage, la tentative suivante échoue aussi.
            _hard_unload(model)
            print(f"  ⚠️  échec en {mode} ({type(e).__name__}) — VRAM libérée, "
                  f"{vram_total_free_gb():.1f} Go disponibles")

    free_vram()
    raise RuntimeError(f"{short} : chargement impossible. Dernière erreur : {last_err}")


def _hard_unload(model):
    """Démonte un modèle, y compris partiellement chargé ou réparti."""
    if model is None:
        free_vram()
        return
    try:
        # détache les paramètres device par device avant de lâcher l'objet
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


def unload(model):
    """À appeler entre deux modèles de la campagne."""
    _hard_unload(model)
    if os.path.isdir(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)