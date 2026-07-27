# ══════════════════════════════════════════════════════════════════════════════
#  VOLET A-JUGE SPECTRAL — PHASE 6 (GSP & interprétabilité mécaniste)
#
#  Juge par log-vraisemblance (Yes/No) + diagnostics spectraux couche par
#  couche sur le graphe d'attention, via `spectral-trust`.
#
#  ── CHAÎNE D'EXÉCUTION ───────────────────────────────────────────────────
#     1. logprob.py                 → mmlu_logprob_results_k{N}.json  (solver)
#     2. volet_a_judge.py           → judge_bank_logprob.json         (la banque)
#     3. volet_a_judge_spectral.py    ← CE SCRIPT
#     4. analyse_cv.py              → tableau M0 / M1 / M2
#
#  Ce script NE CONSTRUIT PAS sa banque : il lit celle exportée à l'étape 2.
#  C'est ce qui garantit que les deux scripts juges portent sur exactement
#  les mêmes items — sinon leurs chiffres ne sont pas comparables.
#
#  ── API réelle de spectral-trust 0.2.x ───────────────────────────────────
#     GSPDiagnosticsFramework(config)  ← le context manager
#     framework.analyze_text(text, save_results=False)
#         → {"layer_diagnostics": [SpectralDiagnostics, ...], ...}
#     SpectralAnalyzer(config) est un composant bas niveau, PAS un CM.
#
#  Usage :  python volet_a_judge_spectral.py
# ══════════════════════════════════════════════════════════════════════════════

import os

# Coupe les barres tqdm « Analyzing layers » (une par item, illisible en log).
os.environ.setdefault("TQDM_DISABLE", "1")

import json, time, traceback, torch
import numpy as np
from collections import defaultdict

# Chargement robuste : préflight VRAM, repli multi-GPU puis déport CPU,
# nettoyage réel après un chargement avorté. Aucune quantification.
from model_loading import (load_model_safe, unload, free_vram,
                           vram_free_gb, estimate_params_billions)

try:
    from spectral_trust import GSPDiagnosticsFramework, GSPConfig
    HAS_SPECTRAL_TRUST = True
    print("✅ `spectral-trust` chargée")
except ImportError:
    HAS_SPECTRAL_TRUST = False
    print("⚠️  `spectral-trust` absente — le run continuera SANS profil spectral.")

# ── Login Hugging Face ────────────────────────────────────────────────────────
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("✅ Login Hugging Face OK")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DEV_FILE  = "mmlu_dev_examples.json"          # exemples few-shot (inutile si k=0)
BANK_FILE = "judge_bank_logprob.json"         # ← produit par volet_a_judge.py

NTRAIN = 0        # zero-shot : aligné sur logprob.py et volet_a_judge.py

# ⚠️ Fenêtre d'analyse spectrale, en tokens.
#    L'eigendécomposition est en O(N³) PAR COUCHE ET PAR ITEM.
#      512  → ~0.4 s/couche   |   1024 → ~3 s/couche  (×8)
#    Le prompt est plafonné à cette valeur : plus de troncature silencieuse.
SPECTRAL_MAX_LEN = 1024

# Chronomètre : s'arrête après N items pour estimer la durée totale.
# Mets None pour lancer la campagne complète.
DRY_RUN_ITEMS = None      # ex. 5 pour un test de vitesse

VERDICTS = ["Yes", "No"]
LETTERS  = ["A", "B", "C", "D"]

JUDGE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
]

# Modèles à ne PAS charger avec déport CPU : le déport fonctionne, mais chaque
# forward retransfère des poids, ce qui rend l'analyse spectrale inexploitable
# en temps. Ils sont ignorés proprement s'ils ne tiennent pas en VRAM.
NO_CPU_OFFLOAD = {"google/gemma-2-27b-it"}

RESULTS_FILE = f"mmlu_judge_spectral_results_k{NTRAIN}_w{SPECTRAL_MAX_LEN}.json"

print(f"✅ Device {DEVICE} | NTRAIN={NTRAIN} | fenêtre={SPECTRAL_MAX_LEN} tokens")
print(f"✅ Sortie : {RESULTS_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
#  BANQUE D'ITEMS (lue, jamais reconstruite)
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(BANK_FILE):
    raise SystemExit(
        f"❌ {BANK_FILE} introuvable.\n"
        f"   Lance d'abord volet_a_judge.py : c'est lui qui construit et exporte\n"
        f"   la banque. Les deux scripts juges doivent porter sur les mêmes items."
    )

with open(BANK_FILE, encoding="utf-8") as f:
    judge_items = json.load(f)

n_pairs = len(judge_items) // 2
print(f"✅ {len(judge_items)} items ({n_pairs} paires) chargés depuis {BANK_FILE}")

src = defaultdict(int)
for it in judge_items:
    src[it.get("neg_source", "inconnue")] += 1
if len(src) > 1:
    print("   provenance des distracteurs : "
          + ", ".join(f"{k}={v // 2}" for k, v in sorted(src.items())))

if n_pairs < 100:
    print(f"   ⚠️  {n_pairs} paires seulement : puissance statistique faible. "
          f"Vérifie que volet_a_judge.py tourne en ITEM_SOURCE='logprob' et "
          f"que le solver couvre tout le panel.")

# ── Exemples few-shot (chargés seulement si NTRAIN > 0) ───────────────────────
dev_by_subject = defaultdict(list)
if NTRAIN > 0:
    if os.path.exists(DEV_FILE):
        with open(DEV_FILE, encoding="utf-8") as f:
            dev_raw = json.load(f)
    else:
        from datasets import load_dataset
        dev_ds = load_dataset("cais/mmlu", "all", split="dev")
        dev_raw = [{"subject": r["subject"], "question": r["question"],
                    "choices": r["choices"], "answer": r["answer"]} for r in dev_ds]
        with open(DEV_FILE, "w", encoding="utf-8") as f:
            json.dump(dev_raw, f, ensure_ascii=False, indent=2)
    for r in dev_raw:
        dev_by_subject[r["subject"]].append(r)

# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT JUGE  (style hendrycks, pas de chat_template)
# ══════════════════════════════════════════════════════════════════════════════
def format_subject(subject: str) -> str:
    return " " + subject.replace("_", " ")


def format_judge_example(question, choices, proposed_letter, verdict=None) -> str:
    lines = [question] + [f"{l}. {c}" for l, c in zip(LETTERS, choices)]
    lines.append(f"Proposed answer: {proposed_letter}. "
                 f"{choices[LETTERS.index(proposed_letter)]}")
    prompt = "\n".join(lines) + "\nIs the proposed answer correct? Answer:"
    if verdict is not None:
        prompt += f" {verdict}\n\n"
    return prompt


def gen_judge_prompt(dev_examples, subject, k) -> str:
    header = ("The following are multiple choice questions about"
              f"{format_subject(subject)}, each followed by a proposed answer. "
              "Decide whether the proposed answer is correct (Yes or No).\n\n")
    if k == 0:
        return header
    import random
    body = ""
    for i, ex in enumerate(dev_examples[:k]):
        gt = LETTERS[ex["answer"]]
        if i % 2 == 0:
            proposed, verdict = gt, "Yes"
        else:
            rng = random.Random(f"42-dev-{subject}-{i}")
            proposed = rng.choice([l for l in LETTERS if l != gt])
            verdict = "No"
        body += format_judge_example(ex["question"], ex["choices"], proposed, verdict)
    return header + body


def fit_judge_prompt(tokenizer, dev_examples, subject, k_max, item, max_len):
    k = k_max
    prompt = None
    while k >= 0:
        prompt = (gen_judge_prompt(dev_examples, subject, k)
                  + format_judge_example(item["question"], item["choices"],
                                         item["proposed_letter"]))
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= max_len or k == 0:
            return prompt, k
        k -= 1
    return prompt, 0


def resolve_verdict_token_ids(tokenizer, sample_context: str) -> dict:
    """Résolution par diff d'encodage sur un contexte réel, + garde anti-collision."""
    base_ids = tokenizer(sample_context, add_special_tokens=False)["input_ids"]
    ids = {}
    for verdict in VERDICTS:
        full_ids = tokenizer(sample_context + " " + verdict,
                             add_special_tokens=False)["input_ids"]
        new_ids = full_ids[len(base_ids):]
        if not new_ids:
            raise ValueError(f"aucun nouveau token pour '{verdict}'")
        ids[verdict] = new_ids[0]
    if len(set(ids.values())) < len(VERDICTS):
        raise ValueError(f"collision de tokens Yes/No : {ids}")
    return ids

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION GSP
# ══════════════════════════════════════════════════════════════════════════════
def build_gsp_config(model_name: str):
    """Les défauts de GSPConfig sont inadaptés à un run par lots.

      torch_dtype "float32" → doublerait l'empreinte mémoire
      max_length 512        → troncature silencieuse du prompt
      save_plots True       → une figure matplotlib PAR ITEM
      normalization "rw"    → laplacien non symétrique → eig() dense général
      verbose True          → un log par couche × par item
    """
    return GSPConfig(
        model_name        = model_name,
        device            = DEVICE,
        torch_dtype       = "float16",
        trust_remote_code = True,
        max_length        = SPECTRAL_MAX_LEN,
        normalization     = "sym",     # laplacien symétrique → solveur symétrique
        eigen_solver      = "dense",   # à N ≤ 1024, eigh dense bat ARPACK 'SM'
        hfer_cutoff_ratio = 0.1,
        save_plots        = False,
        display_plots     = False,
        save_intermediate = False,
        verbose           = False,
        output_dir        = f"./gsp_out/{model_name.split('/')[-1]}",
    )


# Attributs réels de la dataclass SpectralDiagnostics
METRIC_ATTRS = {
    "fiedler":          "fiedler_value",
    "hfer":             "hfer",
    "smoothness":       "smoothness_index",
    "spectral_entropy": "spectral_entropy",
    "energy":           "energy",
    "connectivity":     "connectivity",
}


def extract_spectral(analysis: dict) -> list:
    """Extrait les scalaires par couche.

    `analysis` contient aussi 'model_outputs' (attentions + hidden states
    complets) qu'il ne faut surtout pas sérialiser. Et les éléments de
    'layer_diagnostics' sont des objets, pas des dicts : json.dump planterait.
    """
    rows = []
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
        rows.append(row)
    return rows

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def score_item_spectral(framework, model, tokenizer, prompt, verdict_ids):
    # (a) verdict par log-vraisemblance
    #     output_attentions=False explicite : spectral-trust force
    #     config.output_attentions=True, ce qui ferait allouer les attentions
    #     pour rien sur ce forward-là.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_attentions=False,
                    output_hidden_states=False, use_cache=False)
    logprobs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
    verdict_lp = {v: logprobs[tid].item() for v, tid in verdict_ids.items()}
    pred = max(verdict_lp, key=verdict_lp.get)
    del outputs, logprobs, inputs

    # (b) profil spectral — forward instrumenté du framework
    spectral = None
    if framework is not None:
        try:
            analysis = framework.analyze_text(prompt, save_results=False)
            spectral = extract_spectral(analysis)
            del analysis
        except Exception as e:
            # Le verdict survit à un échec spectral : on ne perd pas l'item.
            spectral = {"error": f"{type(e).__name__}: {e}"}

    return pred, verdict_lp, spectral

# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLE D'ITEMS
# ══════════════════════════════════════════════════════════════════════════════
def _iterate_items(framework, model, tokenizer, todo, on_result):
    probe = todo[0]
    probe_prompt = (gen_judge_prompt(dev_by_subject.get(probe["subject"], []),
                                     probe["subject"], 0)
                    + format_judge_example(probe["question"], probe["choices"],
                                           probe["proposed_letter"]))
    verdict_ids = resolve_verdict_token_ids(tokenizer, probe_prompt)

    model_max = min(getattr(tokenizer, "model_max_length", 4096) or 4096, 4096)
    max_len = min(model_max, SPECTRAL_MAX_LEN) - 8
    print(f"  ℹ️  prompt plafonné à {max_len} tokens | verdict_ids={verdict_ids}")

    limit = DRY_RUN_ITEMS or len(todo)
    durations = []

    for i, item in enumerate(todo[:limit]):
        try:
            t0 = time.time()
            prompt, k_used = fit_judge_prompt(
                tokenizer, dev_by_subject.get(item["subject"], []),
                item["subject"], NTRAIN, item, max_len)

            n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if n_tok > max_len:
                print(f"  ⚠️  {item['item_id']} : {n_tok} tokens > {max_len} — "
                      f"graphe tronqué")

            pred, verdict_lp, spectral = score_item_spectral(
                framework, model, tokenizer, prompt, verdict_ids)
            correct = pred == item["gt_verdict"]
            dt = time.time() - t0
            durations.append(dt)

            print(f"  [{i+1:4d}/{limit}] {item['subject']:<28} "
                  f"{item['item_id'][-4:]:<4} GT={item['gt_verdict']:<3} "
                  f"Pred={pred:<3} k={k_used} {n_tok:>4}tok {dt:5.1f}s "
                  f"{'✅' if correct else '❌'}")

            on_result({
                "item_id":          item["item_id"],
                "question_id":      item["question_id"],
                "subject":          item["subject"],
                "proposed_letter":  item["proposed_letter"],
                "gt_verdict":       item["gt_verdict"],
                "pred_verdict":     pred,
                "is_correct":       correct,
                "verdict_logprobs": verdict_lp,
                "margin_yes_no":    verdict_lp["Yes"] - verdict_lp["No"],
                "k_shot_used":      k_used,
                "n_tokens_prompt":  n_tok,
                "neg_source":       item.get("neg_source"),
                "spectral":         spectral,
            })

            if (i + 1) % 20 == 0:
                free_vram()

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  OOM — {item['item_id']} ignoré")
            free_vram()
        except Exception as e:
            print(f"  ❌ {item['item_id']} : {type(e).__name__}: {e}")
            traceback.print_exc()

    if durations:
        med = float(np.median(durations))
        print(f"  ⏱  médiane {med:.1f} s/item → "
              f"{med * len(todo) / 60:.0f} min pour les {len(todo)} items")
        if DRY_RUN_ITEMS:
            print(f"  🛑 DRY_RUN_ITEMS={DRY_RUN_ITEMS} — mets-le à None pour "
                  f"lancer la campagne complète.")

# ══════════════════════════════════════════════════════════════════════════════
#  RUN PAR MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
def run_model(model_name: str, todo: list, on_result) -> None:
    short = model_name.split("/")[-1]
    print(f"\n{'#'*72}\n  VOLET A-JUGE SPECTRAL : {short}  ({len(todo)} items)\n{'#'*72}")

    model = tokenizer = framework = None
    try:
        # C'est NOUS qui chargeons, pas la librairie : ça donne le contrôle sur
        # le placement, l'attention eager et le nettoyage en cas d'OOM.
        model, tokenizer, info = load_model_safe(
            model_name, allow_cpu_offload=(model_name not in NO_CPU_OFFLOAD))

        def _wrap(res):
            res["model"] = model_name
            res["load_mode"] = info["mode"]   # single_gpu / multi_gpu / cpu_offload
            on_result(res)

        if HAS_SPECTRAL_TRUST:
            framework = GSPDiagnosticsFramework(build_gsp_config(model_name))
            framework.__enter__()
            # On injecte notre modèle au lieu d'appeler instrumenter.load_model(),
            # qui en chargerait un second exemplaire.
            framework.instrumenter.model = model
            framework.instrumenter.tokenizer = tokenizer
            for attr, val in (("device", model.device), ("model_name", model_name)):
                if hasattr(framework.instrumenter, attr):
                    try:
                        setattr(framework.instrumenter, attr, val)
                    except Exception:
                        pass

        _iterate_items(framework, model, tokenizer, todo, _wrap)

    except RuntimeError as e:
        # Modèle trop gros : on le signale, la campagne continue.
        print(f"  ⛔ {short} ignoré — {e}")
    finally:
        if framework is not None:
            try:
                framework.__exit__(None, None, None)
            except Exception:
                pass
            try:
                framework.instrumenter.model = None
            except Exception:
                pass
        unload(model)
        del tokenizer
        free_vram()
        print(f"  🧹 VRAM après déchargement : {vram_free_gb():.1f} Go libres")

# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
results, done = [], set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    done = {(r["model"], r["item_id"]) for r in results}
    print(f"\n♻️  Reprise : {len(done)} paires (modèle, item) déjà faites")


def save_result(r: dict):
    """Écriture atomique : un Ctrl-C pendant json.dump tronquerait le fichier."""
    results.append(r)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, RESULTS_FILE)


# Du plus petit au plus grand : si le plus gros échoue, les autres sont faits.
try:
    JUDGE_MODELS = sorted(JUDGE_MODELS, key=estimate_params_billions)
    print("\nOrdre de passage :")
    for m in JUDGE_MODELS:
        print(f"   {estimate_params_billions(m):5.1f} G params  {m}")
except Exception:
    pass

for model_name in JUDGE_MODELS:
    todo = [it for it in judge_items if (model_name, it["item_id"]) not in done]
    if not todo:
        print(f"  ⏩ {model_name.split('/')[-1]} — déjà complet")
        continue
    try:
        run_model(model_name, todo, on_result=save_result)
    except Exception as e:
        print(f"  ❌ {model_name} : {type(e).__name__}: {e}")
        traceback.print_exc()

# ── Récapitulatif ─────────────────────────────────────────────────────────────
by_model = defaultdict(list)
for r in results:
    by_model[r["model"]].append(r)

if by_model:
    print(f"\n{'#'*72}\n  RÉCAPITULATIF PHASE 6\n{'#'*72}")
    print(f"  {'Modèle':<34} {'Acc':>7} {'TPR':>7} {'TNR':>7} {'Yes%':>7} "
          f"{'spectral':>10} {'N':>6}")
    print(f"  {'-'*34} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*6}")
    for m, rs in sorted(by_model.items()):
        pos = [r["is_correct"] for r in rs if r["gt_verdict"] == "Yes"]
        neg = [r["is_correct"] for r in rs if r["gt_verdict"] == "No"]
        acc = sum(r["is_correct"] for r in rs) / len(rs) * 100
        yes = sum(r["pred_verdict"] == "Yes" for r in rs) / len(rs) * 100
        ok_spec = sum(isinstance(r["spectral"], list) for r in rs)
        print(f"  {m.split('/')[-1]:<34} {acc:>6.1f}% "
              f"{(sum(pos)/len(pos)*100 if pos else 0):>6.1f}% "
              f"{(sum(neg)/len(neg)*100 if neg else 0):>6.1f}% "
              f"{yes:>6.1f}% {ok_spec:>6}/{len(rs):<3} {len(rs):>6}")
    print("   TPR = acc sur items GT=Yes | TNR = acc sur items GT=No")
    print("   Yes% ≈ 50 attendu — un écart fort signale un biais de verdict")

print(f"\n✅ {len(results)} résultats dans `{RESULTS_FILE}`")
print(f"   Étape suivante :  python analyse_cv.py {RESULTS_FILE}")