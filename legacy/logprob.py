# ══════════════════════════════════════════════════════════════════════════════
#  VOLET A — SCORING LOG-VRAISEMBLANCE (méthode canonique MMLU / hendrycks/test)
#
#  Reproduit fidèlement le prompt et le calcul du script officiel eval.py :
#  en-tête sujet + k exemples few-shot + question, prompt terminé par
#  "\nAnswer:" SANS génération de texte. Un seul forward pass par question,
#  lecture des logits sur les 4 lettres A/B/C/D à la position suivant
#  immédiatement "Answer:", argmax.
#
#  Différences volontaires avec Volet B (solver_mmlu.py) :
#    - PAS de chat_template : prompt brut, comme le script officiel (GPT-3
#      n'était pas un modèle "instruct"), appliqué uniformément à tous les
#      modèles du panel, instruct ou base.
#    - PAS de génération : un seul forward pass, donc aucun des artefacts de
#      format diagnostiqués sur Volet B (réponses vides, dérive hors-sujet,
#      budget de tokens insuffisant) ne peut se produire ici.
#
#  NTRAIN=5 par défaut pour matcher le Tableau 1 du papier (5-shot). Mets
#  NTRAIN=0 pour comparer en zero-shot — attention, ça casse la comparabilité
#  directe avec les chiffres publiés du papier.
#
#  Utilise volontairement le MÊME mmlu_questions.json que solver_mmlu.py :
#  c'est ce qui permet de comparer Volet A et Volet B sur exactement les
#  mêmes questions.
#
#  Deux accuracies calculées en parallèle par modèle : weighted (comme le
#  script officiel, pondérée par le nb de questions/matière dans l'échantillon)
#  et unweighted (une matière = une voix — ta métrique principale).
#
#  Usage :  python volet_a_logprob.py
# ══════════════════════════════════════════════════════════════════════════════

import os, json, gc, torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Login Hugging Face ────────────────────────────────────────────────────────
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("✅ Login Hugging Face OK")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
QUESTIONS_FILE = "mmlu_questions.json"        # RÉUTILISE le fichier de Volet B
DEV_FILE       = "mmlu_dev_examples.json"     # exemples few-shot (split "dev")
RESULTS_FILE   = "mmlu_logprob_results.json"

NTRAIN  = 0   # k-shot — 5 = valeur du papier (Table 1). 0 = zero-shot.
LETTERS = ["A", "B", "C", "D"]

JUDGE_MODELS = [
    # TODO : remplace par le panel de modèles plus gros que tu veux tester.
    # Les deux ci-dessous sont juste là pour vérifier que le script tourne.
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-Nemo-Instruct-2407"
]

print(f"✅ Device  : {DEVICE}")
print(f"✅ Modèles : {len(JUDGE_MODELS)}  |  NTRAIN={NTRAIN}")

# ── Questions cibles : réutilise mmlu_questions.json (même échantillon que Volet B)
if not os.path.exists(QUESTIONS_FILE):
    raise SystemExit(
        f"❌ {QUESTIONS_FILE} introuvable — lance d'abord solver_mmlu.py (ou copie "
        f"son mmlu_questions.json ici) pour garder le même échantillon de questions "
        f"entre Volet A et Volet B."
    )
with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)
print(f"✅ {QUESTIONS_FILE} chargé ({len(questions)} questions)")

# ── Exemples few-shot (split "dev" de MMLU, 5 par matière dans le jeu complet)
if os.path.exists(DEV_FILE):
    with open(DEV_FILE, encoding="utf-8") as f:
        dev_raw = json.load(f)
    print(f"✅ {DEV_FILE} chargé ({len(dev_raw)} exemples few-shot)")
else:
    print("Chargement du split dev MMLU (few-shot)...")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")
    dev_raw = [
        {"subject": r["subject"], "question": r["question"],
         "choices": r["choices"], "answer": r["answer"]}
        for r in dev_ds
    ]
    with open(DEV_FILE, "w", encoding="utf-8") as f:
        json.dump(dev_raw, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(dev_raw)} exemples few-shot sauvegardés dans {DEV_FILE}")

dev_by_subject = defaultdict(list)
for r in dev_raw:
    dev_by_subject[r["subject"]].append(r)

# ── Construction du prompt ─────────────────────────────────────────────────────
# Fidèle à format_subject/format_example/gen_prompt du script officiel — y
# compris le double espace après "about", hérité tel quel du script d'origine
# (format_subject y ajoute déjà un espace initial ; ce n'est pas un bug ici).
def format_subject(subject: str) -> str:
    return " " + subject.replace("_", " ")

def format_example(question: str, choices: list, letter: str = None) -> str:
    lines = [question] + [f"{l}. {c}" for l, c in zip(LETTERS, choices)]
    prompt = "\n".join(lines) + "\nAnswer:"
    if letter is not None:
        prompt += f" {letter}\n\n"
    return prompt

def gen_prompt(dev_examples: list, subject: str, k: int) -> str:
    header = ("The following are multiple choice questions (with answers) about "
              f"{format_subject(subject)}.\n\n")
    body = "".join(
        format_example(ex["question"], ex["choices"], LETTERS[ex["answer"]])
        for ex in dev_examples[:k]
    )
    return header + body

def fit_prompt(tokenizer, dev_examples, subject, k_max, question, choices, max_len):
    """Réduit k si le prompt dépasse max_len tokens (équivalent du crop() d'origine)."""
    k = k_max
    while k >= 0:
        prompt = gen_prompt(dev_examples, subject, k) + format_example(question, choices)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= max_len or k == 0:
            return prompt, k
        k -= 1
    return prompt, 0

# ── Résolution des token IDs des 4 lettres, PAR MODÈLE ────────────────────────
# Le piège : " A" ne se découpe pas forcément en un seul token identique selon
# le tokenizer (voir la PR evaluate_hf.py sur hendrycks/test). On mesure par
# diff d'encodage sur un contexte réel plutôt que de supposer un token fixe,
# et on détecte explicitement toute collision entre lettres — une collision
# rend l'argmax single-position non fiable pour ce tokenizer précis, donc le
# modèle est écarté proprement plutôt que de produire un score silencieusement
# faux.
def resolve_letter_token_ids(tokenizer, sample_context: str) -> dict:
    base_ids = tokenizer(sample_context, add_special_tokens=False)["input_ids"]
    ids = {}
    for letter in LETTERS:
        full_ids = tokenizer(sample_context + " " + letter,
                              add_special_tokens=False)["input_ids"]
        new_ids = full_ids[len(base_ids):]
        if not new_ids:
            raise ValueError(f"aucun nouveau token pour '{letter}'")
        ids[letter] = new_ids[0]   # 1er nouveau token = celui lu en position -1
    if len(set(ids.values())) < len(LETTERS):
        raise ValueError(
            f"collision de tokens entre lettres : {ids} — argmax single-position "
            f"non fiable pour ce tokenizer"
        )
    return ids

# ── Chargement modèle ──────────────────────────────────────────────────────────
# Identique à solver_mmlu.py, device_map figé sur cuda:0 pour éviter le split
# multi-GPU qui ralentit tout sans raison sur un pod à plusieurs cartes.
def load_model(model_name: str):
    tok_kwargs = {}
    if "mistral" in model_name.lower() or "ministral" in model_name.lower():
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(dtype=torch.float16, device_map={"": 0}, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except ImportError:
        load_kwargs.pop("trust_remote_code")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer

def _free_vram(*tensors):
    for t in tensors:
        if t is not None:
            del t
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

@torch.no_grad()
def score_question(model, tokenizer, prompt: str, letter_ids: dict):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]                       # next-token logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)          # upcast fp32 pour la stabilité
    letter_lp = {l: logprobs[tid].item() for l, tid in letter_ids.items()}
    pred = max(letter_lp, key=letter_lp.get)
    return pred, letter_lp

# ── Run par modèle ──────────────────────────────────────────────────────────────
def run_model(model_name: str, todo: list, on_result) -> None:
    short = model_name.split("/")[-1]
    print(f"\n{'#'*70}\n  VOLET A : {short}  ({len(todo)} questions)\n{'#'*70}")
    model, tokenizer = None, None
    try:
        model, tokenizer = load_model(model_name)
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  VRAM libre après chargement : {free:.1f} Go")

        probe_q = todo[0]
        probe_prompt = (
            gen_prompt(dev_by_subject.get(probe_q["subject"], []), probe_q["subject"], 0)
            + format_example(probe_q["question"], probe_q["choices"])
        )
        try:
            letter_ids = resolve_letter_token_ids(tokenizer, probe_prompt)
        except ValueError as e:
            print(f"  ⚠️  {short} écarté : {e}")
            return
        print(f"  IDs lettres résolus : {letter_ids}")

        max_len = min(getattr(tokenizer, "model_max_length", 4096) or 4096, 4096) - 8

        for i, q in enumerate(todo):
            try:
                dev_ex = dev_by_subject.get(q["subject"], [])
                prompt, k_used = fit_prompt(tokenizer, dev_ex, q["subject"], NTRAIN,
                                             q["question"], q["choices"], max_len)
                pred, letter_lp = score_question(model, tokenizer, prompt, letter_ids)
                correct = pred == q["gt_letter"]

                mark = "✅" if correct else "❌"
                extra = f"  (k={k_used})" if k_used < NTRAIN else ""
                print(f"  [{i+1:3d}/{len(todo)}] {q['subject']:<30} "
                      f"GT={q['gt_letter']} Pred={pred} {mark}{extra}")

                on_result({
                    "model":           model_name,
                    "question_id":     q["question_id"],
                    "subject":         q["subject"],
                    "gt_letter":       q["gt_letter"],
                    "pred_letter":     pred,
                    "is_correct":      correct,
                    "letter_logprobs": letter_lp,
                    "k_shot_used":     k_used,
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM — {q['question_id']} ignorée")
                _free_vram()
            except Exception as e:
                print(f"  ❌ {q['question_id']} : {e}")
    finally:
        if model is not None:
            model.cpu()
        del model, tokenizer
        _free_vram()

# ── Boucle principale + reprise (même mécanique que solver_mmlu.py) ────────────
results = []
done = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    done = {(r["model"], r["question_id"]) for r in results}
    print(f"\n♻️  Reprise : {len(done)} paires (modèle, question) déjà sauvegardées")

def save_result(r: dict):
    results.append(r)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

for model_name in JUDGE_MODELS:
    short = model_name.split("/")[-1]
    todo = [q for q in questions if (model_name, q["question_id"]) not in done]
    if not todo:
        print(f"  ⏩ {short} — déjà complet")
        continue
    try:
        run_model(model_name, todo, on_result=save_result)
    except Exception as e:
        print(f"  ❌ {short} : {e}")

# ── Récapitulatif : weighted ET unweighted, côte à côte ─────────────────────────
by_model = defaultdict(lambda: defaultdict(list))
for r in results:
    by_model[r["model"]][r["subject"]].append(r["is_correct"])

rows = []
for model, by_subject in by_model.items():
    flat = [c for cors in by_subject.values() for c in cors]
    weighted = sum(flat) / len(flat) * 100 if flat else 0.0
    acc_par_subject = {s: sum(c) / len(c) for s, c in by_subject.items()}
    unweighted = (sum(acc_par_subject.values()) / len(acc_par_subject) * 100
                  if acc_par_subject else 0.0)
    rows.append((model.split("/")[-1], weighted, unweighted, len(flat), len(by_subject)))

rows.sort(key=lambda x: x[2], reverse=True)   # trié par UNWEIGHTED — métrique principale

print(f"\n\n{'#'*78}")
print(f"#  RÉCAPITULATIF VOLET A — {len(questions)} questions, {len(JUDGE_MODELS)} modèles")
print(f"{'#'*78}")
print(f"  {'Modèle':<32} {'Weighted':>10} {'Unweighted':>11} {'N':>6} {'Matières':>9}")
print(f"  {'-'*32} {'-'*10} {'-'*11} {'-'*6} {'-'*9}")
for name, w, u, n, ns in rows:
    print(f"  {name:<32} {w:>9.1f}% {u:>10.1f}% {n:>6} {ns:>9}")

print(f"\n✅ {RESULTS_FILE} — {len(results)} résultats au total")
print("   Weighted   = pondérée par nb de questions/matière (convention du papier)")
print("   Unweighted = une matière = une voix (ta métrique principale)")