# ══════════════════════════════════════════════════════════════════════════════
#  VOLET A-JUGE — ÉVALUATION "JUGE" PAR LOG-VRAISEMBLANCE (Yes/No)
#
#  Symétrique de volet_a_logprob.py : même mécanique (un seul forward pass,
#  lecture des logits à la position suivant "Answer:", argmax), mais appliquée
#  au rôle de JUGE : le modèle voit une question MMLU + une réponse proposée,
#  et on lit les logprobs de " Yes" / " No".
#
#  Design :
#    - Trois modes de construction des items (ITEM_SOURCE) :
#        * "logprob"   → RECOMMANDÉ. neg = mauvaise lettre vers laquelle le
#                        panel penche le plus (moyenne des probabilités
#                        renormalisées sur les 4 lettres). Existe pour TOUTES
#                        les questions → une paire par question.
#        * "bank"      → historique. neg = mauvaise lettre la plus PRÉDITE ;
#                        exige qu'au moins un modèle ait trouvé juste et un
#                        autre faux → ne retient qu'une fraction des questions.
#        * "synthetic" → neg tirée au sort (seedée). Distracteurs souvent
#                        triviaux ; à n'utiliser que sans run solver.
#    - Pour chaque question, 2 items —
#        * item "pos" : réponse proposée = bonne lettre  → verdict attendu Yes
#        * item "neg" : réponse proposée = mauvaise lettre (seed fixe) → No
#      => jeu équilibré 50/50 par construction, identique pour tous les
#      modèles, aucun auto-jugement possible.
#    - Few-shot (NTRAIN exemplaires) construits depuis le split dev, en
#      alternant Yes/No pour ne pas biaiser le prior du verdict.
#    - La marge logprob(Yes) - logprob(No) est sauvegardée par item : c'est
#      un signal interne de confiance, réutilisable en phase spectrale.
#
#  Métriques finales par modèle :
#    - judge accuracy weighted / unweighted (mêmes conventions que Volet A)
#    - TPR (acc sur items corrects) / TNR (acc sur items incorrects) et
#      yes_rate — pour détecter un Yes-bias
#    - corrélation panel solver↔judge (à partir de mmlu_logprob_results.json)
#    - analyse intra-modèle : P(bien juger | solver correct) vs
#      P(bien juger | solver incorrect) — test au niveau question de
#      l'hypothèse good-responder
#
#  Usage :  python volet_a_judge.py   (après volet_a_logprob.py)
# ══════════════════════════════════════════════════════════════════════════════

import os, json, gc, math, random, torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Chargement délégué à model_loading.py : préflight VRAM, repli multi-GPU puis
# déport CPU, et surtout nettoyage réel après un chargement avorté (sans quoi
# les poids déjà transférés bloquent la tentative suivante).
from model_loading import load_model_safe, unload, free_vram, vram_free_gb

# ── Login Hugging Face ────────────────────────────────────────────────────────
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("✅ Login Hugging Face OK")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
QUESTIONS_FILE  = "mmlu_questions.json"          # même échantillon que Volet A
DEV_FILE        = "mmlu_dev_examples.json"       # mêmes exemples few-shot
SOLVER_RESULTS  = "mmlu_logprob_results.json"    # résultats solver (corrélation)
RESULTS_FILE    = None    # défini après NTRAIN (voir plus bas)

SEED     = 42     # tirages aléatoires — NE PAS CHANGER en cours de run
NTRAIN   = 0      # 0 = zero-shot (aligné sur le solver et la Phase 6)

# Source des réponses à juger — voir l'en-tête du fichier.
#   "logprob"   → recommandé : une paire par question (le plus de données)
#   "bank"      → historique : critère restrictif, peu de paires
#   "synthetic" → sans run solver, distracteurs au sort
ITEM_SOURCE = "logprob"

# Fichier de sortie : dépend du protocole, pour ne jamais mélanger deux runs.
RESULTS_FILE = f"mmlu_judge_results_{ITEM_SOURCE}_k{NTRAIN}.json"
BANK_EXPORT  = f"judge_bank_{ITEM_SOURCE}.json"   # banque exportée, relue par la Phase 6
LETTERS  = ["A", "B", "C", "D"]
VERDICTS = ["Yes", "No"]

JUDGE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
]

print(f"✅ Device  : {DEVICE}")
print(f"✅ Modèles : {len(JUDGE_MODELS)}  |  NTRAIN={NTRAIN}  |  SEED={SEED}")

# ── Chargement questions + dev (mêmes fichiers que Volet A) ───────────────────
if not os.path.exists(QUESTIONS_FILE):
    raise SystemExit(f"❌ {QUESTIONS_FILE} introuvable — lance d'abord volet_a_logprob.py")
with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)
print(f"✅ {QUESTIONS_FILE} chargé ({len(questions)} questions)")

if os.path.exists(DEV_FILE):
    with open(DEV_FILE, encoding="utf-8") as f:
        dev_raw = json.load(f)
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
print(f"✅ {len(dev_raw)} exemples few-shot disponibles")

dev_by_subject = defaultdict(list)
for r in dev_raw:
    dev_by_subject[r["subject"]].append(r)

# ── Résultats solver (nécessaires en mode "bank" + corrélation finale) ────────
solver_raw = None
if os.path.exists(SOLVER_RESULTS):
    with open(SOLVER_RESULTS, encoding="utf-8") as f:
        solver_raw = json.load(f)
    print(f"✅ {SOLVER_RESULTS} chargé ({len(solver_raw)} résultats solver)")

# ── Construction des items de jugement (déterministe, seedée) ─────────────────
def _base(q):
    return {"subject": q["subject"], "question": q["question"],
            "choices": q["choices"], "gt_letter": q["gt_letter"],
            "question_id": q["question_id"]}

def make_judge_items_synthetic(questions: list) -> list:
    """pos = bonne lettre ; neg = mauvaise lettre tirée au sort (seedée)."""
    items = []
    for q in questions:
        gt = q["gt_letter"]
        rng = random.Random(f"{SEED}-{q['question_id']}")
        wrong = rng.choice([l for l in LETTERS if l != gt])
        items.append({**_base(q), "item_id": f"{q['question_id']}_pos",
                      "proposed_letter": gt,    "gt_verdict": "Yes"})
        items.append({**_base(q), "item_id": f"{q['question_id']}_neg",
                      "proposed_letter": wrong, "gt_verdict": "No"})
    return items

def _softmax4(lp: dict) -> dict:
    """Renormalise les log-probs sur les 4 lettres → distribution propre.

    Sans cette étape, un modèle très "confiant" (logits de grande amplitude)
    dominerait la moyenne du panel. Après renormalisation, chaque modèle
    apporte une distribution de somme 1 et pèse pareil.
    """
    vals = [lp[l] for l in LETTERS]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    tot = sum(exps)
    return {l: e / tot for l, e in zip(LETTERS, exps)}


def make_judge_items_logprob(questions: list, solver_raw: list) -> list:
    """Items construits depuis les LOG-PROBABILITÉS du panel solver.

    neg = la mauvaise lettre vers laquelle le panel penche le plus, même si
    aucun modèle ne l'a finalement choisie. Cette information existe pour
    TOUTES les questions : on obtient donc une paire par question, au lieu de
    la fraction retenue par make_judge_items_bank.

    Le distracteur reste écologiquement valide : c'est la mauvaise option que
    les modèles trouvent la plus tentante, pas une lettre tirée au sort.
    """
    probs = defaultdict(lambda: defaultdict(list))   # qid -> lettre -> [p]
    preds = defaultdict(list)                        # qid -> [pred_letter]
    n_models = set()
    for r in solver_raw:
        lp = r.get("letter_logprobs")
        if not lp or any(l not in lp for l in LETTERS):
            continue
        p = _softmax4(lp)
        for l in LETTERS:
            probs[r["question_id"]][l].append(p[l])
        preds[r["question_id"]].append(r["pred_letter"])
        n_models.add(r["model"])

    if len(n_models) < 3:
        print(f"  ⚠️  seulement {len(n_models)} modèle(s) dans {SOLVER_RESULTS} : "
              f"le 'panel' n'en est pas un. Les distracteurs refléteront ce seul "
              f"modèle, qui sera ensuite avantagé quand il jugera ses propres "
              f"pièges. Lance le solver sur tout le panel d'abord.")

    items, n_pred, n_lp_only, n_skip = [], 0, 0, 0
    for q in questions:
        qid, gt = q["question_id"], q["gt_letter"]
        if qid not in probs:
            n_skip += 1
            continue
        mean_p = {l: sum(v) / len(v) for l, v in probs[qid].items()}
        wrong = [l for l in LETTERS if l != gt]
        neg_letter = max(wrong, key=lambda l: mean_p[l])

        # Traçabilité : ce distracteur a-t-il été réellement choisi par un
        # modèle (ancien critère) ou vient-il des log-probs seules ?
        was_predicted = neg_letter in preds.get(qid, [])
        n_pred += was_predicted
        n_lp_only += (not was_predicted)

        extra = {"neg_source": "panel_pred" if was_predicted else "panel_logprob",
                 "neg_mean_prob": round(mean_p[neg_letter], 4),
                 "gt_mean_prob":  round(mean_p[gt], 4),
                 "n_models_correct": sum(1 for p in preds.get(qid, []) if p == gt)}

        items.append({**_base(q), **extra, "item_id": f"{qid}_pos",
                      "proposed_letter": gt,         "gt_verdict": "Yes"})
        items.append({**_base(q), **extra, "item_id": f"{qid}_neg",
                      "proposed_letter": neg_letter, "gt_verdict": "No"})

    print(f"  Banque log-prob : {len(items)//2}/{len(questions)} questions retenues "
          f"({n_pred} distracteurs réellement prédits, {n_lp_only} issus des "
          f"log-probs seules"
          + (f", {n_skip} sans log-probs" if n_skip else "") + ")")
    return items


def make_judge_items_bank(questions: list, solver_raw: list) -> list:
    """Items construits depuis les VRAIES prédictions du run solver.
    En QCM une réponse se réduit à sa lettre : le même item est montré à tous
    les juges. Une question n'entre que si ≥1 modèle a prédit la bonne lettre
    ET ≥1 modèle une mauvaise (base commune équilibrée 50/50).
    neg = la mauvaise lettre la plus prédite (départage seedé)."""
    preds = defaultdict(list)                       # question_id -> [pred_letter]
    for r in solver_raw:
        preds[r["question_id"]].append(r["pred_letter"])
    items, drop_all_ok, drop_all_ko, drop_nopred = [], 0, 0, 0
    for q in questions:
        gt = q["gt_letter"]
        p = preds.get(q["question_id"], [])
        if not p:
            drop_nopred += 1
            continue
        wrong = [l for l in p if l != gt and l in LETTERS]
        if gt not in p:
            drop_all_ko += 1                        # personne n'a la bonne lettre
            continue
        if not wrong:
            drop_all_ok += 1                        # personne ne s'est trompé
            continue
        counts = defaultdict(int)
        for l in wrong:
            counts[l] += 1
        top = max(counts.values())
        cands = sorted(l for l, c in counts.items() if c == top)
        rng = random.Random(f"{SEED}-{q['question_id']}")
        neg_letter = rng.choice(cands)
        items.append({**_base(q), "item_id": f"{q['question_id']}_pos",
                      "proposed_letter": gt,         "gt_verdict": "Yes"})
        items.append({**_base(q), "item_id": f"{q['question_id']}_neg",
                      "proposed_letter": neg_letter, "gt_verdict": "No"})
    kept = len(items) // 2
    print(f"  Base commune : {kept}/{len(questions)} questions retenues "
          f"(exclues : {drop_all_ok} tout-le-panel-réussit, "
          f"{drop_all_ko} personne-ne-réussit, {drop_nopred} sans prédiction)")
    return items

if ITEM_SOURCE in ("logprob", "bank"):
    if solver_raw is None:
        raise SystemExit(f"❌ ITEM_SOURCE='{ITEM_SOURCE}' mais {SOLVER_RESULTS} "
                         f"introuvable — lance d'abord logprob.py "
                         f"(ou passe en 'synthetic')")
    judge_items = (make_judge_items_logprob(questions, solver_raw)
                   if ITEM_SOURCE == "logprob"
                   else make_judge_items_bank(questions, solver_raw))
else:
    judge_items = make_judge_items_synthetic(questions)

n_q_kept = len(judge_items) // 2
print(f"✅ {len(judge_items)} items de jugement ({n_q_kept} pos + {n_q_kept} neg) "
      f"[source: {ITEM_SOURCE}]")

# Export de la banque : la Phase 6 doit juger EXACTEMENT les mêmes items.
with open(BANK_EXPORT, "w", encoding="utf-8") as f:
    json.dump(judge_items, f, ensure_ascii=False, indent=2)
print(f"✅ banque exportée dans {BANK_EXPORT} (à lire par volet_a_judge_spectral.py)")

# ── Construction du prompt juge ───────────────────────────────────────────────
# Style hendrycks conservé (prompt brut, pas de chat_template, en-tête sujet),
# adapté au rôle de juge. Chaque exemple montre la question, les choix, la
# réponse proposée (lettre + texte du choix) et le verdict.
def format_subject(subject: str) -> str:
    return " " + subject.replace("_", " ")

def format_judge_example(question: str, choices: list, proposed_letter: str,
                          verdict: str = None) -> str:
    lines = [question] + [f"{l}. {c}" for l, c in zip(LETTERS, choices)]
    prop_text = choices[LETTERS.index(proposed_letter)]
    lines.append(f"Proposed answer: {proposed_letter}. {prop_text}")
    prompt = "\n".join(lines) + "\nIs the proposed answer correct? Answer:"
    if verdict is not None:
        prompt += f" {verdict}\n\n"
    return prompt

def gen_judge_prompt(dev_examples: list, subject: str, k: int) -> str:
    header = ("The following are multiple choice questions about"
              f"{format_subject(subject)}, each followed by a proposed answer. "
              "Decide whether the proposed answer is correct (Yes or No).\n\n")
    body = ""
    for i, ex in enumerate(dev_examples[:k]):
        gt = LETTERS[ex["answer"]]
        if i % 2 == 0:                       # exemplaires pairs → verdict Yes
            proposed, verdict = gt, "Yes"
        else:                                # impairs → mauvaise lettre, No
            rng = random.Random(f"{SEED}-dev-{subject}-{i}")
            proposed, verdict = rng.choice([l for l in LETTERS if l != gt]), "No"
        body += format_judge_example(ex["question"], ex["choices"], proposed, verdict)
    return header + body

def fit_judge_prompt(tokenizer, dev_examples, subject, k_max, item, max_len):
    """Réduit k si le prompt dépasse max_len tokens (même logique que Volet A)."""
    k = k_max
    while k >= 0:
        prompt = (gen_judge_prompt(dev_examples, subject, k)
                  + format_judge_example(item["question"], item["choices"],
                                          item["proposed_letter"]))
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= max_len or k == 0:
            return prompt, k
        k -= 1
    return prompt, 0

# ── Résolution des token IDs de " Yes" / " No", PAR MODÈLE ────────────────────
# Même méthode par diff d'encodage que resolve_letter_token_ids de Volet A,
# même garde-fou anti-collision.
def resolve_verdict_token_ids(tokenizer, sample_context: str) -> dict:
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

@torch.no_grad()
def score_item(model, tokenizer, prompt: str, verdict_ids: dict):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    verdict_lp = {v: logprobs[tid].item() for v, tid in verdict_ids.items()}
    pred = max(verdict_lp, key=verdict_lp.get)
    return pred, verdict_lp

# ── Run par modèle ────────────────────────────────────────────────────────────
def run_model(model_name: str, todo: list, on_result) -> None:
    short = model_name.split("/")[-1]
    print(f"\n{'#'*70}\n  VOLET A-JUGE : {short}  ({len(todo)} items)\n{'#'*70}")
    model, tokenizer = None, None
    try:
        model, tokenizer, _info = load_model_safe(model_name)

        probe = todo[0]
        probe_prompt = (
            gen_judge_prompt(dev_by_subject.get(probe["subject"], []),
                             probe["subject"], 0)
            + format_judge_example(probe["question"], probe["choices"],
                                    probe["proposed_letter"])
        )
        try:
            verdict_ids = resolve_verdict_token_ids(tokenizer, probe_prompt)
        except ValueError as e:
            print(f"  ⚠️  {short} écarté : {e}")
            return
        print(f"  IDs verdicts résolus : {verdict_ids}")

        max_len = min(getattr(tokenizer, "model_max_length", 4096) or 4096, 4096) - 8

        for i, item in enumerate(todo):
            try:
                dev_ex = dev_by_subject.get(item["subject"], [])
                prompt, k_used = fit_judge_prompt(tokenizer, dev_ex, item["subject"],
                                                   NTRAIN, item, max_len)
                pred, verdict_lp = score_item(model, tokenizer, prompt, verdict_ids)
                correct = pred == item["gt_verdict"]

                mark = "✅" if correct else "❌"
                extra = f"  (k={k_used})" if k_used < NTRAIN else ""
                print(f"  [{i+1:3d}/{len(todo)}] {item['subject']:<28} "
                      f"{item['item_id'][-4:]:<4} GT={item['gt_verdict']:<3} "
                      f"Pred={pred:<3} {mark}{extra}")

                on_result({
                    "model":            model_name,
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
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM — {item['item_id']} ignoré")
                free_vram()
            except Exception as e:
                print(f"  ❌ {item['item_id']} : {e}")
    finally:
        unload(model)          # gère aussi un modèle partiellement chargé
        del tokenizer
        free_vram()
        print(f"  🧹 VRAM après déchargement : {vram_free_gb():.1f} Go libres")

# ── Boucle principale + reprise ───────────────────────────────────────────────
results = []
done = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    done = {(r["model"], r["item_id"]) for r in results}
    print(f"\n♻️  Reprise : {len(done)} paires (modèle, item) déjà sauvegardées")

def save_result(r: dict):
    """Écriture atomique : un Ctrl-C pendant json.dump tronquerait le fichier
    et ferait perdre tout le run, pas seulement le dernier item."""
    results.append(r)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, RESULTS_FILE)

for model_name in JUDGE_MODELS:
    short = model_name.split("/")[-1]
    todo = [it for it in judge_items if (model_name, it["item_id"]) not in done]
    if not todo:
        print(f"  ⏩ {short} — déjà complet")
        continue
    try:
        run_model(model_name, todo, on_result=save_result)
    except Exception as e:
        print(f"  ❌ {short} : {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  RÉCAPITULATIF 1 — judge accuracy par modèle (weighted / unweighted / biais)
# ══════════════════════════════════════════════════════════════════════════════
by_model = defaultdict(list)
for r in results:
    by_model[r["model"]].append(r)

judge_stats = {}
for model, rs in by_model.items():
    by_subject = defaultdict(list)
    for r in rs:
        by_subject[r["subject"]].append(r["is_correct"])
    flat = [r["is_correct"] for r in rs]
    weighted = sum(flat) / len(flat) * 100 if flat else 0.0
    acc_subj = {s: sum(c) / len(c) for s, c in by_subject.items()}
    unweighted = sum(acc_subj.values()) / len(acc_subj) * 100 if acc_subj else 0.0
    pos = [r["is_correct"] for r in rs if r["gt_verdict"] == "Yes"]
    neg = [r["is_correct"] for r in rs if r["gt_verdict"] == "No"]
    tpr = sum(pos) / len(pos) * 100 if pos else 0.0
    tnr = sum(neg) / len(neg) * 100 if neg else 0.0
    yes_rate = sum(r["pred_verdict"] == "Yes" for r in rs) / len(rs) * 100
    judge_stats[model] = dict(weighted=weighted, unweighted=unweighted,
                               tpr=tpr, tnr=tnr, yes_rate=yes_rate, n=len(rs))

print(f"\n\n{'#'*88}")
print(f"#  RÉCAPITULATIF JUGE — {len(judge_items)} items ({len(questions)} pos + {len(questions)} neg)")
print(f"{'#'*88}")
print(f"  {'Modèle':<32} {'Weighted':>9} {'Unweight':>9} {'TPR':>7} {'TNR':>7} {'Yes%':>7} {'N':>6}")
print(f"  {'-'*32} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
for model, s in sorted(judge_stats.items(), key=lambda kv: -kv[1]["unweighted"]):
    print(f"  {model.split('/')[-1]:<32} {s['weighted']:>8.1f}% {s['unweighted']:>8.1f}% "
          f"{s['tpr']:>6.1f}% {s['tnr']:>6.1f}% {s['yes_rate']:>6.1f}% {s['n']:>6}")
print("   TPR = acc sur items corrects (GT=Yes) | TNR = acc sur items incorrects (GT=No)")
print("   Yes% ≈ 50 attendu — un écart fort signale un biais de verdict")

# ══════════════════════════════════════════════════════════════════════════════
#  RÉCAPITULATIF 2 — hypothèse good-responder : solver vs judge
# ══════════════════════════════════════════════════════════════════════════════
if solver_raw is not None:
    # Accuracy solver unweighted par modèle + correct/incorrect par question
    solver_by_model = defaultdict(lambda: defaultdict(list))
    solver_correct = {}                     # (model, question_id) -> bool
    for r in solver_raw:
        solver_by_model[r["model"]][r["subject"]].append(r["is_correct"])
        solver_correct[(r["model"], r["question_id"])] = r["is_correct"]

    solver_acc = {}
    for model, by_subject in solver_by_model.items():
        acc_subj = {s: sum(c) / len(c) for s, c in by_subject.items()}
        solver_acc[model] = sum(acc_subj.values()) / len(acc_subj) * 100

    common = sorted(set(solver_acc) & set(judge_stats),
                    key=lambda m: -solver_acc[m])

    print(f"\n{'#'*88}")
    print(f"#  HYPOTHÈSE GOOD-RESPONDER — solver vs judge (unweighted)")
    print(f"{'#'*88}")
    print(f"  {'Modèle':<32} {'Solver':>9} {'Judge':>9} {'Écart':>8}")
    print(f"  {'-'*32} {'-'*9} {'-'*9} {'-'*8}")
    for m in common:
        s_acc, j_acc = solver_acc[m], judge_stats[m]["unweighted"]
        print(f"  {m.split('/')[-1]:<32} {s_acc:>8.1f}% {j_acc:>8.1f}% {j_acc - s_acc:>+7.1f}")

    # Spearman panel (n petit — indicatif seulement)
    if len(common) >= 3:
        def spearman(xs, ys):
            def ranks(v):
                order = sorted(range(len(v)), key=lambda i: v[i])
                rk = [0.0] * len(v)
                for pos, i in enumerate(order):
                    rk[i] = pos + 1
                return rk
            rx, ry = ranks(xs), ranks(ys)
            n = len(xs)
            mx, my = sum(rx) / n, sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
            return num / den if den else float("nan")
        rho = spearman([solver_acc[m] for m in common],
                       [judge_stats[m]["unweighted"] for m in common])
        print(f"\n  Spearman ρ (panel, n={len(common)}) = {rho:.3f}"
              f"   ⚠️ indicatif — puissance faible à n<10")

    # Analyse intra-modèle par question : une question est "bien jugée" si les
    # DEUX items (pos et neg) reçoivent le bon verdict.
    print(f"\n  Analyse intra-modèle (niveau question, {len(questions)} paires appariées/modèle) :")
    print(f"  {'Modèle':<32} {'P(juge✓|slv✓)':>14} {'P(juge✓|slv✗)':>14} {'Δ':>7}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*7}")
    for m in common:
        jok = defaultdict(list)             # question_id -> [is_correct des 2 items]
        for r in by_model[m]:
            jok[r["question_id"]].append(r["is_correct"])
        both, given_c, given_w = {}, [], []
        for qid, cors in jok.items():
            if len(cors) == 2:
                both[qid] = all(cors)
        for qid, ok in both.items():
            sc = solver_correct.get((m, qid))
            if sc is True:
                given_c.append(ok)
            elif sc is False:
                given_w.append(ok)
        p_c = sum(given_c) / len(given_c) * 100 if given_c else float("nan")
        p_w = sum(given_w) / len(given_w) * 100 if given_w else float("nan")
        print(f"  {m.split('/')[-1]:<32} {p_c:>13.1f}% {p_w:>13.1f}% {p_c - p_w:>+6.1f}")
    print("   Δ > 0 = le modèle juge mieux les questions qu'il sait résoudre")
    print("   → version au niveau question de l'hypothèse good-responder")

    # Biais de self-agreement : le juge dit-il plus volontiers Yes quand la
    # lettre proposée coïncide avec SA PROPRE prédiction solver ?
    # → version QCM du self-preference, mesurée sans changer le design.
    solver_pred = {(r["model"], r["question_id"]): r["pred_letter"]
                   for r in solver_raw}
    print(f"\n  Biais de self-agreement (P(verdict=Yes) selon la lettre proposée) :")
    print(f"  {'Modèle':<32} {'=sa préd.':>11} {'≠sa préd.':>11} {'Δ':>7}")
    print(f"  {'-'*32} {'-'*11} {'-'*11} {'-'*7}")
    for m in common:
        own, other = [], []
        for r in by_model[m]:
            mine = solver_pred.get((m, r["question_id"]))
            if mine is None:
                continue
            said_yes = r["pred_verdict"] == "Yes"
            (own if r["proposed_letter"] == mine else other).append(said_yes)
        p_own = sum(own) / len(own) * 100 if own else float("nan")
        p_oth = sum(other) / len(other) * 100 if other else float("nan")
        print(f"  {m.split('/')[-1]:<32} {p_own:>10.1f}% {p_oth:>10.1f}% {p_own - p_oth:>+6.1f}")
    print("   Δ élevé = le juge valide surtout ce qu'il aurait répondu lui-même")
    print("   (à lire avec le tableau intra-modèle : un bon juge a un Δ dû à sa")
    print("    compétence solver, un juge biaisé a un Δ même quand il se trompe)")
else:
    print(f"\n⚠️  {SOLVER_RESULTS} introuvable — corrélation solver↔judge sautée")

print(f"\n✅ {RESULTS_FILE} — {len(results)} résultats au total")