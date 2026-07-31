# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE PRÉDICTIVE — validation croisée groupée par question
#
#  Répond à la question du mémoire : les signaux internes prédisent-ils la
#  justesse du jugement mieux que le verdict ?
#
#  Trois modèles emboîtés, comparés en AUROC hors échantillon :
#    M0  taux de base           (aucune information)
#    M1  marge de log-vraisemblance
#    M2  marge + profil spectral résumé
#
#  ⚠️ POINT CRITIQUE — GroupKFold(groups=question_id).
#     Les items _pos et _neg d'une même question partagent presque tout leur
#     texte. S'ils tombent dans des plis différents, le modèle a déjà vu
#     l'essentiel et le score est faussement bon. Le groupement l'interdit.
#
#  Le profil spectral est résumé par 3 chiffres par métrique (au lieu de
#  choisir une couche après balayage, ce qui revient à garder le meilleur de
#  24 essais) :
#     - moyenne sur le dernier tiers du réseau
#     - pente sur le dernier tiers
#     - position du maximum, normalisée par la profondeur
#
#  Usage :  python analyse_cv.py mmlu_judge_spectral_results.json
# ══════════════════════════════════════════════════════════════════════════════

import sys, json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

METRICS = ["fiedler", "hfer", "smoothness", "spectral_entropy"]
N_SPLITS = 5


def profile_features(spectral: list) -> np.ndarray:
    """Résume le profil couche par couche en 3 nombres par métrique."""
    n = len(spectral)
    late = slice(int(2 * n / 3), n)          # dernier tiers du réseau
    x = np.arange(n)[late]
    feats = []
    for m in METRICS:
        v = np.array([c[m] if c[m] is not None else np.nan for c in spectral], float)
        v = np.nan_to_num(v, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0)
        vl = v[late]
        pente = np.polyfit(x, vl, 1)[0] if len(x) > 1 else 0.0
        feats += [vl.mean(), pente, float(np.argmax(v)) / n]
    return np.array(feats)


def cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple:
    """AUROC hors échantillon, moyenne sur les plis. X vide = taux de base."""
    if X.shape[1] == 0:
        return 0.5, 0.0
    scores = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        pipe.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def analyse(rows: list, model: str):
    rows = [r for r in rows if isinstance(r.get("spectral"), list)]
    if len(rows) < 40:
        print(f"  (trop peu d'items exploitables pour {model})")
        return

    y = np.array([float(r["is_correct"]) for r in rows])
    g = np.array([r["question_id"] for r in rows])
    m = np.array([r["margin_yes_no"] for r in rows])
    S = np.vstack([profile_features(r["spectral"]) for r in rows])

    X0 = np.zeros((len(rows), 0))
    X1 = np.c_[m, np.abs(m)]
    X2 = np.c_[X1, S]

    print(f"\n### {model}   ({len(rows)} items, {len(set(g))} questions)")
    modes = {r.get("load_mode", "?") for r in rows}
    if modes != {"?"}:
        print(f"  placement : {', '.join(sorted(modes))}")
    print(f"  exactitude du verdict : {y.mean():.1%}   "
          f"(taux de base équilibré = 50 %)")
    print(f"\n  {'modèle':<34} {'AUROC hors échantillon':>24}")
    for label, X in [("M0  taux de base", X0),
                     ("M1  marge", X1),
                     ("M2  marge + profil spectral", X2)]:
        mu, sd = cv_auroc(X, y, g)
        print(f"  {label:<34} {mu:>16.3f} ± {sd:.3f}")

    print("\n  Lecture : 0,50 = aucune information. M2 doit dépasser M1 de façon")
    print("  nette (au-delà des écarts-types) pour que le spectral apporte")
    print("  quelque chose que la marge ne contenait pas déjà.")


def main(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for model in sorted({r["model"] for r in rows}):
        analyse([r for r in rows if r["model"] == model], model)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "mmlu_judge_spectral_results.json")