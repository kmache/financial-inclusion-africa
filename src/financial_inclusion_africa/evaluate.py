"""Métriques et protocole d'évaluation : coût métier, seuil, validation croisée, calibration (ECE)
(puis intervalle bootstrap)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from financial_inclusion_africa.config import COST_FN, COST_FP, N_SPLITS, RANDOM_STATE


def theoretical_threshold(cost_fp: float = COST_FP, cost_fn: float = COST_FN) -> float:
    """Seuil de décision optimal pour des probabilités calibrées (slide 44).

    On contacte dès que ``p * cost_fn > (1 - p) * cost_fp``,
    soit ``p > t* = cost_fp / (cost_fp + cost_fn)``.
    """
    if cost_fp < 0 or cost_fn < 0 or cost_fp + cost_fn == 0:
        raise ValueError("Les coûts doivent être positifs et non tous nuls.")
    return cost_fp / (cost_fp + cost_fn)


def total_cost(
    y_true, y_pred, cost_fp: float = COST_FP, cost_fn: float = COST_FN
) -> float:
    """Coût total d'une stratégie de contact : ``C_FP * FP + C_FN * FN``."""
    _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_fp * fp + cost_fn * fn)


def cost_at_threshold(
    y_true,
    y_proba,
    threshold: float,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> float:
    """Coût total quand on contacte toutes les personnes dont ``y_proba >= threshold``."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return total_cost(y_true, y_pred, cost_fp, cost_fn)


def classification_metrics(
    y_true,
    y_proba,
    threshold: float,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> dict[str, float]:
    """Toutes les métriques rapportées dans le projet, pour des probabilités et un seuil.

    - Classement (sans seuil) : PR-AUC (métrique principale), ROC-AUC.
    - Probabilités (sans seuil) : Brier, log-loss.
    - Décision (au seuil) : accuracy, précision, recall, F1, coût total.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cost": total_cost(y_true, y_pred, cost_fp, cost_fn),
    }


def full_classification_report(y_true, y_proba, threshold: float) -> str:
    """Rapport de classification complet (scikit-learn) au seuil donné."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No (0)", "Yes (1)"],
        digits=4,
        zero_division=0,
    )


# --- Protocole de validation croisée (étape 4) ------------------------------------------------

CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
"""L'objet ``cv`` unique du projet : mêmes plis pour tous les modèles, dans tous les notebooks."""

T_STAR = theoretical_threshold()


def _precision_at_t(y_true, y_proba):
    return precision_score(y_true, np.asarray(y_proba) >= T_STAR, zero_division=0)


def _recall_at_t(y_true, y_proba):
    return recall_score(y_true, np.asarray(y_proba) >= T_STAR, zero_division=0)


def _cost_per_person_at_t(y_true, y_proba):
    return cost_at_threshold(y_true, y_proba, T_STAR) / len(y_true)


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
    "logloss": "neg_log_loss",
    "acc": "accuracy",
    "precision_t": make_scorer(_precision_at_t, response_method="predict_proba"),
    "recall_t": make_scorer(_recall_at_t, response_method="predict_proba"),
    "cost_t": make_scorer(
        _cost_per_person_at_t, response_method="predict_proba", greater_is_better=False
    ),
}
"""Métriques calculées dans chaque pli. ``*_t`` : au seuil théorique t* (≈ 0,091) ;
``cost_t`` : coût moyen par personne (USD), comparable quelle que soit la taille du pli.
``acc`` est l'accuracy au seuil 0,5 (convention de scikit-learn)."""

NEGATED = {"brier", "logloss", "cost_t"}
"""Scores renvoyés négatifs par scikit-learn (plus grand = meilleur) : à remettre au signe +."""


def cv_summary(res: dict, name: str | None = None) -> pd.Series:
    """Résultat de ``cross_validate`` → une ligne : moyenne et écart-type de chaque métrique,
    en validation et en entraînement, et temps d'entraînement moyen par pli (secondes)."""
    row = {}
    for metric in [k.removeprefix("test_") for k in res if k.startswith("test_")]:
        sign = -1 if metric in NEGATED else 1
        for part, prefix in [("val", "test_"), ("train", "train_")]:
            if prefix + metric in res:
                values = sign * np.asarray(res[prefix + metric])
                row[(metric, part, "mean")] = values.mean()
                row[(metric, part, "std")] = values.std()
    row[("fit_time", "", "mean")] = np.mean(res["fit_time"])
    return pd.Series(row, name=name)


def format_summary(
    table: pd.DataFrame, metrics=None, part: str = "val", digits: int = 3
):
    """Tableau lisible ``moyenne ± écart-type`` à partir de lignes ``cv_summary``."""
    metrics = metrics or [
        m for m in dict.fromkeys(table.columns.get_level_values(0)) if m != "fit_time"
    ]
    out = pd.DataFrame(index=table.index)
    for m in metrics:
        if (m, part, "mean") in table.columns:
            out[m] = [
                f"{mu:.{digits}f} ± {sd:.{digits}f}"
                for mu, sd in zip(
                    table[(m, part, "mean")], table[(m, part, "std")], strict=True
                )
            ]
    out["temps (s)"] = table[("fit_time", "", "mean")].round(2)
    return out


def fold_scores(res: dict) -> pd.DataFrame:
    """Scores de validation pli par pli (signes remis au positif) : pour comparer des modèles
    sur les mêmes plis (différences appariées)."""
    out = {
        k.removeprefix("test_"): (-1 if k.removeprefix("test_") in NEGATED else 1)
        * np.asarray(v)
        for k, v in res.items()
        if k.startswith("test_")
    }
    return pd.DataFrame(out).rename_axis("pli")


def evaluate_cv(model, X, y, cv=CV, groups=None, scoring=None, name=None):
    """``cross_validate`` avec le protocole du projet ; renvoie (ligne de résumé, résultats bruts)."""
    res = cross_validate(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring or SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    return cv_summary(res, name), res


def per_group_scores(model, X, y, groups, cv) -> pd.DataFrame:
    """Validation croisée groupée : scores du groupe laissé de côté dans chaque pli.

    En plus des métriques, compare la prévalence réelle du groupe à la probabilité moyenne
    prédite (« calibration globale ») et rapporte la PR-AUC relative à la prévalence (lift),
    seule comparable d'un groupe à l'autre.
    """
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        proba = fitted.predict_proba(X.iloc[va])[:, 1]
        y_va = y.iloc[va]
        prevalence = y_va.mean()
        ap = average_precision_score(y_va, proba)
        rows.append(
            {
                "pli": fold,
                "groupe laissé de côté": ", ".join(
                    map(str, pd.unique(groups.iloc[va]))
                ),
                "lignes": len(va),
                "prévalence": prevalence,
                "proba moyenne prédite": proba.mean(),
                "pr_auc": ap,
                "pr_auc / prévalence": ap / prevalence,
                "roc_auc": roc_auc_score(y_va, proba),
                "brier": brier_score_loss(y_va, proba),
            }
        )
    return pd.DataFrame(rows).set_index("pli")


def paired_difference(
    res_a: dict, res_b: dict, metric: str = "pr_auc"
) -> dict[str, float]:
    """Différence B − A pli par pli, pour deux ``cross_validate`` faits sur les **mêmes plis**.

    Renvoie la moyenne, l'écart-type et l'erreur-type (écart-type / √k) des différences, et le
    nombre de plis où B fait mieux. L'écart-type des différences appariées est bien plus petit
    que celui des scores eux-mêmes : la difficulté propre à chaque pli s'annule (Q4.2).
    """
    a = fold_scores(res_a)[metric].to_numpy()
    b = fold_scores(res_b)[metric].to_numpy()
    d = b - a
    return {
        "moyenne A": a.mean(),
        "moyenne B": b.mean(),
        "écart-type A": a.std(),
        "différence moyenne B − A": d.mean(),
        "écart-type des différences": d.std(),
        "erreur-type des différences": d.std(ddof=1) / np.sqrt(len(d)),
        "plis où B > A": int((d > 0).sum()),
        "plis": len(d),
    }


def one_standard_error(
    values, mean_scores, std_scores, n_splits: int = N_SPLITS, simplest="max"
):
    """Règle d'une erreur-type (slide 52) : la valeur la plus simple à moins d'une erreur-type
    (écart-type / √k) du meilleur score moyen.

    ``simplest`` : ``"max"`` si la plus grande valeur est la plus simple (``min_samples_leaf``,
    pénalité…), ``"min"`` sinon (profondeur, nombre de feuilles…).
    """
    values, mean_scores, std_scores = map(np.asarray, (values, mean_scores, std_scores))
    best = int(np.argmax(mean_scores))
    se = std_scores[best] / np.sqrt(n_splits)
    eligible = values[mean_scores >= mean_scores[best] - se]
    chosen = eligible.max() if simplest == "max" else eligible.min()
    return {
        "meilleure valeur": values[best].item(),
        "meilleur score": mean_scores[best].item(),
        "erreur-type": se.item(),
        "seuil (meilleur − SE)": (mean_scores[best] - se).item(),
        "valeur retenue (1 SE)": chosen.item(),
    }


# --- Calibration (étape 8) ----------------------------------------------------------------------


def _bin_index(p, n_bins: int) -> np.ndarray:
    """Intervalles de largeur ``1 / n_bins`` ; ``p = 1`` tombe dans le dernier."""
    return np.clip((np.asarray(p, dtype=float) * n_bins).astype(int), 0, n_bins - 1)


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """ECE (slide 42) : ``Σ_k (n_k / n) · |acc(k) − conf(k)|`` sur ``n_bins`` intervalles égaux.

    ``conf(k)`` : probabilité moyenne annoncée dans l'intervalle k ; ``acc(k)`` : fréquence
    observée des positifs. Les intervalles vides ne comptent pas.
    """
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    bins = _bin_index(p, n_bins)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Le détail de l'ECE, intervalle par intervalle : effectif, probabilité moyenne annoncée,
    fréquence observée, écart (observé − annoncé) et contribution à l'ECE."""
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    bins = _bin_index(p, n_bins)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf, acc = p[m].mean(), y_true[m].mean()
            rows.append(
                {
                    "intervalle": f"[{b / n_bins:.1f} ; {(b + 1) / n_bins:.1f}[",
                    "effectif": int(m.sum()),
                    "annoncé (conf)": conf,
                    "observé (acc)": acc,
                    "écart observé − annoncé": acc - conf,
                    "contribution à l'ECE": m.mean() * abs(acc - conf),
                }
            )
    return pd.DataFrame(rows).set_index("intervalle")


def calibration_metrics(y_true, p, n_bins: int = 10) -> dict[str, float]:
    """Métriques de probabilités (ECE, Brier, log-loss) et de classement (PR-AUC, ROC-AUC),
    plus la prévalence prédite (moyenne des probabilités) à comparer à la prévalence réelle."""
    y_true, p = np.asarray(y_true), np.asarray(p, dtype=float)
    return {
        "ECE": expected_calibration_error(y_true, p, n_bins),
        "Brier": brier_score_loss(y_true, p),
        "log-loss": log_loss(y_true, p, labels=[0, 1]),
        "PR-AUC": average_precision_score(y_true, p),
        "ROC-AUC": roc_auc_score(y_true, p),
        "prévalence prédite": p.mean(),
    }


def ece_noise_floor(
    p, n_bins: int = 10, n_sim: int = 200, seed: int = 0
) -> dict[str, float]:
    """ECE d'un modèle **parfaitement calibré** qui annoncerait les mêmes probabilités ``p`` :
    on tire ``y ~ Bernoulli(p)`` ``n_sim`` fois. L'ECE n'est jamais nulle sur un échantillon
    fini ; une ECE sous le 95e centile de ce plancher est indiscernable du bruit."""
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    sims = [
        expected_calibration_error(rng.uniform(size=p.size) < p, p, n_bins)
        for _ in range(n_sim)
    ]
    return {
        "moyenne": float(np.mean(sims)),
        "95e centile": float(np.percentile(sims, 95)),
    }


def expected_counts(y_true, p, groups) -> pd.DataFrame:
    """Par groupe : nombre réel de positifs, somme des probabilités (nombre attendu, slide 40),
    écart, erreur relative ``(attendu − réel) / réel``, et bruit attendu.

    Même parfaitement calibré, le nombre réel varie autour du nombre attendu : c'est une somme de
    Bernoulli, d'écart-type ``√Σ p(1 − p)``. ``écart / écart-type`` au-delà de 2 en valeur
    absolue signale un biais réel, pas du hasard.
    """
    p = np.asarray(p, dtype=float)
    df = pd.DataFrame(
        {
            "groupe": np.asarray(groups),
            "réel": np.asarray(y_true),
            "attendu": p,
            "variance": p * (1 - p),
        }
    )
    out = df.groupby("groupe")[["réel", "attendu", "variance"]].sum()
    out = pd.concat([out, out.sum().rename("Total").to_frame().T])
    out["écart"] = out["attendu"] - out["réel"]
    out["erreur relative"] = out["écart"] / out["réel"]
    out["écart-type attendu"] = np.sqrt(out.pop("variance"))
    out["écart / écart-type"] = out["écart"] / out["écart-type attendu"]
    return out
