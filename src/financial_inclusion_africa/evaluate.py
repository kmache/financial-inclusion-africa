"""Métriques d'évaluation : coût métier, seuil théorique (puis ECE, intervalle bootstrap)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from financial_inclusion_africa.config import COST_FN, COST_FP


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
