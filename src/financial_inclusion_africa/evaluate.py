"""Métriques d'évaluation : coût métier, seuil théorique (puis ECE, intervalle bootstrap)."""

import numpy as np
from sklearn.metrics import confusion_matrix

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
