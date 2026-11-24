"""Modèles retenus à l'étape 7 : hyperparamètres trouvés par Optuna et fabriques de pipelines.

Les valeurs ci-dessous sont recopiées des études Optuna du notebook 04 (40 essais, TPE, graine
42, validation croisée ``evaluate.CV``, métrique PR-AUC). Le notebook 04 vérifie qu'elles
correspondent toujours aux meilleurs essais. Les notebooks 05 et 06 utilisent ces fabriques pour
travailler exactement sur les mêmes modèles.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from financial_inclusion_africa.config import RANDOM_STATE
from financial_inclusion_africa.features import catboost_cat_features, pipe

LGBM_PARAMS = {
    "learning_rate": 0.04881569095741423,
    "num_leaves": 6,
    "min_child_samples": 23,
    "subsample": 0.5609139378118542,
    "subsample_freq": 1,
    "colsample_bytree": 0.5306753727051603,
    "reg_lambda": 1.9346358011029263,
    "n_estimators": 341,
}
"""LightGBM, catégories natives : meilleur essai Optuna (PR-AUC en CV 0,5815)."""

CATBOOST_PARAMS = {
    "learning_rate": 0.06506110593721001,
    "depth": 4,
    "l2_leaf_reg": 23.778513667224345,
    "iterations": 801,
    "bagging_temperature": 0.5375337345053691,
    "random_strength": 2.821584919847275,
}
"""CatBoost, catégories natives : meilleur essai Optuna (PR-AUC en CV 0,5841)."""

CATBOOST_THREADS = 2
"""CatBoost dépend légèrement du nombre de threads (≈ 0,001 de PR-AUC) : on fixe celui de la
recherche pour que les résultats soient reproductibles."""


def make_baseline(**overrides) -> Pipeline:
    """Modèle de base : régression logistique, one-hot + standardisation."""
    return pipe(LogisticRegression(**{"max_iter": 2000, **overrides}))


def make_lgbm(**overrides) -> Pipeline:
    """LightGBM réglé, catégories natives (un seul cœur : la CV parallélise les plis)."""
    params = {**LGBM_PARAMS, **overrides}
    model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    return pipe(model, encoding="native")


def make_catboost(**overrides) -> Pipeline:
    """CatBoost réglé, catégories natives (texte), ``cat_features`` en tuple."""
    params = {**CATBOOST_PARAMS, **overrides}
    model = CatBoostClassifier(
        **params,
        cat_features=catboost_cat_features(),
        random_seed=RANDOM_STATE,
        thread_count=CATBOOST_THREADS,
        verbose=0,
    )
    return pipe(model, encoding="native_str")


CANDIDATES = {
    "Régression logistique (modèle de base)": make_baseline,
    "LightGBM réglé": make_lgbm,
    "CatBoost réglé": make_catboost,
}
"""Les modèles qui passent à l'étape 8 (calibration) : le modèle de base et les deux candidats."""
