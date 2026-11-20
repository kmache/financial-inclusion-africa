"""Feature engineering et préprocesseurs scikit-learn (étape 3).

Deux familles de transformations :

- ``AddFeatures`` : règles métier fixes, **sans apprentissage** (aucune statistique tirée des
  données) ; elle peut être appliquée à n'importe quelle ligne sans fuite ;
- ``onehot_preprocessor`` : encodage, imputation et standardisation, qui **apprennent** des
  paramètres (catégories, médianes, moyennes) et doivent donc être ajustés dans chaque pli,
  à l'intérieur du pipeline.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financial_inclusion_africa.data import CAT_COLS, NUM_COLS

# Rang ordinal de l'éducation. « Other/Dont know/RTA » (35 lignes, quasi-manquante, Q2.6) est
# volontairement absente : elle devient NaN, puis est imputée dans le pipeline, avec un
# indicateur ``education_unknown``. Elle ne doit jamais devenir silencieusement le rang le plus
# haut, ni le plus bas.
EDU_ORDER = {
    "No formal education": 0,
    "Primary education": 1,
    "Secondary education": 2,
    "Vocational/Specialised training": 3,
    "Tertiary education": 4,
}
FORMAL_JOBS = {"Formally employed Private", "Formally employed Government"}
AGE_BINS = [15, 25, 35, 50, 65, 120]
AGE_LABELS = ["16-25", "26-35", "36-50", "51-65", "66+"]

# Variables du guide (+ education_unknown, ajoutée pour Q2.6) et nos variables (Q3.5)
GUIDE_NUM = [
    "education_rank",
    "education_unknown",
    "is_head",
    "formal_job",
    "urban_phone",
]
GUIDE_CAT = ["age_group"]
CUSTOM_NUM = ["phone_education", "lives_alone"]

FeatureSet = Literal["none", "guide", "all"]


class AddFeatures(BaseEstimator, TransformerMixin):
    """Feature engineering sans apprentissage : aucune statistique tirée des données.

    Parameters
    ----------
    custom : bool
        Ajoute aussi nos variables (``phone_education``, ``lives_alone``) aux variables du guide.
    """

    def __init__(self, custom: bool = True):
        self.custom = custom

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        # --- variables du guide
        X["education_rank"] = (
            X["education_level"].map(EDU_ORDER).astype(float)
        )  # NaN si inconnu
        X["education_unknown"] = X["education_rank"].isna().astype(int)
        X["is_head"] = (X["relationship_with_head"] == "Head of Household").astype(int)
        X["formal_job"] = X["job_type"].isin(FORMAL_JOBS).astype(int)
        X["urban_phone"] = (
            (X["location_type"] == "Urban") & (X["cellphone_access"] == "Yes")
        ).astype(int)
        X["age_group"] = pd.cut(
            X["age_of_respondent"], AGE_BINS, labels=AGE_LABELS
        ).astype(str)
        # --- nos variables (Q3.5)
        if self.custom:
            # Interaction téléphone × éducation : sans téléphone, l'éducation change peu de
            # chose ; avec téléphone, l'effet est fort (étape 2, interactions).
            X["phone_education"] = (X["cellphone_access"] == "Yes") * X[
                "education_rank"
            ]
            # Personne seule : gère elle-même ses finances (19,1 % ont un compte, contre 14,1 %).
            X["lives_alone"] = (X["household_size"] == 1).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        extra = GUIDE_NUM + GUIDE_CAT + (CUSTOM_NUM if self.custom else [])
        return np.asarray([*self.feature_names_in_, *extra], dtype=object)


def onehot_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """One-hot pour les catégorielles ; imputation médiane puis standardisation pour les numériques.

    - ``min_frequency=20`` : les modalités vues moins de 20 fois à l'entraînement sont regroupées
      dans une colonne « infrequent » (Q2.6, Q3.2) ;
    - ``handle_unknown="infrequent_if_exist"`` : une modalité jamais vue à l'entraînement est
      envoyée dans cette colonne si elle existe, sinon codée par des zéros, sans erreur.
    """
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
                cat_cols,
            ),
            ("num", numeric, num_cols),
        ]
    )


def feature_columns(features: FeatureSet = "all") -> tuple[list[str], list[str]]:
    """Colonnes catégorielles et numériques envoyées au préprocesseur."""
    if features == "none":
        return list(CAT_COLS), list(NUM_COLS)
    cat = CAT_COLS + GUIDE_CAT
    num = NUM_COLS + GUIDE_NUM + (CUSTOM_NUM if features == "all" else [])
    return cat, num


def pipe(model, features: FeatureSet = "all") -> Pipeline:
    """Pipeline complet : [variables métier] → encodage/standardisation → modèle.

    ``features`` : ``"none"`` (colonnes brutes), ``"guide"`` (variables du guide) ou
    ``"all"`` (variables du guide + nos variables, Q3.5). Sert à mesurer leur effet à l'étape 5.
    """
    cat, num = feature_columns(features)
    steps = [] if features == "none" else [AddFeatures(custom=features == "all")]
    return make_pipeline(*steps, onehot_preprocessor(cat, num), model)
