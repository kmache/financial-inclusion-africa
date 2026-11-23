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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

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


MIN_FREQUENCY = 20
"""Seuil de rareté commun à tous les encodages (Q2.6, Q3.2)."""
INFREQUENT = "infrequent"


def onehot_preprocessor(
    cat_cols: list[str], num_cols: list[str], scale: bool = True
) -> ColumnTransformer:
    """One-hot pour les catégorielles ; imputation médiane (+ standardisation) des numériques.

    - ``min_frequency=20`` : les modalités vues moins de 20 fois à l'entraînement sont regroupées
      dans une colonne « infrequent » (Q2.6, Q3.2) ;
    - ``handle_unknown="infrequent_if_exist"`` : une modalité jamais vue à l'entraînement est
      envoyée dans cette colonne si elle existe, sinon codée par des zéros, sans erreur ;
    - ``scale`` : standardisation des numériques, indispensable pour la régression logistique,
      inutile (mais sans danger) pour les arbres (Q3.4).
    """
    numeric = make_pipeline(
        SimpleImputer(strategy="median"), *([StandardScaler()] if scale else [])
    )
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_FREQUENCY,
                    sparse_output=False,
                ),
                cat_cols,
            ),
            ("num", numeric, num_cols),
        ]
    )


def ordinal_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """Codes entiers pour les catégorielles, numériques intactes (NaN compris) : pour les arbres.

    Les codes suivent l'ordre alphabétique : sans signification pour une variable nominale, mais
    un arbre peut regrouper les modalités par coupures successives. Modalités rares regroupées
    (``min_frequency``), modalité inconnue codée −1. Pas de standardisation ni d'imputation :
    les forêts (scikit-learn ≥ 1.4) et les boostings gèrent les valeurs manquantes.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        min_frequency=MIN_FREQUENCY,
    )
    return ColumnTransformer(
        [("cat", encoder, cat_cols), ("num", "passthrough", num_cols)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


class NativeCategories(BaseEstimator, TransformerMixin):
    """Prépare les colonnes pour les modèles qui gèrent nativement les catégories.

    Apprend (``fit``) les modalités fréquentes de chaque colonne catégorielle ; les modalités
    rares ou jamais vues deviennent ``"infrequent"`` (même traitement de Q2.6 que les autres
    encodages). Les numériques sont transmises telles quelles (NaN compris).

    Parameters
    ----------
    cat_cols, num_cols : colonnes à garder.
    dtype : ``"category"`` (XGBoost, LightGBM, HistGradientBoosting) ou ``"str"`` (CatBoost).
    min_frequency : seuil de rareté.
    """

    def __init__(
        self, cat_cols, num_cols, dtype="category", min_frequency=MIN_FREQUENCY
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.dtype = dtype
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        self.categories_ = {}
        for col in self.cat_cols:
            counts = X[col].astype(str).value_counts()
            frequent = sorted(counts.index[counts >= self.min_frequency])
            self.categories_[col] = [*frequent, INFREQUENT]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = X[list(self.cat_cols) + list(self.num_cols)].copy()
        for col, cats in self.categories_.items():
            values = out[col].astype(str)
            values = values.where(values.isin(cats), INFREQUENT)
            if self.dtype == "category":
                out[col] = pd.Categorical(values, categories=cats)
            else:
                out[col] = values.astype(object)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.cat_cols, *self.num_cols], dtype=object)


def feature_columns(features: FeatureSet = "all") -> tuple[list[str], list[str]]:
    """Colonnes catégorielles et numériques envoyées au préprocesseur."""
    if features == "none":
        return list(CAT_COLS), list(NUM_COLS)
    cat = CAT_COLS + GUIDE_CAT
    num = NUM_COLS + GUIDE_NUM + (CUSTOM_NUM if features == "all" else [])
    return cat, num


def catboost_cat_features(features: FeatureSet = "all") -> tuple[str, ...]:
    """Colonnes catégorielles à passer à CatBoost : un **tuple** (une liste casse le clonage)."""
    return tuple(feature_columns(features)[0])


Encoding = Literal["onehot", "ordinal", "native", "native_str"]


def pipe(
    model,
    features: FeatureSet = "all",
    encoding: Encoding = "onehot",
    scale: bool = True,
) -> Pipeline:
    """Pipeline complet : [variables métier] → encodage → modèle.

    ``features`` : ``"none"`` (colonnes brutes), ``"guide"`` (variables du guide) ou ``"all"``
    (variables du guide + nos variables, Q3.5) ; sert à mesurer leur effet à l'étape 5.

    ``encoding`` selon la famille de modèle :

    ===============  ==========================================  ================================
    ``encoding``     préparation                                 modèles
    ===============  ==========================================  ================================
    ``"onehot"``     one-hot + imputation (+ standardisation)    régression logistique ; tous
                                                                 (comparaison du guide, étape 6)
    ``"ordinal"``    codes entiers, numériques intactes          forêt aléatoire, boostings
    ``"native"``     dtype ``category``, numériques intactes     XGBoost (``enable_categorical``),
                                                                 LightGBM, HistGradientBoosting
    ``"native_str"`` texte, numériques intactes                  CatBoost (``cat_features`` en
                                                                 tuple)
    ===============  ==========================================  ================================

    ``scale`` ne concerne que ``"onehot"`` : à garder pour la régression logistique, inutile
    pour les arbres (Q3.4).
    """
    cat, num = feature_columns(features)
    steps = [] if features == "none" else [AddFeatures(custom=features == "all")]
    if encoding == "onehot":
        prep = onehot_preprocessor(cat, num, scale=scale)
    elif encoding == "ordinal":
        prep = ordinal_preprocessor(cat, num)
    elif encoding in ("native", "native_str"):
        prep = NativeCategories(
            cat, num, dtype="category" if encoding == "native" else "str"
        )
    else:
        raise ValueError(f"encoding inconnu : {encoding!r}")
    return make_pipeline(*steps, prep, model)


def fold_category_audit(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
    cols: list[str] | None = None,
    min_frequency: int = MIN_FREQUENCY,
    watch: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Q2.6 pli par pli : modalités rares ou inconnues dans chaque pli d'entraînement.

    Pour chaque pli et chaque colonne catégorielle, liste les modalités vues moins de
    ``min_frequency`` fois à l'entraînement (regroupées par l'encodeur) et celles présentes en
    validation mais absentes de l'entraînement (inconnues). ``watch`` ajoute des modalités à
    suivre même quand elles sont fréquentes.
    """
    cols = cols or CAT_COLS
    watch = watch or {}
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        held_out = (
            ", ".join(map(str, pd.unique(groups.iloc[va])))
            if groups is not None
            else ""
        )
        for col in cols:
            n_tr, n_va = X_tr[col].value_counts(), X_va[col].value_counts()
            rare = set(n_tr.index[n_tr < min_frequency])
            unknown = set(n_va.index.difference(n_tr.index))
            for modality in sorted(rare | unknown | set(watch.get(col, []))):
                k_tr, k_va = int(n_tr.get(modality, 0)), int(n_va.get(modality, 0))
                if k_tr == 0:
                    status = "inconnue à l'entraînement"
                elif k_tr < min_frequency:
                    status = "rare : regroupée"
                else:
                    status = "fréquente : colonne propre"
                rows.append(
                    {
                        "pli": fold,
                        "laissé de côté": held_out,
                        "variable": col,
                        "modalité": modality,
                        "effectif entraînement": k_tr,
                        "effectif validation": k_va,
                        "statut": status,
                    }
                )
    return pd.DataFrame(rows)
