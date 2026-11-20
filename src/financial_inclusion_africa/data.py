"""Chargement des données brutes et séparation X / y (étape 3.1)."""

from pathlib import Path

import pandas as pd

from financial_inclusion_africa.config import RAW, TARGET

CAT_COLS = [
    "country",
    "location_type",
    "cellphone_access",
    "gender_of_respondent",
    "relationship_with_head",
    "marital_status",
    "education_level",
    "job_type",
]
NUM_COLS = ["household_size", "age_of_respondent"]

DROPPED_COLS = {
    "uniqueid": "identifiant arbitraire, numérotation par pays, signal parasite (Q2.4)",
    "year": "entièrement déterminée par country, valeur future jamais vue (Q2.3)",
}
"""Colonnes interdites comme variables explicatives, avec la raison."""


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Charge le fichier brut et encode la cible ``bank_account`` en 0/1."""
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare X et y ; retire la cible et les colonnes qui ne doivent pas servir de variables.

    Accepte une cible déjà encodée en 0/1 (``load_raw``) ou brute (« Yes » / « No »).
    """
    target = df[TARGET]
    if pd.api.types.is_numeric_dtype(target):
        y = target.astype(int)
    else:
        y = (target == "Yes").astype(int)
    y = y.rename(TARGET)
    X = df.drop(columns=[TARGET, *DROPPED_COLS])
    return X[CAT_COLS + NUM_COLS], y
