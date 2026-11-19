"""Outils d'exploration (étape 2) : typologie des colonnes, taux par modalité, associations."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from financial_inclusion_africa.config import TARGET

# Typologie des colonnes (Q2.1)
NUMERIC = ["household_size", "age_of_respondent"]
BINARY = ["location_type", "cellphone_access", "gender_of_respondent"]
NOMINAL = ["country", "relationship_with_head", "marital_status", "job_type"]
ORDINAL = ["education_level"]
IDENTIFIERS = ["uniqueid"]
REDUNDANT = ["year"]  # entièrement déterminée par country (Q2.3)

EDUCATION_ORDER = [
    "No formal education",
    "Primary education",
    "Secondary education",
    "Vocational/Specialised training",
    "Tertiary education",
    "Other/Dont know/RTA",
]
"""Ordre naturel des niveaux d'éducation ; la modalité « autre / ne sait pas » est mise à la fin."""

CATEGORICAL = BINARY + NOMINAL + ORDINAL


def column_types() -> pd.Series:
    """Type statistique de chaque colonne du fichier brut."""
    kinds = {
        **dict.fromkeys(NUMERIC, "numérique (discrète)"),
        **dict.fromkeys(BINARY, "binaire"),
        **dict.fromkeys(NOMINAL, "nominale"),
        **dict.fromkeys(ORDINAL, "ordinale"),
        **dict.fromkeys(IDENTIFIERS, "identifiant"),
        **dict.fromkeys(REDUNDANT, "numérique → redondante avec country"),
        TARGET: "cible binaire",
    }
    return pd.Series(kinds, name="type")


def target_rate(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    """Effectif, nombre de positifs et part des positifs pour chaque modalité de ``col``."""
    out = df.groupby(col, observed=True)[target].agg(
        effectif="size", positifs="sum", taux="mean"
    )
    out["part_effectif"] = out["effectif"] / out["effectif"].sum()
    return out


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér (sans correction) : association entre deux variables catégorielles, dans [0, 1]."""
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Matrice symétrique des V de Cramér entre les colonnes ``cols``."""
    m = pd.DataFrame(1.0, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            m.loc[a, b] = m.loc[b, a] = cramers_v(df[a], df[b])
    return m


def duplicate_profiles(
    df: pd.DataFrame, exclude: tuple[str, ...] = ("uniqueid",), target: str = TARGET
) -> dict[str, int]:
    """Profils identiques (hors ``exclude`` et cible) et étiquettes contradictoires (Q2.5).

    ``erreurs_irreductibles`` : nombre minimal d'erreurs de toute fonction des variables, car
    deux lignes de même profil et d'étiquettes différentes ne peuvent pas être toutes deux justes.
    """
    features = [c for c in df.columns if c not in (*exclude, target)]
    g = df.groupby(features, observed=True)[target].agg(n="size", pos="sum")
    repeated = g[g["n"] > 1]
    contradictory = repeated[(repeated["pos"] > 0) & (repeated["pos"] < repeated["n"])]
    minority = np.minimum(
        contradictory["pos"], contradictory["n"] - contradictory["pos"]
    )
    return {
        "profils_distincts": len(g),
        "profils_repetes": len(repeated),
        "lignes_dans_profils_repetes": int(repeated["n"].sum()),
        "profils_contradictoires": len(contradictory),
        "lignes_dans_profils_contradictoires": int(contradictory["n"].sum()),
        "erreurs_irreductibles": int(minority.sum()),
    }


def rare_categories(
    df: pd.DataFrame, cols: list[str], threshold: int = 50
) -> pd.DataFrame:
    """Modalités de moins de ``threshold`` occurrences (Q2.6)."""
    rows = [
        {"variable": c, "modalite": m, "effectif": int(n)}
        for c in cols
        for m, n in df[c].value_counts().items()
        if n < threshold
    ]
    return pd.DataFrame(rows, columns=["variable", "modalite", "effectif"])
