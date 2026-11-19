"""Chargement et nettoyage des données brutes."""

from pathlib import Path

import pandas as pd

from financial_inclusion_africa.config import RAW, TARGET


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Charge le fichier brut et encode la cible ``bank_account`` en 0/1."""
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df
