"""Chemins, graine aléatoire et coûts métier : définis une seule fois pour tout le projet."""

from pathlib import Path

# --- Chemins -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw" / "inclusion-financiere-en-afrique.csv"
VARIABLE_DEFINITIONS = DATA_DIR / "raw" / "VariableDefinitions.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Reproductibilité --------------------------------------------------------
RANDOM_STATE = 42

# --- Cible -------------------------------------------------------------------
TARGET = "bank_account"
ID_COLUMNS = ["uniqueid"]

# --- Coûts métier (scénario KaziPay, hypothèses pédagogiques) ----------------
COST_FP = 0.5  # USD : cout de contacter une personne sans compte
COST_FN = 5.0  # USD : cout de ne pas contacter une personne qui a un compte.
