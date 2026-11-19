# Qui possède un compte bancaire ? Prédire l'inclusion financière en Afrique de l'Est

Cas pratique du **Module 5 · AIMS Sénégal** (Évaluation et optimisation des modèles de ML).
De la donnée brute à un modèle évalué honnêtement : pipeline sans fuite, validation croisée
stratifiée, méthodes d'ensemble, optimisation des hyperparamètres (Random Search, Optuna),
calibration, seuil selon les coûts et interprétation (permutation, SHAP).

> 🚧 Projet en cours : ce README sera complété à l'étape 12 (problème, méthode, résultats, limites).

## Données

Compétition **Financial Inclusion in Africa** ([Zindi](https://zindi.africa)), issue des enquêtes
FinScope 2016–2018 au Kenya, au Rwanda, en Tanzanie et en Ouganda (23 524 personnes, cible
`bank_account`). Les données **ne sont pas versionnées** : téléchargez-les depuis Zindi et placez-les dans
`data/raw/` :

```
data/raw/inclusion-financiere-en-afrique.csv
data/raw/VariableDefinitions.csv
```

## Installation

```bash
git clone git@github.com:kmache/financial-inclusion-africa.git
cd financial-inclusion-africa
uv sync
```

| Commande | Usage |
|---|---|
| `uv run jupyter lab` | Ouvrir les notebooks avec l'environnement du projet |
| `uv run pytest` | Lancer les tests de `tests/` |
| `uv run ruff check --fix src tests` | Vérifier et corriger le style du code |
| `uv sync` | Recréer exactement l'environnement à partir de `uv.lock` |

## Structure

```
financial-inclusion-africa/
├── data/raw/                      # données brutes (non versionnées)
├── data/processed/                # fichiers intermédiaires (non versionnés)
├── notebooks/                     # 01 → 06, exécutables de haut en bas
├── src/financial_inclusion_africa/
│   ├── config.py                  # chemins, graine, coûts métier
│   ├── data.py                    # chargement et nettoyage
│   ├── features.py                # feature engineering et préprocesseurs
│   └── evaluate.py                # ECE, intervalle bootstrap, coût
├── reports/figures/               # figures exportées pour le README
└── tests/
```

## Règles

`random_state = 42` partout · le test reste scellé jusqu'à l'étape 10 · même objet `cv` pour tous
les modèles · toujours la moyenne ± l'écart-type · une étape, un commit.
