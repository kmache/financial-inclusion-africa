import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from financial_inclusion_africa import config
from financial_inclusion_africa.data import load_split, split
from financial_inclusion_africa.evaluate import (
    CV,
    SCORING,
    T_STAR,
    cv_summary,
    fold_scores,
    format_summary,
    one_standard_error,
    paired_difference,
    per_group_scores,
)
from financial_inclusion_africa.features import fold_category_audit

requires_data = pytest.mark.skipif(not config.RAW.exists(), reason="data/raw/ absent")


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "pays": rng.choice(["A", "B", "C", "D"], n),
            "cat": rng.choice(["u", "v"], n),
        }
    )
    X.loc[:4, "cat"] = "rare"
    y = pd.Series((X["x"] + rng.normal(scale=1.0, size=n) > 1.0).astype(int))
    return X, y


def test_split_is_stratified_and_reproducible(toy):
    X, y = toy
    a = split(X, y)
    b = split(X, y)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.01
    assert len(a[1]) == round(len(X) * config.TEST_SIZE)


def test_cv_is_shared_and_stratified(toy):
    X, y = toy
    rates = [y.iloc[va].mean() for _, va in CV.split(X, y)]
    assert max(rates) - min(rates) < 0.01
    assert CV.random_state == config.RANDOM_STATE


def test_cv_summary_flips_negated_scores(toy):
    X, y = toy
    res = cross_validate(
        LogisticRegression(),
        X[["x"]],
        y,
        cv=CV,
        scoring=SCORING,
        return_train_score=True,
    )
    row = cv_summary(res, "lr")
    for metric in ["brier", "logloss", "cost_t"]:
        assert row[(metric, "val", "mean")] > 0
        assert row[(metric, "train", "mean")] > 0
    assert 0 < row[("pr_auc", "val", "mean")] <= 1
    assert ("fit_time", "", "mean") in row.index
    table = format_summary(pd.DataFrame([row]))
    assert "±" in table.loc["lr", "pr_auc"]
    folds = fold_scores(res)
    assert len(folds) == config.N_SPLITS and (folds["brier"] > 0).all()


def test_cost_scorer_matches_contact_everyone(toy):
    X, y = toy
    res = cross_validate(
        DummyClassifier(strategy="prior"), X[["x"]], y, cv=CV, scoring=SCORING
    )
    row = cv_summary(res)
    # la probabilité « prior » dépasse t* : tout le monde est contacté, coût = C_FP × part des négatifs
    assert y.mean() > T_STAR
    expected = config.COST_FP * (1 - y.mean())
    assert row[("cost_t", "val", "mean")] == pytest.approx(expected, abs=0.01)
    assert row[("recall_t", "val", "mean")] == pytest.approx(1.0)


def test_per_group_scores_one_group_per_fold(toy):
    X, y = toy
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    out = per_group_scores(LogisticRegression(), X[["x"]], y, X["pays"], sgkf)
    assert sorted(out["groupe laissé de côté"]) == ["A", "B", "C", "D"]
    assert np.allclose(out["pr_auc / prévalence"], out["pr_auc"] / out["prévalence"])


def test_fold_category_audit_statuses(toy):
    X, y = toy
    audit = fold_category_audit(X, y, CV, cols=["cat"], watch={"cat": ["u"]})
    statuses = set(audit["statut"])
    assert "rare : regroupée" in statuses
    assert "fréquente : colonne propre" in statuses
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    by_country = fold_category_audit(X, y, sgkf, groups=X["pays"], cols=["pays"])
    assert (by_country["statut"] == "inconnue à l'entraînement").sum() == 4


@requires_data
def test_load_split_checkpoint():
    X_train, X_test, y_train, y_test = load_split()
    assert (len(X_train), len(X_test)) == (18_819, 4_705)
    assert abs(y_train.mean() - y_test.mean()) < 0.001


def _fake_res(values):
    return {
        "test_pr_auc": np.asarray(values, dtype=float),
        "fit_time": np.zeros(len(values)),
    }


def test_paired_difference_cancels_fold_difficulty():
    a = _fake_res([0.50, 0.60, 0.55, 0.45, 0.58])
    b = _fake_res([0.51, 0.61, 0.56, 0.46, 0.59])  # +0,01 dans chaque pli
    d = paired_difference(a, b)
    assert d["différence moyenne B − A"] == pytest.approx(0.01)
    assert d["écart-type des différences"] == pytest.approx(0.0, abs=1e-12)
    assert d["écart-type A"] > 0.05
    assert d["plis où B > A"] == 5


def test_one_standard_error_picks_simplest_within_one_se():
    values = [1, 5, 10, 50]
    means = [0.47, 0.57, 0.5725, 0.563]
    stds = [0.03, 0.03, 0.03, 0.03]  # SE = 0,03 / √5 ≈ 0,0134
    rule = one_standard_error(values, means, stds, n_splits=5)
    assert rule["meilleure valeur"] == 10
    assert rule["valeur retenue (1 SE)"] == 50
    assert (
        one_standard_error(values, means, stds, n_splits=5, simplest="min")[
            "valeur retenue (1 SE)"
        ]
        == 5
    )
