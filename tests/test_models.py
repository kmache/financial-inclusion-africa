import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from financial_inclusion_africa.models import (
    CANDIDATES,
    CATBOOST_PARAMS,
    LGBM_PARAMS,
    make_catboost,
    make_lgbm,
)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "country": rng.choice(["Kenya", "Rwanda", "Tanzania", "Uganda"], n),
            "location_type": rng.choice(["Rural", "Urban"], n),
            "cellphone_access": rng.choice(["Yes", "No"], n),
            "gender_of_respondent": rng.choice(["Female", "Male"], n),
            "relationship_with_head": rng.choice(["Head of Household", "Spouse"], n),
            "marital_status": rng.choice(["Married/Living together", "Widowed"], n),
            "education_level": rng.choice(
                ["Primary education", "Tertiary education"], n
            ),
            "job_type": rng.choice(["Self employed", "Formally employed Private"], n),
            "household_size": rng.integers(1, 10, n),
            "age_of_respondent": rng.integers(16, 90, n),
        }
    )


def test_tuned_params_are_used():
    lgbm = make_lgbm()[-1]
    assert lgbm.get_params()["num_leaves"] == LGBM_PARAMS["num_leaves"]
    cat = make_catboost()[-1]
    assert cat.get_params()["depth"] == CATBOOST_PARAMS["depth"]
    assert isinstance(cat.get_params()["cat_features"], tuple)
    assert make_lgbm(n_estimators=10)[-1].get_params()["n_estimators"] == 10


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_candidates_fit_after_clone(name):
    X = _toy()
    y = (X["cellphone_access"] == "Yes").astype(int)
    overrides = {"iterations": 20} if "CatBoost" in name else {}
    factory = CANDIDATES[name]
    model = clone(factory(**overrides) if overrides else factory()).fit(X, y)
    proba = model.predict_proba(X.head(5))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
