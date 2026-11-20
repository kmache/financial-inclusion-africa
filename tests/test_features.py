import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from financial_inclusion_africa import config
from financial_inclusion_africa.data import CAT_COLS, NUM_COLS, clean, load_raw
from financial_inclusion_africa.features import (
    EDU_ORDER,
    AddFeatures,
    NativeCategories,
    catboost_cat_features,
    feature_columns,
    onehot_preprocessor,
    ordinal_preprocessor,
    pipe,
)

requires_data = pytest.mark.skipif(not config.RAW.exists(), reason="data/raw/ absent")

EDUCATION = [*EDU_ORDER, "Other/Dont know/RTA"]


def make_toy(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Jeu synthétique au schéma du vrai jeu (aucune ligne réelle)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "country": rng.choice(["Kenya", "Rwanda", "Tanzania", "Uganda"], n),
            "location_type": rng.choice(["Rural", "Urban"], n),
            "cellphone_access": rng.choice(["Yes", "No"], n),
            "gender_of_respondent": rng.choice(["Female", "Male"], n),
            "relationship_with_head": rng.choice(
                ["Head of Household", "Spouse", "Child"], n
            ),
            "marital_status": rng.choice(
                ["Married/Living together", "Single/Never Married"], n
            ),
            "education_level": rng.choice(list(EDU_ORDER), n),
            "job_type": rng.choice(["Self employed", "Formally employed Private"], n),
            "household_size": rng.integers(1, 10, n),
            "age_of_respondent": rng.integers(16, 90, n),
        }
    )


def test_clean_drops_forbidden_columns_and_encodes_target():
    raw = make_toy(5).assign(
        uniqueid="id", year=2018, bank_account=["Yes", "No"] * 2 + ["No"]
    )
    X, y = clean(raw)
    assert list(X.columns) == CAT_COLS + NUM_COLS
    assert {"uniqueid", "year", "bank_account"}.isdisjoint(X.columns)
    assert y.tolist() == [1, 0, 1, 0, 0]


def test_add_features_is_stateless():
    a, b = make_toy(seed=1), make_toy(seed=2)
    new = make_toy(20, seed=3)
    out_a = AddFeatures().fit(a).transform(new)
    out_b = AddFeatures().fit(b).transform(new)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_education_unknown_is_missing_not_top_rank():
    X = make_toy(4)
    X["education_level"] = [
        "Other/Dont know/RTA",
        "Tertiary education",
        "No formal education",
        "Primary education",
    ]
    X["cellphone_access"] = ["Yes", "Yes", "No", "Yes"]
    out = AddFeatures().fit_transform(X)
    assert np.isnan(out.loc[0, "education_rank"])
    assert out["education_unknown"].tolist() == [1, 0, 0, 0]
    assert out.loc[1, "education_rank"] == 4
    assert out.loc[2, "phone_education"] == 0  # pas de téléphone
    assert out.loc[3, "phone_education"] == 1


def test_custom_flag_controls_extra_columns():
    X = make_toy(10)
    assert "lives_alone" in AddFeatures(custom=True).fit_transform(X)
    assert "lives_alone" not in AddFeatures(custom=False).fit_transform(X)
    names = AddFeatures(custom=True).fit(X).get_feature_names_out()
    assert list(names) == list(AddFeatures(custom=True).fit_transform(X).columns)


def test_rare_category_grouped_and_unknown_category_does_not_crash():
    X = make_toy(300)
    X.loc[:7, "marital_status"] = "Dont know"  # 8 occurrences < min_frequency=20
    cat, num = feature_columns("all")
    pre = onehot_preprocessor(cat, num).fit(AddFeatures().fit_transform(X))
    names = pre.get_feature_names_out()
    assert "cat__marital_status_infrequent_sklearn" in names
    assert "cat__marital_status_Dont know" not in names
    new = AddFeatures().fit_transform(
        make_toy(5, seed=9).assign(marital_status="Jamais vu")
    )
    out = pre.transform(new)  # modalité inconnue : pas d'erreur
    col = list(names).index("cat__marital_status_infrequent_sklearn")
    assert (out[:, col] == 1).all()


def test_imputer_fills_unknown_education_inside_pipeline():
    X = make_toy(100)
    X.loc[:2, "education_level"] = "Other/Dont know/RTA"
    cat, num = feature_columns("all")
    out = onehot_preprocessor(cat, num).fit_transform(AddFeatures().fit_transform(X))
    assert not np.isnan(out).any()


@pytest.mark.parametrize("features", ["none", "guide", "all"])
def test_pipe_fits_and_predicts(features):
    X = make_toy(200)
    y = (X["cellphone_access"] == "Yes").astype(int)
    model = pipe(LogisticRegression(max_iter=2000), features=features).fit(X, y)
    assert model.predict_proba(X.head(3)).shape == (3, 2)


@requires_data
def test_pipe_on_real_rows_smoke():
    X, y = clean(load_raw())
    sample = X.sample(2000, random_state=config.RANDOM_STATE)
    model = pipe(LogisticRegression(max_iter=2000)).fit(sample, y.loc[sample.index])
    proba = model.predict_proba(sample.head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


# --- Encodages pour les modèles à base d'arbres ----------------------------------------------


def test_onehot_without_scaling_keeps_raw_numeric_values():
    X = make_toy(100)
    cat, num = feature_columns("all")
    pre = onehot_preprocessor(cat, num, scale=False).fit(AddFeatures().fit_transform(X))
    out = pre.transform(AddFeatures().fit_transform(X))
    col = list(pre.get_feature_names_out()).index("num__age_of_respondent")
    assert np.array_equal(out[:, col], X["age_of_respondent"].to_numpy())


def test_ordinal_preprocessor_keeps_nan_and_codes_unknown():
    X = make_toy(300)
    X.loc[:2, "education_level"] = "Other/Dont know/RTA"
    Xf = AddFeatures().fit_transform(X)
    cat, num = feature_columns("all")
    pre = ordinal_preprocessor(cat, num).fit(Xf)
    out = pre.transform(Xf)
    assert (
        out["education_rank"].isna().sum() == 3
    )  # pas d'imputation : l'arbre gère NaN
    assert (
        out["age_of_respondent"] == X["age_of_respondent"]
    ).all()  # pas de standardisation
    new = AddFeatures().fit_transform(make_toy(3, seed=5).assign(job_type="Jamais vu"))
    assert (pre.transform(new)["job_type"] == -1).all()


@pytest.mark.parametrize("dtype", ["category", "str"])
def test_native_categories_learns_frequent_and_maps_rare_unknown(dtype):
    X = make_toy(300)
    X.loc[:7, "marital_status"] = "Dont know"  # rare : 8 < 20
    cat, num = feature_columns("all")
    nat = NativeCategories(cat, num, dtype=dtype).fit(AddFeatures().fit_transform(X))
    assert "Dont know" not in nat.categories_["marital_status"]
    new = AddFeatures().fit_transform(
        make_toy(3, seed=7).assign(
            marital_status=["Dont know", "Jamais vue", "Married/Living together"]
        )
    )
    out = nat.transform(new)
    assert out["marital_status"].astype(str).tolist() == [
        "infrequent",
        "infrequent",
        "Married/Living together",
    ]
    assert list(out.columns) == cat + num
    if dtype == "category":
        assert isinstance(out["marital_status"].dtype, pd.CategoricalDtype)


def _tree_models():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "rf_ordinal": (
            RandomForestClassifier(n_estimators=20, random_state=0),
            "ordinal",
        ),
        "hgb_native": (
            HistGradientBoostingClassifier(max_iter=20, random_state=0),
            "native",
        ),
        "xgb_native": (
            XGBClassifier(n_estimators=20, enable_categorical=True),
            "native",
        ),
        "lgbm_native": (LGBMClassifier(n_estimators=20, verbose=-1), "native"),
        "catboost_native": (
            CatBoostClassifier(
                iterations=20, verbose=0, cat_features=catboost_cat_features()
            ),
            "native_str",
        ),
    }


@pytest.mark.parametrize("name", list(_tree_models()))
def test_tree_pipelines_fit_clone_and_handle_unseen_categories(name):
    from sklearn.base import clone

    model, encoding = _tree_models()[name]
    X = make_toy(300)
    y = (X["cellphone_access"] == "Yes").astype(int)
    fitted = clone(pipe(model, encoding=encoding)).fit(
        X, y
    )  # clone : comme cross_validate
    new = make_toy(4, seed=11).assign(marital_status="Jamais vue", job_type="Inconnu")
    proba = fitted.predict_proba(new)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipe_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        pipe(LogisticRegression(), encoding="binaire")
