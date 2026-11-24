import pytest

from financial_inclusion_africa import config
from financial_inclusion_africa.data import load_raw

requires_data = pytest.mark.skipif(
    not config.RAW.exists(),
    reason="data/raw/ absent : télécharger les données depuis Zindi",
)


def test_config_paths_point_inside_project():
    assert (config.ROOT / "pyproject.toml").exists()
    assert config.RAW.is_relative_to(config.ROOT)
    assert config.RANDOM_STATE == 42


@requires_data
def test_load_raw_shape_and_columns():
    df = load_raw()
    assert df.shape == (23524, 13)
    assert config.TARGET in df.columns


@requires_data
def test_target_is_binary_without_missing_values():
    y = load_raw()[config.TARGET]
    assert y.notna().all()
    assert set(y.unique()) == {0, 1}


@requires_data
def test_clean_retire_les_colonnes_interdites():
    """Test minimal du guide (12.2) : ni identifiant, ni année, ni cible parmi les variables."""
    from financial_inclusion_africa.data import clean

    X, y = clean(load_raw())
    assert {"uniqueid", "year", "bank_account"}.isdisjoint(X.columns)
    assert set(y.unique()) == {0, 1}


@requires_data
def test_add_features_garde_toutes_les_lignes():
    """Test minimal du guide (12.2). Écart assumé : une éducation inconnue donne NaN (imputée dans
    le pipeline, avec l'indicateur ``education_unknown``), pas -1 comme dans le guide."""
    from financial_inclusion_africa.data import clean
    from financial_inclusion_africa.features import AddFeatures

    X, _ = clean(load_raw())
    Xf = AddFeatures().fit_transform(X)
    assert len(Xf) == len(X)
    rank = Xf["education_rank"]
    assert rank.dropna().between(0, 4).all()
    assert (rank.isna() == (Xf["education_unknown"] == 1)).all()
