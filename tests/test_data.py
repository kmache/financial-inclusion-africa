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
