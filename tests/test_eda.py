import pandas as pd
import pytest

from financial_inclusion_africa.eda import (
    cramers_v,
    duplicate_profiles,
    rare_categories,
    target_rate,
)


@pytest.fixture
def toy():
    return pd.DataFrame(
        {
            "uniqueid": ["a", "b", "c", "d", "e"],
            "pays": ["K", "K", "K", "R", "R"],
            "tel": ["Yes", "Yes", "No", "No", "No"],
            "bank_account": [1, 0, 0, 0, 1],
        }
    )


def test_target_rate(toy):
    t = target_rate(toy, "pays")
    assert t.loc["K", "effectif"] == 3
    assert t.loc["K", "taux"] == pytest.approx(1 / 3)
    assert t["part_effectif"].sum() == pytest.approx(1.0)


def test_cramers_v_bounds():
    x = pd.Series(list("aabb") * 10)
    assert cramers_v(x, x) == pytest.approx(1.0)
    assert cramers_v(x, pd.Series(list("abab") * 10)) == pytest.approx(0.0)


def test_duplicate_profiles_counts_contradictions(toy):
    d = duplicate_profiles(toy)
    # profils (pays, tel) : (K, Yes) x2 contradictoire ; (R, No) x2 contradictoire ; (K, No) seul
    assert d["profils_repetes"] == 2
    assert d["lignes_dans_profils_repetes"] == 4
    assert d["profils_contradictoires"] == 2
    assert d["erreurs_irreductibles"] == 2


def test_rare_categories(toy):
    r = rare_categories(toy, ["pays"], threshold=3)
    assert r.to_dict("records") == [
        {"variable": "pays", "modalite": "R", "effectif": 2}
    ]
