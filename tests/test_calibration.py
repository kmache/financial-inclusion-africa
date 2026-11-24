import numpy as np
import pytest

from financial_inclusion_africa.evaluate import (
    calibration_metrics,
    ece_noise_floor,
    expected_calibration_error,
    expected_counts,
    reliability_table,
)


def test_ece_slide_42_example():
    # Slide 42 : bins A (500, conf 0,10, acc 0,08), B (300, 0,45, 0,50), C (200, 0,85, 0,95)
    p = np.r_[np.full(500, 0.10), np.full(300, 0.45), np.full(200, 0.85)]
    y = np.r_[
        np.repeat([1, 0], [40, 460]),
        np.repeat([1, 0], [150, 150]),
        np.repeat([1, 0], [190, 10]),
    ]
    assert expected_calibration_error(y, p) == pytest.approx(0.045)
    table = reliability_table(y, p)
    assert table["effectif"].tolist() == [500, 300, 200]
    assert table["contribution à l'ECE"].sum() == pytest.approx(0.045)


def test_ece_perfect_and_extreme_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = rng.uniform(size=p.size) < p  # calibré par construction
    assert expected_calibration_error(y, p) < 0.005
    # p = 1 tombe dans le dernier intervalle, sans erreur d'indice
    assert expected_calibration_error([1, 0], [1.0, 0.0]) == pytest.approx(0.0)


def test_calibration_metrics_and_expected_counts():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6])
    m = calibration_metrics(y, p)
    assert m["ROC-AUC"] == pytest.approx(1.0)
    assert m["prévalence prédite"] == pytest.approx(p.mean())
    out = expected_counts(y, p, ["A", "A", "A", "B", "B", "B"])
    assert out.loc["A", "réel"] == 1 and out.loc["A", "attendu"] == pytest.approx(1.1)
    assert out.loc["Total", "réel"] == 3
    assert out.loc["B", "erreur relative"] == pytest.approx((1.6 - 2) / 2)
    assert out.loc["B", "écart-type attendu"] == pytest.approx(
        np.sqrt(0.21 + 0.21 + 0.24)
    )


def test_ece_noise_floor_is_small_but_positive():
    p = np.random.default_rng(1).beta(1, 6, size=5000)
    floor = ece_noise_floor(p, n_sim=50)
    assert 0 < floor["moyenne"] < floor["95e centile"] < 0.03


def test_bootstrap_ci_contains_value_and_shrinks_with_n():
    from sklearn.metrics import roc_auc_score

    from financial_inclusion_africa.evaluate import bootstrap_ci, bootstrap_metrics

    rng = np.random.default_rng(0)
    widths = []
    for n in (400, 4000):
        p = rng.uniform(size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=200)
        assert lo < roc_auc_score(y, p) < hi
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2  # largeur ≈ 1/√n
    table = bootstrap_metrics(
        y, p, {"auc": roc_auc_score, "moyenne": lambda a, b: b.mean()}, n_boot=100
    )
    assert (
        table.loc["auc", "IC 95 % bas"]
        < table.loc["auc", "test"]
        < table.loc["auc", "IC 95 % haut"]
    )
