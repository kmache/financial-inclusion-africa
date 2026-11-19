import pytest

from financial_inclusion_africa.evaluate import (
    cost_at_threshold,
    theoretical_threshold,
    total_cost,
)


def test_theoretical_threshold_kazipay():
    assert theoretical_threshold(0.5, 5.0) == pytest.approx(1 / 11)
    assert theoretical_threshold(0.5, 2.0) == pytest.approx(0.2)
    assert theoretical_threshold(1.0, 1.0) == pytest.approx(0.5)


def test_theoretical_threshold_rejects_invalid_costs():
    with pytest.raises(ValueError):
        theoretical_threshold(0.0, 0.0)
    with pytest.raises(ValueError):
        theoretical_threshold(-1.0, 5.0)


def test_total_cost_counts_fp_and_fn():
    y_true = [1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0]  # 1 FN, 2 FP
    assert total_cost(y_true, y_pred, cost_fp=0.5, cost_fn=5.0) == pytest.approx(6.0)


def test_cost_at_threshold_contact_nobody_or_everybody():
    y_true = [1, 0, 0, 0]
    y_proba = [0.3, 0.2, 0.1, 0.05]
    # Personne contacté : 1 FN ; tout le monde contacté : 3 FP
    assert cost_at_threshold(y_true, y_proba, 1.01, 0.5, 5.0) == pytest.approx(5.0)
    assert cost_at_threshold(y_true, y_proba, 0.0, 0.5, 5.0) == pytest.approx(1.5)
