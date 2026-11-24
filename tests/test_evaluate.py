import pytest

from financial_inclusion_africa.evaluate import (
    classification_metrics,
    cost_at_threshold,
    full_classification_report,
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


def test_classification_metrics_perfect_model():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]
    m = classification_metrics(y_true, y_proba, threshold=0.5)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == m["recall"] == m["f1"] == m["accuracy"] == 1.0
    assert m["cost"] == 0.0
    assert 0 < m["log_loss"] < 0.25
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)


def test_full_classification_report_lists_both_classes():
    report = full_classification_report(
        [0, 1, 0, 1], [0.1, 0.9, 0.6, 0.4], threshold=0.5
    )
    assert "No (0)" in report and "Yes (1)" in report


def test_cost_curve_matches_cost_at_threshold():
    import numpy as np

    from financial_inclusion_africa.evaluate import cost_curve

    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = rng.uniform(size=500) < p
    grid = [0.0, 0.05, 0.091, 0.5, 1.01]
    expected = [cost_at_threshold(y, p, t) for t in grid]
    assert cost_curve(y, p, grid) == pytest.approx(expected)
    assert cost_curve(y, p, [0.2], 0.5, 2.0)[0] == pytest.approx(
        cost_at_threshold(y, p, 0.2, 0.5, 2.0)
    )
