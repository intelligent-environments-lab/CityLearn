import numpy as np
import pytest

from citylearn.internal.kpi import CityLearnKPIService


def test_equity_relative_benefit_uses_paper_formula():
    value = CityLearnKPIService._equity_relative_benefit_percent(80.0, 100.0)
    assert value == pytest.approx(20.0)


def test_equity_relative_benefit_is_none_for_non_positive_baseline():
    assert CityLearnKPIService._equity_relative_benefit_percent(10.0, 0.0) is None
    assert CityLearnKPIService._equity_relative_benefit_percent(10.0, -5.0) is None


def test_safe_div_handles_near_zero_baseline():
    assert CityLearnKPIService._safe_div(1.0, 1.0e-12) is None
    assert CityLearnKPIService._safe_div(0.0, 1.0e-12) == pytest.approx(1.0)


def test_equity_distribution_metrics_for_equal_benefits():
    metrics = CityLearnKPIService._equity_distribution_metrics(np.array([10.0, 10.0, 10.0], dtype="float64"))
    assert metrics["equity_gini_benefit"] == pytest.approx(0.0)
    assert metrics["equity_cr20_benefit"] == pytest.approx(1.0 / 3.0)
    assert metrics["equity_losers_percent"] == pytest.approx(0.0)


def test_equity_distribution_metrics_for_high_concentration():
    metrics = CityLearnKPIService._equity_distribution_metrics(np.array([100.0, 0.0, 0.0, 0.0, 0.0], dtype="float64"))
    assert metrics["equity_gini_benefit"] == pytest.approx(0.8)
    assert metrics["equity_cr20_benefit"] == pytest.approx(1.0)
    assert metrics["equity_losers_percent"] == pytest.approx(0.0)


def test_equity_distribution_metrics_include_losers():
    metrics = CityLearnKPIService._equity_distribution_metrics(np.array([-10.0, 5.0, 0.0, 4.0], dtype="float64"))
    assert metrics["equity_losers_percent"] == pytest.approx(25.0)


def test_equity_bpr_uses_asset_groups():
    non_negative_benefits = {
        "b1": 2.0,
        "b2": 4.0,
        "b3": 1.0,
        "b4": 3.0,
    }
    groups = {
        "b1": "asset_poor",
        "b2": "asset_rich",
        "b3": "asset_poor",
        "b4": "asset_rich",
    }
    expected = ((2.0 + 1.0) / 2.0) / ((4.0 + 3.0) / 2.0)

    assert CityLearnKPIService._equity_bpr(non_negative_benefits, groups) == pytest.approx(expected)


def test_equity_bpr_returns_none_on_missing_or_invalid_groups():
    non_negative_benefits = {"b1": 2.0, "b2": 4.0}
    missing = {"b1": "asset_poor"}
    invalid = {"b1": "asset_poor", "b2": "unknown"}

    assert CityLearnKPIService._equity_bpr(non_negative_benefits, missing) is None
    assert CityLearnKPIService._equity_bpr(non_negative_benefits, invalid) is None
