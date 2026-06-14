"""Full-coverage tests for the Kairos revenue and retention primitives.

Pure-algorithm money math: every branch is exercised, including the guard
clauses, because the optimizer's recommendations depend on these being correct.
"""

from __future__ import annotations

import pytest

from kairos.optimize.objective import (
    break_revenue,
    clamp,
    fixed_revenue,
    predicted_retention,
    retention_adjusted_revenue,
    weighted_objective,
)


def test_clamp() -> None:
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
    with pytest.raises(ValueError):
        clamp(1, 10, 0)


def test_break_revenue_standard_unit() -> None:
    # 5 rating points, 60s (2 units of 30s), CPP 1000 -> 5 * 1000 * 2 = 10000.
    assert break_revenue(5.0, 60.0, 1000.0) == 10000.0


def test_break_revenue_premium_and_custom_unit() -> None:
    assert break_revenue(4.0, 30.0, 500.0, premium=1.5) == 3000.0
    assert break_revenue(2.0, 20.0, 100.0, unit_seconds=10.0) == 400.0


def test_break_revenue_zero_cases() -> None:
    assert break_revenue(0.0, 60.0, 1000.0) == 0.0
    assert break_revenue(5.0, 0.0, 1000.0) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rating_points": -1.0, "duration_seconds": 30.0, "cpp": 100.0},
        {"rating_points": 1.0, "duration_seconds": -1.0, "cpp": 100.0},
        {"rating_points": 1.0, "duration_seconds": 30.0, "cpp": -100.0},
    ],
)
def test_break_revenue_negative_inputs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        break_revenue(**kwargs)


def test_break_revenue_bad_unit_and_premium_raise() -> None:
    with pytest.raises(ValueError):
        break_revenue(1.0, 30.0, 100.0, unit_seconds=0.0)
    with pytest.raises(ValueError):
        break_revenue(1.0, 30.0, 100.0, premium=-0.5)


def test_fixed_revenue() -> None:
    assert fixed_revenue(2500) == 2500.0
    with pytest.raises(ValueError):
        fixed_revenue(-1)


def test_predicted_retention_decreases_with_breaks() -> None:
    # baseline 0.85, each break costs 0.05 retention.
    assert predicted_retention(0.85, -0.05, 0) == 0.85
    assert predicted_retention(0.85, -0.05, 2) == pytest.approx(0.75)


def test_predicted_retention_clamps_to_floor_and_ceiling() -> None:
    assert predicted_retention(0.85, -0.5, 10, floor=0.2) == 0.2
    assert predicted_retention(0.95, 0.5, 1, ceiling=1.0) == 1.0


def test_predicted_retention_negative_breaks_raise() -> None:
    with pytest.raises(ValueError):
        predicted_retention(0.85, -0.05, -1)


def test_retention_adjusted_revenue() -> None:
    assert retention_adjusted_revenue(10000.0, 0.8) == 8000.0
    # retention is clamped to [0, 1] before applying.
    assert retention_adjusted_revenue(10000.0, 1.5) == 10000.0
    with pytest.raises(ValueError):
        retention_adjusted_revenue(-1.0, 0.8)


def test_weighted_objective_extremes() -> None:
    # revenue_weight 1.0 -> only revenue term (normalised, clamped to 1.0).
    assert weighted_objective(100.0, 0.5, revenue_weight=1.0, revenue_scale=100.0) == 1.0
    # revenue_weight 0.0 -> only retention term.
    assert weighted_objective(100.0, 0.5, revenue_weight=0.0, revenue_scale=100.0) == 0.5


def test_weighted_objective_balanced() -> None:
    score = weighted_objective(50.0, 0.6, revenue_weight=0.5, revenue_scale=100.0)
    assert score == pytest.approx(0.5 * 0.5 + 0.5 * 0.6)


def test_weighted_objective_validation() -> None:
    with pytest.raises(ValueError):
        weighted_objective(100.0, 0.5, revenue_weight=1.5, revenue_scale=100.0)
    with pytest.raises(ValueError):
        weighted_objective(100.0, 0.5, revenue_weight=0.5, revenue_scale=0.0)
    with pytest.raises(ValueError):
        weighted_objective(-1.0, 0.5, revenue_weight=0.5, revenue_scale=100.0)
