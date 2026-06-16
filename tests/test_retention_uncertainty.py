"""Stage 1 retention-uncertainty tests: carry the full posterior end to end.

These cover the three things Stage 1 added so the optimizer's retention cost is
an uncertainty-aware, honestly-labelled quantity rather than a discarded point:

  * :func:`kairos.optimize.objective.conservative_impact` -- the risk-aware
    valuation (default == point estimate; conservative mode values shedding
    higher), with the clamps and the NaN-degrades-to-point guarantee.
  * :func:`kairos.model.measure.confidence_label` -- the high/medium/low logic
    that needs BOTH the n floor and a tight interval for "high".
  * The full posterior round-trip: ``write_coefficients_json`` ->
    ``read_coefficients_detail`` -> :class:`PosteriorImpactModel` ->
    ``estimate_for``, plus ``load_impact_model`` threading the detail end to end,
    while ``read_coefficients_json`` stays the flat back-compat reader.
"""

from __future__ import annotations

import json

import pytest

from kairos.model.impact import (
    AssumptionImpactModel,
    PosteriorImpactModel,
    RetentionEstimate,
    load_impact_model,
)
from kairos.model.measure import (
    CoefficientDetail,
    MeasuredCoefficient,
    confidence_label,
    read_coefficients_detail,
    read_coefficients_json,
    write_coefficients_json,
)
from kairos.optimize.objective import conservative_impact
from kairos.optimize.pricing import OptimizerAssumptions


# ---------------------------------------------------------------------------
# conservative_impact: the risk-aware retention valuation.
# ---------------------------------------------------------------------------


def test_conservative_impact_lambda_zero_is_point_estimate() -> None:
    # Default risk_lambda == 0.0 must reproduce the point estimate exactly, so
    # nothing in the optimizer changes unless a risk preference is opted into.
    assert conservative_impact(-0.03, -0.05, -0.01) == -0.03
    assert conservative_impact(-0.03, -0.05, -0.01, risk_lambda=0.0) == -0.03


def test_conservative_impact_lambda_one_is_lower_bound() -> None:
    # Full risk aversion values the break at the worst plausible cost: the more
    # negative (lower) credible bound, regardless of bound ordering.
    assert conservative_impact(-0.03, -0.05, -0.01, risk_lambda=1.0) == -0.05
    assert conservative_impact(-0.03, -0.01, -0.05, risk_lambda=1.0) == -0.05
    # The lower bound also includes the point when the point is below the interval.
    assert conservative_impact(-0.06, -0.05, -0.01, risk_lambda=1.0) == -0.06


def test_conservative_impact_partial_lambda_between_point_and_bound() -> None:
    # 0 < lambda < 1 applies a partial variance penalty: point - lambda*half_width.
    # half_width for [-0.05, -0.01] is 0.02; lambda 0.5 -> -0.03 - 0.5*0.02 = -0.04.
    value = conservative_impact(-0.03, -0.05, -0.01, risk_lambda=0.5)
    assert value == pytest.approx(-0.04)
    # It sits strictly between the point and the full lower bound.
    point = -0.03
    bound = -0.05
    assert bound < value < point


def test_conservative_impact_never_more_optimistic_than_point() -> None:
    # Even with a wildly asymmetric interval the conservative value never rises
    # above the point estimate.
    for lam in (0.0, 0.25, 0.5, 0.9, 1.0):
        value = conservative_impact(-0.02, -0.10, 0.05, risk_lambda=lam)
        assert value <= -0.02 + 1e-12


def test_conservative_impact_clamped_non_positive() -> None:
    # A positive point (the data showed no shedding) is clamped to 0, never valued
    # as a retention gain the optimizer could chase.
    assert conservative_impact(0.04, 0.02, 0.06, risk_lambda=0.0) == 0.0
    assert conservative_impact(0.04, 0.02, 0.06, risk_lambda=0.5) == 0.0


def test_conservative_impact_nan_interval_degrades_to_point() -> None:
    # A non-finite interval must degrade to the point estimate, never fabricate a
    # cost.
    assert conservative_impact(-0.03, float("nan"), -0.01, risk_lambda=1.0) == -0.03
    assert conservative_impact(-0.03, -0.05, float("nan"), risk_lambda=0.7) == -0.03


def test_conservative_impact_negative_lambda_raises() -> None:
    with pytest.raises(ValueError):
        conservative_impact(-0.03, -0.05, -0.01, risk_lambda=-0.1)


# ---------------------------------------------------------------------------
# confidence_label: high needs BOTH the n floor AND a tight interval.
# ---------------------------------------------------------------------------


def test_confidence_high_needs_both_n_and_tight_interval() -> None:
    # Many breaks AND a tight interval -> high.
    assert confidence_label(80, -0.031, -0.029) == "high"


def test_confidence_many_breaks_wide_interval_is_medium() -> None:
    # A large n but a wide interval (half-width 0.02 < width here) is only medium,
    # never high: width matters, not just count.
    assert confidence_label(200, -0.08, -0.01) == "medium"


def test_confidence_few_breaks_tight_interval_capped_at_medium() -> None:
    # A tight interval on a handful of breaks (which the fragile normal SE can
    # produce) is held to medium at best by the n floor -- never a false high.
    label = confidence_label(20, -0.031, -0.029)
    assert label == "medium"
    # Below the medium n floor it drops to low even with a tight interval.
    assert confidence_label(5, -0.031, -0.029) == "low"


def test_confidence_inverted_or_nan_interval_is_low() -> None:
    # A NaN bound degrades to low, never a false high.
    assert confidence_label(100, float("nan"), -0.01) == "low"
    # An inverted interval is handled by absolute width, but a wide one stays low.
    assert confidence_label(3, -0.2, -0.01) == "low"


# ---------------------------------------------------------------------------
# The new richer JSON loader + flat back-compat reader.
# ---------------------------------------------------------------------------


def _sample_coefficients() -> dict[str, MeasuredCoefficient]:
    return {
        # A well-measured, tight cell -> high confidence.
        "news_first_short": MeasuredCoefficient(
            channel_name="news_first_short",
            coefficient=-0.030,
            raw_delta=-0.030,
            n=80,
            ci_low=-0.031,
            ci_high=-0.029,
        ),
        # A thin cell with a wide interval -> low confidence.
        "drama_last_long": MeasuredCoefficient(
            channel_name="drama_last_long",
            coefficient=-0.050,
            raw_delta=-0.050,
            n=4,
            ci_low=-0.12,
            ci_high=0.01,
        ),
    }


def test_detail_round_trip_preserves_interval_and_n(tmp_path) -> None:
    path = tmp_path / "coeffs.json"
    write_coefficients_json(path, _sample_coefficients(), metadata={"window": "2y"})

    detail = read_coefficients_detail(path)
    assert set(detail) == {"news_first_short", "drama_last_long"}

    high = detail["news_first_short"]
    assert isinstance(high, CoefficientDetail)
    assert high.coefficient == pytest.approx(-0.030)
    assert high.ci_low == pytest.approx(-0.031)
    assert high.ci_high == pytest.approx(-0.029)
    assert high.n == 80
    assert high.confidence == "high"

    low = detail["drama_last_long"]
    assert low.n == 4
    assert low.confidence == "low"


def test_read_coefficients_json_is_flat_back_compat(tmp_path) -> None:
    path = tmp_path / "coeffs.json"
    write_coefficients_json(path, _sample_coefficients())

    flat = read_coefficients_json(path)
    # The flat reader returns ONLY the point map -- no interval, no n.
    assert flat == {
        "news_first_short": pytest.approx(-0.030),
        "drama_last_long": pytest.approx(-0.050),
    }
    assert all(isinstance(v, float) for v in flat.values())


def test_detail_missing_block_degrades_to_point_low(tmp_path) -> None:
    # A coefficients file with only the flat map (no detail) degrades each cell to
    # a point estimate with no interval, n=0 and low confidence.
    path = tmp_path / "flat_only.json"
    path.write_text(
        json.dumps({"coefficients": {"sport_mid_short": -0.04}}), encoding="utf-8"
    )
    detail = read_coefficients_detail(path)
    cell = detail["sport_mid_short"]
    assert cell.coefficient == pytest.approx(-0.04)
    assert cell.ci_low == cell.ci_high == pytest.approx(-0.04)
    assert cell.n == 0
    assert cell.confidence == "low"


def test_read_coefficients_detail_missing_file_is_empty(tmp_path) -> None:
    assert read_coefficients_detail(tmp_path / "nope.json") == {}


# ---------------------------------------------------------------------------
# The full posterior reaches the optimizer via estimate_for.
# ---------------------------------------------------------------------------


def test_posterior_model_estimate_for_carries_interval_and_confidence() -> None:
    coeffs = {"news_first_short": -0.030, "drama_last_long": -0.050}
    detail = {
        "news_first_short": RetentionEstimate(
            coefficient=-0.030, ci_low=-0.031, ci_high=-0.029, n=80, confidence="high",
        ),
    }
    model = PosteriorImpactModel(coeffs, default=-0.03, source="measured", detail=detail)
    assert model.has_detail

    est = model.estimate_for("news", "first", "short")
    assert est.coefficient == pytest.approx(-0.030)
    assert est.ci_low == pytest.approx(-0.031)
    assert est.ci_high == pytest.approx(-0.029)
    assert est.n == 80
    assert est.confidence == "high"


def test_posterior_model_cell_without_detail_degrades() -> None:
    # A channel present in coefficients but with no detail entry degrades to the
    # point coefficient, no interval, n=0, low confidence.
    coeffs = {"drama_last_long": -0.050}
    model = PosteriorImpactModel(coeffs, default=-0.03, source="measured")
    est = model.estimate_for("drama", "last", "long")
    assert est.coefficient == pytest.approx(-0.050)
    assert est.ci_low == est.ci_high == pytest.approx(-0.050)
    assert est.n == 0
    assert est.confidence == "low"


def test_assumption_model_estimate_for_is_degenerate_low_confidence() -> None:
    model = AssumptionImpactModel(OptimizerAssumptions())
    est = model.estimate_for("news", "first", "short")
    point = OptimizerAssumptions().retention_impact_per_break
    assert est.coefficient == pytest.approx(point)
    assert est.ci_low == est.ci_high == pytest.approx(point)
    assert est.n == 0
    assert est.confidence == "low"


def test_load_impact_model_threads_detail_end_to_end(tmp_path) -> None:
    # Prove the detail round-trips through load_impact_model in a real run: write a
    # coefficients JSON (with detail) and load a model whose estimate_for carries
    # the interval + confidence, not just the point.
    coeff_path = tmp_path / "tv_break_coefficients.json"
    write_coefficients_json(coeff_path, _sample_coefficients())

    model = load_impact_model(
        tmp_path / "missing_posterior.pkl",
        coefficients_path=coeff_path,
    )
    assert isinstance(model, PosteriorImpactModel)
    assert model.source == "measured"
    assert model.has_detail

    est = model.estimate_for("news", "first", "short")
    assert est.coefficient == pytest.approx(-0.030)
    assert est.ci_low == pytest.approx(-0.031)
    assert est.ci_high == pytest.approx(-0.029)
    assert est.n == 80
    assert est.confidence == "high"

    # The thin cell still loads, with its low-confidence label preserved.
    thin = model.estimate_for("drama", "last", "long")
    assert thin.n == 4
    assert thin.confidence == "low"


def test_load_impact_model_falls_back_to_assumption_without_coefficients(tmp_path) -> None:
    # With no coefficients file and no posterior, the honest fallback is the
    # assumption model: a degenerate low-confidence estimate.
    model = load_impact_model(
        tmp_path / "missing_posterior.pkl",
        coefficients_path=tmp_path / "missing_coeffs.json",
    )
    assert isinstance(model, AssumptionImpactModel)
    est = model.estimate_for("news", "first", "short")
    assert est.confidence == "low"
    assert est.n == 0
