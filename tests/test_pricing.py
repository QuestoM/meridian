"""Tests for the config-driven pricing model and optimizer assumptions.

These read the real ``config/optimization_weights.yaml`` so the typed view is
proven to match the numbers the owner actually edits.
"""

from __future__ import annotations

import pytest

from kairos.optimize.pricing import OptimizerAssumptions, PricingModel


@pytest.fixture(scope="module")
def pricing() -> PricingModel:
    return PricingModel.from_yaml()


def test_base_price_matches_config(pricing: PricingModel) -> None:
    assert pricing.base_price == 60.0


def test_program_premiums(pricing: PricingModel) -> None:
    assert pricing.program_premium("News") == 1.15
    assert pricing.program_premium("PrimeShow1") == 1.10
    assert pricing.program_premium("PrimeShow2") == 1.00


def test_unknown_program_falls_back_to_other(pricing: PricingModel) -> None:
    assert pricing.program_premium("Drama") == 0.80      # the configured Other value
    assert pricing.program_premium("anything") == 0.80


def test_ad_type_premiums(pricing: PricingModel) -> None:
    assert pricing.ad_type_premium("חסות") == 1.05       # sponsorship
    assert pricing.ad_type_premium("פרסומת") == 1.00     # standard spot
    assert pricing.ad_type_premium("unknown") == 1.0     # safe default


def test_day_premiums(pricing: PricingModel) -> None:
    assert pricing.day_premium(1) == 1.00   # Monday
    assert pricing.day_premium(4) == 1.05   # Thursday
    assert pricing.day_premium(5) == 1.15   # Friday
    assert pricing.day_premium(6) == 1.20   # Saturday
    assert pricing.day_premium(99) == 1.0   # safe default


def test_position_premiums(pricing: PricingModel) -> None:
    assert pricing.position_premium(1, break_size=5) == 1.30
    assert pricing.position_premium(2, break_size=5) == 1.15
    assert pricing.position_premium(3, break_size=5) == 1.05
    # Position 4 of 5 is neither top-three nor last, so it is a middle position.
    assert pricing.position_premium(4, break_size=5) == 1.00
    # The last position beyond the third gets the last-in-break premium.
    assert pricing.position_premium(5, break_size=5) == 1.20
    assert pricing.position_premium(4, break_size=4) == 1.20


def test_position_zero_is_rejected(pricing: PricingModel) -> None:
    with pytest.raises(ValueError):
        pricing.position_premium(0, break_size=3)


def test_segment_premium_combines_program_and_day(pricing: PricingModel) -> None:
    assert pricing.segment_premium(pricing_class="News", weekday_iso=6) == pytest.approx(1.15 * 1.20)
    assert pricing.segment_premium(pricing_class="Drama", weekday_iso=1) == pytest.approx(0.80 * 1.00)


def test_from_weights_reads_a_plain_dict() -> None:
    model = PricingModel.from_weights({
        "base_price_per_second_per_tvr_point": 50.0,
        "premiums": {
            "program_type": {"News": 1.2, "Other": 0.9},
            "day_of_week": {5: 1.1},
            "position_in_break": {1: 1.4, "last": 1.25, "default_middle": 1.0},
        },
    })
    assert model.base_price == 50.0
    assert model.program_premium("News") == 1.2
    assert model.day_premium(5) == 1.1
    assert model.position_premium(1, break_size=2) == 1.4


def test_negative_base_price_is_rejected() -> None:
    with pytest.raises(ValueError):
        PricingModel(base_price_per_second_per_tvr_point=-1.0)


def test_assumption_defaults_are_sane() -> None:
    assumptions = OptimizerAssumptions()
    assert assumptions.retention_baseline == 1.0
    assert assumptions.retention_impact_per_break <= 0
    assert assumptions.default_break_length_seconds > 0
    assert 0.0 <= assumptions.revenue_weight <= 1.0


def test_positive_retention_impact_is_rejected() -> None:
    with pytest.raises(ValueError):
        OptimizerAssumptions(retention_impact_per_break=0.05)


def test_out_of_range_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        OptimizerAssumptions(revenue_weight=1.5)
