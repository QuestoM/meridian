"""Tests for the config-driven pricing model and optimizer assumptions.

These read the real ``config/optimization_weights.yaml`` so the typed view is
proven to match the numbers the owner actually edits.
"""

from __future__ import annotations

import pytest

from kairos.optimize.pricing import OptimizerAssumptions, PriceBreakdown, PricingModel


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


def test_price_slot_default_is_identity_to_segment_premium(pricing: PricingModel) -> None:
    """The composition primitive must reproduce the legacy premium exactly.

    With only the program and day layers active (the default), price_slot's
    total_premium has to equal segment_premium for every class/day, so unifying
    on price_slot changes no revenue number.
    """
    for pricing_class in ("News", "PrimeShow1", "PrimeShow2", "Drama", "Other"):
        for weekday in range(1, 8):
            breakdown = pricing.price_slot(pricing_class=pricing_class, weekday_iso=weekday)
            expected = pricing.segment_premium(pricing_class=pricing_class, weekday_iso=weekday)
            assert breakdown.total_premium == pytest.approx(expected)
            assert breakdown.final_cpp == pytest.approx(pricing.base_price * expected)


def test_price_slot_default_carries_only_named_program_and_day_layers(pricing: PricingModel) -> None:
    breakdown = pricing.price_slot(pricing_class="News", weekday_iso=6)
    assert isinstance(breakdown, PriceBreakdown)
    assert [layer.name for layer in breakdown.layers] == ["program", "day"]
    assert all(layer.source == "rate_card" for layer in breakdown.layers)
    # Every line traces to a real configured premium.
    by_name = {layer.name: layer.multiplier for layer in breakdown.layers}
    assert by_name["program"] == 1.15
    assert by_name["day"] == 1.20


def test_price_slot_position_layer_is_off_by_default(pricing: PricingModel) -> None:
    """Passing a position must not change the price unless the layer is enabled."""
    base_only = pricing.price_slot(pricing_class="News", weekday_iso=6, position=1, break_size=5)
    assert [layer.name for layer in base_only.layers] == ["program", "day"]
    with_position = pricing.price_slot(
        pricing_class="News", weekday_iso=6, position=1, break_size=5, enable_position=True
    )
    assert [layer.name for layer in with_position.layers] == ["program", "day", "position"]
    # Enabling the layer multiplies in the first-position premium (1.30), a real change.
    assert with_position.total_premium == pytest.approx(base_only.total_premium * 1.30)


def test_price_slot_ad_type_layer_is_off_by_default(pricing: PricingModel) -> None:
    base_only = pricing.price_slot(pricing_class="News", weekday_iso=1, ad_type="חסות")
    assert [layer.name for layer in base_only.layers] == ["program", "day"]
    with_ad_type = pricing.price_slot(
        pricing_class="News", weekday_iso=1, ad_type="חסות", enable_ad_type=True
    )
    assert with_ad_type.total_premium == pytest.approx(base_only.total_premium * 1.05)


def test_price_slot_accepts_per_advertiser_base(pricing: PricingModel) -> None:
    breakdown = pricing.price_slot(pricing_class="PrimeShow2", weekday_iso=1, base_cpp=80.0)
    assert breakdown.base_cpp == 80.0
    # PrimeShow2 x Monday is 1.00 x 1.00, so the final CPP is just the negotiated base.
    assert breakdown.final_cpp == pytest.approx(80.0)


def test_from_config_empty_overrides_is_identity_to_from_yaml() -> None:
    """No operator overrides must reproduce the YAML rate card exactly."""
    yaml_model = PricingModel.from_yaml()
    config_model = PricingModel.from_config({})
    assert config_model.base_price == yaml_model.base_price
    assert config_model.program_type_premiums == yaml_model.program_type_premiums
    assert config_model.position_premiums == yaml_model.position_premiums
    # Activation flags ship OFF, so the live premium is unchanged.
    assert config_model.enable_position is False
    assert config_model.enable_ad_type is False
    assert config_model.enable_show is False


def test_from_config_merges_a_single_premium_without_dropping_the_table() -> None:
    """A one-key override edits that value and leaves the rest of the table intact."""
    model = PricingModel.from_config(
        {"premiums": {"program_type": {"News": 1.25}}}
    )
    assert model.program_premium("News") == 1.25       # the operator's edit
    assert model.program_premium("PrimeShow1") == 1.10  # untouched YAML value


def test_from_config_activates_a_layer_via_overrides() -> None:
    model = PricingModel.from_config({"pricing_activation": {"position": True}})
    assert model.enable_position is True
    breakdown = model.price_slot(
        pricing_class="News", weekday_iso=1, position=1, break_size=5
    )
    assert [layer.name for layer in breakdown.layers] == ["program", "day", "position"]
    assert breakdown.total_premium == pytest.approx(1.15 * 1.00 * 1.30)


def test_from_config_position_override_with_string_key_takes_effect() -> None:
    """A dashboard edit arrives as a JSON-string position key ("2"); it must apply.

    JSON object keys are always strings, so an operator edit to position 2 reaches the
    engine as {"2": 1.5}. position_premium looks up the int 2, so from_weights has to
    coerce the numeric key to int or the edit would silently no-op. The named keys
    (last, default_middle) must stay strings.
    """
    model = PricingModel.from_config(
        {"premiums": {"position_in_break": {"2": 1.5}}, "pricing_activation": {"position": True}}
    )
    assert model.position_premium(2, break_size=5) == 1.5      # the operator's edit took effect
    assert model.position_premium(1, break_size=5) == 1.30     # untouched YAML value
    breakdown = model.price_slot(pricing_class="News", weekday_iso=1, position=2, break_size=5)
    assert breakdown.total_premium == pytest.approx(1.15 * 1.00 * 1.5)


def test_show_premium_layer(pricing: PricingModel) -> None:
    assert pricing.show_premium("Big Brother") == 1.0   # nothing configured, no effect
    model = PricingModel.from_config(
        {"premiums": {"show": {"Big Brother": 1.25}}, "pricing_activation": {"show": True}}
    )
    assert model.show_premium("Big Brother") == 1.25
    breakdown = model.price_slot(pricing_class="News", weekday_iso=1, show="Big Brother")
    assert [layer.name for layer in breakdown.layers] == ["program", "day", "show"]
    assert breakdown.total_premium == pytest.approx(1.15 * 1.00 * 1.25)
    # A show with no configured premium leaves the price unchanged even when on.
    plain = model.price_slot(pricing_class="News", weekday_iso=1, show="Unknown Show")
    assert plain.total_premium == pytest.approx(1.15)


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
