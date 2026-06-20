"""Tests for per-layer and per-final advertiser overrides (S5-S7).

These prove the owner's model: a scoped advertiser/campaign rule REPLACES exactly
one named layer (the position-2 example), most-specific-wins resolves competing
rules, campaign scope narrows a rule, and an empty override set is the identity.
"""

from __future__ import annotations

import pytest

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine, Condition
from kairos.optimize.layer_overrides import (
    apply_overrides,
    resolve_layer_overrides,
)
from kairos.optimize.pricing import PricingModel


def _engine(*conditions: Condition) -> AdvertiserRuleEngine:
    by_adv: dict[str, list[Condition]] = {}
    for c in conditions:
        by_adv.setdefault(c.advertiser_id, []).append(c)
    return AdvertiserRuleEngine(baselines={}, conditions=by_adv)


def _slot(model: PricingModel):
    # A News/Saturday breakdown with the position layer active, so a position
    # override has a real layer to replace.
    return model.price_slot(
        pricing_class="News", weekday_iso=6, position=2, break_size=5,
        enable_position=True,
    )


def test_no_targeted_rules_is_identity() -> None:
    """A legacy whole-stack rule (empty target_layer) does not touch the breakdown."""
    engine = _engine(Condition(
        advertiser_id="X", rule_id="r1", effect="premium", value=0.9,
        scope_positions=frozenset({"2"}),  # legacy, no target_layer
    ))
    res = resolve_layer_overrides(engine, "X", position=2)
    assert res.layer_overrides == ()
    assert res.final_overrides == ()


def test_replace_position_layer_for_advertiser() -> None:
    """The owner's case: advertiser X gets a 0.90 position-2 price that replaces
    the general position-2 premium, while program and day still stack."""
    model = PricingModel.from_yaml()
    engine = _engine(Condition(
        advertiser_id="X", rule_id="pos2", effect="premium", value=0.90,
        target_layer="position", scope_positions=frozenset({"2"}),
    ))
    res = resolve_layer_overrides(engine, "X", position=2)
    assert [o.target_layer for o in res.layer_overrides] == ["position"]
    assert res.layer_overrides[0].multiplier == pytest.approx(0.90)

    before = _slot(model)
    after = apply_overrides(before, res)
    by_name = {layer.name: layer for layer in after.layers}
    # Program (1.15) and day (1.20) untouched; position replaced from 1.15 to 0.90.
    assert by_name["program"].multiplier == pytest.approx(1.15)
    assert by_name["day"].multiplier == pytest.approx(1.20)
    assert by_name["position"].multiplier == pytest.approx(0.90)
    assert by_name["position"].source == "override:pos2"
    assert after.total_premium == pytest.approx(1.15 * 1.20 * 0.90)


def test_most_specific_wins_per_layer() -> None:
    """A campaign+position rule beats an advertiser-wide position rule (S7)."""
    engine = _engine(
        Condition(advertiser_id="X", rule_id="wide", effect="premium", value=0.95,
                  target_layer="position", scope_positions=frozenset({"2"})),
        Condition(advertiser_id="X", rule_id="camp", effect="premium", value=0.80,
                  target_layer="position", scope_positions=frozenset({"2"}),
                  scope_campaigns=frozenset({"C1"})),
    )
    res = resolve_layer_overrides(engine, "X", position=2, campaign="C1")
    assert len(res.layer_overrides) == 1
    assert res.layer_overrides[0].rule_id == "camp"
    assert res.layer_overrides[0].multiplier == pytest.approx(0.80)
    assert [s.rule_id for s in res.shadowed] == ["wide"]


def test_campaign_scope_excludes_other_campaign() -> None:
    """A campaign-scoped rule must not match a different campaign (S6)."""
    engine = _engine(Condition(
        advertiser_id="X", rule_id="camp", effect="premium", value=0.80,
        target_layer="position", scope_positions=frozenset({"2"}),
        scope_campaigns=frozenset({"C1"}),
    ))
    assert resolve_layer_overrides(engine, "X", position=2, campaign="C2").layer_overrides == ()
    assert len(resolve_layer_overrides(engine, "X", position=2, campaign="C1").layer_overrides) == 1


def test_position_override_injects_layer_when_general_layer_off() -> None:
    """Advertiser X can be given a position price even when the position layer is
    globally off, so the override injects the layer rather than no-opping."""
    model = PricingModel.from_yaml()
    engine = _engine(Condition(
        advertiser_id="X", rule_id="pos2", effect="premium", value=0.90,
        target_layer="position", scope_positions=frozenset({"2"}),
    ))
    base = model.price_slot(pricing_class="News", weekday_iso=6)  # position layer OFF
    assert [l.name for l in base.layers] == ["program", "day"]
    res = resolve_layer_overrides(engine, "X", position=2)
    after = apply_overrides(base, res)
    by_name = {l.name: l for l in after.layers}
    assert "position" in by_name
    assert by_name["position"].multiplier == pytest.approx(0.90)
    # Canonical order keeps position after day.
    assert [l.name for l in after.layers] == ["program", "day", "position"]


def test_final_override_adjusts_whole_price() -> None:
    """A target_layer='final' percent rule multiplies the whole composed price."""
    model = PricingModel.from_yaml()
    engine = _engine(Condition(
        advertiser_id="X", rule_id="disc", effect="premium", value=-10.0,
        mode="percent", target_layer="final",
    ))
    res = resolve_layer_overrides(engine, "X", position=2)
    assert len(res.final_overrides) == 1
    assert res.final_overrides[0].multiplier == pytest.approx(0.90)
    before = model.price_slot(pricing_class="News", weekday_iso=6)
    after = apply_overrides(before, res)
    assert after.total_premium == pytest.approx(before.total_premium * 0.90)
    assert after.layers[-1].name == "final"
    assert after.layers[-1].source == "override:disc"
