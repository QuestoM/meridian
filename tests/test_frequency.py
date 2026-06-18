"""Tests for ad-frequency and competitive-separation enforcement.

Covers the pure resolution/enforcement layer (most-specific-wins composition,
each limit type, competitive separation, the identity case) and the real-data
wiring through :func:`kairos.export.spots.price_daily_spots`. The pass is pure
and deterministic, so every assertion is exact, not statistical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kairos.export.spots import price_daily_spots
from kairos.optimize._frequency_rules import (
    FrequencyRule,
    FrequencyRuleSet,
    load_frequency_rules,
    resolve_effective,
    rule_from_row,
)
from kairos.optimize.frequency import SpotView, enforce_spots

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"


def _spot(key, adv, break_id, pos, *, campaign="C", ad="A", minute=None):
    return SpotView(
        key=key, advertiser=adv, campaign=campaign, ad=ad,
        break_id=break_id, position=pos, minute=minute,
    )


def _ruleset(*rules):
    return FrequencyRuleSet(rules=list(rules))


# --- composition: most-specific-wins -----------------------------------------


def test_most_specific_rule_wins_ad_over_campaign_over_advertiser_over_default():
    default = FrequencyRule("d", "max_per_break", "default", value=1)
    advertiser = FrequencyRule("a", "max_per_break", "advertiser", advertiser_id="ADV", value=2)
    campaign = FrequencyRule(
        "c", "max_per_break", "campaign", advertiser_id="ADV", campaign="CAM", value=3,
    )
    ad = FrequencyRule(
        "x", "max_per_break", "ad", advertiser_id="ADV", campaign="CAM", ad="AD", value=4,
    )
    rules = [default, advertiser, campaign, ad]
    # An ADV/CAM/AD spot resolves to the ad rule (value 4).
    assert resolve_effective(rules, "ADV", "CAM", "AD").value == 4
    # An ADV/CAM/OTHER spot resolves to the campaign rule (value 3).
    assert resolve_effective(rules, "ADV", "CAM", "OTHER").value == 3
    # An ADV/OTHER spot resolves to the advertiser rule (value 2).
    assert resolve_effective(rules, "ADV", "OTHER", "z").value == 2
    # An unrelated advertiser falls back to the default (value 1).
    assert resolve_effective(rules, "ZZZ", "q", "q").value == 1


def test_resolution_is_deterministic_first_wins_on_ties():
    a = FrequencyRule("a", "max_per_break", "advertiser", advertiser_id="ADV", value=5)
    b = FrequencyRule("b", "max_per_break", "advertiser", advertiser_id="ADV", value=9)
    assert resolve_effective([a, b], "ADV", "C", "A").rule_id == "a"


# --- identity ----------------------------------------------------------------


def test_identity_no_rules_keeps_every_spot_in_order():
    spots = [_spot(i, "ADV", "B1", i) for i in range(5)]
    result = enforce_spots(spots, _ruleset())
    assert result.kept == [0, 1, 2, 3, 4]
    assert result.dropped == []


def test_disabled_rule_is_inert():
    rule = FrequencyRule("d", "max_per_break", "default", value=1, enabled=False)
    spots = [_spot(0, "ADV", "B1", 1), _spot(1, "ADV", "B1", 2)]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 1]


# --- MAX_PER_BREAK -----------------------------------------------------------


def test_max_per_break_drops_the_third_in_a_break_keeps_other_break():
    rule = FrequencyRule("d", "max_per_break", "default", value=2)
    spots = [
        _spot(0, "ADV", "B1", 1), _spot(1, "ADV", "B1", 2),
        _spot(2, "ADV", "B1", 3),   # third in B1 -> dropped
        _spot(3, "ADV", "B2", 1),   # different break -> kept
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 1, 3]
    assert [d.key for d in result.dropped] == [2]
    assert "max_per_break=2" in result.dropped[0].reason


def test_max_per_break_counts_per_advertiser_not_per_ad():
    # default scope counts the advertiser, so two different ads still collide.
    rule = FrequencyRule("d", "max_per_break", "default", value=1)
    spots = [
        _spot(0, "ADV", "B1", 1, ad="A1"),
        _spot(1, "ADV", "B1", 2, ad="A2"),
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0]


# --- MAX_PER_DAY -------------------------------------------------------------


def test_max_per_day_caps_across_breaks():
    rule = FrequencyRule("d", "max_per_day", "default", value=2)
    spots = [
        _spot(0, "ADV", "B1", 1), _spot(1, "ADV", "B2", 1),
        _spot(2, "ADV", "B3", 1),   # third of the day -> dropped
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 1]
    assert result.dropped[0].limit_type == "max_per_day"


# --- MAX_CONSECUTIVE ---------------------------------------------------------


def test_max_consecutive_blocks_adjacent_positions():
    rule = FrequencyRule("d", "max_consecutive", "default", value=1)
    spots = [
        _spot(0, "ADV", "B1", 1),
        _spot(1, "ADV", "B1", 2),   # adjacent to pos 1 -> would exceed run of 1
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0]
    assert result.dropped[0].limit_type == "max_consecutive"


def test_max_consecutive_allows_non_adjacent():
    rule = FrequencyRule("d", "max_consecutive", "default", value=1)
    spots = [_spot(0, "ADV", "B1", 1), _spot(1, "ADV", "B1", 5)]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 1]


# --- MIN_SEPARATION ----------------------------------------------------------


def test_min_separation_minutes_drops_too_close():
    rule = FrequencyRule("d", "min_separation", "default", value=10, unit="minutes")
    spots = [
        _spot(0, "ADV", "B1", 1, minute=100.0),
        _spot(1, "ADV", "B2", 1, minute=105.0),   # 5 min < 10 -> dropped
        _spot(2, "ADV", "B3", 1, minute=120.0),   # 20 min from first -> kept
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 2]
    assert "min_separation" in result.dropped[0].reason


def test_min_separation_positions_within_break():
    rule = FrequencyRule("d", "min_separation", "default", value=3, unit="positions")
    spots = [
        _spot(0, "ADV", "B1", 1),
        _spot(1, "ADV", "B1", 2),   # 1 position < 3 -> dropped
        _spot(2, "ADV", "B1", 6),   # 5 positions from first -> kept
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 2]


# --- COMPETITIVE_SEPARATION --------------------------------------------------


def test_competitive_separation_keeps_two_banks_out_of_one_break():
    rule = FrequencyRule(
        "comp", "competitive_separation", "default",
        competing_group="banks", members=frozenset({"BANK_A", "BANK_B"}),
        value=0, unit="positions",
    )
    spots = [
        _spot(0, "BANK_A", "B1", 1),
        _spot(1, "BANK_B", "B1", 2),   # competitor in same break -> dropped
        _spot(2, "BANK_B", "B2", 1),   # different break -> kept
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 2]
    assert result.dropped[0].limit_type == "competitive_separation"
    assert "BANK_B" in result.dropped[0].advertiser


def test_competitive_separation_minutes_window():
    rule = FrequencyRule(
        "comp", "competitive_separation", "default",
        competing_group="banks", members=frozenset({"BANK_A", "BANK_B"}),
        value=15, unit="minutes",
    )
    spots = [
        _spot(0, "BANK_A", "B1", 1, minute=100.0),
        _spot(1, "BANK_B", "B2", 1, minute=108.0),   # 8 min < 15 -> dropped
        _spot(2, "BANK_B", "B3", 1, minute=130.0),   # 30 min -> kept
    ]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 2]


def test_competitive_separation_ignores_same_advertiser():
    # Two spots of the SAME bank are a frequency concern, not a competitive one.
    rule = FrequencyRule(
        "comp", "competitive_separation", "default",
        competing_group="banks", members=frozenset({"BANK_A", "BANK_B"}),
        value=0, unit="positions",
    )
    spots = [_spot(0, "BANK_A", "B1", 1), _spot(1, "BANK_A", "B1", 2)]
    result = enforce_spots(spots, _ruleset(rule))
    assert result.kept == [0, 1]


# --- parsing / honesty -------------------------------------------------------


def test_malformed_rows_are_skipped_with_a_reason():
    bad_type, reason = rule_from_row({"rule_id": "r", "limit_type": "bogus", "value": "1"})
    assert bad_type is None and "unknown limit_type" in reason
    bad_value, reason = rule_from_row(
        {"rule_id": "r", "limit_type": "max_per_break", "value": "x"})
    assert bad_value is None and "non-numeric" in reason
    bad_comp, reason = rule_from_row(
        {"rule_id": "r", "limit_type": "competitive_separation", "members": "ONE", "value": "0"})
    assert bad_comp is None and ">=2 members" in reason


def test_shipped_csv_loads_a_conservative_default():
    ruleset = load_frequency_rules()
    assert ruleset.skipped == []
    by_break = ruleset.by_limit("max_per_break")
    assert any(r.scope == "default" and r.value == 1 for r in by_break)


# --- real-data wiring through price_daily_spots ------------------------------


def _load_daily():
    from kairos.data.loaders import load_daily_input
    return load_daily_input(DAILY)


def test_real_data_identity_with_empty_ruleset():
    daily = _load_daily()
    result = price_daily_spots(daily, frequency=FrequencyRuleSet())
    # No frequency rule -> nothing dropped by frequency, every priced spot kept.
    assert result.frequency_dropped == []
    assert len(result.priced) == 175


def test_real_data_default_removes_in_break_repetition_and_reports_it():
    daily = _load_daily()
    seeded = price_daily_spots(daily)  # shipped default: 1 per advertiser per break
    assert len(seeded.frequency_dropped) > 0
    # Conservation: identity priced count == kept + frequency-dropped.
    identity = price_daily_spots(daily, frequency=FrequencyRuleSet())
    assert len(identity.priced) == len(seeded.priced) + len(seeded.frequency_dropped)
    # No break holds two spots of one advertiser after enforcement.
    seen = set()
    for spot in seeded.priced:
        marker = (spot.break_id, spot.advertiser)
        assert marker not in seen, f"duplicate {marker} survived"
        seen.add(marker)
    # Every drop carries an explicit, non-empty reason.
    assert all(d.reason for d in seeded.frequency_dropped)
