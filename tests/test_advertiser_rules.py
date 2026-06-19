"""Tests for the scoped, conditional advertiser-rule engine and its API.

These prove the pure engine semantics (premium stacking, is_allowed precedence,
scope intersection in overlaps) and the conditions CRUD round-trip, all on real
files written to a temporary data directory so nothing touches the seeded store.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kairos.optimize.advertiser_rules import (
    AdvertiserRuleEngine,
    Baseline,
    Condition,
)


def _baseline(advertiser_id: str, **kwargs) -> Baseline:
    return Baseline(advertiser_id=advertiser_id, **kwargs)


def _condition(advertiser_id: str, rule_id: str, effect: str, **kwargs) -> Condition:
    return Condition(advertiser_id=advertiser_id, rule_id=rule_id, effect=effect, **kwargs)


# --- effective_premium ------------------------------------------------------


def test_effective_premium_unknown_advertiser_is_one() -> None:
    engine = AdvertiserRuleEngine()
    assert engine.effective_premium("NOBODY", position=1) == 1.0


def test_effective_premium_uses_baseline_default() -> None:
    engine = AdvertiserRuleEngine(baselines={"A": _baseline("A", default_premium=1.2)})
    assert engine.effective_premium("A") == pytest.approx(1.2)


def test_effective_premium_stacks_matching_rules() -> None:
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=2.0)},
        conditions={
            "A": [
                _condition("A", "r1", "premium", value=1.5, scope_positions=frozenset({"1"})),
                _condition("A", "r2", "premium", value=2.0, scope_dayparts=frozenset({"prime"})),
                # This one should NOT match a position-2 prime spot.
                _condition("A", "r3", "premium", value=10.0, scope_positions=frozenset({"3"})),
            ]
        },
    )
    # position 1, prime: baseline 2.0 * r1 1.5 * r2 2.0 = 6.0 (r3 excluded).
    assert engine.effective_premium("A", position=1, daypart="prime") == pytest.approx(6.0)
    # position 2, daytime: only baseline applies.
    assert engine.effective_premium("A", position=2, daypart="daytime") == pytest.approx(2.0)


def test_any_scope_matches_everything() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=1.3)]}  # all-ANY scope
    )
    assert engine.effective_premium("A", position=7, genre="Drama", daypart="overnight") == pytest.approx(1.3)


# --- is_allowed -------------------------------------------------------------


def test_baseline_allow_positions_limits() -> None:
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", allow_positions=frozenset({"1", "2"}))}
    )
    assert engine.is_allowed("A", position=1)
    assert not engine.is_allowed("A", position=3)


def test_baseline_prime_time_only() -> None:
    engine = AdvertiserRuleEngine(baselines={"A": _baseline("A", prime_time_only=True)})
    assert engine.is_allowed("A", daypart="prime")
    assert not engine.is_allowed("A", daypart="daytime")
    assert not engine.is_allowed("A", daypart=None)


def test_forbid_blocks_and_overrides_require() -> None:
    engine = AdvertiserRuleEngine(
        conditions={
            "A": [
                _condition("A", "req", "require", scope_positions=frozenset({"1"})),
                _condition("A", "fbd", "forbid", scope_positions=frozenset({"1"})),
            ]
        }
    )
    # position 1 matches both require and forbid; forbid wins.
    decision = engine.allow_decision("A", position=1)
    assert not decision.allowed
    assert "forbidden" in decision.reason


def test_require_must_match_when_present() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "req", "require", scope_dayparts=frozenset({"prime"}))]}
    )
    assert engine.is_allowed("A", daypart="prime")
    decision = engine.allow_decision("A", daypart="daytime")
    assert not decision.allowed
    assert "require" in decision.reason


def test_unknown_advertiser_is_allowed() -> None:
    engine = AdvertiserRuleEngine()
    assert engine.is_allowed("NOBODY", position=5, genre="News", daypart="prime")


# --- overlaps ---------------------------------------------------------------


def test_overlaps_detects_require_forbid_conflict() -> None:
    engine = AdvertiserRuleEngine(
        conditions={
            "A": [
                _condition("A", "req", "require", scope_genres=frozenset({"News"})),
                _condition("A", "fbd", "forbid", scope_genres=frozenset({"News", "Sports"})),
            ]
        }
    )
    findings = engine.overlaps("A")
    assert len(findings) == 1
    assert findings[0].kind == "conflict"


def test_overlaps_detects_stacked_premiums() -> None:
    engine = AdvertiserRuleEngine(
        conditions={
            "A": [
                _condition("A", "p1", "premium", value=1.2, scope_positions=frozenset({"1"})),
                _condition("A", "p2", "premium", value=1.5, scope_positions=frozenset({"1", "2"})),
            ]
        }
    )
    findings = engine.overlaps("A")
    assert len(findings) == 1
    assert findings[0].kind == "stacked_premium"


def test_overlaps_ignores_disjoint_scopes() -> None:
    engine = AdvertiserRuleEngine(
        conditions={
            "A": [
                _condition("A", "p1", "premium", value=1.2, scope_positions=frozenset({"1"})),
                _condition("A", "p2", "premium", value=1.5, scope_positions=frozenset({"2"})),
            ]
        }
    )
    # positions 1 and 2 do not intersect, so no finding.
    assert engine.overlaps("A") == []


def test_overlaps_any_scope_intersects_everything() -> None:
    engine = AdvertiserRuleEngine(
        conditions={
            "A": [
                _condition("A", "p1", "premium", value=1.2),  # ANY positions
                _condition("A", "p2", "premium", value=1.5, scope_positions=frozenset({"2"})),
            ]
        }
    )
    findings = engine.overlaps("A")
    assert len(findings) == 1
    assert findings[0].kind == "stacked_premium"


# --- file loading + engine integration --------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_from_files_reads_baseline_and_conditions(tmp_path: Path) -> None:
    rules = tmp_path / "advertiser_rules.csv"
    conditions = tmp_path / "advertiser_conditions.csv"
    _write_csv(
        rules,
        ["advertiser_id", "default_premium", "allow_positions", "allow_genres", "prime_time_only", "notes"],
        [{"advertiser_id": "A", "default_premium": "1.5", "allow_positions": "1,2",
          "allow_genres": "ANY", "prime_time_only": "False", "notes": ""}],
    )
    _write_csv(
        conditions,
        ["advertiser_id", "rule_id", "scope_positions", "scope_genres", "scope_dayparts", "effect", "value", "notes"],
        [{"advertiser_id": "A", "rule_id": "r1", "scope_positions": "1", "scope_genres": "ANY",
          "scope_dayparts": "prime", "effect": "premium", "value": "2.0", "notes": ""}],
    )
    engine = AdvertiserRuleEngine.from_files(rules_path=rules, conditions_path=conditions)
    assert engine.effective_premium("A", position=1, daypart="prime") == pytest.approx(3.0)
    assert not engine.is_allowed("A", position=3)


def test_empty_conditions_file_yields_no_conditions(tmp_path: Path) -> None:
    conditions = tmp_path / "advertiser_conditions.csv"
    _write_csv(
        conditions,
        ["advertiser_id", "rule_id", "scope_positions", "scope_genres", "scope_dayparts", "effect", "value", "notes"],
        [],
    )
    engine = AdvertiserRuleEngine.from_files(conditions_path=conditions)
    assert engine.conditions == {}


# --- per-spot pricing wiring ------------------------------------------------


def test_price_daily_spots_applies_premium_and_drops_forbidden() -> None:
    import pandas as pd

    from kairos.export.spots import price_daily_spots
    from kairos.optimize.pricing import PricingModel

    engine = AdvertiserRuleEngine(
        baselines={"BRAND": _baseline("BRAND", default_premium=2.0)},
        conditions={"BRAND": [_condition("BRAND", "fbd", "forbid", scope_positions=frozenset({"1"}))]},
    )
    pricing = PricingModel(base_price_per_second_per_tvr_point=10.0)
    daily = pd.DataFrame([
        {"advertiser": "BRAND", "campaign": "C", "program": "Show", "position_in_break": 2,
         "planned_tvr": 5.0, "duration_sec": 30.0, "pricing_type": "CPP", "price": None, "spot_time": "21:00:00"},
        {"advertiser": "BRAND", "campaign": "C", "program": "Show", "position_in_break": 1,
         "planned_tvr": 5.0, "duration_sec": 30.0, "pricing_type": "CPP", "price": None, "spot_time": "21:05:00"},
    ])
    result = price_daily_spots(daily, engine=engine, pricing=pricing)
    # The position-2 spot is priced; the position-1 spot is forbidden and dropped.
    assert len(result.priced) == 1
    assert len(result.dropped) == 1
    spot = result.priced[0]
    # revenue = 10 base * 5 tvr * (30/30) units * 2.0 premium = 100.0
    assert spot.revenue == pytest.approx(100.0)
    assert result.dropped[0].position == 1


def test_price_daily_spots_honours_programme_scope_and_placement_pressure() -> None:
    # The daily pricing path must pass the spot's programme to the engine so a
    # programme-scoped rule actually bites, and must surface placement_value so a
    # placement-preference (pressure) rule shows a steer without inflating revenue.
    import pandas as pd

    from kairos.export.spots import price_daily_spots
    from kairos.optimize.pricing import PricingModel

    engine = AdvertiserRuleEngine(
        baselines={"BRAND": _baseline("BRAND", default_premium=1.0)},
        conditions={"BRAND": [
            _condition("BRAND", "news_prem", "premium", value=20.0, mode="percent",
                       scope_programmes=frozenset({"News"})),
            _condition("BRAND", "news_push", "pressure", value=10.0,
                       scope_programmes=frozenset({"News"})),
        ]},
    )
    pricing = PricingModel(base_price_per_second_per_tvr_point=10.0)
    daily = pd.DataFrame([
        {"advertiser": "BRAND", "campaign": "C", "program": "News", "position_in_break": 2,
         "planned_tvr": 5.0, "duration_sec": 30.0, "pricing_type": "CPP", "price": None, "spot_time": "21:00:00"},
        {"advertiser": "BRAND", "campaign": "C", "program": "Movie", "position_in_break": 2,
         "planned_tvr": 5.0, "duration_sec": 30.0, "pricing_type": "CPP", "price": None, "spot_time": "22:00:00"},
    ])
    priced = {spot.program: spot for spot in price_daily_spots(daily, engine=engine, pricing=pricing).priced}

    # News matches the programme-scoped +20% premium: revenue = 10 * 5 * 1.20 = 60.
    assert priced["News"].revenue == pytest.approx(60.0)
    # Movie is out of scope, so it stays at the baseline: revenue = 10 * 5 * 1.0 = 50.
    assert priced["Movie"].revenue == pytest.approx(50.0)
    # The +10% placement pressure shows only in News's placement_value (60 * 1.10 = 66),
    # never in its charged revenue; Movie has no pressure, so the two are equal.
    assert priced["News"].placement_value == pytest.approx(66.0)
    assert priced["News"].revenue == pytest.approx(60.0)
    assert priced["Movie"].placement_value == pytest.approx(priced["Movie"].revenue)


# --- API CRUD round-trip ----------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """A TestClient with the conditions store redirected to a temp file."""
    from fastapi.testclient import TestClient

    import kairos_api.advertiser_conditions as conditions_module

    conditions_path = tmp_path / "advertiser_conditions.csv"
    _write_csv(
        conditions_path,
        ["advertiser_id", "rule_id", "scope_positions", "scope_genres", "scope_dayparts", "effect", "value", "notes"],
        [],
    )
    monkeypatch.setattr(conditions_module, "CONDITIONS_PATH", conditions_path)
    monkeypatch.setattr(conditions_module, "BACKUP_DIR", tmp_path / "_backups")
    # Point the engine the overlap views build (via from_files defaults) at the
    # same temp conditions file, so the API reads what the CRUD just wrote.
    import kairos.optimize.advertiser_rules as engine_module

    monkeypatch.setattr(engine_module, "DEFAULT_CONDITIONS_PATH", conditions_path)

    from kairos_api.server import app

    return TestClient(app)


def test_conditions_crud_round_trip(client) -> None:
    # Create.
    create = client.post(
        "/api/advertisers/A/conditions",
        json={"rule_id": "r1", "effect": "premium", "value": 1.5,
              "scope_positions": "1,2", "scope_genres": "ANY", "scope_dayparts": "prime"},
    )
    assert create.status_code == 201
    body = create.json()
    assert body["effect"] == "premium"
    assert body["value"] == 1.5
    assert body["scope_positions"] == "1,2"

    # Duplicate create -> 409.
    dup = client.post("/api/advertisers/A/conditions", json={"rule_id": "r1", "effect": "forbid"})
    assert dup.status_code == 409

    # List.
    listing = client.get("/api/advertisers/A/conditions")
    assert listing.status_code == 200
    assert len(listing.json()["conditions"]) == 1

    # Update (partial).
    update = client.put("/api/advertisers/A/conditions/r1", json={"value": 2.5})
    assert update.status_code == 200
    assert update.json()["value"] == 2.5

    # Update a missing rule -> 404.
    missing = client.put("/api/advertisers/A/conditions/nope", json={"value": 1.0})
    assert missing.status_code == 404

    # Delete.
    deleted = client.delete("/api/advertisers/A/conditions/r1")
    assert deleted.status_code == 200
    assert client.get("/api/advertisers/A/conditions").json()["conditions"] == []


# --- Stage 2 upgrade: programme dimension, placement pressure, gold position ---


def test_premium_can_scope_to_a_specific_programme() -> None:
    # A premium scoped to one programme matches only that programme, leaving every
    # other show at the baseline. This is the new programme dimension.
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=1.0)},
        conditions={"A": [
            _condition("A", "r1", "premium", value=1.3,
                       scope_programmes=frozenset({"חדשות הערב"})),
        ]},
    )
    assert engine.effective_premium("A", programme="חדשות הערב") == pytest.approx(1.3)
    assert engine.effective_premium("A", programme="סרט הערב") == pytest.approx(1.0)


def test_pressure_steers_placement_but_never_revenue() -> None:
    # A +20% pressure rule must leave the real premium (revenue) at the baseline,
    # while the placement value the optimizer ranks on is lifted by 20%.
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=1.0)},
        conditions={"A": [
            _condition("A", "p1", "pressure", value=20.0, scope_dayparts=frozenset({"prime"})),
        ]},
    )
    # Revenue is unchanged: pressure is never charged.
    assert engine.effective_premium("A", daypart="prime") == pytest.approx(1.0)
    # Placement value is lifted, so the optimizer prefers this slot.
    assert engine.pressure_multiplier("A", daypart="prime") == pytest.approx(1.2)
    assert engine.placement_multiplier("A", daypart="prime") == pytest.approx(1.2)
    # Outside the scope there is no steer and no charge.
    assert engine.pressure_multiplier("A", daypart="noon") == pytest.approx(1.0)
    assert engine.placement_multiplier("A", daypart="noon") == pytest.approx(1.0)


def test_pressure_and_premium_compose_correctly() -> None:
    # A real +50% premium and a virtual +20% pressure: revenue carries only the
    # premium (1.5), placement carries both (1.5 * 1.2 = 1.8).
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=1.0)},
        conditions={"A": [
            _condition("A", "r1", "premium", value=1.5, scope_dayparts=frozenset({"prime"})),
            _condition("A", "p1", "pressure", value=20.0, scope_dayparts=frozenset({"prime"})),
        ]},
    )
    assert engine.effective_premium("A", daypart="prime") == pytest.approx(1.5)
    assert engine.placement_multiplier("A", daypart="prime") == pytest.approx(1.8)


def test_gold_break_is_a_scopeable_position() -> None:
    # The gold break (ברייק זהב) is a first-class position token, so a rule can
    # target it like any other position.
    from kairos.optimize.advertiser_rules import GOLD_POSITION

    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=1.0)},
        conditions={"A": [
            _condition("A", "g1", "premium", value=2.0,
                       scope_positions=frozenset({GOLD_POSITION})),
        ]},
    )
    assert engine.effective_premium("A", position=GOLD_POSITION) == pytest.approx(2.0)
    assert engine.effective_premium("A", position=1) == pytest.approx(1.0)


def test_two_pressure_rules_report_stacked_pressure_overlap() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [
            _condition("A", "p1", "pressure", value=10.0, scope_dayparts=frozenset({"prime"})),
            _condition("A", "p2", "pressure", value=15.0, scope_dayparts=frozenset({"prime"})),
        ]},
    )
    kinds = {f.kind for f in engine.overlaps("A")}
    assert "stacked_pressure" in kinds


def test_conditions_csv_round_trip_carries_programme_scope() -> None:
    from kairos.optimize.advertiser_rules import _condition_from_row

    row = {
        "advertiser_id": "A", "rule_id": "r1", "effect": "premium", "value": "1.3",
        "scope_positions": "ANY", "scope_genres": "ANY", "scope_dayparts": "prime",
        "scope_programmes": "חדשות הערב", "notes": "",
    }
    condition = _condition_from_row(row)
    assert condition is not None
    assert condition.scope_programmes == frozenset({"חדשות הערב"})


# --- coefficient modes (percent and cost-per-point) -------------------------


def test_legacy_premium_value_is_a_bare_multiplier() -> None:
    # A rule with no explicit mode keeps the original multiplier semantics: value
    # 1.5 means a 1.5x premium, unchanged by the new mode column.
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=1.5)]},
    )
    assert engine.effective_premium("A") == pytest.approx(1.5)


def test_percent_mode_reads_value_as_a_signed_percent() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [
            _condition("A", "up", "premium", value=15.0, mode="percent",
                       scope_dayparts=frozenset({"prime"})),
            _condition("A", "down", "premium", value=-20.0, mode="percent",
                       scope_dayparts=frozenset({"night"})),
        ]},
    )
    assert engine.effective_premium("A", daypart="prime") == pytest.approx(1.15)
    assert engine.effective_premium("A", daypart="night") == pytest.approx(0.80)


def test_cpp_absolute_sets_the_point_price() -> None:
    # With base_cpp=100, a cpp_absolute rule of 130 prices the spot as if the CPP
    # were 130, a 1.3x factor on the engine's base price.
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=130.0, mode="cpp_absolute")]},
    )
    assert engine.effective_premium("A", base_cpp=100.0) == pytest.approx(1.3)


def test_cpp_absolute_overrides_baseline_not_stacks() -> None:
    # cpp_absolute SETS the effective CPP; it must override the baseline premium,
    # not multiply by it. With a 2.0 baseline and a cpp_absolute of 130 on
    # base_cpp 100, the result is 1.3 (the absolute), not 2.0 * 1.3 = 2.6.
    engine = AdvertiserRuleEngine(
        baselines={"A": _baseline("A", default_premium=2.0)},
        conditions={"A": [_condition("A", "r1", "premium", value=130.0, mode="cpp_absolute")]},
    )
    assert engine.effective_premium("A", base_cpp=100.0) == pytest.approx(1.3)
    # A relative rule appearing after the absolute composes on the absolute's
    # result: cpp_absolute 130 (1.3x) then percent +10 -> 1.3 * 1.10 = 1.43.
    composed = AdvertiserRuleEngine(
        conditions={"A": [
            _condition("A", "r1", "premium", value=130.0, mode="cpp_absolute"),
            _condition("A", "r2", "premium", value=10.0, mode="percent"),
        ]},
    )
    assert composed.effective_premium("A", base_cpp=100.0) == pytest.approx(1.43)


def test_cpp_add_raises_the_point_price() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=20.0, mode="cpp_add")]},
    )
    # base 100 + 20 = 120 -> 1.2x
    assert engine.effective_premium("A", base_cpp=100.0) == pytest.approx(1.2)


def test_cpp_discount_lowers_the_point_price_and_floors_at_zero() -> None:
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=25.0, mode="cpp_discount")]},
    )
    # base 100 - 25 = 75 -> 0.75x
    assert engine.effective_premium("A", base_cpp=100.0) == pytest.approx(0.75)
    # A discount larger than the base never goes negative.
    big = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=500.0, mode="cpp_discount")]},
    )
    assert big.effective_premium("A", base_cpp=100.0) == pytest.approx(0.0)


def test_cpp_mode_without_base_cpp_leaves_premium_unchanged() -> None:
    # No base price to convert a CPP delta into a factor: the rule is a no-op
    # rather than a guess, so revenue is unchanged.
    engine = AdvertiserRuleEngine(
        conditions={"A": [_condition("A", "r1", "premium", value=130.0, mode="cpp_absolute")]},
    )
    assert engine.effective_premium("A") == pytest.approx(1.0)


def test_unknown_mode_falls_back_to_multiplier() -> None:
    from kairos.optimize.advertiser_rules import _condition_from_row

    row = {
        "advertiser_id": "A", "rule_id": "r1", "effect": "premium", "value": "1.4",
        "mode": "nonsense", "scope_positions": "ANY", "scope_genres": "ANY",
        "scope_dayparts": "ANY", "scope_programmes": "ANY", "notes": "",
    }
    condition = _condition_from_row(row)
    assert condition is not None
    assert condition.mode == "multiplier"
    engine = AdvertiserRuleEngine(conditions={"A": [condition]})
    assert engine.effective_premium("A") == pytest.approx(1.4)


def test_placement_multiplier_forwards_base_cpp_to_cpp_modes() -> None:
    # A cpp_add premium plus a pressure lever: placement ranks on both, revenue on
    # the premium only, and the cpp mode resolves with the forwarded base_cpp.
    engine = AdvertiserRuleEngine(
        conditions={"A": [
            _condition("A", "prem", "premium", value=20.0, mode="cpp_add"),
            _condition("A", "press", "pressure", value=10.0),
        ]},
    )
    # premium = 1.2 (120/100), pressure = 1.1 -> placement = 1.32
    assert engine.effective_premium("A", base_cpp=100.0) == pytest.approx(1.2)
    assert engine.placement_multiplier("A", base_cpp=100.0) == pytest.approx(1.32)
