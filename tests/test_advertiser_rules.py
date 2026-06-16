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
