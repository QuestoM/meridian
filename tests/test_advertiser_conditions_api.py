"""Tests for the advertiser-condition CRUD API additions.

These cover the new ``mode`` and ``scope_programmes`` columns (so a percent or
cost-per-point premium and a programme-scoped rule round-trip through the CSV)
and the ``/options`` endpoint that feeds the dashboard real scope vocabularies.
"""

from __future__ import annotations

import importlib

import pytest

import kairos_api.advertiser_conditions as ac


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """Point the CRUD at a throwaway conditions CSV so tests never touch real data."""
    path = tmp_path / "advertiser_conditions.csv"
    monkeypatch.setattr(ac, "CONDITIONS_PATH", path)
    monkeypatch.setattr(ac, "BACKUP_DIR", tmp_path / "_backups")
    return path


def test_create_round_trips_mode_and_programme_scope(temp_store) -> None:
    payload = ac.ConditionCreate(
        rule_id="r1",
        effect="premium",
        value=15.0,
        mode="percent",
        scope_programmes="חדשות הערב",
    )
    record = ac.create_condition("ACME", payload)
    assert record["mode"] == "percent"
    assert record["value"] == pytest.approx(15.0)
    assert record["scope_programmes"] == "חדשות הערב"

    # Reading it back through the list view preserves both new fields.
    stored = ac.conditions_for("ACME")
    assert len(stored) == 1
    assert stored[0]["mode"] == "percent"
    assert stored[0]["scope_programmes"] == "חדשות הערב"


def test_unknown_mode_normalizes_to_multiplier(temp_store) -> None:
    record = ac.create_condition(
        "ACME", ac.ConditionCreate(rule_id="r1", effect="premium", value=1.4, mode="nonsense")
    )
    assert record["mode"] == "multiplier"


def test_pressure_effect_is_accepted(temp_store) -> None:
    record = ac.create_condition(
        "ACME", ac.ConditionCreate(rule_id="p1", effect="pressure", value=10.0)
    )
    assert record["effect"] == "pressure"


def test_update_changes_mode_and_programme_scope(temp_store) -> None:
    ac.create_condition("ACME", ac.ConditionCreate(rule_id="r1", effect="premium", value=1.2))
    updated = ac.update_condition(
        "ACME", "r1", ac.ConditionUpdate(mode="cpp_add", value=20.0, scope_programmes="שישי בלילה")
    )
    assert updated["mode"] == "cpp_add"
    assert updated["value"] == pytest.approx(20.0)
    assert updated["scope_programmes"] == "שישי בלילה"


def test_options_endpoint_serves_real_vocabularies() -> None:
    options = ac.scope_options()
    position_keys = {p["key"] for p in options["positions"]}
    assert "gold" in position_keys
    assert {"first", "middle", "last"}.issubset(position_keys)
    assert {d["key"] for d in options["dayparts"]} == {
        "morning", "noon", "evening", "prime", "night",
    }
    assert set(options["effects"]) == {"premium", "require", "forbid", "pressure"}
    assert options["modes"][0] == "multiplier"
    assert "cpp_discount" in options["modes"]
    # Genres and programmes come from real config/EPG; both should be non-empty here.
    assert len(options["genres"]) > 0
    assert len(options["programmes"]) > 0
