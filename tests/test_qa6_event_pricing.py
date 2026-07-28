"""Contract tests for the operator-asserted event-date pricing layer.

The layer is exactly like the rate-card day premiums: an operator ASSERTION per
calendar event (data/calendar_events.csv, price_multiplier column), gated on
pricing_activation.events which ships OFF. These tests prove the identity
guarantee (OFF means every premium is byte-identical to the pre-layer engine and
the events store is never even read), the exact per-date effect when ON,
open-ended coverage, multiplicative composition of overlapping events, the API
validation bounds, and the price-slot tester's event line. Every store and the
settings file are relocated to tmp so nothing under data/ is ever written.
"""

from __future__ import annotations

import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos.optimize.event_pricing as event_pricing
import kairos_api.core as core
import kairos_api.events_api as events_api
import kairos_api.version_store as vs
from kairos.data import ProgramClassifier
from kairos.data.transform import build_segments_from_programmes
from kairos.optimize.pricing import (
    PricingModel,
    load_event_day_multipliers,
    load_price_events,
    pricing_from_settings,
)

ROOT = Path(__file__).resolve().parents[1]

LEGACY_COLUMNS = ("event_id", "name", "type", "start_date", "end_date",
                  "intensity", "notes", "active")


@pytest.fixture(scope="module")
def classifier() -> ProgramClassifier:
    return ProgramClassifier.from_yaml()


def _write_store(path: Path, rows: list[dict], columns: tuple = None) -> Path:
    """Write a raw events store like the API does (utf-8-sig, str(bool) actives)."""
    fieldnames = list(columns or events_api.COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            base = {"event_id": f"ev{index:02d}", "name": "אירוע", "type": "special",
                    "start_date": "", "end_date": "", "intensity": "3", "notes": "",
                    "active": "True"}
            if "price_multiplier" in fieldnames:
                base["price_multiplier"] = "1.0"
            base.update(row)
            writer.writerow(base)
    return path


def _programmes(dates: list[str]) -> pd.DataFrame:
    rows = []
    for day in dates:
        rows.append(("חדשות הערב", "קשת 12", f"{day} 20:00:00", 3600.0, 5.0))
        rows.append(("תוכנית פריים", "קשת 12", f"{day} 21:00:00", 3600.0, 4.0))
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    return frame


# --- (a) identity guarantee: OFF is byte-identical --------------------------------
def test_off_is_byte_identical_on_a_real_settings_load(tmp_path, monkeypatch, classifier):
    """With the shipped default (events activation OFF), the layer must change
    nothing: same premiums to the last bit, and the events store never read,
    even while an active 2.0x event covers the schedule dates."""
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-04", "end_date": "2024-11-05", "price_multiplier": "2.0"},
    ])
    monkeypatch.setattr(event_pricing, "DEFAULT_EVENTS_PATH", store)

    settings = core._load_settings()   # the real shipped settings file
    model = pricing_from_settings(settings)
    assert model.enable_events is False, "the events layer must ship OFF"
    assert model.event_day_multipliers == {}, "OFF must never even read the store"

    baseline = PricingModel.from_config(getattr(settings, "pricing_overrides", None) or {})
    frame = _programmes(["2024-11-04", "2024-11-05"])
    with_layer = build_segments_from_programmes(frame, classifier, model)
    before = build_segments_from_programmes(frame, classifier, baseline)
    assert len(with_layer) == len(before) == 4
    assert [s.premium for s in with_layer] == [s.premium for s in before]
    assert [s.baseline_tvr for s in with_layer] == [s.baseline_tvr for s in before]
    assert [s.cpp for s in with_layer] == [s.cpp for s in before]

    breakdown = model.price_slot(pricing_class="News", weekday_iso=1, day="2024-11-04")
    assert [layer.name for layer in breakdown.layers] == ["program", "day"]


# --- (b) ON multiplies exactly the covered date -----------------------------------
def test_on_multiplies_exactly_the_covered_date(tmp_path, monkeypatch, classifier):
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-04", "end_date": "2024-11-04", "price_multiplier": "1.5"},
    ])
    monkeypatch.setattr(event_pricing, "DEFAULT_EVENTS_PATH", store)

    model = pricing_from_settings({"pricing_overrides": {"pricing_activation": {"events": True}}})
    assert model.enable_events is True
    assert model.event_day_multipliers == {"2024-11-04": 1.5}

    frame = _programmes(["2024-11-04", "2024-11-05"])
    segments = build_segments_from_programmes(frame, classifier, model)
    plain = build_segments_from_programmes(frame, classifier, PricingModel.from_config({}))
    assert {s.day for s in segments} == {"2024-11-04", "2024-11-05"}
    for seg, base in zip(segments, plain):
        if seg.day == "2024-11-04":
            assert seg.premium == base.premium * 1.5, "the covered date must move by exactly 1.5x"
        else:
            assert seg.premium == base.premium, "an uncovered date must not move at all"

    on_day = model.price_slot(pricing_class="News", weekday_iso=1, day="2024-11-04")
    assert [layer.name for layer in on_day.layers] == ["program", "day", "event"]
    assert on_day.layers[-1].multiplier == 1.5
    assert on_day.layers[-1].source == "operator_event"
    off_day = model.price_slot(pricing_class="News", weekday_iso=1, day="2024-11-05")
    assert [layer.name for layer in off_day.layers] == ["program", "day"]


def test_overlapping_events_compose_multiplicatively(tmp_path):
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-04", "end_date": "2024-11-06", "price_multiplier": "1.5"},
        {"start_date": "2024-11-05", "end_date": "2024-11-05", "price_multiplier": "2.0"},
    ])
    multipliers = load_event_day_multipliers(store)
    assert multipliers["2024-11-04"] == 1.5
    assert multipliers["2024-11-05"] == 1.5 * 2.0
    assert multipliers["2024-11-06"] == 1.5
    assert "2024-11-07" not in multipliers


# --- (c) open-ended coverage ------------------------------------------------------
def test_open_ended_event_covers_from_start_to_the_horizon(tmp_path):
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-01", "end_date": "", "price_multiplier": "1.2"},
    ])
    multipliers = load_event_day_multipliers(store)
    assert "2024-10-31" not in multipliers, "no coverage before the start"
    assert multipliers["2024-11-01"] == 1.2
    assert multipliers["2024-11-30"] == 1.2
    assert multipliers[date.today().isoformat()] == 1.2
    horizon = date.today() + timedelta(days=event_pricing.EVENT_OPEN_HORIZON_DAYS)
    assert multipliers[horizon.isoformat()] == 1.2
    assert (horizon + timedelta(days=1)).isoformat() not in multipliers


def test_inactive_neutral_and_legacy_rows_contribute_nothing(tmp_path):
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-04", "price_multiplier": "2.0", "active": "False"},
        {"start_date": "2024-11-04", "price_multiplier": "1.0"},
        {"start_date": "not-a-date", "price_multiplier": "2.0"},
    ])
    assert load_price_events(store) == []
    assert load_event_day_multipliers(store) == {}
    # A legacy store predating the column reads as all-neutral, and a missing
    # store reads as empty, never an error.
    legacy = _write_store(tmp_path / "legacy.csv",
                          [{"start_date": "2024-11-04"}], columns=LEGACY_COLUMNS)
    assert load_price_events(legacy) == []
    assert load_event_day_multipliers(tmp_path / "absent.csv") == {}


# --- (d) API validation bounds and tolerant read ----------------------------------
@pytest.fixture()
def events_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(events_api, "EVENTS_PATH", tmp_path / "calendar_events.csv")
    app = FastAPI()
    app.include_router(events_api.router)
    return TestClient(app)


def test_price_multiplier_validation_and_roundtrip(events_client):
    payload = {"name": "דרבי", "type": "sport", "start_date": "2026-01-01", "intensity": 3}
    for bad in (0.05, 5.5, 0.0, -1.0):
        response = events_client.post("/api/events", json={**payload, "price_multiplier": bad})
        assert response.status_code == 400, f"{bad} must be rejected: {response.text}"
    assert events_client.get("/api/events").json()["events"] == []

    created = events_client.post("/api/events", json=payload)
    assert created.status_code == 201
    assert created.json()["price_multiplier"] == 1.0, "omitted multiplier defaults neutral"
    for edge in (0.1, 5.0):
        response = events_client.post("/api/events", json={**payload, "price_multiplier": edge})
        assert response.status_code == 201 and response.json()["price_multiplier"] == edge

    event_id = created.json()["event_id"]
    updated = events_client.put(f"/api/events/{event_id}", json={"price_multiplier": 2.5})
    assert updated.status_code == 200 and updated.json()["price_multiplier"] == 2.5
    assert updated.json()["name"] == "דרבי", "untouched fields survive the partial PUT"
    assert events_client.put(f"/api/events/{event_id}",
                             json={"price_multiplier": 9.0}).status_code == 400
    listed = events_client.get("/api/events").json()["events"]
    assert all(isinstance(event["price_multiplier"], float) for event in listed)


def test_legacy_store_without_column_reads_and_updates_as_neutral(events_client, tmp_path):
    _write_store(events_api.EVENTS_PATH, [{"start_date": "2024-11-04"}],
                 columns=LEGACY_COLUMNS)
    listed = events_client.get("/api/events").json()["events"]
    assert listed[0]["price_multiplier"] == 1.0, "a legacy row must read neutral, not missing"
    updated = events_client.put(f"/api/events/{listed[0]['event_id']}",
                                json={"intensity": 2}).json()
    assert updated["price_multiplier"] == 1.0, "an unrelated edit must keep the neutral value"


# --- (e) the price-slot tester and the /api/pricing exposure ----------------------
@pytest.fixture()
def pricing_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    store = _write_store(tmp_path / "calendar_events.csv", [
        {"start_date": "2024-11-08", "end_date": "2024-11-08", "price_multiplier": "2.0"},
    ])
    monkeypatch.setattr(event_pricing, "DEFAULT_EVENTS_PATH", store)
    import kairos_api.pricing_api as pricing_api

    app = FastAPI()
    app.include_router(pricing_api.router)
    return TestClient(app)


def test_tester_and_state_expose_the_event_layer(pricing_client):
    state = pricing_client.get("/api/pricing").json()
    assert state["events"]["enabled"] is False, "the layer must ship OFF"
    assert state["events"]["active_event_count"] == 1

    off = pricing_client.post("/api/pricing/price-slot", json={
        "pricing_class": "News", "weekday_iso": 5, "day": "2024-11-08",
    }).json()
    assert "event" not in [layer["name"] for layer in off["layers"]]
    wired_off = {layer["name"]: layer for layer in off["wired_off_layers"]}
    assert wired_off["event"]["multiplier"] == 2.0 and wired_off["event"]["applied"] is False

    activated = pricing_client.put("/api/pricing", json={
        "overrides": {"pricing_activation": {"events": True}},
    })
    assert activated.status_code == 200, activated.text
    assert activated.json()["events"]["enabled"] is True

    on = pricing_client.post("/api/pricing/price-slot", json={
        "pricing_class": "News", "weekday_iso": 5, "day": "2024-11-08",
    }).json()
    layers = {layer["name"]: layer for layer in on["layers"]}
    assert layers["event"]["multiplier"] == 2.0
    assert layers["event"]["source"] == "operator_event"
    product = on["base_cpp"]
    for layer in on["layers"]:
        product *= layer["multiplier"]
    assert on["final_cpp"] == pytest.approx(product, rel=1e-12)
    assert on["final_cpp"] == pytest.approx(off["final_cpp"] * 2.0, rel=1e-12)

    uncovered = pricing_client.post("/api/pricing/price-slot", json={
        "pricing_class": "News", "weekday_iso": 5, "day": "2024-11-09",
    }).json()
    assert "event" not in [layer["name"] for layer in uncovered["layers"]]
    assert uncovered["final_cpp"] == pytest.approx(off["final_cpp"], rel=1e-12)
