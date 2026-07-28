"""Contract tests for the calendar events store, holiday table and model context.

Every test relocates the events store and the version store into a tmp tree
(EVENTS_PATH module global plus KAIROS_VERSIONS_DIR), so nothing under data/ is
ever written. The model_context assertions compare the API payload against the
REAL sources (config/optimization_weights.yaml and the coefficients metadata)
read independently in the test, so a fabricated value cannot pass.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.events_api as events_api
import kairos_api.version_store as vs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    """Relocate the events store and the version store to tmp."""
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(events_api, "EVENTS_PATH", tmp_path / "calendar_events.csv")
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(events_api.router)
    app.include_router(vs.router)
    return TestClient(app)


def _create(client: TestClient, **overrides) -> dict:
    payload = {"name": "מבצע צפוני", "type": "war", "start_date": "2024-10-01",
               "end_date": "", "intensity": 5, "notes": "", "active": True}
    payload.update(overrides)
    response = client.post("/api/events", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# --- store CRUD ----------------------------------------------------------------
def test_get_on_missing_store_is_honest_and_creates_nothing(client, relocated) -> None:
    body = client.get("/api/events").json()
    assert body["events"] == []
    assert body["holidays"], "the bundled holiday table must be served"
    assert "model_context" in body
    assert not (relocated / "calendar_events.csv").exists(), "a read must not create the store"


def test_create_list_update_delete_roundtrip(client) -> None:
    created = _create(client)
    assert created["end_date"] is None and created["intensity"] == 5 and created["active"]

    listed = client.get("/api/events").json()["events"]
    assert [event["event_id"] for event in listed] == [created["event_id"]]

    updated = client.put(f"/api/events/{created['event_id']}",
                         json={"end_date": "2024-11-27", "intensity": 2,
                               "active": False}).json()
    assert updated["end_date"] == "2024-11-27"
    assert updated["intensity"] == 2 and updated["active"] is False
    assert updated["name"] == created["name"], "untouched fields survive a partial PUT"

    deleted = client.delete(f"/api/events/{created['event_id']}")
    assert deleted.status_code == 200 and deleted.json() == {"deleted": created["event_id"]}
    assert client.get("/api/events").json()["events"] == []


def test_validation_rejects_bad_payloads(client) -> None:
    bad = [
        {"type": "invasion"},                                      # unknown type
        {"intensity": 0},                                          # below range
        {"intensity": 6},                                          # above range
        {"start_date": "01/10/2024"},                              # non-ISO date
        {"end_date": "2024-09-30"},                                # ends before start
        {"name": "   "},                                           # blank name
    ]
    for overrides in bad:
        payload = {"name": "אירוע", "type": "special", "start_date": "2024-10-01",
                   "intensity": 3}
        payload.update(overrides)
        response = client.post("/api/events", json=payload)
        assert response.status_code == 400, f"{overrides} must be rejected: {response.text}"
    assert client.get("/api/events").json()["events"] == [], "rejected payloads store nothing"


def test_unknown_event_id_is_404(client) -> None:
    assert client.put("/api/events/deadbeef", json={"intensity": 1}).status_code == 404
    assert client.delete("/api/events/deadbeef").status_code == 404


# --- version snapshots ---------------------------------------------------------
def test_mutations_snapshot_the_events_store_and_restore_round_trips(client) -> None:
    created = _create(client)
    client.put(f"/api/events/{created['event_id']}", json={"intensity": 1})
    client.delete(f"/api/events/{created['event_id']}")

    entries = client.get("/api/versions").json()["entries"]
    events_versions = [entry for entry in entries if entry["files"] == ["events"]]
    assert len(events_versions) >= 2, "create, update and delete each snapshot first"
    assert all(entry["source"] == "manual_edit" for entry in events_versions)

    # The newest events version captured the store just before the delete, so
    # restoring it brings the event back, and the diff endpoint reports the row.
    newest = events_versions[0]["version_id"]
    diff = client.get(f"/api/versions/{newest}/diff").json()["diff"]["events"]
    assert [row["event_id"] for row in diff["added"]] == [created["event_id"]]
    restore = client.post(f"/api/versions/{newest}/restore", json={"files": ["events"]})
    assert restore.status_code == 200 and restore.json()["restored"] == ["events"]
    restored = client.get("/api/events").json()["events"]
    assert [event["event_id"] for event in restored] == [created["event_id"]]
    assert restored[0]["intensity"] == 1


def test_events_is_a_registered_logical_file() -> None:
    assert "events" in vs._LOGICAL_ORDER
    assert vs._ID_COLUMN["events"] == "event_id"
    assert vs._logical_path("events") == Path(events_api.EVENTS_PATH)


# --- bundled holiday table -----------------------------------------------------
def test_holiday_table_structure() -> None:
    holidays = events_api._load_holidays()
    assert holidays, "the bundled table must parse"
    years = set()
    for row in holidays:
        parsed = date.fromisoformat(row["date"])
        years.add(parsed.year)
        assert row["name"], f"holiday on {row['date']} needs a name"
        assert row["kind"] in ("national", "religious"), row
        assert isinstance(row["is_school_holiday"], bool)
    assert years == {2024, 2025, 2026, 2027}, "the table covers exactly 2024-2027"
    for year in (2024, 2025, 2026, 2027):
        count = sum(1 for row in holidays if row["date"].startswith(str(year)))
        assert count >= 10, f"{year} has only {count} rows"
    dates = [row["date"] for row in holidays]
    assert dates == sorted(dates) and len(dates) == len(set(dates))


def test_holiday_table_anchors() -> None:
    """A few well-known anchor dates, not the whole calendar."""
    by_date = {row["date"]: row for row in events_api._load_holidays()}
    assert by_date["2024-10-12"]["name"] == "יום כיפור"
    assert by_date["2024-10-12"]["kind"] == "religious"
    assert by_date["2025-05-01"]["name"] == "יום העצמאות"
    assert by_date["2025-05-01"]["kind"] == "national"
    assert by_date["2024-05-06"]["kind"] == "national"  # יום השואה


def test_holiday_file_carries_the_verify_note() -> None:
    text = (ROOT / "kairos" / "config" / "israel_holidays.csv").read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any("reference" in line for line in header)
    assert any("verify" in line for line in header)


# --- model context: real sources only ------------------------------------------
def test_model_context_matches_the_real_sources(client) -> None:
    context = client.get("/api/events").json()["model_context"]

    config = yaml.safe_load((ROOT / "config" / "optimization_weights.yaml").read_text())
    expected = {int(k): float(v) for k, v in config["premiums"]["day_of_week"].items()}
    got = {row["iso_weekday"]: row["multiplier"] for row in context["weekday_premiums"]["values"]}
    assert got == expected, "weekday premiums must be the rate-card values, verbatim"
    assert "rate-card assertion" in context["weekday_premiums"]["basis"]

    metadata = json.loads(
        (ROOT / "models" / "tv_break_coefficients.json").read_text())["metadata"]
    measurement = context["measurement"]
    assert measurement["available"] is True
    assert measurement["detrend_baseline_mode"] == metadata["detrend_baseline_mode"]
    assert (measurement["seasonal_baseline"]["recommended"]
            == metadata["detrend_seasonality_recommended"])
    assert (measurement["seasonal_baseline"]["holdout"]["relative_improvement"]
            == metadata["detrend_seasonality_holdout"]["relative_improvement"])
    assert measurement["level_drift"] == metadata["level_drift"]
    assert measurement["computed_at"] == metadata["computed_at"]

    window = context["training_window"]
    assert (window["start"], window["end"], window["days"]) == ("2024-11-01", "2024-11-30", 30)
    assert window["total_breaks_measured"] == metadata["total_breaks_measured"]

    disclosure = context["wartime_disclosure"]
    assert disclosure["ceasefire_date"] == "2024-11-27"
    assert disclosure["post_ceasefire_breaks"] == 132
    assert disclosure["total_breaks_measured"] == metadata["total_breaks_measured"]
    for token in ("2024-11-27", "132", str(metadata["total_breaks_measured"])):
        assert token in disclosure["line"]


def test_model_context_missing_metadata_is_honest(client, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(events_api, "COEFFICIENTS_PATH", tmp_path / "absent.json")
    measurement = client.get("/api/events").json()["model_context"]["measurement"]
    assert measurement["available"] is False
    assert "not found" in measurement["reason"]


# --- overlaps ------------------------------------------------------------------
def test_window_and_plan_overlaps(client, monkeypatch) -> None:
    monkeypatch.setattr(events_api, "_plan_dates",
                        lambda: ["2024-11-28", "2024-12-03", "2025-02-01"])

    bounded = _create(client, name="אירוע תחום", type="special",
                      start_date="2024-11-25", end_date="2024-12-05", intensity=3)
    open_ended = _create(client, name="מלחמה", type="war",
                         start_date="2023-10-07", end_date="")
    outside = _create(client, name="ספורט", type="sport",
                      start_date="2025-01-01", end_date="2025-01-02", intensity=1)

    by_id = {event["event_id"]: event for event in client.get("/api/events").json()["events"]}
    assert by_id[bounded["event_id"]]["window_overlap_days"] == 6  # Nov 25..30
    assert by_id[bounded["event_id"]]["plan_overlap_dates"] == ["2024-11-28", "2024-12-03"]
    assert by_id[open_ended["event_id"]]["window_overlap_days"] == 30
    assert by_id[open_ended["event_id"]]["plan_overlap_dates"] == [
        "2024-11-28", "2024-12-03", "2025-02-01"]
    assert by_id[outside["event_id"]]["window_overlap_days"] == 0
    assert by_id[outside["event_id"]]["plan_overlap_dates"] == []


def test_plan_overlap_empty_when_no_plan(client, monkeypatch) -> None:
    monkeypatch.setattr(events_api, "_plan_dates", lambda: [])
    created = _create(client, start_date="2024-11-01")
    event = client.get("/api/events").json()["events"][0]
    assert event["event_id"] == created["event_id"]
    assert event["plan_overlap_dates"] == []
    assert event["window_overlap_days"] == 30


# --- server wiring -------------------------------------------------------------
def test_server_mounts_the_events_router() -> None:
    from kairos_api import server

    paths = {route.path for route in server.app.routes}
    assert "/api/events" in paths
    assert "/api/events/{event_id}" in paths
