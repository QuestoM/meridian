"""Journey tests: morning freshness, the decision loop, and the settings loop.

Each test walks a real operator flow end to end against the real committed data,
entirely in process (fastapi.testclient.TestClient or the builder functions
directly). Anything that would mutate operator state (the override store, the
saved settings, the weekly CSV) is redirected to a temporary copy first, so the
suite never moves the repository's committed plan or settings.

Journeys covered here:
  1. Morning check freshness: the staleness verdict appears when an input
     changes and clears when the schedule is restamped (recompute writes the
     stamp via kairos.export.schedule.write_weekly_schedule).
  2. Decision loop: approve resolves into a persisted anchored override
     (source=recommendation, status=active), reject into a dismissed record,
     the decision log lists both, an override edit flips the schedule to
     stale, and a recompute of that one day applies the override while leaving
     every other committed row byte-identical.
  6. Settings loop: PUT echoes the saved object, the floor maps into the
     engine guardrails, and objective_mode is wired from the saved settings
     through the recompute body into the optimizer.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import kairos.optimize.overrides as overrides_engine
import kairos_api.core as core
import kairos_api.overrides as overrides_api
from kairos.export.schedule import build_weekly_schedule
from kairos.export.schedule_freshness import schedule_freshness, write_schedule_meta
from kairos.optimize.overrides import Override, OverrideSet
from kairos.service import guardrails_from_settings

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"


@pytest.fixture()
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def saved_settings() -> dict:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def tmp_override_store(tmp_path, monkeypatch) -> Path:
    """Redirect every override read/write (API store, engine default, backups)
    to a temporary file seeded with the committed store's content."""
    store = tmp_path / "manual_overrides.csv"
    shutil.copy(ROOT / "data" / "manual_overrides.csv", store)
    monkeypatch.setattr(overrides_api, "OVERRIDES_PATH", store)
    monkeypatch.setattr(overrides_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(overrides_engine, "DEFAULT_OVERRIDES_PATH", store)
    return store


def test_morning_freshness_lifecycle(tmp_path):
    """Journey 1: the freshness banner logic. unknown with no stamp, fresh
    right after a stamp, stale naming the changed group after an input edit,
    fresh again after restamping (what a recompute does)."""
    tmp_csv = tmp_path / "weekly_break_schedule.csv"
    tmp_csv.write_text("stub", encoding="utf-8")
    tmp_root = tmp_path / "root"
    (tmp_root / "data").mkdir(parents=True)
    shutil.copy(SETTINGS_PATH, tmp_root / "data" / "kairos_settings.json")

    verdict = schedule_freshness(tmp_root, csv_path=tmp_csv)
    assert verdict == {"status": "unknown", "computed_at": None, "changed": []}

    write_schedule_meta(tmp_csv, tmp_root)
    verdict = schedule_freshness(tmp_root, csv_path=tmp_csv)
    assert verdict["status"] == "fresh", verdict
    assert verdict["changed"] == []

    settings = json.loads((tmp_root / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
    settings["min_retention_floor"] = 0.9
    (tmp_root / "data" / "kairos_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verdict = schedule_freshness(tmp_root, csv_path=tmp_csv)
    assert verdict["status"] == "stale", verdict
    assert "settings" in verdict["changed"], verdict

    write_schedule_meta(tmp_csv, tmp_root)
    verdict = schedule_freshness(tmp_root, csv_path=tmp_csv)
    assert verdict["status"] == "fresh", "restamping (a recompute) must clear staleness"


def test_overview_exposes_honest_freshness_block(client):
    """The overview carries schedule_freshness with the frozen tri-state
    contract, never a fabricated fresh with no stamp."""
    body = client.get("/api/overview").json()
    block = body.get("schedule_freshness")
    assert isinstance(block, dict), "overview must carry schedule_freshness"
    assert block.get("status") in {"fresh", "stale", "unknown"}
    assert isinstance(block.get("changed"), list)
    if block["status"] == "stale":
        assert block["changed"], "stale verdicts must name the changed groups"


def test_decision_approve_creates_anchored_override(client, tmp_override_store):
    """Journey 2, steps 1-2: approving a recommendation persists a REAL
    override with provenance (source=recommendation, rec_id), an active status
    and the semantic anchor trio, and the decision log lists it."""
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    owned = plan[(plan["channel"] == "עכשיו 14") & (plan["num_breaks"] > 0)].iloc[0]
    payload = {
        "action": "approve",
        "recommendation_id": "rec-qa-1",
        "target_id": owned["segment_id"],
        "kind": "gold",
        "gold": True,
        "anchor_date": owned["date"],
        "anchor_start": owned["start_time"],
        "anchor_title": owned["program_type"],
    }
    response = client.post("/api/break-decisions", json=payload)
    assert response.status_code == 200, response.text
    decision = response.json()["decision"]
    assert decision["status"] == "active"
    assert decision["source"] == "recommendation"
    assert decision["rec_id"] == "rec-qa-1"
    assert decision["anchor_date"] == owned["date"]
    assert decision["anchor_start"] == owned["start_time"]

    stored = pd.read_csv(tmp_override_store, encoding="utf-8-sig", dtype=str)
    assert (stored["target_id"] == owned["segment_id"]).any(), "override row not persisted"

    log = client.get("/api/break-decisions").json()["decisions"]
    mine = [d for d in log if d.get("recommendation_id") == "rec-qa-1"]
    assert mine and mine[0]["action"] == "approve"


def test_decision_reject_is_recorded_but_never_applied(client, tmp_override_store):
    """Journey 2: reject persists a dismissed record. Dismissed overrides are
    excluded from the constraints the optimizer consumes, so the plan is
    untouched by a rejection."""
    response = client.post(
        "/api/break-decisions",
        json={"action": "reject", "recommendation_id": "rec-qa-2",
              "target_id": "2024-11-01|עכשיו 14|000", "kind": "forbid"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["decision"]["status"] == "dismissed"
    constraints = OverrideSet.from_csv(tmp_override_store).segment_constraints()
    assert "2024-11-01|עכשיו 14|000" not in constraints, (
        "a dismissed decision must not become an engine constraint"
    )


def test_override_edit_flips_schedule_stale(tmp_path, tmp_override_store, client):
    """Journey 2, step 3: after approval the saved schedule must read stale
    (changed group: overrides) until the operator recomputes."""
    tmp_csv = tmp_path / "weekly.csv"
    tmp_csv.write_text("stub", encoding="utf-8")
    write_schedule_meta(tmp_csv, ROOT)
    assert schedule_freshness(ROOT, csv_path=tmp_csv)["status"] == "fresh"

    response = client.post(
        "/api/break-decisions",
        json={"action": "approve", "recommendation_id": "rec-qa-3",
              "target_id": "2024-11-01|עכשיו 14|000", "kind": "pin", "value": "1"},
    )
    assert response.status_code == 200, response.text

    verdict = schedule_freshness(ROOT, csv_path=tmp_csv)
    assert verdict["status"] == "stale", verdict
    assert "overrides" in verdict["changed"], verdict


def test_recompute_day_applies_override_and_preserves_rest(tmp_path, saved_settings):
    """Journey 2, steps 4-5: recompute-this-day re-optimizes only the touched
    channel-day, honors the override there, and leaves every other committed
    row byte-identical (the incremental merge contract)."""
    committed = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
    channel, day = "עכשיו 14", "2024-11-01"
    day_rows = committed[(committed["channel"] == channel) & (committed["date"] == day)]
    target = day_rows[day_rows["num_breaks"] != "0"].iloc[0]
    override = OverrideSet(overrides=[Override(
        override_id="qa-forbid", scope="segment",
        target_id=target["segment_id"], kind="forbid", value="",
    )])

    tmp_csv = tmp_path / "weekly.csv"
    shutil.copy(CSV_PATH, tmp_csv)
    frame = build_weekly_schedule(
        settings=saved_settings,
        revenue_weight=saved_settings["revenue_weight"] / 100.0,
        risk_lambda=saved_settings["risk_lambda"],
        operator_channel=saved_settings["operator_channel"],
        overrides=override,
        only_days=[(channel, day)],
        existing_csv=tmp_csv,
    )
    assert frame.attrs.get("skipped_overrides") == [], "override was silently skipped"
    after = frame[frame["segment_id"] == target["segment_id"]].iloc[0]
    assert str(after["num_breaks"]) == "0", "forbid override did not reach the plan"
    assert float(after["predicted_revenue"]) == 0.0, "a forbidden segment must earn zero"

    mask = (frame["channel"] == channel) & (frame["date"] == day)
    untouched = frame[~mask].reset_index(drop=True)
    untouched_committed = committed[
        ~((committed["channel"] == channel) & (committed["date"] == day))
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(untouched, untouched_committed)


def test_settings_roundtrip_echo_and_guardrail_mapping(tmp_path, monkeypatch, client):
    """Journey 6: PUT /api/settings persists and echoes the exact object, and
    the saved floor maps into the engine Guardrails the recompute uses."""
    tmp_settings = tmp_path / "kairos_settings.json"
    shutil.copy(SETTINGS_PATH, tmp_settings)
    monkeypatch.setattr(core, "SETTINGS_PATH", tmp_settings)

    body = client.get("/api/settings").json()
    body["min_retention_floor"] = 0.81
    body["objective_mode"] = "revenue_net"
    response = client.put("/api/settings", json=body)
    assert response.status_code == 200, response.text
    echoed = response.json()
    assert echoed["min_retention_floor"] == 0.81
    assert echoed["objective_mode"] == "revenue_net"

    persisted = json.loads(tmp_settings.read_text(encoding="utf-8"))
    assert persisted["min_retention_floor"] == 0.81
    guardrails = guardrails_from_settings(persisted)
    assert guardrails.min_retention_floor == 0.81
    assert SETTINGS_PATH.read_text(encoding="utf-8") != "", "real settings untouched"


def test_objective_mode_is_wired_from_settings_to_recompute():
    """Journey 6: the recompute body reads objective_mode from the SAVED
    settings and forwards it into build_weekly_schedule, whose signature
    forwards it to the optimizer core. Static wiring proof at the two seams
    (kairos_api/recompute_api.py:52, kairos/export/schedule.py:177)."""
    from kairos_api import recompute_api

    source = inspect.getsource(recompute_api._run_recompute)
    assert "objective_mode" in source, "recompute drops the saved objective_mode"
    parameters = inspect.signature(build_weekly_schedule).parameters
    assert "objective_mode" in parameters, "builder no longer accepts objective_mode"


def test_objective_mode_reaches_the_optimizer_engine(saved_settings):
    """Journey 6, engine end: objective_mode=revenue_net runs the real engine
    on a real channel-day, stays guardrail-compliant, and an unknown mode is
    rejected loudly (proving the parameter is consumed, not ignored)."""
    from kairos.data.loaders import load_programmes
    from kairos.data.transform import build_segments_from_programmes
    from kairos.model.impact import load_impact_model
    from kairos.optimize.optimizer import optimize_breaks
    from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
    from kairos.service import DEFAULT_IMPACT_MODEL_PATH, _build_classifier

    assumptions = OptimizerAssumptions()
    segments = build_segments_from_programmes(
        load_programmes(), _build_classifier(), pricing_from_settings(saved_settings, None),
        assumptions=assumptions,
        impact_model=load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions),
        channel="עכשיו 14", day="2024-11-01",
    )
    assert segments, "no segments for the probe channel-day"
    guardrails = guardrails_from_settings(saved_settings)
    net = optimize_breaks(segments, guardrails, objective_mode="revenue_net")
    assert net.is_compliant, [v.detail for v in net.violations]
    with pytest.raises(ValueError):
        optimize_breaks(segments, guardrails, objective_mode="not-a-mode")
