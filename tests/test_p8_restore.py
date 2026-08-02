"""P8 History, the other half: the restore, the run comparison, the attestation.

Split from ``tests/test_p8_history.py`` so neither file passes the 450-line cap.
The fixtures are that module's: importing them is how the two halves stay one
environment, which is the pattern ``tests/test_p3_break_store.py`` already
established in this wave.

What these tests are for, in one line each:

- the restore path still covers the same nine logical files and still snapshots
  first, which is this piece's Bar 3 row,
- a version captured against a different store is refused rather than restored,
- reading history and applying a restore are separately permissioned,
- a run delta is a subtraction of two recorded figures, never an invention,
- and an unreadable guardrail store answers unknown, never unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import kairos_api.version_store as vs
from kairos_api import core, history_api_runs, history_api_timeline

from test_p8_history import (  # noqa: F401 - fixtures are used by name
    OWNED,
    _as,
    _run_record,
    auth_env,
    history_env,
)


# --- restore: what is refused, and what still works -----------------------------

def test_a_version_recorded_against_another_store_is_marked_unrestorable(history_env, tmp_path) -> None:
    version_id = vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    manifest_path = vs._manifest_path(version_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "/tmp/pytest-of-somebody/kairos_settings.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    entry = history_env.get("/api/versions").json()["entries"][0]
    assert entry["restorable"] is False
    assert entry["restore_block"] == "foreign_store"

    refused = history_env.post(f"/api/versions/{version_id}/restore", json={})
    assert refused.status_code == 409
    assert "different store location" in refused.json()["detail"]


def test_a_version_missing_its_snapshot_bytes_is_unrestorable(history_env) -> None:
    version_id = vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    (vs._versions_root() / version_id / "settings.json").unlink()
    entry = history_env.get("/api/versions").json()["entries"][0]
    assert entry["restore_block"] == "missing_snapshot"
    assert history_env.post(f"/api/versions/{version_id}/restore", json={}).status_code == 409


def test_a_restorable_version_still_restores_and_snapshots_first(history_env) -> None:
    """Bar 3: the restore path still restores the same logical files and still
    records a pre-restore safety point, so a restore is itself undoable."""
    settings_path = Path(core.SETTINGS_PATH)
    original = settings_path.read_bytes()
    version_id = vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    settings_path.write_text(json.dumps({"operator_channel": OWNED, "revenue_weight": 91}), encoding="utf-8")

    listed = {entry["version_id"]: entry for entry in history_env.get("/api/versions").json()["entries"]}
    assert listed[version_id]["restorable"] is True

    body = history_env.post(f"/api/versions/{version_id}/restore", json={"files": ["settings"]}).json()
    assert body["restored"] == ["settings"]
    assert body["safety_version_id"], "a pre_restore safety point is recorded first"
    assert settings_path.read_bytes() == original

    safety = next(m for m in vs._all_manifests() if m["version_id"] == body["safety_version_id"])
    assert safety["source"] == "pre_restore"


def test_the_restore_path_still_covers_the_same_nine_logical_files(history_env) -> None:
    """The Bar 3 row for this piece names the nine by number. A manual snapshot
    covers all nine, the listing reports all nine, and a restore of all nine
    puts back exactly the nine and records one safety point first."""
    assert len(vs._LOGICAL_ORDER) == 9
    assert set(vs._LOGICAL_ORDER) == {
        "settings", "constraints", "overrides", "advertisers", "conditions",
        "events", "agencies", "agency_links", "agency_conditions",
    }

    created = history_env.post("/api/versions/snapshot", json={"label": "all nine"}).json()
    assert set(created["files"]) == set(vs._LOGICAL_ORDER)
    assert created["restorable"] is True

    settings_path = Path(core.SETTINGS_PATH)
    original = settings_path.read_bytes()
    settings_path.write_text(json.dumps({"operator_channel": OWNED, "revenue_weight": 12}), encoding="utf-8")

    before = len(vs._all_manifests())
    body = history_env.post(f"/api/versions/{created['version_id']}/restore", json={}).json()
    assert body["restored"] == list(vs._LOGICAL_ORDER)
    assert len(vs._all_manifests()) == before + 1, "exactly one safety point was recorded"
    assert settings_path.read_bytes() == original


def test_the_restore_appears_on_the_timeline_with_the_point_that_undoes_it(history_env) -> None:
    version_id = vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    Path(core.SETTINGS_PATH).write_text(json.dumps({"operator_channel": OWNED, "revenue_weight": 77}), encoding="utf-8")
    history_env.post(f"/api/versions/{version_id}/restore", json={"files": ["settings"]})

    rows = history_env.get("/api/history", params={"kind": "restore"}).json()["entries"]
    assert len(rows) == 1
    assert rows[0]["facts"]["version_id"] == version_id
    assert rows[0]["facts"]["restored"] == ["settings"]
    assert rows[0]["facts"]["safety_version_id"], "the reversal is itself an addressable point"


def test_reading_history_and_applying_a_restore_are_separately_permissioned(history_env, auth_env) -> None:
    """Figma's rule: viewing history is available to viewers, restoring is not.
    The refusal is legible before the click, through can_edit on the read."""
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    viewer = _as(history_env, auth_env, "viewer1", "viewer")

    listing = viewer.get("/api/versions")
    assert listing.status_code == 200
    assert listing.json()["can_edit"] is False
    assert listing.json()["can_edit_reason"] == "לחשבון צפייה אין הרשאת עריכה"
    assert viewer.get("/api/history").json()["can_edit"] is False
    assert viewer.get(f"/api/versions/{version_id}/diff").status_code == 200
    assert viewer.post(f"/api/versions/{version_id}/restore", json={}).status_code == 403

    operator = _as(history_env, auth_env, "operator1", "operator")
    assert operator.get("/api/versions").json()["can_edit"] is True


# --- the run, and its comparison ------------------------------------------------

def test_the_run_delta_is_a_subtraction_of_two_recorded_figures(history_env) -> None:
    body = history_env.get(f"/api/history/runs/{'b' * 32}").json()
    assert body["channel"] == OWNED
    assert body["summary"]["projected_revenue"] == 1292939.67
    comparison = body["comparison"]
    assert comparison["state"] == "measured"
    assert comparison["compared_run_id"] == "a" * 32
    revenue = comparison["fields"]["projected_revenue"]
    assert revenue == {"state": "measured", "from": 1414695.20, "to": 1292939.67, "delta": -121755.53}
    assert comparison["fields"]["average_retention"]["delta"] == pytest.approx(-1.9)


def test_the_first_run_in_a_scope_has_no_earlier_run_rather_than_a_zero_delta(history_env) -> None:
    body = history_env.get(f"/api/history/runs/{'a' * 32}").json()
    assert body["comparison"] == {"state": "no_earlier_run", "compared_run_id": None, "fields": {}}


def test_a_field_only_one_run_recorded_is_unavailable_not_zero() -> None:
    now = {"run_id": "x", "summary": {"projected_revenue": 10.0}}
    before = {"run_id": "w", "summary": {}}
    fields = history_api_runs.delta(now, before)["fields"]
    assert fields["projected_revenue"] == {"state": "unavailable", "from": None, "to": 10.0, "delta": None}


def test_the_previous_run_is_the_previous_run_of_the_same_scope() -> None:
    records = [
        _run_record("1" * 32, "2026-07-01T00:00:00+00:00", OWNED, "2024-11-11", 1.0, 1, 90.0),
        _run_record("2" * 32, "2026-07-02T00:00:00+00:00", OWNED, "2024-11-12", 2.0, 2, 90.0),
        _run_record("3" * 32, "2026-07-03T00:00:00+00:00", OWNED, "2024-11-11", 3.0, 3, 90.0),
    ]
    earlier = history_api_runs.previous_run(records, records[2])
    assert earlier is not None and earlier["run_id"] == "1" * 32, "a different day is not a comparison"


# --- the attestation ------------------------------------------------------------

def test_the_attestation_reports_the_guardrail_record_and_where_it_starts(history_env) -> None:
    body = history_env.get("/api/history/since", params={"day": "2026-07-01"}).json()
    guardrails = body["guardrails"]
    assert guardrails["state"] in ("unchanged", "changed")
    assert guardrails["record_starts"], "the evidence names the day the record itself starts"
    assert guardrails["effective_date"]
    assert body["counts"]["run"] == 2


def test_a_day_that_holds_only_previews_attests_as_unchanged(history_env) -> None:
    """The attestation answers "did anything change", so an act that saved
    nothing may not make it read changed. The counts still show the previews,
    because withholding them would be the other kind of dishonesty."""
    history_env.post("/api/plan/day/score", json={})
    history_env.post("/api/pricing/effect", json={})
    today = history_env.get("/api/history").json()["attestation"]
    assert today["counts"]["preview"] >= 2
    assert today["changed"] == 0
    assert today["verdict"] == "unchanged"
    assert today["attested_kinds"] == ["change", "restore", "restore_point"]

    vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    after = history_env.get("/api/history").json()["attestation"]
    assert after["changed"] >= 1
    assert after["verdict"] == "changed"


def test_an_unreadable_guardrail_store_answers_unknown_never_unchanged(history_env, monkeypatch) -> None:
    from kairos_api import guardrail_store

    def _boom(*args, **kwargs):
        raise OSError("the guardrail store is unreadable")

    monkeypatch.setattr(guardrail_store, "load_record", _boom)
    guardrails = history_env.get("/api/history/since", params={"day": "2026-07-01"}).json()["guardrails"]
    assert guardrails["state"] == "unknown"
    assert guardrails["reason"]
    assert guardrails["changes"] == []


def test_a_broadcast_day_is_an_israeli_day_not_a_utc_one(history_env) -> None:
    """A change saved at 01:37 in Tel Aviv belongs to that morning, not to the
    previous evening. Every record stamps UTC, so the day an entry is filed
    under is its stamp read in the broadcast zone, on the server and on the
    surface alike."""
    assert history_api_timeline.broadcast_day("2026-07-31T22:37:46.003+00:00") == "2026-08-01"
    assert history_api_timeline.broadcast_day("2026-07-31T20:00:00+00:00") == "2026-07-31"
    assert history_api_timeline.broadcast_day("not a timestamp") == "not a time"

    entries = [{"ts": "2026-07-31T22:37:46.003+00:00"}, {"ts": "2026-07-31T20:00:00+00:00"}]
    assert history_api_timeline.since_day(entries, "2026-08-01") == [entries[0]]


def test_a_malformed_day_is_refused_rather_than_guessed(history_env) -> None:
    assert history_env.get("/api/history/since", params={"day": "last tuesday"}).status_code == 400
    assert history_env.get("/api/history", params={"since": "2026"}).status_code == 400
    assert history_env.get("/api/history", params={"kind": "everything"}).status_code == 400


def test_the_attestation_answers_bare_with_today_rather_than_refusing(history_env) -> None:
    """Every other GET on this product answers with no arguments, and this one
    declared ``day`` required, so the shared endpoint sweep in
    ``tests/test_api_surface_qa.py`` measured a 422 on it. A required parameter
    was never the contract: the day this route is about when nobody names one is
    today in the broadcast zone, which is the day the timeline read stamps its
    own attestation with and the day ``HistorySince`` opens its date control on.
    So the bare read is the landing verdict, and it is the same body, which is
    what keeps the one-request landing path and this route from disagreeing.
    """
    bare = history_env.get("/api/history/since")
    assert bare.status_code == 200, bare.text
    body = bare.json()
    today = history_api_timeline.broadcast_day(datetime.now(timezone.utc).isoformat())
    assert body["day"] == today, "a bare attestation is about today, and says so"
    assert body == history_env.get("/api/history").json()["attestation"]

    # A cleared date control sends an empty day and means the same as no day,
    # which is how the timeline's own since and until already read a blank.
    assert history_env.get("/api/history/since", params={"day": ""}).json()["day"] == today

    # And the default is a default, not a fallback: a day that was supplied is
    # still the day answered on, and an impossible one is still refused rather
    # than quietly becoming today.
    named = history_env.get("/api/history/since", params={"day": "2026-07-01"}).json()
    assert named["day"] == "2026-07-01" and named["counts"]["run"] == 2
    assert history_env.get("/api/history/since", params={"day": "2026-02-31"}).status_code == 400
