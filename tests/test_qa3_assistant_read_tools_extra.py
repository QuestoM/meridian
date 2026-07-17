"""Contract tests for the eight additional assistant READ tools.

Dispatch goes through the real seam (assistant_tools.handle_tool_use on a
tool_use-shaped block with a trace list), so the registry wiring, the
provenance stamps and the trace steps are all exercised. Below the seam the
default paths run the REAL builders on the repository's saved data; synthetic
monkeypatched payloads are used only to force the cap and scoping branches the
committed data cannot be assumed to reach. No Anthropic client is constructed
and no live server is touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import kairos_api.assistant_tools as tools

NEW_TOOLS = (
    "get_schedule_freshness",
    "get_yield_per_second",
    "get_gold_breaks",
    "get_make_good_alerts",
    "get_run_log_summary",
    "get_upload_status",
    "get_reports_catalog",
    "get_activity_recent",
)


def _dispatch(name: str, args: dict[str, Any] | None = None, user: str | None = None):
    """One tool_use-shaped block through the real handle_tool_use seam."""
    block = SimpleNamespace(name=name, input=dict(args or {}), id=f"tu_{name}")
    trace: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    message = tools.handle_tool_use(block, trace, items, user=user)
    assert message["type"] == "tool_result"
    assert message["tool_use_id"] == f"tu_{name}"
    assert items == [], "read tools must never capture proposal items"
    return json.loads(message["content"]), trace


def _plan_channels() -> tuple[str, list[str]]:
    import kairos_api.assistant_context as assistant_context

    server = assistant_context._server()
    frame = pd.read_csv(server.OUTPUT_DIR / "weekly_break_schedule.csv")
    owned = str(server._load_settings().operator_channel or "").strip()
    channels = {text for text in frame["channel"].astype(str).str.strip().unique() if text}
    return owned, sorted(channels - {owned})


# --- registry wiring ---------------------------------------------------------------
def test_new_tools_are_registered_with_sources_and_schemas() -> None:
    for name in NEW_TOOLS:
        assert name in tools.READ_TOOL_NAMES
        assert name in tools._READ_EXECUTORS
        assert tools.SOURCE_BY_TOOL[name]
    read_only = {schema["name"] for schema in tools.anthropic_tools(include_propose=False)}
    assert set(NEW_TOOLS) <= read_only
    assert not set(NEW_TOOLS) & tools.PROPOSE_TOOL_NAMES
    by_name = {schema["name"]: schema for schema in tools.READ_TOOL_SCHEMAS}
    for name in NEW_TOOLS:
        description = by_name[name]["description"]
        assert "Call this when" in description
        assert "!" not in description


def test_every_new_tool_dispatches_with_a_source_stamp_on_the_trace() -> None:
    for name in NEW_TOOLS:
        payload, trace = _dispatch(name)
        assert isinstance(payload, dict), name
        assert payload.get("source") == tools.SOURCE_BY_TOOL[name], name
        step = trace[-1]
        assert step["tool"] == name
        assert step.get("source") == tools.SOURCE_BY_TOOL[name]


# --- get_schedule_freshness: the staleness banner's truth --------------------------
def test_get_schedule_freshness_matches_the_sidecar_verdict() -> None:
    from kairos.export.schedule_freshness import ROOT, schedule_freshness

    expected = schedule_freshness(ROOT)
    payload, _ = _dispatch("get_schedule_freshness")
    assert payload["status"] == expected["status"]
    assert payload["status"] in {"fresh", "stale", "unknown"}
    assert payload["computed_at"] == expected["computed_at"]
    assert payload["changed"] == [str(group) for group in expected["changed"]][:20]
    if payload["status"] == "unknown":
        assert payload["reason"]


# --- get_yield_per_second: owned scope, band, caps ---------------------------------
def test_get_yield_per_second_is_scoped_with_band_and_capped_lists() -> None:
    payload, _ = _dispatch("get_yield_per_second")
    assert "scope_channel" in payload
    assert "available" in payload
    if payload["available"]:
        assert len(payload["by_daypart"]) <= 12
        assert len(payload["by_programme"]) <= 12
        assert "totals" in payload
        if payload.get("revenue_net_available"):
            # The band keys are part of the contract whenever net is derivable;
            # None values are the honest no-band state, never missing keys.
            assert "retention_cost_low" in payload
            assert "retention_cost_high" in payload
    else:
        assert payload["reason"]


def test_get_yield_per_second_prefers_the_extracted_scoped_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import insights_api

    sentinel = {"available": True, "scope_channel": "owned", "by_daypart": [], "by_programme": [], "totals": {}}
    monkeypatch.setattr(insights_api, "scoped_yield_payload", lambda: dict(sentinel), raising=False)
    payload, _ = _dispatch("get_yield_per_second")
    assert payload["scope_channel"] == "owned"


def test_get_yield_per_second_caps_long_group_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import insights_api

    groups = [{"group": f"g{i}", "yield_per_second": float(i)} for i in range(15)]
    fake = {"available": True, "scope_channel": None, "by_daypart": list(groups), "by_programme": [], "totals": {}}
    monkeypatch.setattr(insights_api, "scoped_yield_payload", lambda: dict(fake), raising=False)
    payload, _ = _dispatch("get_yield_per_second")
    assert len(payload["by_daypart"]) == 12
    assert payload["by_daypart_total"] == 15
    assert payload["by_daypart_omitted"] == 3


# --- get_gold_breaks: honest empty and the 20-row cap ------------------------------
def test_get_gold_breaks_real_payload_is_honest() -> None:
    payload, _ = _dispatch("get_gold_breaks")
    assert "available" in payload
    assert len(payload.get("breaks", [])) <= 20
    if payload.get("count") == 0:
        assert payload["reason"]


def test_get_gold_breaks_caps_rows_and_keeps_the_true_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import insights_api

    rows = [{"segment_id": f"s{i}", "day": "2026-07-14", "revenue": 1.0} for i in range(30)]
    fake = {"available": True, "enabled": True, "count": 30, "breaks": rows, "by_day": [{"day": "2026-07-14", "count": 30}]}
    monkeypatch.setattr(insights_api, "gold_breaks", lambda: dict(fake))
    payload, _ = _dispatch("get_gold_breaks")
    assert len(payload["breaks"]) == 20
    assert payload["breaks_total"] == 30
    assert payload["breaks_omitted"] == 10
    assert payload["count"] == 30


# --- get_make_good_alerts: data_available passthrough and cap ----------------------
def test_get_make_good_alerts_reports_data_availability_honestly() -> None:
    payload, _ = _dispatch("get_make_good_alerts")
    assert "data_available" in payload
    if not payload["data_available"]:
        assert payload["reason"]
        assert payload["alerts"] == []


def test_get_make_good_alerts_caps_the_alert_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import insights_api

    alerts = [{"campaign_id": f"c{i}", "projected_shortfall": 0.1} for i in range(25)]
    fake = {"alerts": alerts, "data_available": True, "count": 25, "as_of": "2026-07-14"}
    monkeypatch.setattr(insights_api, "make_good_alerts", lambda: dict(fake))
    payload, _ = _dispatch("get_make_good_alerts")
    assert len(payload["alerts"]) == 20
    assert payload["alerts_total"] == 25
    assert payload["alerts_omitted"] == 5
    assert payload["data_available"] is True


# --- get_run_log_summary: newest owned record, dp_tier counters, honest absences ---
def test_get_run_log_summary_reads_the_newest_owned_record() -> None:
    from kairos.observability.run_log import read_run_log

    owned, _competitors = _plan_channels()
    records = [r for r in read_run_log() if str(r.get("channel") or "").strip() == owned]
    assert records, "the repository run log carries owned-channel records; this test needs them"
    payload, _ = _dispatch("get_run_log_summary")
    newest = records[-1]
    assert payload["records_total"] == len(records)
    assert payload["run_id"] == newest.get("run_id")
    assert payload["created_at"] == newest.get("created_at")
    assert payload["channel"] == owned
    assert isinstance(payload["summary"], dict)
    assert len(payload["recent_runs"]) <= 10
    assert payload["recent_runs"][0]["created_at"] == newest.get("created_at")
    if "dp_tier" in payload["summary"]:
        assert "groups_total" in payload["summary"]["dp_tier"]
    else:
        assert payload["dp_tier_note"]


def test_get_run_log_summary_unavailable_when_no_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kairos.observability.run_log.read_run_log", lambda: [])
    payload, trace = _dispatch("get_run_log_summary")
    assert payload["status"] == "unavailable"
    assert "no optimization runs" in payload["reason"]
    assert trace[-1]["ok"] is True


def test_get_run_log_summary_surfaces_dp_tier_and_revert_notes_owned_only(monkeypatch: pytest.MonkeyPatch) -> None:
    owned, competitors = _plan_channels()
    tier = {"groups_total": 3, "groups_exact": 2, "groups_adopted": 1, "groups_not_better": 1, "groups_noncompliant": 0, "fallback_reasons": {}}
    with_tier = {
        "run_id": "r2", "created_at": "2026-07-14T10:00:00+00:00", "channel": owned, "day": "2026-07-14",
        "segment_count": 5, "engine_version": "1.0.0",
        "summary": {"total_breaks": 9, "dp_tier": tier, "notes": ["refinement reverted to the pure greedy plan: scores disagreed"]},
    }
    without_tier = {
        "run_id": "r1", "created_at": "2026-07-13T10:00:00+00:00", "channel": owned, "day": "2026-07-13",
        "segment_count": 5, "engine_version": "1.0.0", "summary": {"total_breaks": 8},
    }
    competitor_run = {
        "run_id": "r3", "created_at": "2026-07-14T11:00:00+00:00", "channel": competitors[0], "day": "2026-07-14",
        "segment_count": 4, "engine_version": "1.0.0", "summary": {"total_breaks": 7},
    }
    monkeypatch.setattr(
        "kairos.observability.run_log.read_run_log", lambda: [without_tier, with_tier, competitor_run]
    )
    payload, _ = _dispatch("get_run_log_summary")
    # The competitor's newer run is invisible: the owned r2 is the newest surfaced.
    assert payload["run_id"] == "r2"
    assert payload["records_total"] == 2
    assert competitors[0] not in json.dumps(payload, ensure_ascii=False, default=str)
    assert payload["summary"]["dp_tier"]["groups_exact"] == 2
    assert "reverted" in payload["summary"]["notes"][0]
    assert "dp_tier_note" not in payload
    digests = payload["recent_runs"]
    assert digests[0]["day"] == "2026-07-14" and digests[0]["dp_tier"]["groups_adopted"] == 1
    assert digests[0]["notes_count"] == 1
    assert digests[1]["day"] == "2026-07-13" and "dp_tier" not in digests[1]

    # A newest record without counters gets the honest absence note instead.
    monkeypatch.setattr("kairos.observability.run_log.read_run_log", lambda: [without_tier])
    payload, _ = _dispatch("get_run_log_summary")
    assert "dp_tier" not in payload["summary"]
    assert "dp_tier" in payload["dp_tier_note"]

    # Only competitor runs in the log: an honest unavailable, never a leak.
    monkeypatch.setattr("kairos.observability.run_log.read_run_log", lambda: [competitor_run])
    payload, _ = _dispatch("get_run_log_summary")
    assert payload["status"] == "unavailable"
    assert competitors[0] not in json.dumps(payload, ensure_ascii=False, default=str)


# --- get_upload_status: slim per-kind entries with honest amber state --------------
def test_get_upload_status_real_payload_carries_in_use_and_validation() -> None:
    payload, _ = _dispatch("get_upload_status")
    assert payload["count"] == len(payload["inputs"]) > 0
    for entry in payload["inputs"]:
        assert {"kind", "exists", "valid", "in_use", "in_use_reason", "engine_reads", "last_validation"} <= set(entry)
        assert "columns" not in entry, "the column list is dropped for output-size discipline"
        assert len(entry["warnings"]) <= 5


def test_get_upload_status_caps_warnings_and_validation_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import uploads

    entry = {
        "kind": "daily", "label_en": "Daily", "cadence": "daily", "filename": "d.xlsx", "exists": True,
        "rows": 9, "last_modified": None, "valid": False, "in_use": False, "in_use_reason": "shadowed by the xlsx",
        "engine_reads": "data/reference/Spots.xlsx", "columns": ["a"] * 30,
        "warnings": [f"warning {i}" for i in range(9)],
        "last_validation": {"dataset": "daily", "filename": "d.xlsx", "checked_at": "t", "accepted": False,
                           "is_valid": False, "rows_loaded": 9,
                           "errors": [f"error {i}" for i in range(5)], "warnings": ["w1"]},
    }
    monkeypatch.setattr(uploads, "upload_status", lambda: {"inputs": [dict(entry)]})
    payload, _ = _dispatch("get_upload_status")
    slim = payload["inputs"][0]
    assert slim["in_use"] is False and slim["in_use_reason"] == "shadowed by the xlsx"
    assert len(slim["warnings"]) == 5 and slim["warnings_omitted"] == 4
    report = slim["last_validation"]
    assert report["accepted"] is False
    assert len(report["errors"]) == 3 and report["errors_omitted"] == 2
    assert "columns" not in slim


# --- get_reports_catalog: slim rows ------------------------------------------------
def test_get_reports_catalog_returns_slim_rows() -> None:
    payload, _ = _dispatch("get_reports_catalog")
    assert payload["count"] == len(payload["reports"]) >= 1
    for row in payload["reports"]:
        assert set(row) == {"id", "title", "status", "rows"}


# --- get_activity_recent: role scoping, metadata only, cap -------------------------
def _fake_entries() -> list[dict[str, Any]]:
    entries = []
    for i in range(30):
        entries.append({
            "ts": f"2026-07-14T10:{i:02d}:00.000+00:00", "user": "alice" if i % 2 else "bob",
            "role": "admin" if i % 2 else "editor", "event": "request", "method": "POST",
            "path": f"/api/overrides/{i}", "status": 201, "duration_ms": 4.2, "via": "dashboard",
        })
    entries.append({"ts": "2026-07-14T11:00:00.000+00:00", "user": "bob", "role": "editor",
                    "event": "login", "method": None, "path": None, "status": None,
                    "duration_ms": None, "via": "dashboard"})
    return entries


def test_get_activity_recent_scopes_non_admin_to_self(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kairos_api.auth.auth_active", lambda: True)
    monkeypatch.setattr("kairos_api.auth_store.get_user", lambda username: {"username": username, "role": "editor"})
    monkeypatch.setattr("kairos_api.activity_log._read_entries", _fake_entries)
    payload, _ = _dispatch("get_activity_recent", user="bob")
    assert payload["scope"] == "self"
    assert payload["entries"], "bob has entries in the fake log"
    assert all(entry["user"] == "bob" for entry in payload["entries"])
    # Newest first: the login event is the latest bob entry.
    assert payload["entries"][0]["action"] == "login"
    assert payload["entries"][1]["action"] == "POST"
    for entry in payload["entries"]:
        assert set(entry) == {"at", "user", "action", "path", "status"}


def test_get_activity_recent_admin_sees_all_and_caps_at_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kairos_api.auth.auth_active", lambda: True)
    monkeypatch.setattr("kairos_api.auth_store.get_user", lambda username: {"username": username, "role": "admin"})
    monkeypatch.setattr("kairos_api.activity_log._read_entries", _fake_entries)
    payload, _ = _dispatch("get_activity_recent", user="alice")
    assert payload["scope"] == "all"
    assert payload["count"] == len(payload["entries"]) == 20
    assert payload["entries_total"] == 31
    assert {entry["user"] for entry in payload["entries"]} == {"alice", "bob"}


def test_get_activity_recent_requires_identity_when_auth_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kairos_api.auth.auth_active", lambda: True)
    payload, trace = _dispatch("get_activity_recent", user=None)
    assert payload["status"] == "unavailable"
    assert "signed-in session" in payload["reason"]
    assert trace[-1]["ok"] is True


def test_get_activity_recent_open_scope_when_auth_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kairos_api.auth.auth_active", lambda: False)
    monkeypatch.setattr("kairos_api.activity_log._read_entries", _fake_entries)
    payload, _ = _dispatch("get_activity_recent", user=None)
    assert payload["scope"] == "all"
    assert payload["count"] == 20


# --- failure honesty: a crashing builder becomes an error, still stamped -----------
def test_a_crashing_builder_reports_an_honest_error_with_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import insights_api

    def _boom() -> dict[str, Any]:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(insights_api, "gold_breaks", _boom)
    payload, trace = _dispatch("get_gold_breaks")
    assert "get_gold_breaks failed" in payload["error"]
    assert "disk on fire" not in payload["error"], "internals never leak to the model"
    assert payload["source"] == tools.SOURCE_BY_TOOL["get_gold_breaks"]
    assert trace[-1]["ok"] is False


# --- audit: money tools stay owned-scope, stores are capped ------------------------
def test_money_tools_never_leak_competitor_channel_names() -> None:
    owned, competitors = _plan_channels()
    assert owned and competitors, "the saved plan must carry competitor channels for this test to bite"
    for name in ("get_net_comparison", "get_yield_per_second"):
        serialized = json.dumps(tools.execute_read_tool(name, {}), ensure_ascii=False, default=str)
        for competitor in competitors:
            assert competitor not in serialized, f"{name} leaked {competitor}"


def test_list_constraints_and_overrides_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import constraints as constraints_api
    from kairos_api import overrides as overrides_api

    fake_constraints = [{"id": f"c{i}", "effect": "forbid"} for i in range(60)]
    monkeypatch.setattr(constraints_api, "list_constraints", lambda: {"constraints": list(fake_constraints), "columns": []})
    payload = tools.execute_read_tool("list_constraints", {})
    assert len(payload["constraints"]) == 50
    assert payload["count"] == 60
    assert payload["truncated"] is True and payload["constraints_omitted"] == 10

    fake_overrides = {"segment": [{"id": f"o{i}"} for i in range(55)], "spot": [{"id": "s0"}]}
    monkeypatch.setattr(overrides_api, "list_overrides", lambda: {"overrides": dict(fake_overrides), "columns": []})
    payload = tools.execute_read_tool("list_overrides", {})
    assert len(payload["overrides"]["segment"]) == 50
    assert len(payload["overrides"]["spot"]) == 1
    assert payload["count"] == 56
    assert payload["truncated"] is True and payload["segment_omitted"] == 5
