"""Additional READ tool executors for the assistant's action plane.

Eight more read tools, split out of kairos_api.assistant_read_tools so both
files stay under the size cap. The conventions are identical: each executor
reuses a real builder or store of the owning module, returns an honest
``{"error": ...}`` or an explicit unavailable-with-reason payload instead of
fabricating, caps every list it returns and records the true total beside the
cap, and is stamped with a provenance source by ``execute_read_tool``.
:func:`register` merges these executors and their source labels into the shared
registry; kairos_api.assistant_read_tools calls it at import time so the
combined registry stays the only dispatch surface.

Executors take ``(args, user)`` like the core ones. ``user`` is the
authenticated session username; only the activity-log tool consults it, to
apply the same role scoping the /api/activity-log route enforces.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Output-size discipline: every list a tool returns is capped, with the honest
# overflow recorded beside it so a cap never hides how much exists.
MAX_LIST_ROWS = 20
MAX_YIELD_GROUPS = 12
MAX_RECENT_RUNS = 10
MAX_FINDINGS = 3
MAX_WARNINGS = 5


def _cap(payload: dict[str, Any], key: str, limit: int) -> None:
    """Cap ``payload[key]`` to ``limit`` rows in place, recording the overflow."""
    rows = list(payload.get(key) or [])
    payload[key] = rows[:limit]
    if len(rows) > limit:
        payload[f"{key}_total"] = len(rows)
        payload[f"{key}_omitted"] = len(rows) - limit


# --- schedule freshness: the staleness banner's truth -----------------------------
def _read_get_schedule_freshness(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos.export.schedule_freshness import ROOT, schedule_freshness

    verdict = schedule_freshness(ROOT)
    payload: dict[str, Any] = {
        "status": verdict.get("status"),
        "computed_at": verdict.get("computed_at"),
        "changed": [str(group) for group in (verdict.get("changed") or [])],
    }
    if payload["status"] == "unknown":
        payload["reason"] = "no freshness stamp exists for the saved schedule, so freshness cannot be verified"
    _cap(payload, "changed", MAX_LIST_ROWS)
    return payload


# --- yield per ad-second, owned scope ---------------------------------------------
def _read_get_yield_per_second(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    # Prefer the extracted scoped builder; the route function computes the
    # identical scoped payload (operator-channel rows only, with scope_channel
    # and the retention-cost band), so this works on either side of that seam.
    from kairos_api import insights_api

    builder = getattr(insights_api, "scoped_yield_payload", None) or insights_api.yield_per_second
    payload = dict(builder())
    _cap(payload, "by_daypart", MAX_YIELD_GROUPS)
    _cap(payload, "by_programme", MAX_YIELD_GROUPS)
    return payload


# --- gold breaks in the committed plan --------------------------------------------
def _read_get_gold_breaks(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.insights_api import gold_breaks

    # The builder already reports honest empties (disabled in settings, a
    # pre-tracking CSV, or simply no gold segments) with the reason; ``count``
    # stays the builder's true total even when the row list is capped.
    payload = dict(gold_breaks())
    _cap(payload, "breaks", MAX_LIST_ROWS)
    _cap(payload, "by_day", MAX_LIST_ROWS)
    return payload


# --- make-good alerts from the pacing projection ----------------------------------
def _read_get_make_good_alerts(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.insights_api import make_good_alerts

    # data_available and reason pass through untouched: when campaign_flights.csv
    # carries no rows the payload says so instead of inventing an alert.
    payload = dict(make_good_alerts())
    _cap(payload, "alerts", MAX_LIST_ROWS)
    return payload


# --- optimizer run log: the newest record plus a recent digest --------------------
def _run_digest(record: dict[str, Any]) -> dict[str, Any]:
    """One slim digest row: which channel-day ran, the DP-tier coverage counters
    when the tier ran, and how many optimizer notes (revert labels) it left."""
    summary = record.get("summary") or {}
    entry: dict[str, Any] = {
        "created_at": record.get("created_at"),
        "channel": record.get("channel"),
        "day": record.get("day"),
        "notes_count": len(summary.get("notes") or []),
    }
    tier = summary.get("dp_tier")
    if isinstance(tier, dict):
        entry["dp_tier"] = {
            key: tier.get(key)
            for key in (
                "groups_total",
                "groups_exact",
                "groups_adopted",
                "groups_not_better",
                "groups_noncompliant",
            )
        }
    return entry


def _read_get_run_log_summary(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos.observability.run_log import read_run_log
    from kairos_api.core import _load_settings

    # The audit trail records runs across every channel in the source data, so it
    # is scoped to the operator's channel here: a competitor's runs (names and
    # money) must never surface, matching every other read tool's boundary.
    owned = str(getattr(_load_settings(), "operator_channel", "") or "").strip()
    if not owned:
        return {"status": "unavailable", "reason": "no operator channel is configured in settings, so run-log records cannot be scoped"}
    records = read_run_log()
    if not records:
        return {"status": "unavailable", "reason": "no optimization runs have been logged yet"}
    records = [record for record in records if str(record.get("channel") or "").strip() == owned]
    if not records:
        return {"status": "unavailable", "reason": "the run log carries no records for the configured operator channel"}
    newest = dict(records[-1])
    summary = dict(newest.get("summary") or {})
    if isinstance(summary.get("notes"), list):
        notes = [str(note) for note in summary["notes"]]
        summary["notes"] = notes[:MAX_LIST_ROWS]
        if len(notes) > MAX_LIST_ROWS:
            summary["notes_omitted"] = len(notes) - MAX_LIST_ROWS
    payload: dict[str, Any] = {
        "records_total": len(records),
        "run_id": newest.get("run_id"),
        "created_at": newest.get("created_at"),
        "channel": newest.get("channel"),
        "day": newest.get("day"),
        "segment_count": newest.get("segment_count"),
        "engine_version": newest.get("engine_version"),
        "summary": summary,
    }
    if "dp_tier" not in summary:
        payload["dp_tier_note"] = "this run's summary carries no dp_tier counters; the DP tier did not run or the run predates the counters"
    # Newest first, so "which channel-days ran the exact tier lately" is answerable
    # from one call without replaying the whole log.
    payload["recent_runs"] = [_run_digest(record) for record in records[-MAX_RECENT_RUNS:][::-1]]
    return payload


# --- per-kind input upload status -------------------------------------------------
def _slim_validation(report: Any) -> Any:
    """The last data-contract report with its finding lists capped. ``None``
    stays ``None``, the honest never-validated state."""
    if not isinstance(report, dict):
        return None
    slim: dict[str, Any] = {
        key: report.get(key)
        for key in ("dataset", "filename", "checked_at", "accepted", "is_valid", "rows_loaded")
    }
    for key in ("errors", "warnings"):
        findings = [str(item) for item in (report.get(key) or [])]
        slim[key] = findings[:MAX_FINDINGS]
        if len(findings) > MAX_FINDINGS:
            slim[f"{key}_omitted"] = len(findings) - MAX_FINDINGS
    return slim


def _read_get_upload_status(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.uploads import upload_status

    inputs: list[dict[str, Any]] = []
    for entry in (upload_status().get("inputs") or [])[:MAX_LIST_ROWS]:
        slim: dict[str, Any] = {
            key: entry.get(key)
            for key in (
                "kind",
                "label_en",
                "cadence",
                "filename",
                "exists",
                "rows",
                "last_modified",
                "valid",
                "in_use",
                "in_use_reason",
                "engine_reads",
            )
        }
        warnings = [str(warning) for warning in (entry.get("warnings") or [])]
        slim["warnings"] = warnings[:MAX_WARNINGS]
        if len(warnings) > MAX_WARNINGS:
            slim["warnings_omitted"] = len(warnings) - MAX_WARNINGS
        slim["last_validation"] = _slim_validation(entry.get("last_validation"))
        inputs.append(slim)
    return {"inputs": inputs, "count": len(inputs)}


# --- reports catalog --------------------------------------------------------------
def _read_get_reports_catalog(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.catalog_api import reports

    rows = [
        {key: row.get(key) for key in ("id", "title", "status", "rows")}
        for row in (reports().get("reports") or [])[:MAX_LIST_ROWS]
    ]
    return {"reports": rows, "count": len(rows)}


# --- recent activity, metadata only -----------------------------------------------
def _read_get_activity_recent(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.activity_log import _read_entries
    from kairos_api.auth import auth_active

    # Mirror the /api/activity-log visibility rule: the admin role sees every
    # entry, any other role sees only its own, and no identity sees nothing.
    scope = "all"
    if auth_active():
        if not user:
            return {"status": "unavailable", "reason": "a signed-in session is required to read the activity log"}
        from kairos_api import auth_store

        record = auth_store.get_user(user)
        if not isinstance(record, dict) or record.get("role") != "admin":
            scope = "self"
    entries = list(_read_entries())
    entries.reverse()
    if scope == "self":
        entries = [entry for entry in entries if entry.get("user") == user]
    # Metadata only, by construction: exactly these five fields, never a body.
    slim = [
        {
            "at": entry.get("ts"),
            "user": entry.get("user"),
            "action": entry.get("event") if entry.get("event") != "request" else (entry.get("method") or "request"),
            "path": entry.get("path"),
            "status": entry.get("status"),
        }
        for entry in entries[:MAX_LIST_ROWS]
    ]
    return {"entries": slim, "count": len(slim), "entries_total": len(entries), "scope": scope}


_EXTRA_READ_EXECUTORS = {
    "get_schedule_freshness": _read_get_schedule_freshness,
    "get_yield_per_second": _read_get_yield_per_second,
    "get_gold_breaks": _read_get_gold_breaks,
    "get_make_good_alerts": _read_get_make_good_alerts,
    "get_run_log_summary": _read_get_run_log_summary,
    "get_upload_status": _read_get_upload_status,
    "get_reports_catalog": _read_get_reports_catalog,
    "get_activity_recent": _read_get_activity_recent,
}

# Provenance stamps, same vocabulary as SOURCE_BY_TOOL in assistant_read_tools:
# the endpoint or dataset the figures came from, surfaced on the trace.
EXTRA_SOURCE_BY_TOOL = {
    "get_schedule_freshness": "schedule freshness sidecar (input fingerprints)",
    "get_yield_per_second": "saved weekly plan, owned-scope yield",
    "get_gold_breaks": "saved weekly plan, gold segments",
    "get_make_good_alerts": "pacing make-good projection",
    "get_run_log_summary": "optimizer run log",
    "get_upload_status": "input upload status",
    "get_reports_catalog": "reports catalog",
    "get_activity_recent": "activity log (metadata only)",
}


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge these executors and their source labels into the shared registry."""
    executors.update(_EXTRA_READ_EXECUTORS)
    sources.update(EXTRA_SOURCE_BY_TOOL)
