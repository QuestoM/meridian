"""Kai reads the two kinds of version without crossing their ownership lines.

Plan versions are run-side records, so every account may read the operator's
own summary. Model candidates and adoption decisions are company-side training
content, so the shared model-disclosure wall replaces that tool for a channel
account with only the shipped version and release note.
"""

from __future__ import annotations

from typing import Any

MAX_PLAN_VERSIONS = 20
MAX_CANDIDATES = 10
MAX_DECISIONS = 20


VERSION_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_plan_versions",
        "description": (
            "Read the live saved weekly plan and its named frozen plan versions. Returns "
            "only the operator-channel totals, whether the live bytes are already frozen, "
            "and each version's owned-channel delta from the previous freeze. It never "
            "returns all-channel totals or rival exclusion counts. Call this when the "
            "operator asks which plan version is current or what changed between runs."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_model_adoption",
        "description": (
            "Read the company-side model-candidate shelf and adoption decisions: the "
            "current model version, what each candidate changes, its owned-channel money "
            "measurement and the recorded ship/no-ship decision. Model internals are "
            "company-only; a channel-affiliated account receives only the shipped model "
            "version and release note from the standard disclosure wall. This tool reads "
            "decisions and never records or applies one."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

VERSION_SOURCE_BY_TOOL = {
    "get_plan_versions": "named plan-version freezes, operator-channel summaries only",
    "get_model_adoption": "company model-candidate artifacts and adoption decision ledger",
}


def _owned_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict) or not isinstance(summary.get("owned"), dict):
        return None
    return dict(summary["owned"])


def _delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(current.get("rows") or 0) - int(previous.get("rows") or 0),
        "breaks": int(current.get("breaks") or 0) - int(previous.get("breaks") or 0),
        "ad_seconds": int(current.get("ad_seconds") or 0) - int(previous.get("ad_seconds") or 0),
        "revenue": round(float(current.get("revenue") or 0) - float(previous.get("revenue") or 0), 2),
    }


def _version_totals(store: Any, channel_scope: Any, version_id: Any, channel: str) -> dict[str, Any] | None:
    """Re-scope frozen bytes at read time; never trust their historical label."""
    frame = store._frame_for(str(version_id))  # the store's frozen plan reader
    if frame is None:
        return None
    owned, note = channel_scope.scope_frame(frame, channel=channel)
    if not note.get("scoped"):
        return None
    return store._totals(owned)


def _plan_version(
    record: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    channel: str,
    store: Any,
    channel_scope: Any,
) -> dict[str, Any]:
    settings = dict(record.get("settings_basis") or {})
    # The channel frozen in the historical settings is not the current caller's
    # scope and may now be a rival. Numeric settings remain useful; the old
    # channel label never leaves this reader.
    settings.pop("operator_channel", None)
    owned = _version_totals(store, channel_scope, record.get("version_id"), channel)
    previous_owned = (
        _version_totals(store, channel_scope, previous.get("version_id"), channel)
        if previous else None
    )
    return {
        "version_id": record.get("version_id"),
        "seq": record.get("seq"),
        "name": record.get("name"),
        "note": record.get("note"),
        "created_at": record.get("created_at"),
        "actor": record.get("actor"),
        "source_kind": record.get("source"),
        "computed_at": record.get("computed_at"),
        "settings_basis": settings,
        "owned_summary": owned,
        "previous_version_id": record.get("previous_version_id"),
        "owned_delta_from_previous": (
            _delta(owned, previous_owned)
            if isinstance(owned, dict) and isinstance(previous_owned, dict) else None
        ),
    }


def _read_get_plan_versions(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    del args, user
    from kairos_api import channel_scope, plan_version_store as store

    channel = channel_scope.operator_channel()
    if not channel:
        return {
            "error": "the operator channel is not configured, so plan-version totals cannot be scoped",
            "scope_available": False,
        }
    live = store.live_state()
    manifests = store.all_manifests()
    limited = manifests[:MAX_PLAN_VERSIONS]
    by_id = {str(record.get("version_id")): record for record in manifests}
    versions = [
        _plan_version(
            record,
            by_id.get(str(record.get("previous_version_id"))),
            channel=channel,
            store=store,
            channel_scope=channel_scope,
        )
        for record in limited
    ]
    live_owned = _owned_summary(live.get("summary"))
    latest_owned = versions[0].get("owned_summary") if versions else None
    payload: dict[str, Any] = {
        "scope": {"scope_channel": channel, "scoped": True},
        "live": {
            "exists": bool(live.get("exists")),
            "computed_at": live.get("computed_at"),
            "frozen_as": live.get("frozen_as"),
            "owned_summary": live_owned,
            "comparison_to_latest": (
                {
                    "version_id": versions[0].get("version_id"),
                    "name": versions[0].get("name"),
                    "delta": _delta(live_owned, latest_owned),
                }
                if isinstance(live_owned, dict) and isinstance(latest_owned, dict) else None
            ),
        },
        "versions": versions,
        "versions_count": len(manifests),
        "proposing_or_restoring": "not available in this read tool",
    }
    if len(manifests) > len(versions):
        payload["versions_omitted"] = len(manifests) - len(versions)
    return payload


def _decision_view(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    evidence = dict(record.get("evidence") or {})
    scope = dict(evidence.get("scope") or {})
    safe_evidence = {
        key: evidence.get(key)
        for key in ("gate_counts", "gate_total", "money_state", "revenue_delta",
                    "revenue_delta_pct", "measured_at")
        if key in evidence
    }
    if scope:
        safe_evidence["scope"] = {
            key: scope.get(key) for key in ("rows", "basis") if key in scope
        }
    return {
        "decision_id": record.get("decision_id"),
        "recorded_at": record.get("recorded_at"),
        "actor": record.get("actor"),
        "model_version_id": record.get("model_version_id"),
        "subject": record.get("subject"),
        "candidate_id": record.get("candidate_id"),
        "decision": record.get("decision"),
        "reason": record.get("reason"),
        "release_note_he": record.get("release_note_he"),
        "release_note_en": record.get("release_note_en"),
        "money_direction": record.get("money_direction"),
        "evidence": safe_evidence,
        "adoption": record.get("adoption"),
    }


def _candidate_view(row: dict[str, Any], decision: Any) -> dict[str, Any]:
    money = dict(row.get("money") or {})
    safe_money = {
        key: money.get(key)
        for key in ("state", "changed", "reason_en", "reason_he", "measured_at")
        if key in money
    }
    if isinstance(money.get("operator_channel_delta"), dict):
        safe_money["operator_channel_delta"] = dict(money["operator_channel_delta"])
    gates = [
        {key: item.get(key) for key in
         ("key", "shipped", "candidate", "shipped_absent", "candidate_absent")}
        for item in list(row.get("gate_deltas") or [])
    ]
    held_out = [
        {key: item.get(key) for key in
         ("gate_id", "label_en", "label_he", "moved", "reason_shipped", "reason_candidate")}
        for item in list(row.get("held_out_deltas") or [])
    ]
    return {
        "candidate_id": row.get("id"),
        "computed_at": row.get("computed_at"),
        "purpose": row.get("purpose"),
        "subject_en": row.get("subject_en"),
        "subject_he": row.get("subject_he"),
        "differences": row.get("differences"),
        "coefficient_deltas": row.get("coefficient_deltas"),
        "gate_deltas": gates,
        "held_out_deltas": held_out,
        "money": safe_money,
        "latest_decision": _decision_view(decision),
    }


def _compact_model_version(version: Any) -> dict[str, Any]:
    if not isinstance(version, dict):
        return {"available": False}
    return {
        key: version.get(key)
        for key in ("available", "id", "name", "short", "trained_at", "recorded")
        if key in version
    }


def _read_get_model_adoption(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    del args, user
    from kairos_api import model_console_artifacts as artifacts
    from kairos_api import model_console_candidates as candidates
    from kairos_api import model_console_api_payloads as payloads
    from kairos_api import model_version_store as store

    version = payloads.current_version()
    version_id = str(version.get("id") or "")
    paths = candidates.candidate_paths()
    measurements = store.measurements()
    rows = []
    for path in paths[:MAX_CANDIDATES]:
        candidate_id = candidates.candidate_id(path)
        raw = candidates.summary_row(path, artifacts.retention_metadata(),
                                     measurements.get(candidate_id))
        decision = store.latest_decision(version_id, "candidate", candidate_id)
        rows.append(_candidate_view(raw, decision))
    decisions_all = store.decisions()
    decisions = [_decision_view(item) for item in decisions_all[:MAX_DECISIONS]]
    payload: dict[str, Any] = {
        "available": bool(version.get("available")),
        "model_version": _compact_model_version(version),
        "candidates": rows,
        "candidates_count": len(paths),
        "decisions": decisions,
        "decisions_count": len(decisions_all),
        "current_model_decision": _decision_view(store.latest_decision(version_id)),
        "proposing_or_adopting": "not available in this read tool",
    }
    if len(paths) > len(rows):
        payload["candidates_omitted"] = len(paths) - len(rows)
    if len(decisions_all) > len(decisions):
        payload["decisions_omitted"] = len(decisions_all) - len(decisions)
    return payload


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    executors.update({
        "get_plan_versions": _read_get_plan_versions,
        "get_model_adoption": _read_get_model_adoption,
    })
    sources.update(VERSION_SOURCE_BY_TOOL)
