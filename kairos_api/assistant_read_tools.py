"""READ tool executors for the assistant's action plane.

Each executor reuses a real dashboard builder or store of the owning module and
returns an honest ``{"error": ...}`` on failure rather than fabricating or
crashing; every result is stamped with a provenance source in
``execute_read_tool``. This module is split out of kairos_api.assistant_tools so
that both files stay under the size cap; the public names (``execute_read_tool``,
``_READ_EXECUTORS``, ``SOURCE_BY_TOOL``) are re-exported from assistant_tools for
back-compat.

Executors take ``(args, user)``. ``user`` is the authenticated session username
resolved by the caller; only the per-user upload tools consult it, and they are
strictly keyed by it so no request can read another operator's uploads.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _allowed_settings_fields() -> frozenset[str]:
    from kairos_api.assistant_tools import ALLOWED_SETTINGS_FIELDS

    return ALLOWED_SETTINGS_FIELDS


# --- optimizer/plan/pricing reads -------------------------------------------------
def _read_get_settings(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.core import _load_settings, _model_dump

    saved = _model_dump(_load_settings())
    subset = {field: saved.get(field) for field in sorted(_allowed_settings_fields())}
    subset["operator_channel"] = saved.get("operator_channel") or None
    return subset


def _read_get_day_detail(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import assistant_context

    date = str(args.get("date", "")).strip()
    if not _ISO_DAY_RE.fullmatch(date):
        return {"error": f"date must be YYYY-MM-DD, got {date!r}"}
    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"error": reason or "no owned-channel plan is available"}
    if date not in set(frame["date_text"]):
        return {"error": f"the saved plan has no rows for {date} on {owned}"}
    return assistant_context._day_detail_section(frame, date, [], [])


def _read_list_constraints(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.constraints import list_constraints

    # Capped: the store is an unbounded operator CSV, and the honest total rides
    # beside the cap so a truncation never hides how many constraints exist.
    records = list(list_constraints()["constraints"])
    payload: dict[str, Any] = {"constraints": records[:50], "count": len(records)}
    if len(records) > 50:
        payload["truncated"] = True
        payload["constraints_omitted"] = len(records) - 50
    return payload


def _read_list_overrides(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.overrides import list_overrides

    # Same cap discipline as list_constraints, applied per scope group.
    grouped = list_overrides()["overrides"]
    capped: dict[str, list[dict[str, Any]]] = {}
    payload: dict[str, Any] = {"overrides": capped, "count": 0}
    for scope, records in grouped.items():
        records = list(records)
        payload["count"] += len(records)
        capped[scope] = records[:50]
        if len(records) > 50:
            payload["truncated"] = True
            payload[f"{scope}_omitted"] = len(records) - 50
    return payload


def _read_get_pricing(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.pricing_api import get_pricing

    state = get_pricing()
    layers = [
        {
            "name": layer["name"],
            "live_today": layer["live_today"],
            **({"value": layer["value"]} if layer.get("kind") == "base" else {"values": layer.get("values", {})}),
        }
        for layer in state["layers"]
    ]
    return {
        "currency": state["currency"],
        "units": state["units"],
        "base": state["base"],
        "layers": layers,
        "activation": state["activation"],
        # The event-date layer state rides along untouched (enabled flag, count
        # of active non-1.0 events, assertion basis), so the model knows the
        # events layer exists; get_event_pricing has the per-event detail.
        "events": state.get("events"),
        "has_operator_overrides": state["has_overrides"],
    }


def _read_get_net_comparison(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.scenario_api import optimizer_net_comparison

    return optimizer_net_comparison()


def _read_get_compliance(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.dashboard_api import compliance

    verdict = compliance()
    checks = [
        {key: check.get(key) for key in ("id", "status", "observed", "limit", "unit")}
        for check in verdict.get("checks", [])
    ]
    violations = verdict.get("violations") or []
    return {
        "status": verdict.get("status"),
        "profile": verdict.get("profile"),
        "checks": checks,
        "violations_total": len(violations),
        "violations_sample": violations[:5],
    }


def _read_simulate_settings_change(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    # The primitive validates the changes against the SAME allowlist and returns an
    # honest unavailable for a forbidden/unknown field or a bad value, never crashing.
    from kairos_api import assistant_simulate

    return assistant_simulate.simulate_settings_change(args.get("changes"))


def _read_get_recommendations(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.core import _load_break_schedule
    from kairos_api.dashboard_api import _build_recommendations

    rows = _build_recommendations(_load_break_schedule())
    slim = [
        {
            "title": row.get("title"),
            "title_he": row.get("title_he"),
            "risk": row.get("risk"),
            "channel": row.get("channel"),
            "segment_id": row.get("segment_id"),
            "program_type": row.get("program_type"),
            "date": row.get("date"),
            "weekday": row.get("weekday"),
            "start_clock": row.get("start_clock"),
            "num_breaks": row.get("num_breaks"),
            "impact_ils": row.get("impact"),
            "retention_pct": row.get("retention"),
            "rationale": row.get("rationale"),
            "proposed_kind": row.get("proposed_kind"),
            "actionable": row.get("actionable"),
        }
        for row in rows[:5]
    ]
    payload: dict[str, Any] = {"recommendations": slim, "count": len(slim)}
    if not slim:
        payload["note"] = "the overview builder produced no recommendations for the owned channel"
    return payload


def _read_get_frontier(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.core import _load_settings
    from kairos_api.dashboard_api import _frontier_state

    points, net_bundle, status = _frontier_state(_load_settings())
    payload: dict[str, Any] = {"status": status, "points": [dict(point) for point in points]}
    if status == "no_channel":
        payload["reason"] = "no operator channel is configured in settings"
    elif status == "computing":
        payload["reason"] = "the frontier sweep is still computing in the background; try again shortly"
    net_point = (net_bundle or {}).get("net_point")
    current = next((dict(point) for point in points if point.get("selected")), None)
    if isinstance(net_point, dict) and net_point.get("selected"):
        current = dict(net_point)
    payload["current_plan_point"] = current
    payload["net_focused_point"] = dict(net_point) if isinstance(net_point, dict) else None
    return payload


def _read_get_audience_stability(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.catalog_api import impact

    drift = impact().get("drift")
    if not isinstance(drift, dict) or not drift:
        return {
            "status": "unavailable",
            "reason": "the coefficients artifact carries no level-drift measurement",
        }
    payload = {key: value for key, value in drift.items() if key != "weekly_levels"}
    levels = drift.get("weekly_levels")
    if isinstance(levels, list):
        payload["weekly_levels"] = levels[-12:]
        if len(levels) > 12:
            payload["weekly_levels_truncated"] = True
            payload["weekly_levels_total"] = len(levels)
    return payload


def _read_get_plan_days(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos.optimize.revenue_net import frame_revenue_net
    from kairos_api import assistant_context

    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"error": reason or "no owned-channel plan is available"}
    days: list[dict[str, Any]] = []
    cost_missing_reason: str | None = None
    for date_text, group in frame.groupby("date_text", sort=True):
        entry: dict[str, Any] = {
            "date": str(date_text),
            "weekday": assistant_context._weekday_label(str(date_text), group),
            "breaks": int(group["num_breaks"].sum()),
            "revenue_ils": int(round(float(group["predicted_revenue"].sum()))),
        }
        money = frame_revenue_net(group)
        if money.get("available"):
            entry["retention_cost_ils"] = round(float(money["retention_cost_ils"]), 2)
            entry["revenue_net_ils"] = round(float(money["revenue_net_ils"]), 2)
        else:
            entry["retention_cost_ils"] = None
            cost_missing_reason = cost_missing_reason or str(money.get("reason") or "")
        days.append(entry)
    payload: dict[str, Any] = {"channel": owned, "days_total": len(days), "days": days[:31]}
    if len(days) > 31:
        payload["truncated"] = True
        payload["days_omitted"] = len(days) - 31
    if cost_missing_reason:
        payload["retention_cost_note"] = cost_missing_reason
    return payload


# --- agreement reads: uploads (own only) and advertiser matching ------------------
def _read_list_uploads(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import assistant_uploads

    summaries = assistant_uploads.list_summaries(user)
    uploads = [
        {
            "upload_id": item["upload_id"],
            "filename": item["filename"],
            "uploaded_at": item["uploaded_at"],
            "sheets": [sheet.get("name") for sheet in item.get("sheets", [])],
            "total_rows": sum(int(sheet.get("total_rows", 0)) for sheet in item.get("sheets", [])),
        }
        for item in summaries
    ]
    return {"uploads": uploads, "count": len(uploads)}


def _read_get_upload(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import assistant_uploads

    upload_id = str(args.get("upload_id", "") or "").strip()
    if not upload_id:
        return {"status": "not_found", "reason": "provide an upload_id to read a stored upload"}
    summary = assistant_uploads.get_summary(user, upload_id)
    if summary is None:
        return {"status": "not_found", "reason": f"no upload {upload_id!r} for this operator"}
    sheets = []
    capped_any = False
    for sheet in summary.get("sheets", []):
        rows = sheet.get("rows", []) or []
        total = int(sheet.get("total_rows", len(rows)))
        entry: dict[str, Any] = {
            "name": sheet.get("name"),
            "columns": sheet.get("columns", []),
            "rows": rows,
            "total_rows": total,
            "rows_shown": len(rows),
        }
        if total > len(rows):
            entry["rows_capped"] = True
            entry["rows_omitted"] = total - len(rows)
            capped_any = True
        sheets.append(entry)
    payload: dict[str, Any] = {
        "upload_id": summary["upload_id"],
        "filename": summary["filename"],
        "uploaded_at": summary["uploaded_at"],
        "sheets": sheets,
        "source": f"uploaded file {summary['filename']}",
    }
    if capped_any:
        payload["cap_note"] = ("some sheets were capped to fit; rows_shown is fewer than total_rows, "
                               "and the omitted rows are not in this result")
    return payload


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _read_find_advertiser(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from difflib import SequenceMatcher

    from kairos_api.advertisers import _load_frame, _row_to_record

    query = str(args.get("name", "") or "").strip()
    if not query:
        return {"candidates": [], "count": 0, "note": "provide a name to match against the advertiser rules"}
    target = _normalize_name(query)
    frame = _load_frame()
    scored: list[tuple[float, dict[str, Any]]] = []
    for _, row in frame.iterrows():
        record = _row_to_record(row)
        name = _normalize_name(record["advertiser_id"])
        if not name:
            continue
        if target and (target in name or name in target):
            score = 1.0
        else:
            score = SequenceMatcher(None, target, name).ratio()
        scored.append((score, record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    candidates = [record for score, record in scored[:5] if score >= 0.3]
    return {"query": query, "candidates": candidates, "count": len(candidates)}


_READ_EXECUTORS = {
    "get_settings": _read_get_settings,
    "get_day_detail": _read_get_day_detail,
    "list_constraints": _read_list_constraints,
    "list_overrides": _read_list_overrides,
    "get_pricing": _read_get_pricing,
    "get_net_comparison": _read_get_net_comparison,
    "get_compliance": _read_get_compliance,
    "simulate_settings_change": _read_simulate_settings_change,
    "get_recommendations": _read_get_recommendations,
    "get_frontier": _read_get_frontier,
    "get_audience_stability": _read_get_audience_stability,
    "get_plan_days": _read_get_plan_days,
    "list_uploads": _read_list_uploads,
    "get_upload": _read_get_upload,
    "find_advertiser": _read_find_advertiser,
}

# The provenance stamp for each read tool result: the endpoint or dataset the
# figures came from. Attached uniformly in execute_read_tool and surfaced on the trace.
SOURCE_BY_TOOL = {
    "get_settings": "saved settings",
    "get_day_detail": "saved weekly plan, owned channel",
    "list_constraints": "stored placement constraints",
    "list_overrides": "stored manual overrides",
    "get_pricing": "pricing hierarchy (rate card and operator overrides)",
    "get_net_comparison": "owned-channel scenario runner, net comparison",
    "get_compliance": "compliance verdict over the committed plan",
    "simulate_settings_change": "owned-channel scenario runner, representative day",
    "get_recommendations": "overview recommendations, owned channel",
    "get_frontier": "owned-channel frontier sweep",
    "get_audience_stability": "measured coefficients artifact, level-drift monitor",
    "get_plan_days": "saved weekly plan, owned channel",
    "list_uploads": "assistant uploads (own)",
    "get_upload": "assistant uploads (own)",
    "find_advertiser": "advertiser rules store",
}

# The additional read executors (freshness, yield, gold, make-goods, run log,
# upload status, reports catalog, activity) live in
# kairos_api.assistant_read_tools_extra so this file stays under the size cap;
# registering them here keeps one combined dispatch registry.
from kairos_api.assistant_read_tools_extra import register as _register_extra  # noqa: E402

_register_extra(_READ_EXECUTORS, SOURCE_BY_TOOL)

# The agencies, calendar-events and money-coverage executors live in
# kairos_api.assistant_read_tools_catalog (size cap); same one-registry rule.
from kairos_api.assistant_read_tools_catalog import register as _register_catalog  # noqa: E402

_register_catalog(_READ_EXECUTORS, SOURCE_BY_TOOL)


def execute_read_tool(name: str, args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    """Run one READ tool, stamping the result with its provenance source.

    Failures come back as {"error": ...}, never raised. Every result dict carries
    a non-empty "source" so the model can name where each figure came from. ``user``
    is passed to every executor; only the per-user upload tools consult it.
    """
    executor = _READ_EXECUTORS.get(name)
    if executor is None:
        return {"error": f"unknown read tool {name!r}", "source": "unknown tool"}
    try:
        result = executor(args, user)
    except HTTPException as exc:
        result = {"error": str(exc.detail)}
    except Exception as exc:  # noqa: BLE001 - surfaced honestly, without internals
        logger.exception("assistant read tool %s failed", name)
        result = {"error": f"{name} failed ({type(exc).__name__}); details are in the server log"}
    if isinstance(result, dict):
        result.setdefault("source", SOURCE_BY_TOOL.get(name, name))
    return result
