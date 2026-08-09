"""READ tool executors for the agencies, calendar-events, money-coverage and
audience-model tools.

These read tools are split out of kairos_api.assistant_read_tools so every file
stays under the size cap. Conventions are identical to the sibling modules:
each executor reuses the real store or builder of the owning module, returns an
honest ``{"error": ...}`` or an explicit unavailable-with-reason payload
instead of fabricating, caps every list with the true total beside the cap, and
is stamped with a provenance source by ``execute_read_tool``. The two nets stay
disambiguated everywhere: the weekly plan's net is revenue net of modeled
retention cost, while the daily ledger's net is gross minus agency rebates
(reporting only); each payload names its own basis explicitly.

Tolerant by design: advertiser condition records pass through whatever fields
the store carries (including custom-pricing fields such as ``scope_weekdays``
or the ``premium_discount`` mode when present), so this module works with or
without those columns.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

MAX_AGENCIES = 20
MAX_EVENTS = 30
MAX_CONDITIONS = 20
MAX_OVERLAP_FINDINGS = 10
MAX_PLAN_DATES = 14
MAX_TOP_ADVERTISERS = 20

LEDGER_BASIS = (
    "daily per-spot ledger (the newest daily spot file, one broadcast day); net "
    "means gross minus agency rebates, reporting only, NOT the weekly plan's "
    "retention-net"
)
AGENCY_BASIS = (
    "agency terms and conditions affect only the daily per-spot ledger and its "
    "reporting-only net figure; the weekly plan and retention math are untouched"
)


def _cap(payload: dict[str, Any], key: str, limit: int) -> None:
    rows = list(payload.get(key) or [])
    payload[key] = rows[:limit]
    if len(rows) > limit:
        payload[f"{key}_total"] = len(rows)
        payload[f"{key}_omitted"] = len(rows) - limit


# --- agencies ---------------------------------------------------------------------
def _agency_terms_row(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("agency_id", "name", "display_name", "agency_type", "status",
            "payment_terms_days", "rebate_percent", "commission_percent",
            "credit_limit_ils", "data_source")
    return {key: record.get(key) for key in keys}


def _read_get_agencies(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import agencies, agency_conditions

    frame = agencies._load_frame()
    conditions = agency_conditions._load_csv(
        agency_conditions.CONDITIONS_PATH, agency_conditions.CONDITION_COLUMNS)
    condition_counts = conditions["agency_id"].astype(str).value_counts().to_dict() if len(conditions) else {}
    links = agency_conditions._load_csv(agency_conditions.LINKS_PATH, agency_conditions.LINK_COLUMNS)
    link_counts = links["agency_id"].astype(str).value_counts().to_dict() if len(links) else {}
    rows = []
    for _, row in frame.iterrows():
        record = agencies._row_to_record(row)
        entry = _agency_terms_row(record)
        entry["conditions_count"] = int(condition_counts.get(record["agency_id"], 0))
        entry["stored_links_count"] = int(link_counts.get(record["agency_id"], 0))
        rows.append(entry)
    payload: dict[str, Any] = {"agencies": rows, "count": len(rows), "basis": AGENCY_BASIS}
    if not rows:
        payload["note"] = "the agencies store is empty; no agency records exist yet"
    _cap(payload, "agencies", MAX_AGENCIES)
    return payload


def _resolve_agency_id(agency_id: str, name: str) -> "str | None":
    """The stored agency_id for an exact id, or a name/display/alias match."""
    from kairos_api import agencies

    frame = agencies._load_frame()
    ids = frame["agency_id"].astype(str)
    if agency_id and (ids == agency_id).any():
        return agency_id
    query = name.strip() or agency_id.strip()
    if not query:
        return None
    for _, row in frame.iterrows():
        tokens = {str(row.get("name", "")).strip(), str(row.get("display_name", "")).strip()}
        tokens.update(part.strip() for part in str(row.get("aliases", "")).split("|"))
        tokens.discard("")
        if query in tokens or any(query in token for token in tokens):
            return str(row.get("agency_id", ""))
    return None


def _read_get_agency_detail(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import agencies, agency_conditions

    agency_id = str(args.get("agency_id", "") or "").strip()
    name = str(args.get("name", "") or "").strip()
    resolved = _resolve_agency_id(agency_id, name)
    if resolved is None:
        return {"error": f"no agency matches id {agency_id!r} or name {name!r}; list them with get_agencies"}
    frame = agencies._load_frame()
    record = agencies._row_to_record(frame.loc[agencies._locate(frame, resolved)])
    payload: dict[str, Any] = dict(record)
    links = agency_conditions.links_for(resolved)
    payload["links"] = {
        "observed": links["observed"][:MAX_CONDITIONS],
        "manual": links["manual"][:MAX_CONDITIONS],
        "effective_count": len(links["effective"]),
        "observed_source_file": links["observed_source_file"],
        "note": "observed links derive live from the newest daily file; manual links are operator-created and win per advertiser",
    }
    payload["conditions"] = agency_conditions.conditions_for(resolved)
    _cap(payload, "conditions", MAX_CONDITIONS)
    try:
        payload["overlaps"] = agency_conditions.overlaps_for(resolved)
        _cap(payload, "overlaps", MAX_OVERLAP_FINDINGS)
    except Exception:  # noqa: BLE001 - findings are additive, never fail the read
        payload["overlaps"] = []
        payload["overlaps_note"] = "overlap findings could not be computed"
    payload["basis"] = AGENCY_BASIS
    return payload


# --- calendar events --------------------------------------------------------------
def _parse_filter_date(raw: Any, field: str) -> "date | None":
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD), got {text!r}")


def _event_in_range(record: dict[str, Any], low: "date | None", high: "date | None") -> bool:
    from kairos_api.events_api import _event_span

    span = _event_span(record)
    if span is None:
        return low is None and high is None
    start, end = span
    if high is not None and start > high:
        return False
    if low is not None and end is not None and end < low:
        return False
    return True


def _read_get_calendar_events(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import events_api

    try:
        low = _parse_filter_date(args.get("date_from"), "date_from")
        high = _parse_filter_date(args.get("date_to"), "date_to")
    except ValueError as exc:
        return {"error": str(exc)}
    type_filter = str(args.get("type", "") or "").strip().lower()
    if type_filter and type_filter not in events_api.EVENT_TYPES:
        return {"error": f"type must be one of {list(events_api.EVENT_TYPES)}, got {type_filter!r}"}
    include_inactive = args.get("include_inactive")
    include_inactive = True if include_inactive is None else bool(include_inactive)
    frame = events_api._load_frame()
    plan_dates = events_api._plan_dates()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = events_api._record(row)
        if not include_inactive and not record["active"]:
            continue
        if type_filter and record["type"] != type_filter:
            continue
        if not _event_in_range(record, low, high):
            continue
        overlap = events_api._plan_overlap_dates(record, plan_dates)
        record["plan_overlap_dates"] = overlap[:MAX_PLAN_DATES]
        record["plan_overlap_count"] = len(overlap)
        rows.append(record)
    payload: dict[str, Any] = {
        "events": rows,
        "count": len(rows),
        "events_total_stored": int(len(frame)),
        "multiplier_basis": "price_multiplier is an operator assertion, never a measurement; it moves forecast revenue only while pricing_activation.events is on",
    }
    if not rows:
        payload["note"] = "no stored event matches these filters"
    _cap(payload, "events", MAX_EVENTS)
    return payload


# --- event pricing layer ----------------------------------------------------------
def _read_get_event_pricing(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos.optimize.pricing import PricingModel
    from kairos_api import events_api
    from kairos_api.core import _load_settings

    overrides = getattr(_load_settings(), "pricing_overrides", None) or {}
    enabled = bool(PricingModel.from_config(overrides).enable_events)
    frame = events_api._load_frame()
    plan_dates = events_api._plan_dates()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = events_api._record(row)
        if not record["active"] or record["price_multiplier"] == 1.0:
            continue
        overlap = events_api._plan_overlap_dates(record, plan_dates)
        rows.append({
            "event_id": record["event_id"], "name": record["name"], "type": record["type"],
            "start_date": record["start_date"], "end_date": record["end_date"],
            "price_multiplier": record["price_multiplier"],
            "plan_overlap_dates": overlap[:MAX_PLAN_DATES],
            "plan_overlap_count": len(overlap),
        })
    payload: dict[str, Any] = {
        "enabled": enabled,
        "activation_flag": "pricing_activation.events",
        "nonneutral_active_events": rows,
        "count": len(rows),
        "basis": "operator assertion per calendar event, not measured; retention coefficients are untouched",
    }
    if not enabled:
        payload["note"] = "the events layer is OFF, so these multipliers change no forecast until the operator activates pricing_activation.events"
    _cap(payload, "nonneutral_active_events", MAX_EVENTS)
    return payload


# --- one advertiser's money rules -------------------------------------------------
def _read_get_advertiser_pricing(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import advertiser_conditions
    from kairos_api.advertisers import _load_frame, _row_to_record

    advertiser = str(args.get("advertiser", "") or "").strip()
    if not advertiser:
        return {"error": "provide the advertiser id (exact store name); find it with find_advertiser"}
    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == advertiser
    if not mask.any():
        return {"error": f"no advertiser {advertiser!r} in the rules store; match the name with find_advertiser first"}
    record = _row_to_record(frame.loc[frame.index[mask][0]])
    # Condition rows pass through as the store serializes them, so custom-pricing
    # fields (scope_weekdays, premium_discount mode) appear whenever they exist.
    conditions = advertiser_conditions.conditions_for(advertiser)
    payload: dict[str, Any] = {
        "advertiser": advertiser,
        "baseline": record,
        "conditions": conditions,
        "conditions_count": len(conditions),
        "basis": "advertiser rules bite on the daily per-spot pricing path; the weekly break-count plan does not attribute breaks to advertisers",
    }
    _cap(payload, "conditions", MAX_CONDITIONS)
    try:
        payload["overlaps"] = advertiser_conditions.overlaps_for(advertiser)
        _cap(payload, "overlaps", MAX_OVERLAP_FINDINGS)
    except Exception:  # noqa: BLE001 - findings are additive, never fail the read
        payload["overlaps"] = []
        payload["overlaps_note"] = "overlap findings could not be computed"
    return payload


# --- daily-ledger advertiser ranking ----------------------------------------------
def _read_get_top_advertisers(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos.export.spots import price_daily_file
    from kairos_api.uploads import _newest_daily

    try:
        limit = int(args.get("limit") or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, MAX_TOP_ADVERTISERS))
    path = _newest_daily()
    if path is None:
        return {"status": "unavailable",
                "reason": "no daily spot file exists, so the per-spot ledger cannot be built"}
    result = price_daily_file(path)
    by_advertiser: dict[str, dict[str, Any]] = {}
    for spot in result.priced:
        entry = by_advertiser.setdefault(spot.advertiser, {
            "advertiser": spot.advertiser, "spots": 0,
            "gross_revenue_ils": 0.0, "net_revenue_ils": 0.0, "agencies": set(),
        })
        entry["spots"] += 1
        entry["gross_revenue_ils"] += spot.revenue
        entry["net_revenue_ils"] += spot.net_revenue
        if spot.agency:
            entry["agencies"].add(spot.agency)
    ranked = sorted(by_advertiser.values(),
                    key=lambda entry: (-entry["gross_revenue_ils"], entry["advertiser"]))
    rows = [
        {
            "advertiser": entry["advertiser"], "spots": entry["spots"],
            "gross_revenue_ils": round(entry["gross_revenue_ils"], 2),
            "net_revenue_ils": round(entry["net_revenue_ils"], 2),
            "agencies": sorted(entry["agencies"]),
        }
        for entry in ranked[:limit]
    ]
    return {
        "source_file": path.name,
        "advertisers": rows,
        "advertisers_total": len(ranked),
        "spots_priced": len(result.priced),
        "spots_dropped_by_rules": len(result.dropped) + len(result.frequency_dropped),
        "totals": {"gross_revenue_ils": result.total_revenue,
                   "net_revenue_ils": result.total_net_revenue},
        "basis": LEDGER_BASIS,
        "currency": "ILS",
    }


_CATALOG_READ_EXECUTORS = {
    "get_agencies": _read_get_agencies,
    "get_agency_detail": _read_get_agency_detail,
    "get_calendar_events": _read_get_calendar_events,
    "get_event_pricing": _read_get_event_pricing,
    "get_advertiser_pricing": _read_get_advertiser_pricing,
    "get_top_advertisers": _read_get_top_advertisers,
}

CATALOG_SOURCE_BY_TOOL = {
    "get_agencies": "agencies store",
    "get_agency_detail": "agencies store (record, links, conditions)",
    "get_calendar_events": "calendar events store",
    "get_event_pricing": "calendar events store and rate-card activation",
    "get_advertiser_pricing": "advertiser rules and scoped conditions stores",
    "get_top_advertisers": "daily per-spot ledger (newest daily file)",
}


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge these executors and their source labels into the shared registry."""
    executors.update(_CATALOG_READ_EXECUTORS)
    sources.update(CATALOG_SOURCE_BY_TOOL)
    # The event-pipeline, audience-model, pod, break and pacing executors live in
    # their own modules (size cap); registering them here keeps the
    # one-registry rule.
    from kairos_api.assistant_audience_model import register as register_audience
    from kairos_api.assistant_event_pipeline import register as register_pipeline
    from kairos_api.assistant_read_tools_break import register as register_break
    from kairos_api.assistant_read_tools_pacing import register as register_pacing
    from kairos_api.assistant_read_tools_pod import register as register_pod

    register_pipeline(executors, sources)
    register_audience(executors, sources)
    register_pod(executors, sources)
    register_break(executors, sources)
    register_pacing(executors, sources)
