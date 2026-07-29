"""Question-keyword grounding sections for the assistant context.

When the operator's question carries a matching Hebrew or English keyword, ONE
compact hard-capped section per topic is attached to the composed context:
``gold_breaks``, ``active_constraints``, ``active_overrides``,
``pricing_state``, ``pacing_status``, ``agencies_state``, ``calendar_events``,
``event_pricing``, ``custom_pricing`` and ``event_pipeline``. Each section
reuses the real builder
behind the matching dashboard surface (the insights gold builder, the
constraints and overrides stores, the pricing hierarchy payload, the pacing
loader plus the make-good projection), so the assistant reads exactly what the
operator's own pages render. Keyword matching is conservative: Hebrew words are
canonicalized with the same prefix-stripping idiom the day-grounding module
uses and compared whole, English keywords match on word boundaries only, and no
match adds nothing. A section whose builder fails is listed absent by the
caller, never substituted. The competitor boundary holds: gold rows are scoped
to the operator's own channel, with excluded rows surfacing only as a count.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from kairos_api.assistant_context import _strip_hebrew_prefixes

GOLD_ROW_CAP = 15
CONSTRAINT_ROW_CAP = 20
OVERRIDE_ROW_CAP = 20
AGENCY_ROW_CAP = 10
EVENT_ROW_CAP = 12
CONDITION_ROW_CAP = 12

_HEBREW_WORD_RE = re.compile(r"[א-ת]+")

# Per-section triggers: canonical Hebrew keywords (matched whole-word after the
# conservative prefix strip, applied to both sides), literal Hebrew phrases
# (substring), and English word-boundary patterns on the lowercased question.
_TRIGGERS: dict[str, dict[str, Any]] = {
    "gold_breaks": {
        "hebrew": ("זהב",),
        "english": re.compile(r"\bgold\b"),
    },
    "active_constraints": {
        "hebrew": ("אילוץ", "אילוצים"),
        "english": re.compile(r"\bconstraints?\b"),
    },
    "active_overrides": {
        "hebrew": ("עקיפה", "עקיפות", "נעיצה", "נעיצות"),
        "english": re.compile(r"\boverrides?\b|\bpins?\b"),
    },
    "pricing_state": {
        "hebrew": ("מחירון", "תמחור", "מחיר"),
        "english": re.compile(r"\bpricing\b|\bcpp\b"),
    },
    "pacing_status": {
        "hebrew": ("פייסינג", "קמפיין"),
        "phrases": ("מייק גוד",),
        "english": re.compile(r"\bpacing\b|\bcampaigns?\b|\bmake[- ]goods?\b"),
    },
    "agencies_state": {
        "hebrew": ("סוכנות", "סוכנויות", "משרד", "משרדים"),
        "phrases": ("אייג'נסי",),
        "english": re.compile(r"\bagenc(?:y|ies)\b|\brebates?\b"),
    },
    "calendar_events": {
        "hebrew": ("אירוע", "אירועים", "מלחמה", "חג", "חגים"),
        "phrases": ("לוח שנה", "לוח השנה"),
        "english": re.compile(r"\bevents?\b|\bholidays?\b|\bwars?\b|\bcalendar\b"),
    },
    "event_pricing": {
        "hebrew": ("מכפיל", "מכפילים"),
        "phrases": ("תמחור אירועים", "תמחור אירוע"),
        "english": re.compile(r"\bmultipliers?\b|\bevent[- ]pricing\b"),
    },
    "custom_pricing": {
        "hebrew": ("הנחה", "הנחות", "שבת"),
        "phrases": ("תמחור אישי", "תמחור מותאם"),
        "english": re.compile(r"\bdiscounts?\b|\bcustom[- ]pricing\b|\bsaturday\b"),
    },
    "event_pipeline": {
        "hebrew": ("פרץ", "פרצה"),
        "phrases": ("מלחמה חדשה", "אירוע חדש", "עדכון אירועים", "עדכון האירועים"),
        "english": re.compile(r"\bnew (?:war|event)\b|\bwar (?:broke|started)\b|\bevent pipeline\b"),
    },
}


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _matches(question: str, name: str) -> bool:
    trigger = _TRIGGERS[name]
    targets = {_strip_hebrew_prefixes(keyword) for keyword in trigger.get("hebrew", ())}
    if targets:
        for word in _HEBREW_WORD_RE.findall(question):
            if _strip_hebrew_prefixes(word) in targets:
                return True
    for phrase in trigger.get("phrases", ()):
        if phrase in question:
            return True
    english = trigger.get("english")
    return bool(english and english.search(question.lower()))


def _cap_rows(section: dict[str, Any], field: str, rows: list[dict[str, Any]], cap: int) -> None:
    """Attach rows hard-capped at ``cap`` with an honest omission count."""
    section[field] = rows[:cap]
    if len(rows) > cap:
        section["truncated"] = True
        section["rows_omitted"] = len(rows) - cap


def _section_gold_breaks() -> dict[str, Any]:
    """The gold list from the insights builder, operator-channel scoped."""
    from kairos_api.insights_api import _build_gold_breaks

    payload = _build_gold_breaks()
    section = {
        key: payload[key]
        for key in ("available", "enabled", "count", "reason", "max_per_day")
        if key in payload
    }
    rows = list(payload.get("breaks") or [])
    owned = str(getattr(_server()._load_settings(), "operator_channel", "") or "").strip()
    if owned:
        scoped = [row for row in rows if str(row.get("channel") or "").strip() == owned]
        section["scope_channel"] = owned
        if len(scoped) != len(rows):
            section["competitor_rows_excluded"] = len(rows) - len(scoped)
            section["count"] = len(scoped)
        rows = scoped
    compact = [
        {
            key: row.get(key)
            for key in ("segment_id", "day", "start_time", "program_type", "duration_seconds", "revenue")
        }
        for row in rows
    ]
    _cap_rows(section, "breaks", compact, GOLD_ROW_CAP)
    return section


def _section_active_constraints() -> dict[str, Any]:
    """The stored scheduling constraints, compacted to id, scope and effect."""
    from kairos_api import constraints

    rows = constraints.list_constraints()["constraints"]
    compact = [
        {
            key: row.get(key)
            for key in ("constraint_id", "scope_type", "scope_value", "channel", "effect")
        }
        for row in rows
    ]
    section: dict[str, Any] = {"count": len(rows)}
    _cap_rows(section, "constraints", compact, CONSTRAINT_ROW_CAP)
    return section


def _section_active_overrides() -> dict[str, Any]:
    """The stored manual overrides (pins), compacted to kind and scope."""
    from kairos_api import overrides

    grouped = overrides.list_overrides()["overrides"]
    rows = [row for scope in sorted(grouped) for row in grouped[scope]]
    compact = [
        {key: row.get(key) for key in ("override_id", "scope", "kind", "target_id", "value")}
        for row in rows
    ]
    section: dict[str, Any] = {"count": len(rows)}
    _cap_rows(section, "overrides", compact, OVERRIDE_ROW_CAP)
    return section


def _section_pricing_state() -> dict[str, Any]:
    """The effective rate card: base price, per-layer live states, overrides flag."""
    from kairos_api import pricing_api

    payload = pricing_api._state_payload(_server()._load_settings())
    return {
        "currency": payload.get("currency"),
        "units": payload.get("units"),
        "base": payload.get("base"),
        "layers": [
            {"name": layer.get("name"), "live_today": bool(layer.get("live_today"))}
            for layer in payload.get("layers", [])
        ],
        "activation": payload.get("activation"),
        # The event-date layer block rides along untouched, so a pricing
        # question surfaces that the events layer exists and whether it is on.
        "events": payload.get("events"),
        "has_overrides": bool(payload.get("has_overrides")),
    }


def _section_pacing_status() -> dict[str, Any]:
    """Campaign flight count plus the make-good projection's honest status."""
    from kairos.optimize.pacing import load_campaigns
    from kairos_api.insights_api import make_good_alerts

    alerts = make_good_alerts()
    section: dict[str, Any] = {
        "flights_count": len(load_campaigns()),
        "make_good_available": bool(alerts.get("data_available")),
        "make_good_reason": alerts.get("reason"),
        "as_of": alerts.get("as_of"),
    }
    if alerts.get("data_available"):
        section["make_good_alerts_count"] = alerts.get("count")
    return section


def _section_agencies_state() -> dict[str, Any]:
    """Agency records with their commercial terms, from the agencies store."""
    from kairos_api import agencies, agency_conditions

    frame = agencies._load_frame()
    conditions = agency_conditions._load_csv(
        agency_conditions.CONDITIONS_PATH, agency_conditions.CONDITION_COLUMNS)
    counts = conditions["agency_id"].astype(str).value_counts().to_dict() if len(conditions) else {}
    rows = []
    for _, row in frame.iterrows():
        record = agencies._row_to_record(row)
        rows.append({
            "agency_id": record["agency_id"], "name": record["name"],
            "agency_type": record["agency_type"], "status": record["status"],
            "payment_terms_days": record["payment_terms_days"],
            "rebate_percent": record["rebate_percent"],
            "commission_percent": record["commission_percent"],
            "conditions_count": int(counts.get(record["agency_id"], 0)),
        })
    section: dict[str, Any] = {
        "count": len(rows),
        "active_count": sum(1 for row in rows if row["status"] == "active"),
        "basis": "agency terms bite only on the daily per-spot ledger's reporting net, never the weekly plan",
    }
    _cap_rows(section, "agencies", rows, AGENCY_ROW_CAP)
    return section


def _compact_event(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("event_id", "name", "type", "start_date", "end_date", "intensity",
            "active", "price_multiplier")
    return {key: record.get(key) for key in keys}


def _section_calendar_events() -> dict[str, Any]:
    """The stored events overlapping the saved plan, plus honest store totals."""
    from kairos_api import events_api

    frame = events_api._load_frame()
    plan_dates = events_api._plan_dates()
    overlapping: list[dict[str, Any]] = []
    active_count = 0
    nonneutral = 0
    for _, row in frame.iterrows():
        record = events_api._record(row)
        if record["active"]:
            active_count += 1
            if record["price_multiplier"] != 1.0:
                nonneutral += 1
        overlap = events_api._plan_overlap_dates(record, plan_dates)
        if overlap:
            entry = _compact_event(record)
            entry["plan_overlap_count"] = len(overlap)
            overlapping.append(entry)
    section: dict[str, Any] = {
        "count": int(len(frame)),
        "active_count": active_count,
        "nonneutral_multiplier_count": nonneutral,
        "note": "plan_overlapping lists only events covering saved plan days; get_calendar_events reads the full store",
    }
    _cap_rows(section, "plan_overlapping", overlapping, EVENT_ROW_CAP)
    return section


def _section_event_pricing() -> dict[str, Any]:
    """The event-date price layer: activation state and the non-neutral events."""
    from kairos.optimize.pricing import PricingModel
    from kairos_api import events_api

    overrides = getattr(_server()._load_settings(), "pricing_overrides", None) or {}
    frame = events_api._load_frame()
    rows = []
    for _, row in frame.iterrows():
        record = events_api._record(row)
        if record["active"] and record["price_multiplier"] != 1.0:
            rows.append(_compact_event(record))
    section: dict[str, Any] = {
        "enabled": bool(PricingModel.from_config(overrides).enable_events),
        "activation_flag": "pricing_activation.events",
        "count": len(rows),
        "basis": "operator assertion per calendar event, not measured; while the layer is off the multipliers change no forecast",
    }
    _cap_rows(section, "nonneutral_active_events", rows, EVENT_ROW_CAP)
    return section


def _section_custom_pricing() -> dict[str, Any]:
    """Scoped advertiser money rules, tolerant of the custom-pricing fields."""
    from kairos_api import advertiser_conditions

    frame = advertiser_conditions._load_frame()
    rows = [advertiser_conditions._row_to_record(row) for _, row in frame.iterrows()]
    compact = [
        {key: record.get(key) for key in (
            "advertiser_id", "rule_id", "effect", "mode", "value",
            "scope_positions", "scope_genres", "scope_dayparts",
            "scope_programmes", "scope_weekdays",
        )}
        for record in rows
    ]
    section: dict[str, Any] = {
        "count": len(rows),
        "advertisers_with_rules": len({record["advertiser_id"] for record in rows}),
        "basis": "these rules bite on the daily per-spot pricing path; weekday scopes are ISO numbers (Saturday is 6)",
    }
    _cap_rows(section, "conditions", compact, CONDITION_ROW_CAP)
    return section


def _section_event_pipeline() -> dict[str, Any]:
    """The full event pipeline in operational order, for a new-war or new-event
    question: the four-step playbook plus the live state of every stage. The
    permissions stage is omitted here because the composer has no actor; the
    get_event_pipeline read tool carries it per acting account."""
    from kairos_api.assistant_event_pipeline import pipeline_snapshot

    return pipeline_snapshot(None, include_permissions=False)


_SECTIONS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("gold_breaks", _section_gold_breaks),
    ("active_constraints", _section_active_constraints),
    ("active_overrides", _section_active_overrides),
    ("pricing_state", _section_pricing_state),
    ("pacing_status", _section_pacing_status),
    ("agencies_state", _section_agencies_state),
    ("calendar_events", _section_calendar_events),
    ("event_pricing", _section_event_pricing),
    ("custom_pricing", _section_custom_pricing),
    ("event_pipeline", _section_event_pipeline),
)

SECTION_NAMES = tuple(name for name, _ in _SECTIONS)


def extend_with_keyword_sections(context: dict[str, Any], sources: list[str], question: str) -> None:
    """Attach each keyword-matched section, absent-marked on failure.

    Mutates context and sources in place under the composer's contract: a
    matched section that cannot be built is listed with an absent marker and
    omitted from the context, never substituted; an unmatched section adds
    nothing at all.
    """
    for name, build in _SECTIONS:
        try:
            matched = _matches(question, name)
        except Exception:
            matched = False
        if not matched:
            continue
        try:
            context[name] = build()
            sources.append(name)
        except Exception:
            sources.append(f"{name} (absent)")
