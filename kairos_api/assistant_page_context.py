"""The current_location grounding section built from the dock's page context.

The ask and stream endpoints accept an OPTIONAL ``page_context`` object from
the dashboard dock: ``{"view": "<nav key>", "label": "<Hebrew page title>",
"entity": {"type": "advertiser"|"agency"|"event"|"program", "id": "...",
"label": "..."} or null}``. When present and valid, one ``current_location``
section is attached to the composed context: where the operator is and, when an
entity is open, that entity's own data pulled from the real store, so a vague
question like מה התנאים המיוחדים שלו resolves against the entity the operator
is looking at. Advisory only, per the frozen contract: it never restricts
tools, the response shape does not change, and an absent or invalid
page_context degrades to exactly today's behavior (no section, no source).
"""

from __future__ import annotations

from typing import Any

SECTION_NAME = "current_location"
ENTITY_TYPES = ("advertiser", "agency", "event", "program")
VIEW_MAX = 60
LABEL_MAX = 120
ID_MAX = 200
CONDITIONS_CAP = 15
SEGMENTS_CAP = 8


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def parse_page_context(raw: Any) -> "dict[str, Any] | None":
    """A validated shallow copy of the page context, or None to degrade.

    Conservative on purpose: anything that is not the contract shape yields
    None, and None means the ask behaves exactly as it does without the field.
    """
    if not isinstance(raw, dict):
        return None
    view = _clip(raw.get("view"), VIEW_MAX)
    label = _clip(raw.get("label"), LABEL_MAX)
    entity_raw = raw.get("entity")
    entity: "dict[str, Any] | None" = None
    if isinstance(entity_raw, dict):
        entity_type = _clip(entity_raw.get("type"), VIEW_MAX)
        entity_id = _clip(entity_raw.get("id"), ID_MAX)
        if entity_type in ENTITY_TYPES and entity_id:
            entity = {"type": entity_type, "id": entity_id,
                      "label": _clip(entity_raw.get("label"), LABEL_MAX) or None}
    if not view and not label and entity is None:
        return None
    return {"view": view or None, "label": label or None, "entity": entity}


# --- per-type entity data, from the real stores only ------------------------------
def _entity_advertiser(entity_id: str) -> dict[str, Any]:
    from kairos_api import advertiser_conditions
    from kairos_api.advertisers import _load_frame, _row_to_record

    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == entity_id
    if not mask.any():
        return {"status": "not_found", "reason": f"no advertiser {entity_id!r} in the rules store"}
    record = _row_to_record(frame.loc[frame.index[mask][0]])
    conditions = advertiser_conditions.conditions_for(entity_id)
    data: dict[str, Any] = {"record": record, "conditions": conditions[:CONDITIONS_CAP],
                            "conditions_count": len(conditions)}
    if len(conditions) > CONDITIONS_CAP:
        data["conditions_omitted"] = len(conditions) - CONDITIONS_CAP
    return data


def _entity_agency(entity_id: str) -> dict[str, Any]:
    from kairos_api import agencies, agency_conditions

    frame = agencies._load_frame()
    mask = frame["agency_id"].astype(str) == entity_id
    if not mask.any():
        return {"status": "not_found", "reason": f"no agency {entity_id!r} in the agencies store"}
    record = agencies._row_to_record(frame.loc[frame.index[mask][0]])
    conditions = agency_conditions.conditions_for(entity_id)
    data: dict[str, Any] = {
        "record": record,
        "conditions": conditions[:CONDITIONS_CAP],
        "conditions_count": len(conditions),
        "links": agency_conditions.link_summary_for(entity_id),
        "basis": "agency terms bite only on the daily per-spot ledger's reporting net, never the weekly plan",
    }
    if len(conditions) > CONDITIONS_CAP:
        data["conditions_omitted"] = len(conditions) - CONDITIONS_CAP
    return data


def _entity_event(entity_id: str) -> dict[str, Any]:
    from kairos_api import events_api

    frame = events_api._load_frame()
    mask = frame["event_id"].astype(str) == entity_id
    if not mask.any():
        return {"status": "not_found", "reason": f"no event {entity_id!r} in the calendar events store"}
    record = events_api._record(frame.loc[frame.index[mask][0]])
    record["window_overlap_days"] = events_api._window_overlap_days(record)
    overlap = events_api._plan_overlap_dates(record, events_api._plan_dates())
    record["plan_overlap_dates"] = overlap[:CONDITIONS_CAP]
    record["plan_overlap_count"] = len(overlap)
    record["multiplier_basis"] = "price_multiplier is an operator assertion; it moves forecast revenue only while pricing_activation.events is on"
    return record


def _entity_program(entity_id: str) -> dict[str, Any]:
    from kairos_api import assistant_context

    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"status": "unavailable", "reason": reason or "no owned-channel plan is available"}
    column = "program_title" if "program_title" in frame.columns else "program_type"
    matched = frame[frame[column].astype(str).str.strip() == entity_id]
    if matched.empty:
        return {"status": "not_found",
                "reason": f"the saved plan has no {column} rows matching {entity_id!r} on {owned}"}
    data: dict[str, Any] = {
        "channel": owned,
        "matched_on": column,
        "days": sorted(set(matched["date_text"].astype(str))),
        "segments_total": int(len(matched)),
        "breaks": int(matched["num_breaks"].sum()),
        "revenue_ils": int(round(float(matched["predicted_revenue"].sum()))),
        "avg_retention_pct": assistant_context._retention_pct(matched["predicted_retention"].mean()),
    }
    top = matched.sort_values("predicted_revenue", ascending=False).head(SEGMENTS_CAP)
    data["segments"] = [assistant_context._compact_row(row) for _, row in top.iterrows()]
    if len(matched) > SEGMENTS_CAP:
        data["segments_omitted"] = int(len(matched)) - SEGMENTS_CAP
    return data


_ENTITY_BUILDERS = {
    "advertiser": _entity_advertiser,
    "agency": _entity_agency,
    "event": _entity_event,
    "program": _entity_program,
}


def extend_with_current_location(context: dict[str, Any], sources: list[str], raw: Any) -> None:
    """Attach the current_location section when a valid page context arrived.

    Mutates context and sources in place under the composer's contract. An
    invalid or absent page context adds nothing at all (today's behavior); a
    valid one whose entity data cannot be built still attaches the location
    with an honest status instead of fabricated entity data.
    """
    parsed = parse_page_context(raw)
    if parsed is None:
        return
    try:
        section: dict[str, Any] = {
            "view": parsed["view"],
            "label": parsed["label"],
            "note": "the operator is currently viewing this page; resolve vague references against the open entity, and answer global questions from the full context and tools as usual",
        }
        entity = parsed["entity"]
        if entity is not None:
            section["entity"] = entity
            try:
                section["entity_data"] = _ENTITY_BUILDERS[entity["type"]](entity["id"])
            except Exception:  # noqa: BLE001 - honest absence beats a crashed ask
                section["entity_data"] = {"status": "unavailable",
                                          "reason": "the entity's store could not be read"}
        context[SECTION_NAME] = section
        sources.append(SECTION_NAME)
    except Exception:  # noqa: BLE001 - the section is advisory, never fail the ask
        sources.append(f"{SECTION_NAME} (absent)")
