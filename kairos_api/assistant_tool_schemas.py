"""Additional Anthropic tool schemas for the assistant's action plane.

The agencies, calendar-events and money-coverage tools: six READ tools (agency
records with terms and links, calendar events with plan overlap, the event
pricing layer state, one advertiser's money rules, the daily-ledger advertiser
ranking) and two PROPOSE tools (calendar-event changes, agency changes). Split
out of kairos_api.assistant_tools so that file stays under the size cap;
assistant_tools extends its schema lists with these at import time, so the
model sees one flat tool list and the name registries stay complete. This
module defines schemas only: executors live in
kairos_api.assistant_read_tools_catalog, validators and appliers in
kairos_api.assistant_propose_extra.
"""

from __future__ import annotations

from typing import Any

_REASON = {
    "type": "string",
    "description": "Why this change is being proposed, in the operator's language.",
}


def _tool(name: str, description: str, properties: dict[str, Any] | None = None,
          required: list[str] | None = None) -> dict[str, Any]:
    """One Anthropic tool schema in the messages.create format (same shape as
    kairos_api.assistant_tools._tool, redefined here to avoid an import cycle)."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema}


EXTRA_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "get_agencies",
        "List the media agencies (סוכנויות) with their commercial terms: agency type, "
        "status, payment terms, rebate and commission percent, credit limit and data "
        "source, plus each agency's condition count. Agency terms affect only the daily "
        "per-spot ledger's net figure, never the weekly plan. Call this when the operator "
        "asks about agencies or agency terms.",
    ),
    _tool(
        "get_agency_detail",
        "Read one agency in full: its record and terms, its advertiser links (observed "
        "from the daily file versus manual, with the effective set), its scoped "
        "conditions, and any overlap findings. Call this when the operator asks about a "
        "specific agency, who buys through it, or its special conditions.",
        {"agency_id": {"type": "string", "description": "The agency id from get_agencies."},
         "name": {"type": "string", "description": "An agency name or alias, when the id is unknown."}},
    ),
    _tool(
        "get_calendar_events",
        "List the stored calendar events (holidays, wars, sport, special periods): dates, "
        "intensity, active flag, the operator-asserted price_multiplier, and which saved "
        "plan days each event overlaps. Optional filters: type (holiday/war/special/"
        "sport/other) and an ISO date range. Call this when the operator asks about "
        "events, holidays, wars or the calendar.",
        {"type": {"type": "string", "description": "Filter to one event type."},
         "date_from": {"type": "string", "description": "Only events overlapping on/after this ISO date."},
         "date_to": {"type": "string", "description": "Only events overlapping on/before this ISO date."},
         "include_inactive": {"type": "boolean", "description": "Include inactive events (default true)."}},
    ),
    _tool(
        "get_event_pricing",
        "Read the event pricing layer state: whether pricing_activation.events is on, and "
        "the active events carrying a non-1.0 price multiplier with the plan days they "
        "cover. Multipliers are operator assertions, never measurements; while the layer "
        "is off they change no forecast. Call this when the operator asks about event "
        "pricing, price multipliers or holiday pricing.",
    ),
    _tool(
        "get_advertiser_pricing",
        "Read one advertiser's money rules as structured rows: the baseline record "
        "(default premium, allowed positions and genres, prime-time flag) and every "
        "scoped condition (effect, value, mode, scopes, and custom fields such as weekday "
        "scope or discount mode when the store carries them). These bite on the daily "
        "per-spot pricing path. Call this for questions about an advertiser's special "
        "terms, discounts or custom pricing.",
        {"advertiser": {"type": "string", "description": "The advertiser id (exact store name)."}},
        ["advertiser"],
    ),
    _tool(
        "get_top_advertisers",
        "Rank advertisers by money from the daily per-spot ledger: spots priced, gross "
        "revenue and net revenue (after agency rebates, reporting only), per advertiser "
        "with its agencies. The basis is the newest daily file, a single broadcast day, "
        "not the weekly plan. Call this when the operator asks who advertises the most or "
        "which advertisers bring the most revenue.",
        {"limit": {"type": "integer", "description": "How many advertisers to return (1-20, default 10)."}},
    ),
]

# The pod tools' schemas are defined beside their executors, so the description
# and what the executor returns cannot drift apart. They ride in this list so
# READ_TOOL_NAMES, which freezes at import, carries them.
from kairos_api.assistant_read_tools_pod import POD_READ_TOOL_SCHEMAS  # noqa: E402

EXTRA_READ_TOOL_SCHEMAS.extend(POD_READ_TOOL_SCHEMAS)

EXTRA_PROPOSE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "propose_event_change",
        "Propose creating, updating or deactivating a calendar event (holiday, war, "
        "special period), including its intensity and its operator-asserted "
        "price_multiplier (0.1-5.0, 1.0 is neutral). The operator must approve before "
        "anything is saved; a multiplier change moves forecast revenue on the event's "
        "days only while the event pricing layer is active.",
        {
            "action": {"type": "string", "description": "'create', 'update' or 'deactivate'."},
            "event_id": {"type": "string", "description": "The stored event id (required for update/deactivate)."},
            "event": {"type": "object", "description": "Event fields: name, type (holiday/war/special/sport/other), start_date, end_date (empty for open-ended), intensity 1-5, notes, active, price_multiplier."},
            "reason": _REASON,
        },
        ["action", "reason"],
    ),
    _tool(
        "propose_agency_change",
        "Propose a change to one agency: create/update its record and commercial terms "
        "(rebate, commission, payment terms, contacts), deactivate it, link or unlink an "
        "advertiser manually, or add/update/delete one of its scoped conditions. Agency "
        "terms and conditions affect only the daily per-spot ledger (net revenue is "
        "reporting-only); the weekly plan is untouched. The operator must approve before "
        "anything is saved.",
        {
            "agency_id": {"type": "string", "description": "The agency id (new id for create)."},
            "action": {"type": "string", "description": "'create', 'update', 'deactivate', 'link_advertiser', 'unlink_advertiser', 'add_condition', 'update_condition' or 'delete_condition'."},
            "changes": {"type": "object", "description": "For create/update: agency record fields (name, display_name, agency_type, contacts, payment_terms_days, rebate_percent, commission_percent, credit_limit_ils, status, notes)."},
            "advertiser": {"type": "string", "description": "For link_advertiser/unlink_advertiser: the advertiser name."},
            "condition": {"type": "object", "description": "For condition actions: rule_id plus effect, value, mode and scope_* fields."},
            "reason": _REASON,
        },
        ["agency_id", "action", "reason"],
    ),
]

EXTRA_KIND_BY_TOOL = {
    "propose_event_change": "event_change",
    "propose_agency_change": "agency_change",
}
