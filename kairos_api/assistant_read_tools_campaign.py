"""Kai's scoped read of one complete campaign record.

Pacing is one projection of a campaign, not the campaign itself. This tool reads
the commercial commitment, order kind, flights, creative assets and delivery
basis from the same builders as the campaign drawer. The lookup is scoped before
it distinguishes missing data, so a rival id and a typo produce the same answer.
"""

from __future__ import annotations

from typing import Any

MAX_FLIGHTS = 30
MAX_ASSETS = 30
MAX_DELIVERY_DAYS = 45


CAMPAIGN_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_campaign",
        "description": (
            "Read one campaign on the operator's own channel in full: commercial "
            "commitment and order kind, flights, creative assets and their QC state, "
            "plus delivery days and the basis of every progress figure. Use "
            "get_campaign_pacing for the pacing remedy and make-good decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "The campaign id from the campaign or pacing board.",
                }
            },
            "required": ["campaign_id"],
        },
    }
]

CAMPAIGN_SOURCE_BY_TOOL = {
    "get_campaign": "campaign store, delivery ledger and creative-assets ledger, owned channel",
}


def _cap(record: dict[str, Any], key: str, limit: int) -> None:
    rows = list(record.get(key) or [])
    record[key] = rows[:limit]
    record[f"{key}_count"] = len(rows)
    if len(rows) > limit:
        record[f"{key}_omitted"] = len(rows) - limit


def _read_get_campaign(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import campaigns_api_store as store
    from kairos_api import campaigns_delivery, channel_scope

    campaign_id = str(args.get("campaign_id", "") or "").strip()
    if not campaign_id:
        return {"error": "provide the campaign id; list owned campaigns with get_pacing_board"}
    channel = channel_scope.operator_channel()
    if not channel:
        return {
            "error": "the operator channel is not configured, so campaign records cannot be scoped",
            "scope_available": False,
        }
    rows = store.campaigns_with_flights(store.load_frame())
    campaign = next(
        (
            row for row in rows
            if row.get("campaign_id") == campaign_id and row.get("channel") == channel
        ),
        None,
    )
    if campaign is None:
        return {
            "error": "no campaign matches that id on the operator's channel",
            "campaign_id": campaign_id,
        }

    campaigns_delivery.attach([campaign])
    _cap(campaign, "flights", MAX_FLIGHTS)
    _cap(campaign, "assets", MAX_ASSETS)
    delivery = dict(campaign.get("delivery") or {})
    _cap(delivery, "days", MAX_DELIVERY_DAYS)
    campaign["delivery"] = delivery
    goal_preflight = None
    if (campaign.get("order") or {}).get("kind") == "goal_based":
        from kairos_api import campaigns_goal_order

        reads = campaigns_goal_order.goal_orders_read(include_demo=True, channel=channel)
        goal_preflight = next(
            (
                row for row in reads.get("orders", [])
                if row.get("campaign_id") == campaign_id
            ),
            None,
        )
        if goal_preflight is not None:
            goal_preflight = {"as_of": reads.get("as_of"), **goal_preflight}
    return {
        "campaign": campaign,
        "goal_preflight": goal_preflight,
        "scope": {"channel": channel, "scoped": True},
        "pacing_tool": "get_campaign_pacing",
        "basis": (
            "the campaign store states the booking; the delivery ledger states what aired, "
            "and unsourced days remain unknown rather than zero"
        ),
    }


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    executors["get_campaign"] = _read_get_campaign
    sources.update(CAMPAIGN_SOURCE_BY_TOOL)
