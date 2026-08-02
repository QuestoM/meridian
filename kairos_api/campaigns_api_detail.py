"""One campaign in full: its flights, its creative and every day of its delivery.

Split out of :mod:`kairos_api.campaigns_api` to keep that module under the
project line limit. The board read already carries these fields for every
campaign; these two routes exist so a drawer can open one campaign, and a
creative panel can open one campaign's tapes, without pulling the whole board
through the wire.

Nothing here computes anything of its own. The campaign comes from the store,
the delivery from :mod:`kairos_api.campaigns_delivery` and the creative from
:mod:`kairos_api.campaigns_assets`, so a figure read here is the same object the
board read, and neither can drift from the other.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from kairos_api import campaigns_api_store as store
from kairos_api import campaigns_assets, campaigns_commitment, campaigns_delivery

router = APIRouter()


def _one(campaign_id: str) -> dict[str, Any]:
    """The campaign, with delivery and creative attached, or the store's 404."""
    frame = store.load_frame()
    store.locate_campaign(frame, campaign_id)
    campaigns = [
        campaign for campaign in store.campaigns_with_flights(frame)
        if campaign["campaign_id"] == campaign_id
    ]
    campaigns_delivery.attach(campaigns)
    return campaigns[0]


@router.get("/campaigns/{campaign_id}/detail")
def campaign_detail(campaign_id: str, request: Request = None) -> dict[str, Any]:
    """One campaign: identity, commitment, flights, creative and delivery days."""
    from kairos_api.campaigns_api import CLIENTS_WALL

    return CLIENTS_WALL.stamp({
        "campaign": _one(campaign_id),
        **campaigns_assets.vocabularies(),
        **campaigns_commitment.vocabularies(),
    }, request)


@router.get("/campaigns/{campaign_id}/assets")
def campaign_assets(campaign_id: str, request: Request = None) -> dict[str, Any]:
    """One campaign's creative, with every unreadable property named as unknown."""
    from kairos_api.campaigns_api import CLIENTS_WALL

    campaign = _one(campaign_id)
    return CLIENTS_WALL.stamp({
        "campaign_id": campaign["campaign_id"],
        "campaign_name": campaign["name"],
        "advertiser": campaign["advertiser"],
        "channel": campaign["channel"],
        "is_demo": campaign["is_demo"],
        "demo": campaign["demo"],
        "assets": campaign["assets"],
        "summary": campaign["assets_summary"],
        **campaigns_assets.vocabularies(),
    }, request)
