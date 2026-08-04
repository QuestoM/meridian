"""The request bodies the campaign and flight routes accept.

Split out of :mod:`kairos_api.campaigns_api` to keep that module under the
project line limit, and named for its parent so the pair is obvious. Nothing
moved changed: the four models are the same classes with the same fields and the
same defaults, and ``campaigns_api`` re-exports each of them under the name it
always had, so ``from kairos_api.campaigns_api import CampaignCreate`` still
resolves for every caller.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CampaignCreate(BaseModel):
    """A new campaign. ``advertiser`` is the client the campaign is for.

    The commitment half is optional on every field, because an insertion order
    that names a budget and no rating goal is a real order and the store must be
    able to hold it without inventing the half it was not told.
    """

    name: str
    advertiser: str
    agency_id: str = ""
    campaign_id: str = ""
    starts_on: str = ""
    ends_on: str = ""
    rebate_percent: Optional[float] = None
    surcharge_discount_percent: Optional[float] = None
    surcharge_weekdays: str = ""
    notes: str = ""
    brand: str = ""
    category: str = ""
    budget_ils: Optional[float] = None
    bonus_ils: Optional[float] = None
    rating_goal_points: Optional[float] = None
    rating_goal_audience: str = ""
    price_model: str = ""
    priority: str = ""
    pacing_mode: str = ""


class CampaignUpdate(BaseModel):
    """Editable fields for a campaign. All optional for PATCH-style PUT."""

    name: Optional[str] = None
    advertiser: Optional[str] = None
    agency_id: Optional[str] = None
    status: Optional[str] = None
    starts_on: Optional[str] = None
    ends_on: Optional[str] = None
    rebate_percent: Optional[float] = None
    surcharge_discount_percent: Optional[float] = None
    surcharge_weekdays: Optional[str] = None
    notes: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    budget_ils: Optional[float] = None
    bonus_ils: Optional[float] = None
    rating_goal_points: Optional[float] = None
    rating_goal_audience: Optional[str] = None
    price_model: Optional[str] = None
    priority: Optional[str] = None
    pacing_mode: Optional[str] = None


class FlightCreate(BaseModel):
    """A flight: a window of the campaign with its own booked goal."""

    starts_on: str
    ends_on: str
    goal_kind: str
    goal_value: float
    name: str = ""
    flight_id: str = ""
    notes: str = ""


class FlightUpdate(BaseModel):
    """Editable fields for a flight. All optional for PATCH-style PUT."""

    starts_on: Optional[str] = None
    ends_on: Optional[str] = None
    goal_kind: Optional[str] = None
    goal_value: Optional[float] = None
    name: Optional[str] = None
    notes: Optional[str] = None
