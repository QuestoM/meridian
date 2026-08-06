"""The request body the one-flow onboarding route accepts.

Split out of :mod:`kairos_api.campaigns_api_onboarding` to keep that module
under the project line limit, and named for its parent so the pair is obvious.
Nothing moved changed: the three models are the same classes with the same
fields and the same defaults, and ``campaigns_api_onboarding`` re-exports each
of them under the name it always had, so
``from kairos_api.campaigns_api_onboarding import OnboardRequest`` still
resolves for every caller.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AgencyInput(BaseModel):
    """The agency step: an existing id, or the fields to create one."""

    agency_id: str = ""
    name: str = ""
    agency_type: str = ""
    contact_name: str = ""
    contact_role: str = ""
    contact_phone: str = ""
    contact_email: str = ""
    vat_id: str = ""
    payment_terms_days: int = 60
    rebate_percent: float = 0.0
    commission_percent: float = 0.0
    credit_limit_ils: float = 0.0
    notes: str = ""


class FlightInput(BaseModel):
    starts_on: str
    ends_on: str
    goal_kind: str
    goal_value: float
    name: str = ""
    notes: str = ""


class OnboardRequest(BaseModel):
    """One signed insertion order, as the account manager holds it."""

    agency: AgencyInput
    advertiser: str
    campaign_name: str
    campaign_starts_on: str
    campaign_ends_on: str
    flights: list[FlightInput] = []
    campaign_id: str = ""
    rebate_percent: Optional[float] = None
    surcharge_discount_percent: Optional[float] = None
    surcharge_weekdays: str = ""
    apply_surcharge_as_agency_rule: bool = False
    notes: str = ""
    # The commitment half, all optional: an insertion order that names a budget
    # and no rating goal is a real order, and the flow must hold it as it is.
    brand: str = ""
    category: str = ""
    budget_ils: Optional[float] = None
    bonus_ils: Optional[float] = None
    rating_goal_points: Optional[float] = None
    rating_goal_audience: str = ""
    price_model: str = ""
    priority: str = ""
    pacing_mode: str = ""
