"""One flow: an agency, the client under it, the campaign, its flights, its terms.

Split out of :mod:`kairos_api.campaigns_api` to keep that module under the
project line limit. It is JS-5 in one request, and the reason it is one request
rather than four is measured: today two of the three entities have no creation
path at all, and the third, the advertiser-to-agency link, runs through a
different endpoint on a different screen, so the account manager has to leave
and come back to link what they just made.

Nothing here is new machinery. Every step calls the store that already owns it,
so there is exactly one implementation of each write and this module cannot
drift from the pages that perform the same writes one at a time.

**Zero duplicates is enforced, not hoped for.** An agency named in the request
that already exists is reused and reported as reused, never created twice. An
advertiser already linked to the agency is left alone. A campaign whose name and
advertiser match an existing one is refused with the id that already holds it.
So running the same insertion order twice produces one agency, one link and one
refusal, and the response says which of the three happened for each step.

**The Saturday discount is written where it prices, or not written at all.**
The condition grammar has no campaign scope, so a campaign-level term prices
nothing. When the flow is asked to apply the discount, it writes an agency
condition with the weekday scope, which the daily pricing path really evaluates,
and the response states plainly that an agency rule covers every campaign bought
through that agency. When it is not asked, the term stays on the campaign as the
agreed record and the response says it prices nothing.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from kairos.optimize.advertiser_rules import PREMIUM
from kairos.optimize._rule_helpers import PREMIUM_DISCOUNT
from kairos_api.campaigns_api_store import refuse

AGENCY_RULE_COVERS = (
    "An agency condition covers every campaign bought through that agency, because the condition "
    "grammar has no campaign scope. This is the only level at which the discount prices spots."
)
AGENCY_RULE_COVERS_HE = (
    "תנאי סוכנות חל על כל קמפיין שנקנה דרך אותה סוכנות, מפני שאין בדקדוק התנאים היקף לקמפיין. זו "
    "הרמה היחידה שבה ההנחה מתמחרת תשדירים."
)
TERM_ONLY = (
    "Stored as the agreed term on the campaign. It prices nothing until it is written as an agency "
    "or advertiser condition."
)
TERM_ONLY_HE = (
    "נשמר כתנאי המוסכם על הקמפיין. אינו מתמחר דבר עד שייכתב כתנאי סוכנות או תנאי מפרסם."
)
LINK_MEANING = (
    "The link makes this advertiser the agency's on the daily pricing path. The advertiser gets a "
    "named identity record of its own the first time it appears in a daily file."
)
LINK_MEANING_HE = (
    "השיוך הופך את המפרסם הזה לשל הסוכנות בנתיב התמחור היומי. המפרסם מקבל כרטיס זהות בשם משלו "
    "בפעם הראשונה שהוא מופיע בקובץ יומי."
)


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


def _agencies_frame():
    from kairos_api.agencies import _load_frame

    return _load_frame()


def next_agency_id() -> str:
    """The next free AGY_nn, so the form never asks a person to invent one."""
    frame = _agencies_frame()
    used = {str(value) for value in frame["agency_id"].astype(str)}
    index = 1
    while f"AGY_{index:02d}" in used:
        index += 1
    return f"AGY_{index:02d}"


def _existing_agency(payload: AgencyInput) -> Optional[str]:
    """The id of the agency this step names, if the product already holds it."""
    frame = _agencies_frame()
    wanted_id = payload.agency_id.strip()
    if wanted_id and (frame["agency_id"].astype(str) == wanted_id).any():
        return wanted_id
    wanted_name = payload.name.strip()
    if wanted_name:
        match = frame[frame["name"].astype(str) == wanted_name]
        if not match.empty:
            return str(match.iloc[0]["agency_id"])
    return None


def _create_agency(payload: AgencyInput, request: Any) -> dict[str, Any]:
    from kairos_api.agencies import AgencyCreate, create_agency

    if not payload.name.strip():
        raise refuse(
            400,
            "An agency needs a name, because the name is what a daily file carries",
            "לסוכנות צריך שם, מפני שהשם הוא מה שקובץ יומי נושא",
        )
    record = create_agency(
        AgencyCreate(
            agency_id=payload.agency_id.strip() or next_agency_id(),
            name=payload.name.strip(),
            display_name=payload.name.strip(),
            agency_type=payload.agency_type,
            contact_name=payload.contact_name,
            contact_role=payload.contact_role,
            contact_phone=payload.contact_phone,
            contact_email=payload.contact_email,
            vat_id=payload.vat_id,
            payment_terms_days=payload.payment_terms_days,
            rebate_percent=payload.rebate_percent,
            commission_percent=payload.commission_percent,
            credit_limit_ils=payload.credit_limit_ils,
            notes=payload.notes,
            data_source="manual",
        ),
        request,
    )
    return {"agency_id": record["agency_id"], "name": record["name"], "outcome": "created", "record": record}


def _link_advertiser(agency_id: str, advertiser: str, request: Any) -> dict[str, Any]:
    """Link the client to the agency, or report that it already was."""
    from kairos_api.agency_conditions import LinkCreate, create_link, links_for

    name = advertiser.strip()
    if not name:
        raise refuse(
            400,
            "A campaign needs a client, because a campaign belongs to one",
            "לקמפיין צריך לקוח, מפני שקמפיין שייך ללקוח",
        )
    links = links_for(agency_id)
    if name in links["effective"]:
        source = "manual" if name in links["manual"] else "observed"
        return {"advertiser": name, "outcome": "already_linked", "source": source,
                "meaning_en": LINK_MEANING, "meaning_he": LINK_MEANING_HE}
    create_link(agency_id, LinkCreate(advertiser=name), request)
    return {"advertiser": name, "outcome": "linked", "source": "manual",
            "meaning_en": LINK_MEANING, "meaning_he": LINK_MEANING_HE}


def _apply_agency_discount(agency_id: str, campaign_id: str, percent: float,
                           weekdays: str, request: Any) -> dict[str, Any]:
    """Write the weekday discount as the agency condition that really prices it."""
    from kairos_api.agency_conditions import ConditionCreate, create_condition

    rule_id = f"{campaign_id}_DISCOUNT"
    record = create_condition(
        agency_id,
        ConditionCreate(
            rule_id=rule_id,
            effect=PREMIUM,
            value=float(percent),
            mode=PREMIUM_DISCOUNT,
            scope_weekdays=weekdays,
            notes=f"Agreed on campaign {campaign_id}",
        ),
        request,
    )
    return {
        "outcome": "priced_as_agency_condition",
        "agency_id": agency_id,
        "rule_id": rule_id,
        "covers_en": AGENCY_RULE_COVERS,
        "covers_he": AGENCY_RULE_COVERS_HE,
        "condition": record,
    }


def onboard_client(payload: OnboardRequest, request: Any) -> dict[str, Any]:
    """Agency, client, campaign, flights and terms, created and linked in one pass."""
    from kairos_api.campaigns_api import CampaignCreate, FlightCreate, create_campaign_row, create_flight_row

    existing = _existing_agency(payload.agency)
    if existing is not None:
        agency = {
            "agency_id": existing,
            "name": payload.agency.name.strip(),
            "outcome": "reused",
            "record": None,
        }
    else:
        agency = _create_agency(payload.agency, request)

    link = _link_advertiser(agency["agency_id"], payload.advertiser, request)

    campaign = create_campaign_row(
        CampaignCreate(
            name=payload.campaign_name,
            advertiser=payload.advertiser,
            agency_id=agency["agency_id"],
            campaign_id=payload.campaign_id,
            starts_on=payload.campaign_starts_on,
            ends_on=payload.campaign_ends_on,
            rebate_percent=payload.rebate_percent,
            surcharge_discount_percent=payload.surcharge_discount_percent,
            surcharge_weekdays=payload.surcharge_weekdays,
            notes=payload.notes,
            brand=payload.brand,
            category=payload.category,
            budget_ils=payload.budget_ils,
            bonus_ils=payload.bonus_ils,
            rating_goal_points=payload.rating_goal_points,
            rating_goal_audience=payload.rating_goal_audience,
            price_model=payload.price_model,
            priority=payload.priority,
            pacing_mode=payload.pacing_mode,
        ),
        request,
    )

    flights = [
        create_flight_row(
            campaign["campaign_id"],
            FlightCreate(
                starts_on=flight.starts_on,
                ends_on=flight.ends_on,
                goal_kind=flight.goal_kind,
                goal_value=flight.goal_value,
                name=flight.name,
                notes=flight.notes,
            ),
            request,
        )
        for flight in payload.flights
    ]

    discount: dict[str, Any] = {
        "outcome": "stored_on_the_campaign",
        "note_en": TERM_ONLY,
        "note_he": TERM_ONLY_HE,
    }
    wants_rule = payload.apply_surcharge_as_agency_rule and payload.surcharge_discount_percent
    if wants_rule:
        discount = _apply_agency_discount(
            agency["agency_id"],
            campaign["campaign_id"],
            float(payload.surcharge_discount_percent or 0.0),
            payload.surcharge_weekdays or "",
            request,
        )

    return {
        "agency": agency,
        "advertiser": link,
        "campaign": campaign,
        "flights": flights,
        "discount": discount,
        "created": {
            "agency": agency["outcome"] == "created",
            "advertiser_link": link["outcome"] == "linked",
            "campaign": True,
            "flights": len(flights),
        },
    }


def options() -> dict[str, Any]:
    """Every choice the form offers, from the real stores, in one call."""
    from kairos_api import campaigns_api_store as store
    from kairos_api import campaigns_commitment
    from kairos_api.agencies import AGENCY_TYPES
    from kairos_api.condition_validation import weekday_options

    frame = _agencies_frame()
    agencies = [
        {
            "agency_id": str(row.get("agency_id", "")),
            "name": str(row.get("name", "")),
            "status": str(row.get("status", "active")) or "active",
            "rebate_percent": _percent(row.get("rebate_percent")),
        }
        for _, row in frame.iterrows()
    ]
    return {
        "agencies": agencies,
        "advertisers": _known_advertisers(),
        "agency_types": list(AGENCY_TYPES),
        "goal_kinds": list(store.GOAL_KINDS),
        "goal_kind_vocabulary": [dict(entry) for entry in store.GOAL_KIND_VOCABULARY],
        "weekdays": weekday_options(),
        "next_agency_id": next_agency_id(),
        "next_campaign_id": store.next_campaign_id(store.load_frame()),
        "operator_channel": campaigns_commitment.operator_channel(),
        **campaigns_commitment.vocabularies(),
    }


def _percent(raw: Any) -> Optional[float]:
    try:
        return round(float(raw), 4)
    except (TypeError, ValueError):
        return None


def _known_advertisers() -> list[dict[str, Any]]:
    """The observed name space, so the form offers real clients before free text."""
    try:
        from kairos.optimize.advertiser_rules_identity import (
            DEFAULT_NAMES_PATH,
            load_advertiser_names,
        )

        names = load_advertiser_names(DEFAULT_NAMES_PATH)
    except Exception:  # noqa: BLE001 - a missing name space is an empty list, not an error
        return []
    return [
        {"advertiser": record.name, "shown_name": record.shown_name, "source": record.source}
        for record in sorted(names.values(), key=lambda item: item.name)
    ]
