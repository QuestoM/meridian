"""Campaign and flight CRUD, the commercial spine's missing two levels.

Before this module the product had an agency and an advertiser and nothing
under them: ``GET /api/campaigns`` was a read-only rollup of what had already
aired, and zero of the fifty-six write operations on the live API created a
campaign. So an account manager holding a signed insertion order had no place
to put it, which is the whole of JS-5.

The shape follows :mod:`kairos_api.agencies`, which is the sibling that already
works: one row per record, validated at the door, written under a lock through a
temp file, versioned before the write, and ended rather than deleted. The store
itself is :mod:`kairos_api.campaigns_api_store`; this module is the HTTP layer
over it plus the duplicate refusals.

Two honest boundaries are enforced here rather than explained later.

**A booked goal is not a delivered figure.** A flight carries what was booked,
in a named unit, and nothing else. Nothing in this repository observes delivery,
so no pace, no shortfall and no projection is computed from a goal. The read
says so with ``delivery`` naming the missing feed, and the surface renders that
sentence instead of a number.

**A campaign term is not yet a priced term.** The condition grammar the pricing
path evaluates scopes on positions, genres, dayparts, programmes and weekdays,
and has no campaign dimension. So a campaign's rebate and its weekday surcharge
discount are stored as the agreed terms and reported with ``priced_by_engine``
false and the reason. The one-flow onboarding offers to write the discount where
it does price, as an agency condition, and says exactly what that covers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from kairos_api import campaigns_api_store as store
from kairos_api import campaigns_commitment as commitment
from kairos_api.affiliation_wall import Wall
from kairos_api.campaigns_api_models import (  # noqa: F401 - re-exported for every caller
    CampaignCreate,
    CampaignUpdate,
    FlightCreate,
    FlightUpdate,
)
from kairos_api.condition_validation import validate_weekday_scope

router = APIRouter(prefix="/api/clients", tags=["clients"])

# Booking a client is a run-side commercial act, so affiliation does not gate it
# and role does: any account may read the commercial spine, a write role may
# change it. The wall exists here to put that answer on the read, because a
# refusal a person meets after the click is the defect, not the refusal itself.
# The server's own middleware still closes the write; this is what makes the
# control tell the truth before it is pressed.
CLIENTS_WALL = Wall(company_only=False)

TERMS_NOT_PRICED = (
    "Campaign terms are the agreed commercial record. The pricing path scopes conditions on "
    "positions, genres, dayparts, programmes and weekdays, and has no campaign scope, so this "
    "term prices nothing until it is written as an agency or advertiser condition."
)
TERMS_NOT_PRICED_HE = (
    "תנאי הקמפיין הם הרישום המסחרי המוסכם. נתיב התמחור מגדיר תנאים לפי מיקום, ז׳אנר, רצועת שידור, "
    "תוכנית ויום בשבוע, ואין בו היקף לקמפיין, ולכן התנאי הזה אינו מתמחר דבר עד שייכתב כתנאי סוכנות "
    "או תנאי מפרסם."
)

def _actor(request: "Request | None") -> str:
    """Who made the change, when the session says so and blank when it does not."""
    if request is None:
        return ""
    try:
        session = getattr(request.state, "session", None)
        return str(getattr(session, "username", "") or "")
    except Exception:  # noqa: BLE001 - attribution must never fail an edit
        return ""


def require_agency(agency_id: str) -> str:
    """An agency id that exists, so a campaign is never orphaned on a typo."""
    wanted = str(agency_id or "").strip()
    if not wanted:
        return ""
    from kairos_api.agencies import _load_frame as load_agencies

    if not (load_agencies()["agency_id"].astype(str) == wanted).any():
        raise store.refuse(
            400,
            f"Agency '{wanted}' does not exist, so no campaign can be booked through it",
            f"הסוכנות ⁦{wanted}⁩ אינה קיימת, ולכן אי אפשר להזמין דרכה קמפיין",
        )
    return wanted


def _weekday_scope(value: Any) -> str:
    """The weekday scope, with the shared validator's refusal said in Hebrew too.

    ``condition_validation`` is a wave-0 module this piece reads and never
    writes, and its refusals are English only. Re-raising here is the honest way
    to keep this destination bilingual without touching a frozen file: the
    validator's own sentence travels as the English half.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return validate_weekday_scope(text)
    except HTTPException as exc:
        raise store.refuse(
            int(exc.status_code),
            str(exc.detail),
            "היקף הימים בשבוע מקבל ANY או רשימת ימי ISO מופרדת בפסיקים, 1 עד 7, שני הוא 1, שבת היא 6, ראשון הוא 7",
        ) from None


def _refuse_duplicate(frame: Any, name: str, advertiser: str, campaign_id: str) -> None:
    """One advertiser cannot hold two campaigns of the same name.

    The sentence says to open the one that already holds the name, so it carries
    that campaign's id as the address of the record it names.
    """
    for _, row in frame[frame["record_type"].astype(str) == store.CAMPAIGN].iterrows():
        held = str(row.get("campaign_id", ""))
        if held == campaign_id:
            continue
        same_name = str(row.get("name", "")).strip() == name
        same_advertiser = str(row.get("advertiser", "")).strip() == advertiser
        if same_name and same_advertiser:
            raise store.refuse(
                409,
                f"'{advertiser}' already has a campaign named '{name}', as {held}. Open that one instead of booking a second.",
                f"ל⁦{advertiser}⁩ כבר יש קמפיין בשם ⁦{name}⁩, שמספרו ⁦{held}⁩. פתחו אותו במקום להזמין קמפיין שני.",
                opens={"kind": "campaign", "id": held},
            )


def _campaign_row(payload: CampaignCreate, campaign_id: str, actor: str) -> dict[str, str]:
    row = store.blank_row()
    row.update({
        "record_type": store.CAMPAIGN,
        "campaign_id": campaign_id,
        "name": payload.name.strip(),
        "advertiser": payload.advertiser.strip(),
        "agency_id": require_agency(payload.agency_id),
        "status": "active",
        "starts_on": store.validate_date(payload.starts_on, "starts_on"),
        "ends_on": store.validate_date(payload.ends_on, "ends_on"),
        "rebate_percent": store.validate_percent(payload.rebate_percent, "rebate_percent"),
        "surcharge_discount_percent": store.validate_percent(
            payload.surcharge_discount_percent, "surcharge_discount_percent"
        ),
        "surcharge_weekdays": _weekday_scope(payload.surcharge_weekdays),
        "notes": payload.notes.strip(),
        "created_by": actor,
        "data_source": "manual",
        # A campaign booked here is a real booking, so it is not a demo row. The
        # channel is stamped from settings and never taken from the request.
        "is_demo": "false",
        **commitment.row_values(payload),
    })
    store.validate_window(row["starts_on"], row["ends_on"])
    return row


def create_campaign_row(payload: CampaignCreate, request: "Request | None") -> dict[str, Any]:
    """Create one campaign under the store lock. Shared with the onboarding flow."""
    if not payload.name.strip():
        raise store.refuse(
            400,
            "A campaign needs a name",
            "לקמפיין צריך שם",
        )
    if not payload.advertiser.strip():
        raise store.refuse(
            400,
            "A campaign needs a client, because a campaign belongs to one",
            "לקמפיין צריך לקוח, מפני שקמפיין שייך ללקוח",
        )
    with store.lock():
        frame = store.load_frame()
        campaign_id = payload.campaign_id.strip() or store.next_campaign_id(frame)
        if (
            (frame["record_type"].astype(str) == store.CAMPAIGN)
            & (frame["campaign_id"].astype(str) == campaign_id)
        ).any():
            raise store.refuse(
                409,
                f"Campaign '{campaign_id}' already exists",
                f"הקמפיין ⁦{campaign_id}⁩ כבר קיים",
                opens={"kind": "campaign", "id": campaign_id},
            )
        _refuse_duplicate(frame, payload.name.strip(), payload.advertiser.strip(), campaign_id)
        frame = store.append(frame, _campaign_row(payload, campaign_id, _actor(request)))
        store.snapshot_before_write(request)
        store.write_frame(frame)
        return store.campaign_record(frame.iloc[-1])


def create_flight_row(campaign_id: str, payload: FlightCreate, request: "Request | None") -> dict[str, Any]:
    """Add one flight to an existing campaign. Shared with the onboarding flow."""
    with store.lock():
        frame = store.load_frame()
        store.locate_campaign(frame, campaign_id)
        flight_id = payload.flight_id.strip() or store.next_flight_id(frame, campaign_id)
        row = store.blank_row()
        row.update({
            "record_type": store.FLIGHT,
            "campaign_id": campaign_id,
            "flight_id": flight_id,
            "name": payload.name.strip(),
            "starts_on": store.validate_date(payload.starts_on, "starts_on"),
            "ends_on": store.validate_date(payload.ends_on, "ends_on"),
            "goal_kind": store.validate_choice(payload.goal_kind, store.GOAL_KINDS, "goal_kind"),
            "goal_value": store.validate_goal(payload.goal_value),
            "notes": payload.notes.strip(),
            "created_by": _actor(request),
        })
        store.validate_window(row["starts_on"], row["ends_on"])
        if (
            (frame["record_type"].astype(str) == store.FLIGHT)
            & (frame["campaign_id"].astype(str) == campaign_id)
            & (frame["flight_id"].astype(str) == flight_id)
        ).any():
            raise store.refuse(
                409,
                f"Flight '{flight_id}' already exists on campaign '{campaign_id}'",
                f"טיסת השידור ⁦{flight_id}⁩ כבר קיימת בקמפיין ⁦{campaign_id}⁩",
            )
        frame = store.append(frame, row)
        store.snapshot_before_write(request)
        store.write_frame(frame)
        return store.flight_record(frame.iloc[-1])


@router.get("/campaigns")
def list_campaigns(request: Request = None) -> dict[str, Any]:
    """Every campaign with its flights, its creative, its delivery and the limits.

    Delivery is attached rather than declared missing, because there now is a
    ledger: :mod:`kairos_api.campaigns_delivery` derives it from the traffic log
    on disk. It stays honest by staying unavailable when no campaign here has a
    sourced day, and by reporting every counted figure as a floor over the days
    a source exists for.
    """
    from kairos_api import campaigns_delivery

    campaigns = store.campaigns_with_flights(store.load_frame())
    delivery = campaigns_delivery.attach(campaigns)
    demo_rows = sum(1 for campaign in campaigns if campaign["is_demo"])
    return CLIENTS_WALL.stamp({
        "campaigns": campaigns,
        "count": len(campaigns),
        "demo_count": demo_rows,
        "booked_count": len(campaigns) - demo_rows,
        "goal_kinds": list(store.GOAL_KINDS),
        "statuses": list(store.STATUSES),
        "status_vocabulary": [dict(entry) for entry in store.STATUS_VOCABULARY],
        "goal_kind_vocabulary": [dict(entry) for entry in store.GOAL_KIND_VOCABULARY],
        "delivery": delivery,
        "terms": {
            "priced_by_engine": False,
            "reason_en": TERMS_NOT_PRICED,
            "reason_he": TERMS_NOT_PRICED_HE,
        },
        **commitment.vocabularies(),
    }, request)


@router.post("/campaigns", status_code=201)
def create_campaign(payload: CampaignCreate, request: Request = None) -> dict[str, Any]:
    return create_campaign_row(payload, request)


@router.put("/campaigns/{campaign_id}")
def update_campaign(campaign_id: str, payload: CampaignUpdate, request: Request = None) -> dict[str, Any]:
    values = payload.model_dump(exclude_unset=True)
    with store.lock():
        frame = store.load_frame()
        index = store.locate_campaign(frame, campaign_id)
        name = str(values.get("name", frame.at[index, "name"]) or "").strip()
        advertiser = str(values.get("advertiser", frame.at[index, "advertiser"]) or "").strip()
        if "name" in values or "advertiser" in values:
            _refuse_duplicate(frame, name, advertiser, campaign_id)
        for key, value in values.items():
            if value is None:
                continue
            frame.at[index, key] = _coerced(key, value)
        store.validate_window(str(frame.at[index, "starts_on"]), str(frame.at[index, "ends_on"]))
        store.snapshot_before_write(request)
        store.write_frame(frame)
        return store.campaign_record(frame.loc[index])


def _coerced(key: str, value: Any) -> str:
    """One field's validated string form, refused at the door when it is wrong."""
    committed = commitment.coerce_field(key, value)
    if committed is not None:
        return committed
    if key in {"starts_on", "ends_on"}:
        return store.validate_date(value, key)
    if key in {"rebate_percent", "surcharge_discount_percent"}:
        return store.validate_percent(value, key)
    if key == "surcharge_weekdays":
        return _weekday_scope(value)
    if key == "status":
        return store.validate_choice(value, store.STATUSES, "status")
    if key == "agency_id":
        return require_agency(value)
    return str(value).strip()


@router.post("/campaigns/{campaign_id}/deactivate")
def deactivate_campaign(campaign_id: str, request: Request = None) -> dict[str, Any]:
    """End a campaign. Its flights and its history stay; nothing is deleted."""
    with store.lock():
        frame = store.load_frame()
        index = store.locate_campaign(frame, campaign_id)
        frame.at[index, "status"] = "ended"
        store.snapshot_before_write(request)
        store.write_frame(frame)
        return store.campaign_record(frame.loc[index])


@router.post("/campaigns/{campaign_id}/flights", status_code=201)
def create_flight(campaign_id: str, payload: FlightCreate, request: Request = None) -> dict[str, Any]:
    return create_flight_row(campaign_id, payload, request)


@router.put("/campaigns/{campaign_id}/flights/{flight_id}")
def update_flight(campaign_id: str, flight_id: str, payload: FlightUpdate,
                  request: Request = None) -> dict[str, Any]:
    values = payload.model_dump(exclude_unset=True)
    with store.lock():
        frame = store.load_frame()
        index = store.locate_flight(frame, campaign_id, flight_id)
        for key, value in values.items():
            if value is None:
                continue
            if key in {"starts_on", "ends_on"}:
                frame.at[index, key] = store.validate_date(value, key)
            elif key == "goal_kind":
                frame.at[index, key] = store.validate_choice(value, store.GOAL_KINDS, "goal_kind")
            elif key == "goal_value":
                frame.at[index, key] = store.validate_goal(value)
            else:
                frame.at[index, key] = str(value).strip()
        store.validate_window(str(frame.at[index, "starts_on"]), str(frame.at[index, "ends_on"]))
        store.snapshot_before_write(request)
        store.write_frame(frame)
        return store.flight_record(frame.loc[index])


@router.delete("/campaigns/{campaign_id}/flights/{flight_id}")
def delete_flight(campaign_id: str, flight_id: str, request: Request = None) -> dict[str, Any]:
    """Remove one flight. The campaign is ended, never deleted; a flight is a line."""
    with store.lock():
        frame = store.load_frame()
        index = store.locate_flight(frame, campaign_id, flight_id)
        frame = frame.drop(index=index).reset_index(drop=True)
        store.snapshot_before_write(request)
        store.write_frame(frame)
    return {"deleted": flight_id, "campaign_id": campaign_id}


@router.get("/onboarding/options")
def onboarding_options(request: Request = None) -> dict[str, Any]:
    """Everything the one-flow form needs, in one call, so nobody has to guess."""
    from kairos_api.campaigns_api_onboarding import options

    return CLIENTS_WALL.stamp(options(), request)


@router.post("/onboarding", status_code=201)
def onboard(payload: dict[str, Any], request: Request = None) -> dict[str, Any]:
    """Agency, advertiser link, campaign, flights and terms, in one pass."""
    from kairos_api.campaigns_api_onboarding import OnboardRequest, onboard_client

    return onboard_client(OnboardRequest(**payload), request)


# The per-campaign detail reads mount here, so they publish under this prefix.
from kairos_api.campaigns_api_detail import router as _detail_router  # noqa: E402

router.include_router(_detail_router)
