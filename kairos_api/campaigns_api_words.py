"""The words this store speaks and the refusals it raises, in both languages.

Split out of :mod:`kairos_api.campaigns_api_store` when the campaign entity grew
its commitment half, so that module stays under the project line limit and stays
about persistence. Nothing moved changed: the vocabularies, the field words, the
bilingual refusal and every validator are the same objects, and the store
re-exports each of them under the name it always had, so ``store.validate_date``
and ``store.STATUS_VOCABULARY`` still resolve for every caller.

Two rules govern everything here.

**A closed set beats free text.** A status, a goal unit, a price model and a
target audience are chosen from a published list, because a free-text unit is a
figure nobody can compare and a free-text audience is a rating point nobody can
count.

**A refusal is addressed to a person.** It names the field the way a person
names it, not the way a column names it, and it arrives in Hebrew and in English
so the Hebrew flow never answers in the language the code was written in.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import HTTPException

STATUSES = ("active", "ended")
# The units a booked goal can be stated in. Closed, because a free-text unit is
# a figure nobody can compare and this list is what a delivery feed must speak.
GOAL_KINDS = ("spots", "seconds", "impressions", "grp", "ils")

# Every closed value set this store speaks, with the word a person reads in each
# language and, for a state, what to do about it. Google Ads pairs each status
# with a "what to do" column rather than leaving the reader to infer one, and a
# raw English token on a Hebrew surface is not a word anybody reads. The tables
# travel on the payload so no surface has to hold a second copy of them.
STATUS_VOCABULARY = (
    {
        "value": "active",
        "label_en": "Active",
        "label_he": "פעיל",
        "what_to_do_en": "Nothing. It is booked and it stands until you end it.",
        "what_to_do_he": "דבר. הקמפיין מוזמן ועומד עד שתסיימו אותו.",
    },
    {
        "value": "ended",
        "label_en": "Ended",
        "label_he": "הסתיים",
        "what_to_do_en": "Book a new campaign. An ended one keeps its flights and its history.",
        "what_to_do_he": "הזמינו קמפיין חדש. קמפיין שהסתיים שומר את הטיסות ואת ההיסטוריה שלו.",
    },
)

GOAL_KIND_VOCABULARY = (
    {"value": "spots", "label_en": "spots", "label_he": "תשדירים"},
    {"value": "seconds", "label_en": "seconds", "label_he": "שניות"},
    {"value": "impressions", "label_en": "impressions", "label_he": "חשיפות"},
    {"value": "grp", "label_en": "GRP", "label_he": "GRP"},
    {"value": "ils", "label_en": "ILS", "label_he": "ש״ח"},
)

# The words a person reads for a field this store refuses. A refusal that names
# ``starts_on`` at an account manager is a column name, not a sentence, and the
# same rule that keeps AGY_10 off the campaign board keeps it out of here.
FIELD_WORDS = {
    "starts_on": ("The start date", "תאריך ההתחלה"),
    "ends_on": ("The end date", "תאריך הסיום"),
    "goal_kind": ("The goal unit", "יחידת היעד"),
    "goal_value": ("The goal", "היעד"),
    "status": ("The status", "הסטטוס"),
    "rebate_percent": ("The rebate percent", "אחוז הרבייט"),
    "surcharge_discount_percent": ("The discount percent", "אחוז ההנחה"),
    "surcharge_weekdays": ("The weekday scope", "היקף הימים בשבוע"),
    "agency_id": ("The agency", "הסוכנות"),
    "name": ("The name", "השם"),
    "advertiser": ("The client", "הלקוח"),
    "budget_ils": ("The budget", "התקציב"),
    "bonus_ils": ("The added value budget", "תקציב הערך המוסף"),
    "rating_goal_points": ("The rating point goal", "יעד נקודות הרייטינג"),
    "rating_goal_audience": ("The target audience", "קהל היעד"),
    "price_model": ("The price model", "מודל התמחור"),
    "priority": ("The priority", "העדיפות"),
    "pacing_mode": ("The pacing", "קצב הפריסה"),
    "brand": ("The brand", "המותג"),
    "category": ("The category", "הקטגוריה"),
}


def field_words(field: str) -> tuple[str, str]:
    """One field named as a person would name it, or its raw key when it is new."""
    return FIELD_WORDS.get(str(field), (str(field), str(field)))


def refuse(
    status_code: int,
    message_en: str,
    message_he: str,
    opens: dict[str, str] | None = None,
) -> HTTPException:
    """One refusal, carried in both languages the way every read sentence is.

    The reads on this destination already cross the wire as ``*_en`` and
    ``*_he`` pairs, and the writes did not: a duplicate campaign rendered in the
    Hebrew flow as an English sentence, which is a refusal the person it is
    addressed to cannot act on. ``detail`` is the pair, so the surface picks the
    reader's language instead of printing the one the code was written in.

    ``opens`` is the address of the record the sentence names, as
    ``{"kind": "campaign" | "agency", "id": "..."}``. Two refusals here tell the
    reader to open a record that already exists and neither said where it was, so
    the sentence named a place and gave no way to it. The address travels with
    the refusal rather than being parsed back out of the prose, and it is absent
    from ``detail`` entirely when the refusal names no record, so a surface can
    never grow a control that opens nothing.
    """
    detail: dict[str, Any] = {"message_en": message_en, "message_he": message_he}
    if opens and opens.get("kind") and opens.get("id"):
        detail["opens"] = {"kind": str(opens["kind"]), "id": str(opens["id"])}
    return HTTPException(status_code=status_code, detail=detail)


def choice_words(allowed: tuple[str, ...], locale: str) -> str:
    """The allowed values as words, from the vocabularies this store publishes."""
    from kairos_api.campaigns_commitment import (
        PACING_MODES,
        PRICE_MODELS,
        PRIORITIES,
        TARGET_AUDIENCES,
    )

    published = (
        *STATUS_VOCABULARY, *GOAL_KIND_VOCABULARY, *TARGET_AUDIENCES,
        *PRICE_MODELS, *PRIORITIES, *PACING_MODES,
    )
    table = {entry["value"]: entry for entry in published}
    labels = []
    for value in allowed:
        entry = table.get(value)
        if entry is None:
            labels.append(value)
        else:
            labels.append(entry["label_he"] if locale == "he" else entry["label_en"])
    return ", ".join(labels)


def validate_date(raw: Any, field: str, *, required: bool = True) -> str:
    """An ISO calendar date, or a stated refusal. Never a silently coerced one."""
    english, hebrew = field_words(field)
    text = str(raw or "").strip()
    if not text:
        if required:
            raise refuse(
                400,
                f"{english} is required, as an ISO date, YYYY-MM-DD",
                f"חובה למלא את {hebrew}, בתאריך ISO, YYYY-MM-DD",
            )
        return ""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise refuse(
            400,
            f"{english} must be an ISO date, YYYY-MM-DD",
            f"יש להזין את {hebrew} כתאריך ISO, YYYY-MM-DD",
        ) from None


def validate_window(starts_on: str, ends_on: str) -> None:
    if starts_on and ends_on and ends_on < starts_on:
        raise refuse(
            400,
            "The end date cannot be earlier than the start date",
            "תאריך הסיום אינו יכול להיות מוקדם מתאריך ההתחלה",
        )


def validate_choice(raw: Any, allowed: tuple[str, ...], field: str, *, allow_blank: bool = False) -> str:
    english, hebrew = field_words(field)
    text = str(raw or "").strip()
    if text and text in allowed:
        return text
    if not text and allow_blank:
        return ""
    raise refuse(
        400,
        f"{english} must be one of: {choice_words(allowed, 'en')}",
        f"יש לבחור את {hebrew} מתוך: {choice_words(allowed, 'he')}",
    )


def validate_percent(raw: Any, field: str) -> str:
    """A percent between 0 and 100, or blank when the term was not agreed."""
    english, hebrew = field_words(field)
    text = str(raw if raw is not None else "").strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise refuse(
            400,
            f"{english} must be a number between 0 and 100",
            f"יש להזין את {hebrew} כמספר בין 0 ל־100",
        ) from None
    if not 0.0 <= value <= 100.0:
        raise refuse(
            400,
            f"{english} must be between 0 and 100",
            f"יש להזין את {hebrew} בין 0 ל־100",
        )
    return str(value)


def validate_amount(raw: Any, field: str) -> str:
    """A figure of zero or more, or blank when the commitment does not state one.

    Blank is a real answer and it is not zero. A campaign booked without a
    rating goal has no rating goal, and a board that renders that as ``0`` has
    told the reader the buyer committed to nothing, which is a different fact.
    """
    english, hebrew = field_words(field)
    text = str(raw if raw is not None else "").strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise refuse(
            400,
            f"{english} must be a number",
            f"יש להזין את {hebrew} כמספר",
        ) from None
    if value < 0:
        raise refuse(
            400,
            f"{english} must be zero or greater",
            f"יש להזין את {hebrew} כאפס או יותר",
        )
    return str(round(value, 2))


def validate_goal(raw: Any) -> str:
    text = str(raw if raw is not None else "").strip()
    if not text:
        raise refuse(
            400,
            "A flight needs a goal, because a flight without one cannot be measured",
            "טיסת שידור צריכה יעד, מפני שטיסה בלי יעד אי אפשר למדוד",
        )
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise refuse(400, "The goal must be a number", "יש להזין את היעד כמספר") from None
    if value < 0:
        raise refuse(
            400,
            "The goal must be zero or greater",
            "יש להזין יעד של אפס או יותר",
        )
    return str(value)
