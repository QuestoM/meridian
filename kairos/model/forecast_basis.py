"""What the rating forecast is measured on, and when it is allowed to answer.

Two refusals live here, and they are the reason the forecast surface can be
shown to a buyer at all.

**The currency.** The audience model's base is the planned break rating on the
ALL-VIEWERS base: the mean measured spot rating in the programme's clock hour,
over everyone the panel counts. The Israeli trade settles on something else --
cost per rating point of JEWISH HOUSEHOLDS, on the QUARTER-HOUR rating, with
OVERNIGHT +1 deferred viewing (``docs/trade/domain.md`` section 3, one published
sentence carrying all three). ``docs/trade/domain.md`` states the rule this
module enforces: *the model must never conflate the two bases.* So a forecast
requested in the settlement currency is REFUSED, and refused with its own
reason rather than the generic one, because that particular substitution -- an
all-viewers number handed over as a household number -- is the one the domain
document names. A demographic the product does not measure at all is refused
too. Only the model's own base is served.

This is deliberately STRICTER than :mod:`kairos.trade.obligations`, which
measures a delivered standing and labels the base it counted on. A standing is
a count of something that already happened with its basis attached; a forecast
is a forward number with a band, published under whatever audience it was asked
for, and a band around the wrong base is a fabrication. Where the two modules
differ, this one refuses.

**The window.** The fit carries no trend and no season term (the season family
is off for lack of contrast in a one-month window), so beyond one annual cycle
past the last observation nothing measured bears on the date. Outside the
bundled Israeli calendar the holiday features read false rather than measured,
which would score the forecast on fabricated calendar context. Both are
refused. A date INSIDE the measured window is answered, and says so: it is a
fit, not an out-of-sample forecast, and the payload points at the backtest for
out-of-sample accuracy.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Optional

# The default confidence level of the published range.
DEFAULT_LEVEL = 0.80

# How far past the last measured observation the fit is allowed to speak.
MAX_HORIZON_DAYS = 365

# The audience the model's own base is measured on, and the one the market
# settles on. Kept apart on every payload, on purpose.
MODEL_AUDIENCE_HE = "כלל הצופים"
MODEL_AUDIENCE_EN = "all viewers"
TRADE_CURRENCY_HE = "בתי אב יהודיים, רייטינג רבעי שעה, overnight +1"
TRADE_CURRENCY_EN = "Jewish households, quarter-hour rating, overnight +1"

# The three states an audience request can be in.
AUDIENCE_SERVED = "served"
AUDIENCE_TRADE_CURRENCY = "trade_currency"
AUDIENCE_UNMEASURED = "unmeasured"

# Wordings that name the model's own base. An empty request is also served: it
# asks for the forecast's own basis, which the basis block then states.
_MODEL_BASE_MARKERS = ("כלל הצופים", "כלל צופים", "all viewers", "total viewers")

# Wordings that name the settlement currency. Checked FIRST, so a request
# naming both bases takes the stricter refusal.
_TRADE_MARKERS = (
    "בתי אב", "household", "יהודי", "jewish",
    "רבעי שעה", "רבע שעה", "quarter-hour", "quarter hour", "overnight",
)

LEVEL_LABELS_HE = {
    "series": "סדרה",
    "genre": "ז'אנר",
    "slot": "רצועה",
    "channel": "ערוץ",
    "global": "ממוצע כללי",
}

FAMILY_LABELS_HE = {
    "weekday_slot": "יום בשבוע ורצועה",
    "series": "סדרה",
    "competitor_lineup": "לוח המתחרים מול השידור",
    "season": "עונה",
    "operator_events": "אירועי המפעיל",
    "calendar_hanukkah": "חנוכה",
    "calendar_school_and_chol_hamoed": "חופשת בית ספר וחול המועד",
    "calendar_religious_blackout": "שבת ומועד",
}


def unavailable(reason_he: str, reason_en: str, **extra: Any) -> dict[str, Any]:
    """The frozen refusal shape every surface here returns."""
    return {"available": False, "reason_he": reason_he, "reason_en": reason_en, **extra}


# ---------------------------------------------------------------- the currency

def classify_audience(audience: str) -> str:
    """Which of the three audience states a request falls in.

    Empty asks for the model's own base and is served. A wording naming the
    settlement currency is :data:`AUDIENCE_TRADE_CURRENCY`; one naming the
    model's base is :data:`AUDIENCE_SERVED`; anything else named is
    :data:`AUDIENCE_UNMEASURED`.
    """
    text = str(audience or "").strip().lower()
    if not text:
        return AUDIENCE_SERVED
    if any(marker in text for marker in _TRADE_MARKERS):
        return AUDIENCE_TRADE_CURRENCY
    if any(marker in text for marker in _MODEL_BASE_MARKERS):
        return AUDIENCE_SERVED
    return AUDIENCE_UNMEASURED


def audience_is_servable(audience: str) -> bool:
    """Whether the all-viewers planned-break-rating base can answer this audience."""
    return classify_audience(audience) == AUDIENCE_SERVED


def audience_basis_block(audience: str = "") -> dict[str, Any]:
    """What the forecast is measured on, and what it is not."""
    return {
        "audience": MODEL_AUDIENCE_HE,
        "audience_en": MODEL_AUDIENCE_EN,
        "measure_he": "רייטינג ברייק מתוכנן: ממוצע הרייטינג הנמדד של התשדירים בשעת השידור של התוכנית",
        "measure_en": "planned break rating: the mean measured spot rating in the programme's clock hour",
        "trade_currency_he": TRADE_CURRENCY_HE,
        "trade_currency_en": TRADE_CURRENCY_EN,
        "serves_trade_currency": False,
        "requested_audience": str(audience or ""),
        "requested_state": classify_audience(audience),
        "note_he": "התחזית אינה נקובה במטבע המסחרי ואינה מדידת בתי אב יהודיים ברבע שעה; אין להציג אותה כהתחייבות מסירה",
    }


def audience_refusal(audience: str) -> Optional[dict[str, Any]]:
    """The refusal for an audience the model cannot serve, or None when it can.

    Two distinct reasons, because they are two distinct mistakes: asking for
    the settlement currency is asking the model to speak money's language, and
    asking for a demographic is asking for a panel cut nothing here measures.
    """
    state = classify_audience(audience)
    if state == AUDIENCE_SERVED:
        return None
    basis = audience_basis_block(audience)
    if state == AUDIENCE_TRADE_CURRENCY:
        return unavailable(
            f"התחזית התבקשה במטבע ההתחשבנות של השוק ({audience}); בסיס המודל הוא "
            f"{MODEL_AUDIENCE_HE} ברייטינג ברייק מתוכנן, והמסחר מסתמך על "
            f"{TRADE_CURRENCY_HE}. אלה שני מטבעות שונים ולא נציג אחד כשני",
            f"forecast requested in the market settlement currency ({audience!r}); "
            f"the model's base is the {MODEL_AUDIENCE_EN} planned break rating while "
            f"the trade settles on {TRADE_CURRENCY_EN}. These are two different "
            "currencies and one is never presented as the other",
            audience_basis=basis,
            audience_state=state,
        )
    return unavailable(
        f"התחזית נקובה בקהל יעד ({audience}) שאין למוצר מדידה עבורו; בסיס המודל הוא "
        f"{MODEL_AUDIENCE_HE} בלבד",
        f"forecast requested for audience {audience!r}, which the product does not "
        f"measure; the model's base is {MODEL_AUDIENCE_EN} only",
        audience_basis=basis,
        audience_state=state,
    )


# ------------------------------------------------------------------ the window

@lru_cache(maxsize=1)
def calendar_span() -> tuple[Optional[date], Optional[date]]:
    """The span the bundled Israeli calendar table actually covers.

    Outside it the holiday features come back silently false rather than
    measured, so a forecast there would be scored on fabricated calendar
    context. The forecast refuses instead.
    """
    try:
        from kairos.data.israel_calendar import load_calendar

        rows = load_calendar()
    except Exception:  # noqa: BLE001 - an unreadable table must not break the surface
        return None, None
    if not rows:
        return None, None
    return min(r.start_date for r in rows), max(r.end_date for r in rows)


def horizon_for(
    day: date, window: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """``(refusal, horizon block)`` for one date. Refusal is None when servable.

    ``window`` is a measured-window block (``from``, ``to``, ``available``); an
    unknown window yields no refusal and says the distance could not be checked.
    """
    first, last = calendar_span()
    if first is not None and not (first <= day <= last):
        return unavailable(
            f"התאריך {day.isoformat()} מחוץ ללוח השנה הישראלי המובנה "
            f"({first.isoformat()} עד {last.isoformat()}); מחוצה לו מאפייני החג "
            "היו נקראים כשקריים ולא כנמדדים",
            f"date {day.isoformat()} falls outside the bundled Israeli calendar "
            f"({first.isoformat()} to {last.isoformat()}); outside it the holiday "
            "features read false rather than measured",
        ), {}
    block: dict[str, Any] = {"measured_window": window, "date": day.isoformat()}
    if not window.get("to"):
        block.update({
            "days_after_window": None, "inside_measured_window": None,
            "note_he": "טווח המדידה אינו ידוע בתהליך הזה; לא ניתן לבדוק את מרחק התחזית ממנו",
        })
        return None, block
    last_measured = date.fromisoformat(str(window["to"])[:10])
    days_after = (day - last_measured).days
    block["days_after_window"] = days_after
    block["inside_measured_window"] = days_after <= 0
    if days_after > MAX_HORIZON_DAYS:
        return unavailable(
            f"התאריך {day.isoformat()} רחוק {days_after} ימים מסוף חלון המדידה "
            f"({last_measured.isoformat()}); למודל אין רכיב מגמה או עונה פעיל, ולכן "
            f"מעבר ל-{MAX_HORIZON_DAYS} ימים אין במדידה דבר שנוגע לתאריך הזה",
            f"date {day.isoformat()} is {days_after} days past the end of the "
            f"measured window ({last_measured.isoformat()}); the fit carries no "
            f"trend or season term, so beyond {MAX_HORIZON_DAYS} days nothing "
            "measured bears on it",
            horizon=block,
        ), block
    if days_after > 0:
        block["note_he"] = (
            "תאריך מעבר לחלון המדידה: הרמה נישאת קדימה ללא רכיב מגמה או עונה"
        )
    else:
        block["note_he"] = (
            "התאריך בתוך חלון המדידה של המודל: המספר הוא התאמה, לא תחזית מחוץ למדגם; "
            "לדיוק מחוץ למדגם ראו את הבקטסט"
        )
    return None, block
