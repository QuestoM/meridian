"""One evaluator per obliging term kind: target, standing, pace, alarm.

Split out of :mod:`kairos.trade.obligations`, which keeps the Obligation record,
the scope resolution and the shared pace/alarm machinery. The seam is real: that
module answers "which rows does this commitment measure over, and how far through
its window is it", and this one answers "what does THIS KIND of commitment count,
and against what". A new committed term kind is a new function here and a line in
:data:`EVALUATORS`, with nothing above it to touch.

The honesty rules are inherited unchanged. Counted figures are floors: a day with
no per-spot source is unknown, never zero. A guarantee in a currency the product
does not hold reports UNKNOWN with the reason rather than comparing two different
things. A term kind with no continuous measurement behind it yet says so by name
in :data:`UNTRACKED_NOTES` instead of reporting a silent zero.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from kairos.trade.obligations import (
    AT_RISK,
    DEFAULT_RISK_RATIO,
    NO_DELIVERY_REASON,
    ON_TRACK,
    UNKNOWN,
    WATCH,
    Inputs,
    Obligation,
    _delivery_slice,
    _floor_sums,
    _pace_block,
    campaigns_for,
    no_basis_pace,
)

# The audience wording the planned-break-rating basis can honestly serve.
_ALL_VIEWERS_MARKERS = ("כלל", "all viewers", "בתי אב", "households")


def _eval_budget(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    amount = ob.params.get("amount", {})
    target = float(amount.get("amount") or 0) or None
    w_from, w_to = ob.window_dates(agreement_window)
    if sliced.empty:
        # No ledger rows resolve for this side in this window. That is an
        # absence of measurement, not a measured zero: alarming "materially
        # behind pace" off nothing on file would be an invented number.
        counted = None
        pace = no_basis_pace(w_from, w_to, inputs.today, NO_DELIVERY_REASON)
    else:
        counted = sums["aired_spend"]
        pace = _pace_block(counted, sums["scheduled_spend"], target, w_from, w_to,
                           inputs.today, _tolerance(ob))
    return {
        "target": {"value": target, "unit": "ILS",
                   "basis": str(amount.get("basis", "unstated"))},
        "standing": {
            "counted": counted, "unit": "ILS",
            "scheduled_ahead": sums["scheduled_spend"],
            "floor_note": "רצפת מדידה: ימים ללא מקור אינם נספרים כאפס",
            "unknown_days": sums["unknown_days"],
            "basis": "מתומחר מנוע מתוך ספר התשדירים; אינו חשבונית",
        },
        "resolution": resolution,
        **pace,
    }


def _audience_is_measurable(audience: str) -> bool:
    text = str(audience or "").strip().lower()
    if not text:
        return True
    return any(marker in text for marker in _ALL_VIEWERS_MARKERS)


def _eval_trp(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    audience = str(ob.params.get("audience", ""))
    points = ob.params.get("points")
    target = float(points) if points is not None else None
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    if not _audience_is_measurable(audience):
        return {
            "target": {"value": target, "unit": "rating_points", "audience": audience},
            "standing": {
                "counted": None, "unit": "rating_points",
                "basis": "רייטינג מתוכנן על בסיס כלל הצופים בלבד",
            },
            "resolution": resolution,
            "alarm": UNKNOWN,
            "alarm_reason": (
                "ההתחייבות נקובה בקהל יעד שאין למוצר מדידה עבורו; השוואה בין "
                "שני מטבעות שונים לא תיעשה"
            ),
            "used_default_bands": False,
        }
    w_from, w_to = ob.window_dates(agreement_window)
    if sliced.empty:
        # Same law as the budget floor: no ledger rows is no measurement.
        counted = None
        pace = no_basis_pace(w_from, w_to, inputs.today, NO_DELIVERY_REASON)
        return {
            "target": {"value": target, "unit": "rating_points", "audience": audience,
                       "incomplete": target is None},
            "standing": {
                "counted": counted, "unit": "rating_points",
                "scheduled_ahead": sums["scheduled_points"],
                "floor_note": "רצפת מדידה: ימים ללא מקור אינם נספרים כאפס",
                "unknown_days": sums["unknown_days"],
                "basis": "רייטינג ברייק מתוכנן מיומן השידור, בסיס כלל הצופים",
            },
            "resolution": resolution,
            **pace,
        }
    counted = sums["aired_points"]
    forecast_forward = None
    if inputs.forecast_points is not None:
        try:
            forecast_forward = inputs.forecast_points(
                counted=counted, scheduled=sliced[sliced["air_state"] == "scheduled"],
            )
        except Exception as exc:  # noqa: BLE001 - a broken forecast must not break a standing
            forecast_forward = {"available": False, "reason_en": str(exc)}
    pace = _pace_block(counted, sums["scheduled_points"], target, w_from, w_to,
                       inputs.today, _tolerance(ob), forecast_forward)
    return {
        "target": {"value": target, "unit": "rating_points", "audience": audience,
                   "incomplete": target is None},
        "standing": {
            "counted": counted, "unit": "rating_points",
            "scheduled_ahead": sums["scheduled_points"],
            "floor_note": "רצפת מדידה: ימים ללא מקור אינם נספרים כאפס",
            "unknown_days": sums["unknown_days"],
            "basis": "רייטינג ברייק מתוכנן מיומן השידור, בסיס כלל הצופים",
        },
        "resolution": resolution,
        **pace,
    }


def _eval_effective_cpp(ob: Obligation, inputs: Inputs, agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    sliced, resolution = _delivery_slice(ob, inputs, agreement_window)
    sums = _floor_sums(sliced)
    cap = ob.params.get("cap")
    cap_value = float(cap) if cap is not None else None
    audience = str(ob.params.get("audience", ""))
    if not _audience_is_measurable(audience):
        return {
            "target": {"value": cap_value, "unit": "ILS_per_point", "audience": audience},
            "standing": {"counted": None, "unit": "ILS_per_point"},
            "resolution": resolution,
            "alarm": UNKNOWN,
            "alarm_reason": "התקרה נקובה בקהל יעד שאין למוצר מדידה עבורו",
            "used_default_bands": False,
        }
    points = sums["aired_points"]
    spend = sums["aired_spend"]
    effective = round(spend / points, 2) if points > 0 else None
    if effective is None or cap_value is None:
        level, reason = UNKNOWN, "אין עדיין נקודות מדודות לחישוב CPP אפקטיבי"
    elif effective <= cap_value:
        level, reason = ON_TRACK, "ה-CPP האפקטיבי בתוך התקרה"
    elif effective <= cap_value * 1.05:
        level, reason = WATCH, "ה-CPP האפקטיבי צמוד לתקרה"
    else:
        level, reason = AT_RISK, "ה-CPP האפקטיבי חורג מהתקרה"
    return {
        "target": {"value": cap_value, "unit": "ILS_per_point", "audience": audience},
        "standing": {
            "counted": effective, "unit": "ILS_per_point",
            "spend_floor": spend, "points_floor": points,
            "unknown_days": sums["unknown_days"],
            "basis": "הוצאה מתומחרת-מנוע חלקי נקודות מתוכננות; רצפת מדידה",
        },
        "resolution": resolution,
        "alarm": level,
        "alarm_reason": reason,
        "used_default_bands": False,
    }


def _eval_preferred_positions(ob: Obligation, inputs: Inputs,
                              agreement_window: Mapping[str, Any]) -> dict[str, Any]:
    target = ob.params.get("target_percent")
    method = str(ob.params.get("counting_method", "unstated"))
    positions = [str(p) for p in ob.params.get("preferred_positions", [])]
    if inputs.preferred_rate is None:
        return {
            "target": {"value": float(target) if target is not None else None,
                       "unit": "percent", "method": method, "positions": positions},
            "standing": {"counted": None, "unit": "percent"},
            "alarm": UNKNOWN,
            "alarm_reason": "מדד המיקומים המועדפים אינו זמין בהקשר זה",
            "used_default_bands": False,
        }
    campaign_ids, how = campaigns_for(ob, inputs)
    measured = inputs.preferred_rate(
        positions=positions, method=method, campaigns=campaign_ids,
    )
    counted = measured.get("rate_percent")
    if counted is None or target is None:
        level, reason = UNKNOWN, str(measured.get("reason") or "אין מדידה זמינה")
    elif float(counted) >= float(target):
        level, reason = ON_TRACK, "שיעור המיקומים המועדפים עומד ביעד"
    elif float(counted) >= float(target) * DEFAULT_RISK_RATIO:
        level, reason = WATCH, "שיעור המיקומים המועדפים מעט מתחת ליעד"
    else:
        level, reason = AT_RISK, "שיעור המיקומים המועדפים מתחת ליעד מהותית"
    return {
        "target": {"value": float(target) if target is not None else None,
                   "unit": "percent", "method": method, "positions": positions},
        "standing": {"counted": counted, "unit": "percent",
                     "method": method, "detail": measured,
                     "basis": "נמדד בשיטה הנקובה בהסכם; השיטה חלק מהמספר"},
        "resolution": {"campaigns": campaign_ids, "resolved": how},
        "alarm": level,
        "alarm_reason": reason,
        "used_default_bands": target is not None and counted is not None,
    }


def eval_untracked(ob: Obligation, note_he: str) -> dict[str, Any]:
    return {
        "target": {"value": None, "unit": None},
        "standing": {"counted": None, "unit": None},
        "alarm": UNKNOWN,
        "alarm_reason": note_he,
        "used_default_bands": False,
    }


def _tolerance(ob: Obligation) -> Optional[float]:
    raw = ob.params.get("tolerance_percent")
    return float(raw) if raw is not None else None


EVALUATORS: dict[str, Callable[[Obligation, Inputs, Mapping[str, Any]], dict[str, Any]]] = {
    "budget-commitment": _eval_budget,
    "trp-delivery-guarantee": _eval_trp,
    "effective-cpp-cap": _eval_effective_cpp,
    "preferred-position-guarantee": _eval_preferred_positions,
}

UNTRACKED_NOTES: dict[str, str] = {
    "share-commitment": "נתח דורש מכנה חיצוני (הצהרת תקציב כוללת) שטרם נמסר; לא ננחש",
    "daypart-mix": "תמהיל רצועות נמדד עם חיבור ספר התשדירים המפולח; בהמתנה",
    "length-mix": "תמהיל אורכים נמדד עם חיבור ספר התשדירים המפולח; בהמתנה",
    "flighting-obligation": "רציפות נמדדת על לוח הפעילות; בהמתנה לחיבור",
    "makegood-accrual-policy": "מדיניות הצבירה מופעלת דרך ספר הצבירות; ראו יתרות שם",
    "added-value-media": "הענקת מדיה נוספת נרשמת בספר הצבירות; ראו יתרות שם",
    "gold-break-allocation": "מעקב הקצאת ברייקי זהב יחובר ללוח השיבוץ",
}
