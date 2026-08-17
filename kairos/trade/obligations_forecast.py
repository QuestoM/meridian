"""The forecast-based forward line on a TRP obligation, and when it refuses.

A delivery guarantee's committed projection is ``counted + scheduled``: the
rating points already measured plus the points the traffic log carries for spots
still ahead. Those booked points are the PLANNED break rating, which is the
historical mean for the programme -- the number this product priced on before the
audience model existed. The forecast's entire claim is that a forward date's
expected rating differs from that mean. So the honest forecast-based projection
is not a second guess at the total; it is the booked points RE-RATED by the ratio
the model expects between the two:

    forecast_forward = counted + scheduled * (sum expected / sum historical)

with both sums taken over the same channel-day's programmes, from the same
forecast payload, so the ratio is a property of one model run rather than two
numbers from different places.

**Why this needs an injected schedule, and refuses without one.** The delivery
ledger is one row per campaign per broadcast day
(:mod:`kairos_api.campaigns_delivery`). It records the channel, the day, the spot
count and the planned points -- and no programme title and no clock time, which
are exactly the two inputs the forecast keys on. The re-rating ratio therefore
cannot come from the ledger alone; a caller must supply the channel-day's
programmes. Given none, this module returns unavailable and names the missing
input. It never spreads a channel-day's points over a guessed programme list,
and it never falls back to the plain historical mean and calls that a forecast.

Every refusal is per DAY as well as overall: a channel-day whose programmes are
unknown, or whose historical sum is too near zero to divide by, is counted and
named while the days that could be re-rated still contribute. A partial answer
that says which part is partial is worth more than a whole answer that is not.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Below this summed historical rating a channel-day's ratio is not formed: the
# denominator is too near zero for the quotient to mean anything.
MIN_HISTORICAL_SUM = 0.05

# Sanity bounds on the re-rating ratio. A ratio outside them is reported, not
# applied: it would mean the model and the traffic log disagree by more than any
# plausible audience movement, which is a data question, not a forecast.
MIN_RATIO = 0.2
MAX_RATIO = 5.0

BASIS_HE = "נקודות מתוזמנות שתומחרו לפי הרייטינג ההיסטורי, מוכפלות ביחס שהמודל מצפה לו בין הרייטינג הצפוי לממוצע ההיסטורי באותם ימי-ערוץ"
BASIS_EN = (
    "the scheduled points, which the traffic log priced at the historical mean, "
    "multiplied by the ratio the audience model expects between the forward "
    "expected rating and that same mean over the same channel-days"
)


def _unavailable(reason_he: str, reason_en: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason_he": reason_he, "reason_en": reason_en, **extra}


def day_ratio(
    payloads: list[dict[str, Any]]
) -> tuple[Optional[float], dict[str, Any]]:
    """The expected-over-historical ratio implied by one channel-day's forecasts.

    Both sums come from the same payload list, so a programme the forecast
    refused is absent from both and cannot tilt the quotient.
    """
    expected = 0.0
    historical = 0.0
    used = 0
    for payload in payloads:
        if not payload.get("available"):
            continue
        history = payload.get("history") or {}
        if history.get("historical_tvr") is None:
            continue
        expected += float(payload["expected_tvr"])
        historical += float(history["historical_tvr"])
        used += 1
    detail = {"n_programmes": used, "expected_sum": round(expected, 4),
              "historical_sum": round(historical, 4)}
    if used == 0:
        return None, {**detail, "reason": "no programme on this channel-day returned a forecast"}
    if historical < MIN_HISTORICAL_SUM:
        return None, {**detail, "reason": (
            f"the channel-day's historical rating sums to {historical:.4f}, below "
            f"{MIN_HISTORICAL_SUM}; the re-rating ratio has no usable denominator"
        )}
    ratio = expected / historical
    if not (MIN_RATIO <= ratio <= MAX_RATIO):
        return None, {**detail, "ratio": round(ratio, 4), "reason": (
            f"the re-rating ratio {ratio:.3f} falls outside the plausible band "
            f"{MIN_RATIO} to {MAX_RATIO}; it is reported and not applied"
        )}
    return ratio, {**detail, "ratio": round(ratio, 4)}


def rerated_points(
    scheduled: pd.DataFrame, *,
    schedule_fn: Optional[Callable[[str, str], pd.DataFrame]],
    forecast_rows_fn: Callable[[pd.DataFrame], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Re-rate a scheduled delivery slice through the forecast, or say why not.

    ``schedule_fn(channel, day)`` returns that channel-day's programmes with
    ``program_title`` and ``start_seconds``; ``forecast_rows_fn`` is the batched
    forecast over prediction rows. Both are injected so this module reads no
    disk and the tests need none.
    """
    if scheduled is None or scheduled.empty:
        return _unavailable(
            "אין שורות מתוזמנות קדימה בהתחייבות הזאת, ולכן אין מה לתמחר מחדש",
            "the obligation carries no scheduled-ahead rows, so there is nothing to re-rate",
        )
    if schedule_fn is None:
        return _unavailable(
            "ספר המסירה מחזיק ערוץ ויום אך לא שם תוכנית ולא שעת שידור, ואלה שני הקלטים שהתחזית נשענת עליהם; בלי לוח שידורים לימים האלה לא ניתן לגזור יחס תמחור מחדש",
            "the delivery ledger carries a channel and a day but no programme title "
            "and no clock time, the two inputs the forecast keys on; without a "
            "schedule for those days no re-rating ratio can be derived",
        )

    points_column = pd.to_numeric(
        scheduled.get("rating_points_planned"), errors="coerce"
    ).fillna(0.0)
    total_rerated = 0.0
    total_unrated = 0.0
    days: list[dict[str, Any]] = []
    for position, (_index, row) in enumerate(scheduled.iterrows()):
        channel = str(row.get("channel", ""))
        day = str(row.get("broadcast_date", ""))[:10]
        booked = float(points_column.iloc[position])
        entry: dict[str, Any] = {"channel_scoped": bool(channel), "date": day,
                                 "booked_points": round(booked, 4)}
        try:
            programmes = schedule_fn(channel, day)
        except Exception as exc:  # noqa: BLE001 - a broken schedule is an honest gap
            logger.warning("obligations forecast: schedule_fn failed for %s: %s", day, exc)
            programmes = None
        if programmes is None or len(programmes) == 0:
            total_unrated += booked
            days.append({**entry, "applied": False,
                         "reason": "no programme list on file for this channel-day"})
            continue
        rows = pd.DataFrame({
            "date": [day] * len(programmes),
            "channel": [channel] * len(programmes),
            "program_title": programmes["program_title"].astype(str).tolist(),
            "start_seconds": pd.to_numeric(
                programmes["start_seconds"], errors="coerce").fillna(0.0).tolist(),
            "duration_seconds": [0.0] * len(programmes),
        })
        ratio, detail = day_ratio(forecast_rows_fn(rows))
        if ratio is None:
            total_unrated += booked
            days.append({**entry, "applied": False, **detail})
            continue
        total_rerated += booked * ratio
        days.append({**entry, "applied": True,
                     "rerated_points": round(booked * ratio, 4), **detail})

    applied = [d for d in days if d.get("applied")]
    if not applied:
        return _unavailable(
            "אף יום-ערוץ מתוזמן לא ניתן לתמחור מחדש דרך התחזית; הפירוט לפי יום מצורף",
            "no scheduled channel-day could be re-rated through the forecast; the "
            "per-day detail says why for each",
            days=days,
        )
    return {
        "available": True,
        "scheduled_points_rerated": round(total_rerated, 2),
        "scheduled_points_not_rerated": round(total_unrated, 2),
        "n_days_rerated": len(applied),
        "n_days_not_rerated": len(days) - len(applied),
        "basis_he": BASIS_HE,
        "basis_en": BASIS_EN,
        "days": days,
    }


def forward_line(
    counted: float, scheduled: pd.DataFrame, *,
    schedule_fn: Optional[Callable[[str, str], pd.DataFrame]],
    forecast_rows_fn: Callable[[pd.DataFrame], list[dict[str, Any]]],
) -> dict[str, Any]:
    """The projection line itself: measured points plus re-rated booked points.

    The shape :data:`kairos.trade.obligations.Inputs.forecast_points` is expected
    to return. Unavailable carries the reason and no number, so the obligations
    payload can omit the line entirely rather than show a blank one.
    """
    block = rerated_points(
        scheduled, schedule_fn=schedule_fn, forecast_rows_fn=forecast_rows_fn,
    )
    if not block.get("available"):
        return block
    value = counted + float(block["scheduled_points_rerated"]) + float(
        block["scheduled_points_not_rerated"]
    )
    return {
        **block,
        "value": round(value, 2),
        "counted_component": round(float(counted), 2),
        "label_he": "תחזית-מודל קדימה",
        "label_en": "forecast forward",
        "note_he": "שורה נוספת לצד המדוד+מתוזמן ולצד קצב-קדימה; אינה מחליפה אותן ואינה התחייבות מסירה",
    }
