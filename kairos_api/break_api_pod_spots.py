"""One spot inside a pod, the position model the trade actually uses, and the
check between a copy version's own declared length and its booked duration.

Split out of :mod:`kairos_api.break_api_pod` under the 450-line cap. Everything
here turns one row of a traffic file into one addressable spot, and every field it
emits is either something it read or something it names as missing.

**Positions are 1 to 5 plus L, and L is not the fifth ordinal.** The trade note is
explicit about this and about why: one campaign can hold both the first and the
last spot of the same break, which is two positions in one pod, and a product that
numbered Last could not express it. The file encodes Last as 99 and an unrequested
position as 0, so neither is returned as an ordinal.

Which codes count as preferred is agreed per client, so the preferred set travels
with the payload as a stated default rather than as a reading of anybody's
contract.

**A blank field is never an empty string standing in for a value.** An advertiser,
a creative, a house number or a length that the file does not carry comes back as
unknown with the reason in both languages, because a spot whose advertiser was
rendered as nothing looks like a spot with no advertiser sold.
"""

from __future__ import annotations

import re
from typing import Any, Optional

LAST_POSITION_CODE = 99
UNPOSITIONED_CODE = 0
PREFERRED_POSITIONS = ("1", "2", "3", "4", "5", "L")
PREFERRED_BASIS = "The preferred set here is 1 to 5 and Last, which is the trade default. Which positions count as preferred is agreed per client, so this is a default and not a reading of any agreement."
PREFERRED_BASIS_HE = "קבוצת המיקומים המועדפים כאן היא 1 עד 5 ואחרון, שהיא ברירת המחדל בענף. אילו מיקומים נחשבים מועדפים נקבע בהסכם מול כל לקוח, ולכן זו ברירת מחדל ולא קריאה של הסכם כלשהו."

# A run of digits immediately followed by a seconds mark is a length. The three
# marks really present in the shipped file are the double quote (28", 35"), the
# apostrophe (35'), and the word שניות with at most one space before it (10
# שניות). A bare number with no mark is never a length, even when it is the
# whole field, because the 15 in "סרט 15 ימי מכירות" is a count of sale days on
# a 14 s spot, and reading it as a duration would manufacture a false alarm on
# the very first pod this surface reads.
_COPY_LENGTH_MARK = re.compile(r"(\d+)(?:[\"']|\s?שניות)")
NO_COPY_LENGTH = "The copy version names no length."
NO_COPY_LENGTH_HE = "שם הגרסה אינו נוקב באורך."
NO_BOOKED_LENGTH = "The copy version names a length, but this spot declares no booked duration to compare it against."
NO_BOOKED_LENGTH_HE = "שם הגרסה נוקב באורך, אך לתשדיר הזה אין אורך מוצהר להשוואה."
COPY_LENGTH_DISAGREES = "The copy version names a different length than the booked duration."
COPY_LENGTH_DISAGREES_HE = "שם הגרסה נוקב באורך שונה מהאורך המוצהר בהזמנה."


def clock_seconds(text: Any) -> Optional[float]:
    """A clock string as seconds from midnight, or None when it is not one."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if " " in raw:
        raw = raw.split(" ")[-1]
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    while len(parts) < 3:
        parts.append("0")
    try:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except (TypeError, ValueError):
        return None


def clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = int(round(float(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def number(value: Any) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN never stands for a length or a position
        return None
    return parsed


def text(value: Any) -> str:
    raw = str(value if value is not None else "").strip()
    return "" if raw.lower() in {"nan", "none"} else raw


def known(value: Any, missing: str, missing_he: str) -> dict[str, Any]:
    """A field that is either read or named as missing, and never an empty stand-in."""
    read = text(value)
    if read:
        return {"state": "real", "value": read}
    return {"state": "unknown", "value": None, "reason": missing, "reason_he": missing_he}


def duration_of(value: Any) -> dict[str, Any]:
    """One spot's declared length. A blank or a zero is an absence, not a length."""
    seconds = number(value)
    if seconds is None or seconds <= 0:
        return {
            "state": "unknown",
            "seconds": None,
            "reason": "This spot declares no length in the traffic file.",
            "reason_he": "התשדיר הזה אינו מצהיר על אורך בקובץ הטראפיק.",
        }
    return {"state": "real", "seconds": round(seconds, 1)}


def position_of(raw: Any) -> dict[str, Any]:
    """One spot's position, with Last as its own thing rather than an ordinal.

    Last comes back as the code ``L`` with no ordinal, an unrequested position
    comes back with no code at all, and a field the file does not carry comes back
    unknown. None of the three is ever a number.
    """
    value = number(raw)
    if value is None:
        return {
            "state": "unknown",
            "code": None,
            "kind": "unknown",
            "ordinal": None,
            "preferred": None,
            "reason": "This spot carries no position in the traffic file.",
            "reason_he": "לתשדיר הזה אין מיקום בקובץ הטראפיק.",
        }
    code = int(round(value))
    if code == LAST_POSITION_CODE:
        return {"state": "real", "code": "L", "kind": "last", "ordinal": None, "preferred": "L" in PREFERRED_POSITIONS}
    if code == UNPOSITIONED_CODE:
        return {
            "state": "real",
            "code": None,
            "kind": "unpositioned",
            "ordinal": None,
            "preferred": False,
            "reason": "No position was requested for this spot.",
            "reason_he": "לא התבקש מיקום עבור התשדיר הזה.",
        }
    return {
        "state": "real",
        "code": str(code),
        "kind": "ordinal",
        "ordinal": code,
        "preferred": str(code) in PREFERRED_POSITIONS,
    }


def copy_declared_seconds(creative_text: Any) -> Optional[float]:
    """A length declared inside the copy version's own name, or ``None``."""
    match = _COPY_LENGTH_MARK.search(str(creative_text or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def copy_length_check(creative: dict[str, Any], duration: dict[str, Any]) -> dict[str, Any]:
    """The copy version's own declared length against this spot's booked duration.

    JS-7's own done condition: any ad whose booked duration disagrees with its
    copy must be impossible to miss. Three states and never a fourth: the copy
    names a length and it agrees, it names one and it disagrees, or it names
    none at all, which is the honest answer for most rows.
    """
    copy_seconds = copy_declared_seconds(creative.get("value") if isinstance(creative, dict) else None)
    if copy_seconds is None:
        return {"state": "none", "copy_seconds": None, "reason": NO_COPY_LENGTH, "reason_he": NO_COPY_LENGTH_HE}
    booked = duration.get("seconds") if isinstance(duration, dict) else None
    if booked is None:
        return {
            "state": "none",
            "copy_seconds": copy_seconds,
            "reason": NO_BOOKED_LENGTH,
            "reason_he": NO_BOOKED_LENGTH_HE,
        }
    if round(copy_seconds, 1) == round(float(booked), 1):
        return {"state": "agrees", "copy_seconds": copy_seconds, "booked_seconds": round(float(booked), 1)}
    return {
        "state": "disagrees",
        "copy_seconds": copy_seconds,
        "booked_seconds": round(float(booked), 1),
        "difference_seconds": round(abs(copy_seconds - float(booked)), 1),
        "reason": COPY_LENGTH_DISAGREES,
        "reason_he": COPY_LENGTH_DISAGREES_HE,
    }


def spot(index: int, row: Any) -> dict[str, Any]:
    """One row of a traffic file as one addressable spot inside its pod."""
    start = clock_seconds(row.get("spot_time"))
    duration = duration_of(row.get("duration_sec"))
    creative = known(row.get("creative"), "This spot names no creative version in the traffic file.", "התשדיר הזה אינו נוקב בגרסת קריאייטיב בקובץ הטראפיק.")
    end = None if start is None or duration["seconds"] is None else start + duration["seconds"]
    return {
        "spot_key": f"s{index}",
        "start_seconds": None if start is None else round(start, 1),
        "start_clock": clock(start),
        "end_seconds": None if end is None else round(end, 1),
        "end_clock": clock(end),
        "duration": duration,
        "position": position_of(row.get("position_in_break")),
        "advertiser": known(row.get("advertiser"), "This spot names no advertiser in the traffic file.", "התשדיר הזה אינו נוקב במפרסם בקובץ הטראפיק."),
        "campaign": known(row.get("campaign"), "This spot names no campaign in the traffic file.", "התשדיר הזה אינו נוקב בקמפיין בקובץ הטראפיק."),
        "creative": creative,
        "copy_length": copy_length_check(creative, duration),
        "house_number": known(row.get("house_number"), "This creative carries no house number in the traffic file.", "לקריאייטיב הזה אין מספר בית בקובץ הטראפיק."),
        "agency": known(row.get("agency"), "This spot names no agency in the traffic file.", "התשדיר הזה אינו נוקב במשרד בקובץ הטראפיק."),
        "spot_type": text(row.get("spot_type")),
        "pricing_type": text(row.get("pricing_type")),
        "break_type": text(row.get("break_type")),
    }


def top_and_tail(spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The campaigns holding both the first spot and the Last spot of this pod.

    Two positions in one break, which the trade note names as the case that makes
    counting contested. It is reported as a fact about this pod and no percentage
    is computed from it, because the agency counts out of the breaks a campaign
    appeared in and the channel counts out of total broadcasts, and this surface
    has no mandate to pick one of the two.
    """
    heads = [item for item in spots if item["position"].get("ordinal") == 1]
    tails = [item for item in spots if item["position"].get("kind") == "last"]
    held: list[dict[str, Any]] = []
    for head in heads:
        for tail in tails:
            if head["campaign"]["value"] and head["campaign"]["value"] == tail["campaign"]["value"]:
                held.append({
                    "campaign": head["campaign"]["value"],
                    "advertiser": head["advertiser"]["value"],
                    "positions": ["1", "L"],
                })
    return held
