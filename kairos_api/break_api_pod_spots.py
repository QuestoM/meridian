"""One spot inside a pod, and the position model the trade actually uses.

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

from typing import Any, Optional

LAST_POSITION_CODE = 99
UNPOSITIONED_CODE = 0
PREFERRED_POSITIONS = ("1", "2", "3", "4", "5", "L")
PREFERRED_BASIS = "The preferred set here is 1 to 5 and Last, which is the trade default. Which positions count as preferred is agreed per client, so this is a default and not a reading of any agreement."
PREFERRED_BASIS_HE = "קבוצת המיקומים המועדפים כאן היא 1 עד 5 ואחרון, שהיא ברירת המחדל בענף. אילו מיקומים נחשבים מועדפים נקבע בהסכם מול כל לקוח, ולכן זו ברירת מחדל ולא קריאה של הסכם כלשהו."


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


def spot(index: int, row: Any) -> dict[str, Any]:
    """One row of a traffic file as one addressable spot inside its pod."""
    start = clock_seconds(row.get("spot_time"))
    duration = duration_of(row.get("duration_sec"))
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
        "creative": known(row.get("creative"), "This spot names no creative version in the traffic file.", "התשדיר הזה אינו נוקב בגרסת קריאייטיב בקובץ הטראפיק."),
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
