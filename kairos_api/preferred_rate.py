"""The preferred-position percentage, reachable at last.

``kairos.optimize.positions.preferred_position_rate`` was built, tested,
bilingual and had ZERO CALLERS outside its own test. Measured 2026-08-09 and
confirmed by the adversarial re-audit: the number the channel and the agency
audit each other with could not be computed by anyone using the product.

That is a worse defect than a missing feature, because the code says the work is
done. This module is the seam between the counting function and the traffic file
it needs, and a route so a person can reach it.

WHAT IT DOES NOT DO, and each omission is deliberate.

It does not GUESS THE PREFERRED SET. Which positions count as preferred is agreed
per client, and the pricing screen already says in both languages that a guessed
percentage is worse than none, precisely because two parties audit each other
with it. With no configured set this surface answers ``unset`` and computes
nothing. On this tree that is what it will answer, and that is the honest state
rather than a broken one: it makes the operator's missing configuration VISIBLE,
where before the whole capability was invisible.

It does not PICK A METHOD. The trade runs two live counting methods and they
disagree, so every reply carries BOTH, each welded to its own bilingual label.
A surface that showed one number without its method would be the exact thing the
positions module was written to prevent.

It does not INVENT A BREAK SIZE. ``occupied_tokens`` needs the break's spot count
to know whether a spot is also Last, and the traffic file gives it: the size is
counted from the rows of that break. Where a break cannot be identified the
appearance is dropped and the count of dropped rows travels in the reply, because
a denominator that quietly shrank is a lie about coverage.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query

from kairos.optimize.positions import (
    AGENCY_METHOD,
    CHANNEL_METHOD,
    Appearance,
    preferred_position_rate,
)

# The pod owns reading the traffic file and grouping it into breaks. Reading it a
# second way here would be a second answer to "what is one break", which the pod
# module's own docstring calls the question it exists to settle.
from kairos_api import break_api_pod as pod
from kairos_api.break_api_pod_spots import preferred_reading


def _appearances(day: str) -> tuple[dict[str, list[Appearance]], int]:
    """Every campaign's broadcasts on one day, and how many rows were unusable.

    The break identity is the pod's own, so this cannot disagree with the board
    an operator is looking at.
    """
    by_campaign: dict[str, list[Appearance]] = {}
    dropped = 0
    for record in pod.pods_for_day(day):
        spots = record.get("spots") or []
        size = len(spots)
        break_id = str(record.get("pod_id") or "")
        for spot in spots:
            campaign = (spot.get("campaign") or {}).get("value")
            position = (spot.get("position") or {}).get("ordinal")
            kind = (spot.get("position") or {}).get("kind")
            if kind == "last":
                # Last is not an ordinal, and the counting function reads a
                # position number. A break's last spot IS its size-th spot, which
                # is the one number that identifies it without inventing one.
                position = size
            if not campaign or not break_id or not position:
                dropped += 1
                continue
            by_campaign.setdefault(str(campaign), []).append(
                Appearance(break_id=break_id, position=int(position), break_size=size)
            )
    return by_campaign, dropped


def rates_for_day(day: str, campaign: Optional[str] = None) -> dict[str, Any]:
    """Both counting methods, for one campaign or for every campaign on a day."""
    reading = preferred_reading()
    preferred = sorted(reading["codes"]) if reading["codes"] else None
    by_campaign, dropped = _appearances(day)
    wanted = [campaign] if campaign else sorted(by_campaign)
    rows = []
    for name in wanted:
        appearances = by_campaign.get(str(name), [])
        rows.append({
            "campaign": name,
            "broadcasts": len(appearances),
            # BOTH methods, always. One number without its method is what the
            # positions module exists to prevent.
            "agency": preferred_position_rate(appearances, preferred, AGENCY_METHOD).as_dict(),
            "channel": preferred_position_rate(appearances, preferred, CHANNEL_METHOD).as_dict(),
        })
    return {
        "day": day,
        "preferred_set": preferred,
        "preferred_state": reading["state"],
        "preferred_unreadable_reason": reading.get("reason"),
        "campaigns": rows,
        # A denominator that quietly shrank is a lie about coverage, so the rows
        # this could not place travel with the answer instead of vanishing.
        "rows_without_a_campaign_or_position": dropped,
        "channel": pod._channel_note(),
    }


router = APIRouter(tags=["plan-day"])


@router.get("/api/preferred-position-rate")
def preferred_position_rate_route(
    day: str = Query("", description="ISO broadcast date, the first covered day when omitted"),
    campaign: str = Query("", description="one campaign, or every campaign when omitted"),
) -> dict[str, Any]:
    """The percentage, under BOTH counting methods, for a day of the traffic file."""
    days = pod.covered_days()
    chosen = str(day).strip() or (days[0] if days else "")
    if not chosen:
        return {
            "day": None,
            "state": "unavailable",
            "reason": "No traffic file is loaded, so no broadcast can be counted.",
            "reason_he": "לא נטען קובץ טראפיק, ולכן אין שידור שניתן לספור.",
            "covered_days": days,
        }
    answer = rates_for_day(chosen, str(campaign).strip() or None)
    answer["covered_days"] = days
    return answer
