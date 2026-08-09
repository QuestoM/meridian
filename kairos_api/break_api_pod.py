"""The pod: the ordered list of spots inside one break, and its four routes.

P3 built the break as an entity and left its contents as an honest state, because
the plan carries no object below the break. This module fills that state with the
one source that genuinely names a break per advertisement: the daily traffic file
under ``data/daily_input``, whose ``break_start`` column is the same value for
every spot in one break. That is the explicit per-ad break identifier P3's
``path_forward`` asked for, and it is already on disk.

**What a pod is.** One break start time on one broadcast day, and the spots that
declare it. Each spot carries an advertiser, a campaign, a creative, a house
number, a declared duration and a position. Measured on the shipped file, break
``2025-04-27 20:40:09``: 28 spots, 569 s of declared duration, positions 1 to 18
with one Last and nine unpositioned sponsorship billboards at the head and tail.

**The arithmetic, which is the point of the surface**, is in
:mod:`kairos_api.break_api_pod_math`, and the spot and its position model are in
:mod:`kairos_api.break_api_pod_spots`. Both were split out under the 450-line cap
and both are coherent on their own: one subtracts, one shapes, and this module
reads the files and answers the routes.

**Why the pod is not joined to a planned break by time.** Measured on
``רשת 13 / 2024-11-01``, the plan's 80 breaks and the spot ledger's 72 pods sit a
median 156 s apart and only 21 of 72 fall within 60 s of each other. So a join on
proximity would attach real spots to a break that did not carry them. The pod is
therefore addressed by its own declared start, and a planned break claims a pod
only when that pod starts inside the break's own window.

**The competitor boundary.** The traffic file carries no channel column at all, so
no rival channel name can reach this surface from it. The channel printed beside a
pod is the operator's own from settings, and it says that is where it came from.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from kairos_api import campaigns_assets_constraints as pair_constraints
from kairos_api import read_cache
from kairos_api.break_api_pod_math import (
    against_declared,
    declared_length,
    pod_arithmetic,
    position_violations,
    verification_errors,
)
from kairos_api.break_api_pod_spots import (
    LAST_POSITION_CODE,
    PREFERRED_BASIS,
    PREFERRED_BASIS_HE,
    PREFERRED_UNSET,
    PREFERRED_UNSET_HE,
    positions_summary,
    preferred_reading,
    UNPOSITIONED_CODE,
    clock_seconds,
    duration_of,
    known,
    position_of,
    spot,
    text,
    top_and_tail,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan-day"])

ROOT = Path(__file__).resolve().parents[1]
DAILY_INPUT_DIR = ROOT / "data" / "daily_input"

CACHE_NAMESPACE = "break_pods"
read_cache.configure(CACHE_NAMESPACE, capacity=8)

POD_ID_SEPARATOR = "~"

NO_COVERAGE = "No traffic file on disk covers this broadcast day, so the breaks on it have no contents to read."
NO_COVERAGE_HE = "אין קובץ טראפיק בדיסק המכסה את יום השידור הזה, ולכן אין תוכן לברייקים שבו."
# Named by the door an operator actually clicks. The shell's own navigation
# entry is Data, and the page behind it is titled Sources, so a path forward
# that said only Sources sent a reader looking for a name the sidebar does not
# carry.
COVERAGE_FORWARD = "Upload the daily traffic file for this day on the Data page, under Sources."
COVERAGE_FORWARD_HE = "העלו את קובץ הטראפיק היומי ליום הזה בעמוד נתונים, תחת מקורות."
NO_BREAK_IN_WINDOW = "A traffic file covers this day, but it declares no break starting inside this one's window."
NO_BREAK_IN_WINDOW_HE = "קובץ טראפיק מכסה את היום הזה, אך אינו מצהיר על ברייק שמתחיל בתוך החלון של הברייק הזה."

# The names a reader of this module reaches for, kept on the parent so a caller
# has one door rather than three. Nothing is reimplemented here.
__all__ = [
    "LAST_POSITION_CODE",
    "PREFERRED_BASIS",
    "PREFERRED_UNSET",
    "UNPOSITIONED_CODE",
    "against_declared",
    "build_pod",
    "contents_state",
    "covered_days",
    "declared_length",
    "pod_arithmetic",
    "pod_id",
    "pods_for_day",
    "parse_pod_id",
    "position_of",
    "router",
]

_duration = duration_of


def pod_id(day: str, break_start: str) -> str:
    """The addressable id of one pod: its broadcast day and its break start clock.

    The separator is a tilde for the reason P3 gives for the break id: a hash in a
    URL is a fragment delimiter and never reaches a server unencoded, while a
    tilde is unreserved in RFC 3986 and survives every hop untouched.
    """
    return f"{str(day).strip()}{POD_ID_SEPARATOR}{str(break_start).strip()}"


def parse_pod_id(value: str) -> tuple[str, str]:
    """Split a pod id back into (day, break start clock), or raise ValueError."""
    raw = str(value or "").strip()
    day, separator, clock = raw.partition(POD_ID_SEPARATOR)
    if not separator or not day.strip() or not clock.strip():
        raise ValueError("a pod id reads <broadcast day>~<break start clock>")
    return day.strip(), clock.strip()


def _fingerprint_of(spots: list[dict[str, Any]]) -> str:
    """A digest of the pod as read, so a saved order knows the file moved under it."""
    parts = [
        f"{item['house_number']['value']}|{item['duration']['seconds']}|{item['advertiser']['value']}"
        for item in spots
    ]
    return hashlib.sha256("~".join(parts).encode("utf-8")).hexdigest()[:16]


def _channel_note() -> dict[str, Any]:
    from kairos_api import break_store

    return {
        "value": break_store.operator_channel() or None,
        "basis": "the operator's own channel from settings. The traffic file carries no channel column, so no other channel can reach this surface from it.",
        "basis_he": "הערוץ של המפעיל מתוך ההגדרות. קובץ הטראפיק אינו כולל עמודת ערוץ, ולכן שום ערוץ אחר אינו יכול להגיע למסך הזה דרכו.",
    }


def _boundary_note() -> dict[str, Any]:
    """What column decides where one pod ends and the next begins, said once.

    Named on the surface rather than only reasoned about here, per
    ``decisions-for-owner.md`` section 2, which records the pod boundary as an
    open owner decision blocking this piece. The file's own break-start column
    is the same value for every spot in one break, which is the explicit
    per-ad break identifier the boundary needs, so it is used as read.
    """
    return {
        "value": "שעת התחלת ברייק",
        "basis": "the traffic file's own break-start column, which carries the same value for every spot in one break",
        "basis_he": "עמודת שעת התחלת הברייק בקובץ הטראפיק עצמו, הנושאת את אותו ערך לכל תשדיר בברייק אחד",
    }


def _read_days() -> dict[str, Any]:
    import pandas as pd

    from kairos.data.loaders import load_daily_input

    frames = []
    for path in sorted(DAILY_INPUT_DIR.glob("*.csv")) if DAILY_INPUT_DIR.exists() else []:
        try:
            frames.append(load_daily_input(path))
        except Exception:  # noqa: BLE001 - one unreadable file is not the answer
            logger.exception("daily traffic file unreadable: %s", path)
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    frame["_day"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    grouped: dict[str, Any] = {}
    for day, rows in frame.dropna(subset=["_day"]).groupby("_day"):
        starts = sorted(rows["break_start"].dropna().unique())
        grouped[str(day)] = [rows[rows["break_start"] == start] for start in starts]
    return grouped


def _traffic_days() -> dict[str, Any]:
    return read_cache.cached(
        CACHE_NAMESPACE,
        key=("traffic-days",),
        fingerprint=read_cache.directory_signatures(DAILY_INPUT_DIR, "*.csv"),
        build=_read_days,
    )


def build_pod(day: str, rows: Any) -> dict[str, Any]:
    """One break's pod: its spots in the order they air, and the arithmetic on them."""
    from kairos_api import break_api_pod_order as order_store

    ordered = rows.sort_values("spot_time")
    # ONE read for the whole pod. It was 60 on a 28-spot pod, each re-reading
    # settings off disk to answer one question, and two could have disagreed.
    preferred = preferred_reading()
    spots = [spot(index, row, preferred["codes"]) for index, (_, row) in enumerate(ordered.iterrows())]
    for sequence, item in enumerate(spots, start=1):
        item["sequence"] = sequence
    start_clock = text(ordered.iloc[0]["break_start"])
    break_start = clock_seconds(start_clock)
    identifier = pod_id(day, start_clock)
    arithmetic = pod_arithmetic(break_start, spots)
    declared = declared_length(day, break_start)
    saved = order_store.applied(identifier, spots, _fingerprint_of(spots))
    violations = position_violations(saved["spots"])
    copy_disagreements = sum(1 for item in spots if item["copy_length"]["state"] == "disagrees")
    # Judged against the SAVED order, so a reordering that breaks a pair is
    # caught by the same check that caught the file breaking it.
    pairs = pair_constraints.pod_pair_block(saved["spots"], start_clock)
    errors = verification_errors(saved["spots"], violations) + pairs["errors"]
    return {
        "pod_id": identifier,
        "day": day,
        "break_start_clock": start_clock,
        "break_start_seconds": None if break_start is None else round(break_start, 1),
        "programme": known(ordered.iloc[0].get("program"), "This break names no programme in the traffic file.", "הברייק הזה אינו נוקב בתוכנית בקובץ הטראפיק."),
        "break_type": text(ordered.iloc[0].get("break_type")),
        "channel": _channel_note(),
        "boundary": _boundary_note(),
        "spots": saved["spots"],
        "order": saved["order"],
        "arithmetic": arithmetic,
        "declared_break_length": declared,
        "against_declared": against_declared(declared, arithmetic),
        "copy_length_disagreements": copy_disagreements,
        "positions": positions_summary(spots, preferred, violations, top_and_tail(spots)),
        # A lead spot and its short closer. NOT ``positions.top_and_tail`` above,
        # the other thing the trade calls Top and Tail: one campaign holding both
        # position 1 and Last. See docs/top-and-tail-design.md.
        "creative_pairs": pairs,
        "verification": {"errors": errors, "count": len(errors)},
        "fingerprint": _fingerprint_of(spots),
    }


def pods_for_day(day: str) -> list[dict[str, Any]]:
    """Every pod the traffic files declare on one broadcast day, in time order."""
    groups = _traffic_days().get(str(day).strip(), [])
    return [build_pod(str(day).strip(), rows) for rows in groups]


def covered_days() -> list[str]:
    """The broadcast days a traffic file on disk actually covers."""
    return sorted(_traffic_days().keys())


def contents_state(day: str, break_start: Optional[float], duration: Optional[float]) -> dict[str, Any]:
    """What a planned break can say about its own contents, read rather than assumed.

    This is the seam P3 left open. A planned break whose day has a traffic file
    reports the pod that starts inside its own window; one whose day has none
    reports the same honest state it always did, except that it now names the days
    that are covered instead of only naming the missing input.
    """
    days = covered_days()
    wanted = str(day).strip()
    if wanted not in days:
        return {
            "state": "unavailable",
            "spots": [],
            "covered_days": days,
            "reason": NO_COVERAGE,
            "reason_he": NO_COVERAGE_HE,
            "path_forward": COVERAGE_FORWARD,
            "path_forward_he": COVERAGE_FORWARD_HE,
        }
    end = None if break_start is None or duration is None else break_start + duration
    for pod in pods_for_day(wanted):
        start = pod["break_start_seconds"]
        if start is None or break_start is None or end is None:
            continue
        if break_start <= start < end:
            # The whole pod rather than a summary of it, so the drawer renders the
            # same component the contents page does. Two shapes of one pod would
            # be two answers to one question on the day they disagreed.
            return {"state": "real", "pod_id": pod["pod_id"], "pod": pod, "covered_days": days, "reason": "", "reason_he": ""}
    return {
        "state": "unavailable",
        "spots": [],
        "covered_days": days,
        "reason": NO_BREAK_IN_WINDOW,
        "reason_he": NO_BREAK_IN_WINDOW_HE,
        "path_forward": COVERAGE_FORWARD,
        "path_forward_he": COVERAGE_FORWARD_HE,
    }


class PodOrder(BaseModel):
    """A reordering of one pod: its spot keys, in the order they should air."""

    spot_keys: list[str]
    note: str = ""


def _actor(request: "Request | None") -> str:
    from kairos_api.affiliation_wall import session_for

    session = session_for(request) or {}
    return str(session.get("username", "") or "")


@router.get("/api/breaks/pods")
def list_pods(day: str = Query("", description="ISO broadcast date, the first covered day when omitted")) -> dict[str, Any]:
    """Every break's pod on one broadcast day, with the arithmetic on each.

    A day with no traffic file behind it is a state of the data rather than a
    fault of the request, so it answers 200 with the reason, the days that are
    covered and an empty list, the way every other honest empty state here does.
    """
    days = covered_days()
    wanted = str(day or "").strip() or (days[0] if days else "")
    if not wanted or wanted not in days:
        return {
            "available": False,
            "day": wanted or None,
            "covered_days": days,
            "pods": [],
            "count": 0,
            "channel": _channel_note(),
            "reason": NO_COVERAGE,
            "reason_he": NO_COVERAGE_HE,
            "path_forward": COVERAGE_FORWARD,
            "path_forward_he": COVERAGE_FORWARD_HE,
        }
    pods = pods_for_day(wanted)
    return {
        "available": True,
        "day": wanted,
        "covered_days": days,
        "pods": pods,
        "count": len(pods),
        "channel": _channel_note(),
        "source": "data/daily_input, the daily traffic files on disk",
        "source_he": "data/daily_input, קובצי הטראפיק היומיים בדיסק",
    }


def _pod_or_404(identifier: str) -> dict[str, Any]:
    try:
        day, clock = parse_pod_id(identifier)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    for pod in pods_for_day(day):
        if pod["break_start_clock"] == clock:
            return pod
    raise HTTPException(status_code=404, detail="No traffic file declares a break at that time on that day")


@router.get("/api/breaks/pod/{pod_id}")
def pod_detail(pod_id: str) -> dict[str, Any]:
    """One pod, everything the traffic file declares about it and nothing else."""
    return _pod_or_404(pod_id)


@router.put("/api/breaks/pod/{pod_id}/order", status_code=200)
def save_order(pod_id: str, payload: PodOrder, request: Request = None) -> dict[str, Any]:
    """Record the order an operator wants this pod's spots to air in.

    The keys must be exactly the pod's own keys, each once. A partial or unknown
    list is refused rather than half applied, because a pod that half took an
    order is a pod nobody can read the arithmetic of.
    """
    from kairos_api import break_api_pod_order as order_store

    pod = _pod_or_404(pod_id)
    if pod["order"].get("locked"):
        raise HTTPException(status_code=423, detail="This pod is locked. Unlock it before changing the order.")
    keys = [item["spot_key"] for item in pod["spots"]]
    if sorted(payload.spot_keys) != sorted(keys):
        raise HTTPException(status_code=422, detail="The order must hold exactly this pod's own spot keys, each once")
    record = order_store.save(pod_id, payload.spot_keys, pod["fingerprint"], _actor(request), payload.note)
    return {"pod_id": pod_id, "saved": record, "pod": _pod_or_404(pod_id)}


@router.delete("/api/breaks/pod/{pod_id}/order")
def forget_order(pod_id: str) -> dict[str, Any]:
    """Drop the saved order and put the pod back in the order the file declares.

    A locked pod refuses this exactly as it refuses a write of a new order.
    Dropping the row is the larger of the two changes, not the smaller: it takes
    the frozen order away and clears the lock with it, so a route that allowed it
    would be a way to unfinalise a pod without ever pressing unlock, and the
    register would lose the fact that anybody had finalised it.
    """
    from kairos_api import break_api_pod_order as order_store

    pod = _pod_or_404(pod_id)
    if pod["order"].get("locked"):
        raise HTTPException(status_code=423, detail="This pod is locked. Unlock it before changing the order.")
    dropped = order_store.forget(pod_id)
    if dropped is None:
        raise HTTPException(status_code=404, detail="This pod carries no saved order")
    return {"forgotten": dropped, "pod": _pod_or_404(pod_id)}


@router.put("/api/breaks/pod/{pod_id}/lock", status_code=200)
def lock_pod(pod_id: str, request: Request = None) -> dict[str, Any]:
    """Finalise this pod: freeze the order it is currently shown in.

    The trade's own step is verification, then finalising. This is that second
    step. Locking always pins whichever order is on screen, file or operator,
    so the frozen order is never ambiguous, and a further order write is
    refused until the pod is unlocked. Nothing here requires the verification
    list to be clear first, because a traffic operator sometimes finalises a
    pod that carries a known, already-handled disagreement. Locking never
    invents an operator order: a pod nobody reordered is frozen as the file's
    own order, and stays reported that way, rather than being recorded as a
    decision an operator made.
    """
    from kairos_api import break_api_pod_order as order_store

    pod = _pod_or_404(pod_id)
    if pod["order"].get("locked"):
        raise HTTPException(status_code=409, detail="This pod is already locked")
    record = order_store.lock(pod_id, pod["fingerprint"], _actor(request))
    return {"pod_id": pod_id, "locked": record, "pod": _pod_or_404(pod_id)}


@router.delete("/api/breaks/pod/{pod_id}/lock")
def unlock_pod(pod_id: str) -> dict[str, Any]:
    """Clear a pod's lock. The order it carries is left exactly as it stood."""
    from kairos_api import break_api_pod_order as order_store

    dropped = order_store.unlock(pod_id)
    if dropped is None:
        raise HTTPException(status_code=404, detail="This pod is not locked")
    return {"unlocked": dropped, "pod": _pod_or_404(pod_id)}
