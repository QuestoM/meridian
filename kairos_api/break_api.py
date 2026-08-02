"""Plan, the day and the break: the board, the live score, and one break's detail.

Eight routes: four reads, and two reversible acts a scheduler performs on a
break, each with its own inverse.

``GET /api/plan/days``            the broadcast days the operator's plan covers
``GET /api/plan/day``             one day as programmes and addressable breaks
``POST /api/plan/day/score``      what a rearrangement is worth, in microseconds
``GET /api/breaks/{break_id}``    one break, everything the plan knows about it
``POST /api/breaks/{break_id}/gold``    mark this break's programme gold
``DELETE /api/breaks/{break_id}/gold``  and take the mark off again
``POST /api/breaks/{break_id}/placement``   record a saved position and its rule
``DELETE /api/breaks/{break_id}/placement`` and forget it again, which is the undo

One thing is deliberately not here.

**Writing the restriction that carries a saved move is not a route in this
module.** The only store the weekly commit path reads for a placement is the
scoped constraint store, so a save has to write a constraint or it never reaches
the plan. That store belongs to the Rules piece and already has a shipped, tested
write route, so the surface calls it and then records the break's own side of the
transaction through ``POST /api/breaks/{break_id}/placement`` above, which writes
only this piece's register. A second implementation of a constraint write would
be a second set of rules.

Measured end to end over HTTP on ``רשת 13 / 2024-11-01``: the save moved exactly
one first break, left the other 44 where the plan put them, and the undo restored
the day's totals byte for byte in 30 ms.

**Delivered money is a state, not a figure.** The plan covers 2024-11-01 to
2024-11-30 and the one daily spot file covers 2025-04-27, so no break in any plan
has a spot ledger behind it. Every break therefore reports delivered as
unavailable with the two dates named and the path to supply a delivery feed. It
will never report a guess.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from kairos_api import break_api_board as board
from kairos_api import break_api_states as states
from kairos_api import break_store, break_store_pins

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan-day"])



class BreakMove(BaseModel):
    """One edit to one break: where it starts, how long it runs, whether it is gold."""

    break_id: str
    offset_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    is_gold: Optional[bool] = None


class ScoreRequest(BaseModel):
    """A day plus the edits to score against it. No edits scores the day as saved."""

    day: str
    moves: list[BreakMove] = []


class PlacementRecord(BaseModel):
    """The break's own record of a placement the operator saved as a constraint."""

    constraint_id: str
    offset_seconds: float
    duration_seconds: float
    is_gold: bool = False
    note: str = ""


def _day_plan_or_404(day: str) -> break_store.DayPlan:
    try:
        return break_store.day_plan(day)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - a data or environment fault is a state
        logger.exception("day plan build failed for %s", day)
        raise HTTPException(status_code=503, detail=f"Could not build this day: {exc}") from None


def _actor(request: "Request | None") -> str:
    """Who is saving, from the one implementation of the identity question."""
    from kairos_api.affiliation_wall import session_for

    session = session_for(request) or {}
    return str(session.get("username", "") or "")


@router.get("/api/plan/days")
def plan_days() -> dict[str, Any]:
    """The broadcast days the operator's own plan covers, ISO-keyed and ascending.

    Presentation orders the week Sunday first; the data stays ISO so nothing
    downstream has to know which day a week starts on.
    """
    channel = break_store.operator_channel()
    days = break_store.plan_days() if channel else []
    return {
        "operator_channel": channel or None,
        "days": days,
        "count": len(days),
        "available": bool(days),
        "reason": "" if days else (states.NO_CHANNEL if not channel else states.NO_PLAN),
        "reason_he": "" if days else (states.NO_CHANNEL_HE if not channel else states.NO_PLAN_HE),
    }


@router.get("/api/plan/day")
def plan_day(day: str = Query("", description="ISO broadcast date, the first planned day when omitted")) -> dict[str, Any]:
    """One broadcast day of the operator's own channel, as a board of breaks.

    Programmes carry their window and their planned break count; breaks carry
    their own clock position, length, ordinal, gold state and the revenue the
    optimizer credited to them. Every figure names the basis it was computed on.

    Omitting the day opens the first day the plan covers, so the route always
    names a real day rather than refusing on a missing parameter.

    With no channel configured and no plan there is no day, and that is a state
    rather than a fault: the route answers 200 with ``available: false``, the
    reason in both languages and empty collections, exactly as
    ``/api/plan/days``, ``/api/schedule/segments`` and ``/api/gold-breaks``
    already do. A named day that carries no segments is a genuine 404, because
    then the caller asked for something that does not exist.

    ``unbound_placements`` carries the saved placements this day holds that no
    longer name a break the plan carries, because a save can change how many
    breaks a programme gets and the ids after it renumber. Without them the money
    such a save spent would have no route back from this surface: see
    :func:`kairos_api.break_api_states.unbound_placements` for the measurement.
    """
    wanted = str(day or "").strip()
    if not wanted:
        available = break_store.plan_days()
        if not available:
            return states.no_day_payload(break_store.operator_channel())
        wanted = available[0]
    if not break_store.operator_channel():
        return states.no_day_payload(break_store.operator_channel())
    plan = _day_plan_or_404(wanted)
    counts, pins = break_store.arrangement(plan)
    items = board._breaks_for_guardrails(plan, pins)
    from kairos.optimize.evaluate import score

    evaluation = score(plan.basis, counts, revenue_weight=plan.revenue_weight, placements=pins)
    saved = break_store_pins.for_day(plan.day)
    breaks = break_store.break_records(plan)
    for record in breaks:
        record["saved_placement"] = saved.get(record["break_id"])
        record["placement_source"] = "operator" if record["break_id"] in saved else "plan"
        record["delivered"] = states.delivered_state(plan.day)
    return {
        "available": True,
        "basis": board.basis(plan),
        "operator_channel": plan.channel,
        "day": plan.day,
        "programmes": [_programme_record(plan, segment, counts) for segment in plan.segments],
        "breaks": breaks,
        "unbound_placements": states.unbound_placements(plan, saved, breaks),
        "totals": board.totals(plan, evaluation, items),
        "compliance": board.compliance(items, plan.guardrails),
        "hours": board.hour_load(items, plan.guardrails),
        "restrictions": states.restrictions_for(plan),
        "gold": {
            "enabled": _gold_enabled(),
            "max_per_day": int(plan.guardrails.gold_breaks_max_per_day),
            "count": sum(1 for item in items if item.is_gold),
        },
        "guardrails": {
            "max_ad_seconds_per_hour": float(plan.guardrails.max_ad_seconds_per_hour),
            "max_breaks_per_hour": int(plan.guardrails.max_breaks_per_hour),
            "min_break_spacing_seconds": float(plan.guardrails.min_break_spacing_seconds),
            "max_daily_ad_seconds": float(plan.guardrails.max_daily_ad_seconds),
        },
    }


def _programme_record(plan: break_store.DayPlan, segment: Any, counts: dict[str, int]) -> dict[str, Any]:
    start = float(segment.start_seconds)
    return {
        "segment_id": segment.segment_id,
        "title": segment.program_title,
        "genre": segment.program_type,
        "channel": segment.channel,
        "day": segment.day,
        "start_seconds": round(start, 1),
        "end_seconds": round(start + float(segment.duration_seconds), 1),
        "duration_seconds": round(float(segment.duration_seconds), 1),
        "breaks": int(counts.get(segment.segment_id, 0)),
        "max_breaks": int(segment.max_breaks),
        "baseline_rating": round(float(segment.baseline_tvr), 3),
        "break_length_seconds": round(float(segment.break_length_seconds), 1),
    }


@router.post("/api/plan/day/score")
def score_day(payload: ScoreRequest) -> dict[str, Any]:
    """What this arrangement is worth, and whether it is still compliant.

    The cheap call. It scores against a basis built once when the day opened, so
    a person dragging a break gets the plan's own revenue, retention and
    objective rather than an approximation, and gets them inside a frame.
    """
    plan = _day_plan_or_404(payload.day)
    moves = [move.model_dump() for move in payload.moves]
    try:
        return board.score_arrangement(plan, moves)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.post("/api/plan/day/save-effect")
def save_effect(payload: ScoreRequest) -> dict[str, Any]:
    """What saving these moves would do to the day, measured before anything is written.

    The expensive call, and the honest one. The cheap score holds the break counts
    the plan already chose, so it answers under the hand during a drag and cannot
    see what a save does: writing a restriction makes the engine plan the whole day
    again with it in force. Measured on ``רשת 13 / 2024-11-01``, pinning a break at
    exactly its own offset and duration costs 30,575.55 ILS against a cheap
    prediction of 0.00.

    So this runs that second plan with nothing written and reports the engine's own
    figures for it. One optimization, about a second, which is why it is an act a
    person takes rather than something a drag triggers.
    """
    plan = _day_plan_or_404(payload.day)
    try:
        return board.save_effect(plan, [move.model_dump() for move in payload.moves])
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _gold_enabled() -> bool:
    from kairos_api.core import _load_settings

    settings = _load_settings()
    return bool(settings.sponsorships_enabled and settings.gold_breaks_enabled)


@router.get("/api/breaks/{break_id}")
def break_detail(break_id: str) -> dict[str, Any]:
    """Everything the plan knows about one break, and nothing it does not.

    The competitor boundary holds by construction: a day plan is only ever built
    for the operator's own channel, so a break id naming any other channel does
    not resolve to a day and answers 404.
    """
    from kairos_api.break_api_detail import build_detail

    try:
        segment_id, ordinal = break_store.parse_break_id(break_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    day = segment_id.split("|")[0].strip()
    plan = _day_plan_or_404(day)
    detail = build_detail(plan, segment_id, ordinal)
    if detail is None:
        raise HTTPException(status_code=404, detail="No such break in the saved plan for that day")
    return detail


def _gold_after_write(day: str, segment_id: str, override_id: str) -> dict[str, Any]:
    """How many breaks in this programme the plan ACTUALLY carries as gold now.

    Measured after the write, on the plan the board is about to show, because the
    only honest answer to how many breaks a mark reached is the one the engine
    produced. When the plan comes back with none, the engine's own reason for
    refusing the override is reported verbatim, the same way the override preview
    reports a rejected override rather than paraphrasing it.
    """
    from kairos_api import overrides as override_api

    rebuilt = _day_plan_or_404(day)
    marked = sum(
        1 for record in break_store.break_records(rebuilt)
        if record["segment_id"] == segment_id and record["is_gold"]
    )
    if marked:
        return {"breaks_marked": marked, "bound": True, "reason": "", "reason_he": ""}
    _active, stale = override_api._resolved_store_overrides(list(rebuilt.segments))
    refused = next((row for row in stale if row.get("override_id") == override_id), None)
    if refused is not None:
        return {
            "breaks_marked": 0,
            "bound": False,
            "reason": str(refused.get("reason", "")),
            "reason_he": states.GOLD_REFUSED_HE,
        }
    return {
        "breaks_marked": 0,
        "bound": False,
        "reason": states.GOLD_UNMARKED,
        "reason_he": states.GOLD_UNMARKED_HE,
    }


@router.post("/api/breaks/{break_id}/gold", status_code=201)
def mark_gold(break_id: str, request: Request = None) -> dict[str, Any]:
    """Mark this break's programme gold, through the operator override store.

    Stated plainly because it is not what a reader would assume: the engine
    carries gold on the programme segment, not on one break inside it, so this
    marks every break in the programme. The response says how many that is.

    Two things here are the result of a measurement rather than a preference.

    The anchor carries all three fields. The engine's re-ingest guard compares a
    stored override's ``(date, start clock, programme)`` against the segment it
    names, and reads a blank field as a mismatch. Measured on
    ``רשת 13 / 2024-11-01`` before this route filled the clock: the override went
    in, the guard reported it stale, active overrides were 0, every one of the 80
    breaks came back ``is_gold: false``, the day's revenue moved by exactly 0.00,
    and the route still answered ``breaks_marked: 4``.

    And the count is read from the plan as it stands after the write rather than
    from the plan that was on screen before it, so it is a number the engine
    produced. When the mark reaches nothing the response says so with the engine's
    own reason instead of reporting a success the plan does not show.
    """
    from kairos_api import overrides as override_api

    try:
        segment_id, ordinal = break_store.parse_break_id(break_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if not _gold_enabled():
        raise HTTPException(status_code=409, detail="Gold breaks are switched off in settings")
    day = segment_id.split("|")[0].strip()
    plan = _day_plan_or_404(day)
    segment = plan.segment(segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail="No such segment in the saved plan for that day")
    payload = override_api.OverrideCreate(
        scope="segment",
        target_id=segment_id,
        kind="gold",
        gold=True,
        notes=f"gold from the day board, break {ordinal}",
        anchor_date=str(segment.day),
        anchor_start=override_api.segment_clock(float(segment.start_seconds)),
        anchor_title=str(segment.program_type),
    )
    record = override_api.create_override(payload, request)
    break_store.invalidate()
    measured = _gold_after_write(day, segment_id, str(record.get("override_id", "")))
    return {
        "break_id": break_id,
        "segment_id": segment_id,
        "override": record,
        "breaks_marked": measured["breaks_marked"],
        "bound": measured["bound"],
        "reason": measured["reason"],
        "reason_he": measured["reason_he"],
        "scope": "programme",
    }


@router.delete("/api/breaks/{break_id}/gold")
def clear_gold(break_id: str, request: Request = None) -> dict[str, Any]:
    """Remove every gold override on this break's programme."""
    from kairos_api import overrides as override_api

    try:
        segment_id, _ordinal = break_store.parse_break_id(break_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    removed: list[str] = []
    for record in override_api.list_overrides()["overrides"].get("segment", []):
        if record.get("target_id") == segment_id and record.get("kind") == "gold":
            override_api.delete_override(record["override_id"], request)
            removed.append(record["override_id"])
    if not removed:
        raise HTTPException(status_code=404, detail="No gold override on this programme")
    break_store.invalidate()
    return {"break_id": break_id, "segment_id": segment_id, "removed": removed}


@router.post("/api/breaks/{break_id}/placement", status_code=201)
def record_placement(break_id: str, payload: PlacementRecord, request: Request = None) -> dict[str, Any]:
    """Record that this break's position was saved, and which constraint carries it.

    The constraint itself is written by the Rules store. This is the break's own
    half of that transaction, and it is what makes an undo exact: it names the
    constraint to delete rather than leaving a surface to guess.
    """
    try:
        segment_id, ordinal = break_store.parse_break_id(break_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    day = segment_id.split("|")[0].strip()
    plan = _day_plan_or_404(day)
    segment = plan.segment(segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail="No such segment in the saved plan for that day")
    record = break_store_pins.save({
        "break_id": break_id,
        "segment_id": segment_id,
        "ordinal": ordinal,
        "channel": segment.channel,
        "day": segment.day,
        "programme": segment.program_title,
        "offset_seconds": round(float(payload.offset_seconds), 1),
        "duration_seconds": round(float(payload.duration_seconds), 1),
        "is_gold": bool(payload.is_gold),
        "constraint_id": payload.constraint_id,
        "actor": _actor(request),
        "note": payload.note,
    })
    break_store.invalidate()
    return record


@router.delete("/api/breaks/{break_id}/placement")
def forget_placement(break_id: str) -> dict[str, Any]:
    """Drop this break's saved-placement record. The undo half of the save."""
    dropped = break_store_pins.forget(break_id)
    if dropped is None:
        raise HTTPException(status_code=404, detail="This break carries no saved placement")
    break_store.invalidate()
    return {"forgotten": dropped}
