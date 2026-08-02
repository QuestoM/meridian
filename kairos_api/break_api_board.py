"""Shaping the day board, and scoring a rearrangement of it in microseconds.

Two jobs, both of them about telling the truth quickly.

**The board.** One operator channel-day as programmes and breaks, each figure
carrying the scope it was computed on, because a figure without its scope is not
a figure. The money is the break's own: the optimizer credits a marginal revenue
to every break it places, and those credits sum back to the day's revenue
exactly, so a break's share of the day is a fact and not a division.

**The score.** A person moving a break wants to know what it cost. The honest
answer has two halves and the second one is the surprising one.

Measured on ``רשת 13 / 2024-11-01``, 82 segments, with the engine's own seam:

* Moving a break within its programme changes revenue by **exactly 0.0**,
  retention by **exactly 0.0** and the objective by **exactly 0.0**. That is
  structural, not a rounding artefact: a break's price is
  ``cpp * rating * duration / unit * premium`` and every one of those is a
  property of the programme it sits in, not of the minute it starts at.
* Changing its length by 60 seconds changes revenue by **5,670.90 ILS**.
* Adding one break to the segment changes revenue by **10,760.41 ILS** and
  retention by **-0.000544**.

So a surface that animates a revenue figure on a horizontal drag is showing a
number that did not move. This module instead reports what genuinely moved: the
hour's ad load, the gap to the neighbouring break and the compliance verdict,
which are all functions of the clock and all of them change. The revenue answer
is still on screen, still real, and stated as unchanged with the reason.

The cost of that answer, measured: :func:`kairos.optimize.evaluate.score` is 105
microseconds and :func:`kairos.optimize.guardrails.evaluate` is 53 microseconds
over the same 82 segments, against 1.13 s to re-optimize the day.
"""

from __future__ import annotations

from typing import Any, Optional

from kairos_api import break_store

# Every guardrail the engine runs on a day, named so a surface can say how many
# checks were run rather than only how many failed.
GUARDRAIL_CHECKS = (
    "retention_floor",
    "breaks_per_hour",
    "hourly_ad_load",
    "break_spacing",
    "daily_ad_load",
    "gold_breaks",
)


def _breaks_for_guardrails(plan: break_store.DayPlan, pins: dict[str, tuple[Any, ...]]) -> list[Any]:
    """The arrangement as the guardrail engine's own Break records.

    ``hour`` is derived exactly as :class:`~kairos.optimize._types.BreakPlacement`
    derives it, so a verdict computed here and a verdict computed by the
    optimizer agree by construction rather than by luck.
    """
    from kairos.optimize.guardrails import Break

    retention = {plan_row.segment_id: plan_row.retention for plan_row in plan.result.segments}
    items: list[Any] = []
    for segment in plan.segments:
        for pin in pins.get(segment.segment_id, ()):  # noqa: B905 - tuple of pins
            start = float(segment.start_seconds) + float(pin.offset_seconds)
            items.append(Break(
                channel=segment.channel,
                day=segment.day,
                hour=int(start // 3600.0),
                start_seconds=start,
                duration_seconds=float(pin.duration_seconds),
                program_type=segment.program_type,
                retention=float(retention.get(segment.segment_id, 1.0)),
                is_gold=bool(pin.is_gold),
            ))
    items.sort(key=lambda item: item.start_seconds)
    return items


def hour_load(items: list[Any], guardrails: Any) -> list[dict[str, Any]]:
    """Ad seconds and break count per clock hour, against the licence limits."""
    buckets: dict[int, dict[str, Any]] = {}
    for item in items:
        bucket = buckets.setdefault(item.hour, {"hour": item.hour, "breaks": 0, "ad_seconds": 0.0})
        bucket["breaks"] += 1
        bucket["ad_seconds"] += float(item.duration_seconds)
    rows = []
    for hour in sorted(buckets):
        bucket = buckets[hour]
        rows.append({
            "hour": hour,
            "breaks": bucket["breaks"],
            "ad_seconds": round(bucket["ad_seconds"], 1),
            "max_breaks": int(guardrails.max_breaks_per_hour),
            "max_ad_seconds": float(guardrails.max_ad_seconds_per_hour),
            "over_breaks": bucket["breaks"] > guardrails.max_breaks_per_hour,
            "over_ad_seconds": bucket["ad_seconds"] > guardrails.max_ad_seconds_per_hour,
        })
    return rows


def _violation(item: Any) -> dict[str, Any]:
    return {
        "code": item.code,
        "scope": item.scope,
        "observed": item.observed,
        "limit": item.limit,
        "detail": item.detail,
    }


def compliance(items: list[Any], guardrails: Any) -> dict[str, Any]:
    """The engine's own verdict on this arrangement, checks run and all."""
    from kairos.optimize.guardrails import evaluate

    violations = evaluate(items, guardrails)
    return {
        "compliant": not violations,
        "checks_run": len(GUARDRAIL_CHECKS),
        "checks": list(GUARDRAIL_CHECKS),
        "violations": [_violation(item) for item in violations],
    }


def spacing_around(items: list[Any], start_seconds: float) -> dict[str, Any]:
    """The gap to the break before and after a position, and the limit on both."""
    before: Optional[float] = None
    after: Optional[float] = None
    for item in items:
        end = item.start_seconds + item.duration_seconds
        if end <= start_seconds:
            gap = start_seconds - end
            before = gap if before is None else min(before, gap)
        elif item.start_seconds >= start_seconds:
            gap = item.start_seconds - start_seconds
            after = gap if after is None else min(after, gap)
    return {
        "gap_before_seconds": None if before is None else round(before, 1),
        "gap_after_seconds": None if after is None else round(after, 1),
    }


def totals(plan: break_store.DayPlan, evaluation: Any, items: list[Any]) -> dict[str, Any]:
    """One arrangement's headline figures, every one from the engine's own basis."""
    return {
        "objective": round(float(evaluation.objective), 6),
        "revenue": round(float(evaluation.revenue), 2),
        "retention": round(float(evaluation.retention), 6),
        "breaks": len(items),
        "ad_seconds": round(sum(float(item.duration_seconds) for item in items), 1),
        "gold_breaks": sum(1 for item in items if item.is_gold),
    }


def basis(plan: break_store.DayPlan) -> dict[str, Any]:
    """What every figure on this board was computed on, printed with the figure."""
    return {
        "channel": plan.channel,
        "day": plan.day,
        "segments": len(plan.segments),
        "revenue_weight": plan.revenue_weight,
        "risk_lambda": float(plan.engine_kwargs["risk_lambda"]),
        "objective_mode": str(plan.engine_kwargs.get("objective_mode", "blend")),
        "source": "weekly plan optimizer, this channel-day",
        "currency": "ILS",
    }


def apply_moves(
    plan: break_store.DayPlan,
    pins: dict[str, tuple[Any, ...]],
    moves: list[dict[str, Any]],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, bool]]:
    """Fold a list of per-break edits into a new pin map, and say what changed.

    A move names one break by id and any of a new offset, a new duration or a new
    gold flag. Unknown break ids raise, because silently dropping an edit is how a
    surface ends up reporting a score for an arrangement nobody asked about.
    """
    from kairos.optimize._types import PlacementPin

    working = {segment_id: list(items) for segment_id, items in pins.items()}
    changed = {"placement": False, "duration": False, "gold": False}
    for move in moves:
        segment_id, ordinal = break_store.parse_break_id(str(move.get("break_id", "")))
        row = working.get(segment_id)
        if row is None or ordinal > len(row):
            raise LookupError(f"no break {move.get('break_id')!r} in this day")
        pin = row[ordinal - 1]
        offset = move.get("offset_seconds")
        duration = move.get("duration_seconds")
        gold = move.get("is_gold")
        new_offset = float(pin.offset_seconds if offset is None else offset)
        new_duration = float(pin.duration_seconds if duration is None else duration)
        new_gold = bool(pin.is_gold if gold is None else gold)
        if abs(new_offset - float(pin.offset_seconds)) > 1e-9:
            changed["placement"] = True
        if abs(new_duration - float(pin.duration_seconds)) > 1e-9:
            changed["duration"] = True
        if new_gold != bool(pin.is_gold):
            changed["gold"] = True
        row[ordinal - 1] = PlacementPin(
            offset_seconds=new_offset, duration_seconds=new_duration, is_gold=new_gold,
        )
    return {segment_id: tuple(items) for segment_id, items in working.items()}, changed


def break_predicate(segment: Any) -> dict[str, Any]:
    """The predicate a save writes, on the frozen contract: date, programme, hour.

    Built here from the plan's own segment rather than taken from the caller, so
    the figure the surface reads before the click and the restriction it writes on
    the click cannot describe two different airings. Channel is never in a
    predicate: the engine scopes every restriction to the operator's own channel.
    """
    return {
        "combinator": "and",
        "conditions": [
            {"field": "date", "operator": "is", "value": str(segment.day)},
            {"field": "programme", "operator": "is", "value": str(segment.program_title)},
            {"field": "hour", "operator": "eq", "value": int(float(segment.start_seconds) // 3600) % 24},
        ],
    }


def candidate_restrictions(plan: break_store.DayPlan, moves: list[dict[str, Any]]) -> list[Any]:
    """The restrictions a save of these moves would write, unwritten.

    Same dataclass the store loads back, so nothing here is a second set of rules:
    it is the row the Rules store would hold, held in memory instead of on disk.
    """
    from kairos.optimize.constraints_store import PlacementConstraint

    _counts, pins = break_store.arrangement(plan)
    rows: list[Any] = []
    for index, move in enumerate(moves, start=1):
        segment_id, ordinal = break_store.parse_break_id(str(move.get("break_id", "")))
        segment = plan.segment(segment_id)
        row = pins.get(segment_id)
        if segment is None or row is None or ordinal > len(row):
            raise LookupError(f"no break {move.get('break_id')!r} in this day")
        pin = row[ordinal - 1]
        offset = move.get("offset_seconds")
        duration = move.get("duration_seconds")
        rows.append(PlacementConstraint(
            constraint_id=f"candidate-{index}",
            scope_type="always",
            effect="fix_offset",
            offset_seconds=round(float(pin.offset_seconds if offset is None else offset)),
            duration_seconds=round(float(pin.duration_seconds if duration is None else duration)),
            order_index=ordinal,
            where=break_predicate(segment),
        ))
    return rows


def _rearrangement(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the day a second run placed differently, counted from the ids."""
    first = {row["break_id"]: row for row in before}
    second = {row["break_id"]: row for row in after}
    touched: set[str] = set()
    moved = added = removed = 0
    for break_id, row in first.items():
        other = second.get(break_id)
        if other is None:
            removed += 1
            touched.add(row["segment_id"])
        elif abs(other["start_seconds"] - row["start_seconds"]) > 0.05 or abs(
            other["duration_seconds"] - row["duration_seconds"]) > 0.05:
            moved += 1
            touched.add(row["segment_id"])
    for break_id, row in second.items():
        if break_id not in first:
            added += 1
            touched.add(row["segment_id"])
    return {
        "moved": moved,
        "added": added,
        "removed": removed,
        "programmes": len(touched),
        "changed": moved + added + removed,
    }


def save_effect(plan: break_store.DayPlan, moves: list[dict[str, Any]]) -> dict[str, Any]:
    """What saving these moves would actually do to the day, by doing it.

    The cheap score beside it prices the arrangement on screen while holding the
    break counts the plan already chose, which is what makes it answer under the
    hand during a drag. A save does something that answer cannot see: it writes a
    restriction, and the engine then plans the whole day again with it in force
    and is free to place the rest of the day differently.

    Measured on ``רשת 13 / 2024-11-01``: pinning a break at exactly the offset,
    duration and gold flag the plan had already given it moves the day from
    1,067,845.55 to 1,037,270.00, a fall of 30,575.55 ILS against a cheap
    prediction of 0.00. Pinning one break one snap unit right can re-plan its
    programme from four breaks to one and cost 47,444.20. So a scheduler was
    clicking Save with nothing on the surface saying that the click re-runs the
    day.

    This runs that second plan, with the restrictions the save would write and
    with nothing written, and reports the engine's own figures for it. It is a
    prediction of the same thing the settlement afterwards measures, so the two
    are comparable by construction: same segments, same overrides, same engine
    arguments, and the totals on both sides come out of the same three functions.
    It costs one optimization, so it is an act a person takes and never something
    a drag triggers.
    """
    import time

    from kairos.optimize.day_core import _optimize_one_day
    from kairos.optimize.evaluate import score
    from kairos_api.overrides import _resolved_store_overrides, _stored_constraints

    counts, pins = break_store.arrangement(plan)
    before_items = _breaks_for_guardrails(plan, pins)
    before_eval = score(plan.basis, counts, revenue_weight=plan.revenue_weight, placements=pins)
    before_totals = totals(plan, before_eval, before_items)
    candidates = candidate_restrictions(plan, moves)
    active, _stale = _resolved_store_overrides(list(plan.segments))
    started = time.perf_counter()
    result = _optimize_one_day(
        list(plan.segments),
        constraints=list(_stored_constraints()) + candidates,
        overrides=active if active.overrides else None,
        **plan.engine_kwargs,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after_plan = break_store.DayPlan(
        channel=plan.channel,
        day=plan.day,
        segments=plan.segments,
        result=result,
        basis=plan.basis,
        engine_kwargs=plan.engine_kwargs,
    )
    after_counts, after_pins = break_store.arrangement(after_plan)
    after_items = _breaks_for_guardrails(after_plan, after_pins)
    after_eval = score(plan.basis, after_counts, revenue_weight=plan.revenue_weight, placements=after_pins)
    after_totals = totals(after_plan, after_eval, after_items)
    return {
        "basis": basis(plan),
        "before": before_totals,
        "after": after_totals,
        "delta": {
            key: round(after_totals[key] - before_totals[key], 6)
            for key in ("objective", "revenue", "retention", "breaks", "ad_seconds", "gold_breaks")
        },
        "rearranged": _rearrangement(break_store.break_records(plan), break_store.break_records(after_plan)),
        "restrictions": [
            {
                "break_id": str(move.get("break_id", "")),
                "offset_seconds": row.offset_seconds,
                "duration_seconds": row.duration_seconds,
                "order_index": row.order_index,
                "where": row.where,
            }
            for move, row in zip(moves, candidates)
        ],
        "compliance": compliance(after_items, plan.guardrails),
        "measured": True,
        "method": "the plan's own optimizer, run again for this day with these restrictions in force and nothing written",
        "engine_ms": round(elapsed_ms, 3),
    }


def score_arrangement(
    plan: break_store.DayPlan,
    moves: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the plan with these edits applied, against the plan as it stands.

    Returns both arrangements, their difference, the compliance verdict on each,
    and a machine statement of which inputs the caller actually changed, so the
    surface can name the reason a revenue figure did or did not move instead of
    animating one that did not.
    """
    import time

    from kairos.optimize.evaluate import score

    counts, pins = break_store.arrangement(plan)
    started = time.perf_counter()
    moved_pins, changed = apply_moves(plan, pins, moves)
    saved_eval = score(
        plan.basis, counts, revenue_weight=plan.revenue_weight, placements=pins,
    )
    current_eval = score(
        plan.basis, counts, revenue_weight=plan.revenue_weight, placements=moved_pins,
    )
    saved_items = _breaks_for_guardrails(plan, pins)
    current_items = _breaks_for_guardrails(plan, moved_pins)
    saved_totals = totals(plan, saved_eval, saved_items)
    current_totals = totals(plan, current_eval, current_items)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "basis": basis(plan),
        "saved": saved_totals,
        "current": current_totals,
        "delta": {
            key: round(current_totals[key] - saved_totals[key], 6)
            for key in ("objective", "revenue", "retention", "breaks", "ad_seconds", "gold_breaks")
        },
        "changed_inputs": changed,
        "revenue_responds_to": ["duration_seconds", "break_count", "programme rating and rate"],
        "revenue_ignores": ["offset_seconds"],
        "compliance": compliance(current_items, plan.guardrails),
        "saved_compliance": compliance(saved_items, plan.guardrails),
        "hours": hour_load(current_items, plan.guardrails),
        "engine_ms": round(elapsed_ms, 3),
    }
