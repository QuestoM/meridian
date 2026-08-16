"""Plan, week, compare: both legs run over the plan's own week, day by day.

Split out of ``scenario_compare_api`` under the 450-line law and named by the
helper rule. It holds the part of the comparison that makes it a comparison of
the week rather than of one day: which dates the plan's own week carries, one
cached optimizer run per leg per day, the accumulation of those days into the
two figures the planner reads, and the route that streams a day the moment it is
decided.

**Why this exists.** JS-2's comparison is on next week, and the panel used to run
both legs on a single representative broadcast day while every other figure on
the destination was the week, so one label carried two different quantities on
one screen. The window here is the same one
:func:`kairos_api.core._summarize_schedule` gives the goal strip and
``/api/plan-progress``, so the comparison and the goal it is measured against are
the same seven dates.

**What it costs, measured 2026-08-01 on the reference data.** One leg on one real
broadcast day of ``רשת 13`` is 0.69 to 1.30 s of refined optimizer, so two legs
over the plan's seven days is 14 runs and about 12.9 s. That is over JS-2's 5 s
bar and it is reported rather than hidden: every payload carries ``elapsed_ms``
and how many of its runs were computed against how many were reused, and the
stream puts the first comparable day on screen in under 2 s. Two things make the
next comparison cheaper. A run is deterministic, proven by running the same call
twice and getting a byte-identical summary and segment list, so ``(day, levers)``
is cached against the plan's own file signature; and two legs that resolve to the
same levers share one run per day.

The greedy optimum would be 2.5x faster and is refused: measured on
``רשת 13 / 2024-11-04`` it reports 1,555,641.19 where the refined optimum the
saved plan is built with reports 1,698,547.39, so it would compare two plans the
operator would never run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from functools import lru_cache
from typing import Any, Iterator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from kairos.optimize.inventory import load_inventory
from kairos_api.core import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _plan_segment_index,
    _reference_today,
    _safe_number,
    _signature,
    _summarize_schedule,
)
from kairos_api.scenario_compare_api_money import _priced, _scenario_summary, compare_body
from kairos_api.scenario_compare_levers import LEVER_FIELDS, ScenarioCompareRequest

logger = logging.getLogger(__name__)

router = APIRouter()


def _week_signature() -> tuple[tuple[str, int, int], ...]:
    """The files a run stands on. A change to any of them retires every cached
    day, because a cached optimizer run is only honest while its inputs are the
    ones on disk."""
    return _signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        SETTINGS_PATH,
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots - inventory.csv",
    ])


_KEY_LOCKS: dict[Any, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()
# Which keys the cached body actually ran for, recorded by the body itself. The
# payload tells the planner how many of its runs were computed and how many were
# reused, so that count has to be exact: a cache hit counter would be wrong the
# moment a second request runs a different day at the same time, and an eviction
# would make a recomputed day read as reused.
_COMPUTED_KEYS: set[Any] = set()


def _key_lock(key: Any) -> threading.Lock:
    """One lock per cache key, so two callers asking for the same day wait for
    one optimization instead of running it twice."""
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def plan_week_window(settings: KairosSettings) -> dict[str, Any]:
    """The plan's own week for the owned channel, or an honest refusal.

    The bounds and the basis come from the saved plan's planning-week slice, the
    same slice the goal strip reads, so the comparison cannot quietly run on a
    different week from the goal it is measured against. The dates are the owned
    channel's own dates inside those bounds: a date the operator has no plan rows
    for is not the operator's week.
    """
    channel = str(settings.operator_channel or "").strip()
    if not channel:
        return {
            "available": False,
            "reason": "no operator channel is configured, so the plan's own week cannot be scoped",
        }
    schedule = _load_break_schedule()
    week = _summarize_schedule(schedule).get("week")
    if not isinstance(week, dict) or not week.get("date_from") or not week.get("date_to"):
        return {
            "available": False,
            "reason": "the saved plan carries no week for your channel, so there is no week to compare",
        }
    date_from = str(week["date_from"])[:10]
    date_to = str(week["date_to"])[:10]
    dates: list[str] = []
    if not schedule.empty and {"channel", "date"}.issubset(set(schedule.columns)):
        owned = schedule[schedule["channel"].astype(str).str.strip() == channel]
        text = owned["date"].astype(str).str.strip()
        dates = sorted({value[:10] for value in text if date_from <= value[:10] <= date_to})
    if not dates:
        return {
            "available": False,
            "reason": "the saved plan carries no dates for your channel inside its own week",
        }
    return {
        "available": True,
        "channel": channel,
        "dates": dates,
        "date_from": date_from,
        "date_to": date_to,
        "n_dates": len(dates),
        "basis": str(week.get("basis") or ""),
    }


def _levers_key(levers: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((field, levers.get(field)) for field in LEVER_FIELDS)


_EMPTY = (None, None, 0, 0, None, None, False, None, None, None, None, 0.0, 0)


@lru_cache(maxsize=192)
def _day_leg_cached(
    signature: tuple[tuple[str, int, int], ...],
    channel: str,
    day: str,
    levers_key: tuple[tuple[str, Any], ...],
) -> tuple[Any, ...]:
    """One leg on one broadcast day, as a tuple of primitives.

    A tuple rather than a dict on purpose: an ``lru_cache`` that hands back a
    mutable object hands every later caller whatever the first one did to it.
    The caller rebuilds its own record from these values.
    """
    with _KEY_LOCKS_GUARD:
        if len(_COMPUTED_KEYS) > 512:
            _COMPUTED_KEYS.clear()
        _COMPUTED_KEYS.add((signature, channel, day, levers_key))
    levers = dict(levers_key)
    settings = _load_settings()
    settings_map = _model_dump(settings)
    from kairos.service import run_scenario

    try:
        payload = run_scenario(
            revenue_weight=levers["revenue_weight"],
            retention_floor=levers["retention_floor"],
            max_breaks_per_hour=levers["max_breaks_per_hour"],
            risk_lambda=levers["risk_lambda"],
            objective_mode=levers["objective_mode"],
            today=_reference_today(settings),
            settings=settings_map,
            channel=channel,
            day=day,
            require_usable_inventory=True,
        )
    except Exception as exc:  # pragma: no cover - data/environment dependent
        logger.exception("the weekly comparison failed on %s", day)
        return (False, f"the optimizer failed on {day}: {str(exc)[:160]}", *_EMPTY)
    if not (payload.get("segments") or []):
        return (False, f"the programme source carries no segments for {day}", *_EMPTY)
    segments = list(_plan_segment_index(((channel, day),), settings_map).values())
    priced = _priced(_scenario_summary(payload, levers), payload, segments, levers["risk_lambda"])
    total_tvr = float(sum(float(getattr(item, "baseline_tvr", 0.0) or 0.0) for item in segments))
    return (
        True,
        None,
        priced.get("projected_revenue"),
        priced.get("average_retention"),
        int(_safe_number(priced.get("total_breaks"))),
        int(_safe_number(priced.get("total_ad_seconds"))),
        priced.get("objective"),
        priced.get("compliant"),
        bool(priced.get("money_available")),
        priced.get("money_reason"),
        priced.get("gross"),
        priced.get("retention_cost"),
        priced.get("revenue_net"),
        total_tvr,
        len(segments),
    )


_FIELDS = (
    "available", "reason", "projected_revenue", "average_retention", "total_breaks",
    "total_ad_seconds", "objective", "compliant", "money_available", "money_reason",
    "gross", "retention_cost", "revenue_net", "total_tvr", "segments",
)


def day_leg(channel: str, day: str, levers: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """One leg on one day as a record, and whether it had to be computed.

    The key lock serialises everything about this key, so the mark the cached
    body leaves is this call's own answer and not another request's.
    """
    # Validate before the cache lookup. A comparison prepared while inventory
    # was valid may stream for several seconds; an all-invalid replacement must
    # stop the next leg instead of letting an old cached figure escape.
    load_inventory(require_usable=True)
    key = (_week_signature(), channel, day, _levers_key(levers))
    with _key_lock(key):
        with _KEY_LOCKS_GUARD:
            _COMPUTED_KEYS.discard(key)
        values = _day_leg_cached(*key)
        with _KEY_LOCKS_GUARD:
            computed = key in _COMPUTED_KEYS
    return dict(zip(_FIELDS, values)), computed


def _sum(values: list[Any]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers), 2) if numbers else None


def _leg_total(days: list[dict[str, Any]], levers: dict[str, Any], channel: str) -> dict[str, Any]:
    """The days of one leg, added up the way the engine adds them.

    Revenue, breaks, ad seconds and the three money figures are sums. Retention
    is weighted by each day's own baseline rating total, which is the rule the
    optimizer applies inside a day and the rule the saved plan's own week metric
    applies across rows. The blended score is not a sum of any kind: it is a
    normalised per-day balance, so what is reported is the mean of the days and
    ``objective_basis`` says so rather than letting a reader assume otherwise.
    """
    priced = [day for day in days if day["available"]]
    weight = sum(day["total_tvr"] for day in priced) or 0.0
    retention = None
    if weight > 0:
        weighted = sum(float(day["average_retention"] or 0.0) * float(day["total_tvr"]) for day in priced)
        retention = round(weighted / weight, 1)
    scores = [float(day["objective"]) for day in priced if day["objective"] is not None]
    # A money total that silently drops a day is worse than no money total: it
    # reads as the week and is not. So the week is priced only when every day in
    # the window ran and every one of them priced, and the days that did not are
    # named on the leg itself.
    missing = [day["date"] for day in days if not day["available"]]
    money = bool(priced) and not missing and all(day["money_available"] for day in priced)
    summary: dict[str, Any] = {
        "revenue_weight": levers.get("revenue_weight"),
        "projected_revenue": _sum([day["projected_revenue"] for day in priced]),
        "average_retention": retention,
        "total_breaks": sum(day["total_breaks"] for day in priced),
        "total_ad_seconds": sum(day["total_ad_seconds"] for day in priced),
        "objective": round(sum(scores) / len(scores), 4) if scores else None,
        "objective_basis": "mean_of_days",
        "compliant": all(bool(day["compliant"]) for day in priced) if priced else None,
        "days_breaching": [day["date"] for day in priced if not day["compliant"]],
        "channel": channel,
        "day": None,
        "days": len(priced),
        "days_expected": len(days),
        "days_missing": missing,
        "levers": dict(levers),
        "money_available": money,
    }
    if money:
        summary["gross"] = _sum([day["gross"] for day in priced])
        summary["retention_cost"] = _sum([day["retention_cost"] for day in priced])
        summary["revenue_net"] = _sum([day["revenue_net"] for day in priced])
    else:
        refused = [day for day in days if not day["available"] or not day["money_available"]]
        first = next((day for day in refused if day.get("reason") or day.get("money_reason")), None)
        summary["money_reason"] = (
            None if first is None else str(first.get("reason") or first.get("money_reason"))
        )
    return summary


def _pair(record: dict[str, Any]) -> dict[str, Any]:
    """The fields one day of one leg publishes. Everything else stays internal."""
    return {
        "available": record["available"],
        "reason": record["reason"],
        "revenue": record["projected_revenue"],
        "retention": record["average_retention"],
        "breaks": record["total_breaks"],
        "money_available": record["money_available"],
        "gross": record.get("gross"),
        "retention_cost": record.get("retention_cost"),
        "revenue_net": record.get("revenue_net"),
    }


def _weekday(day: str) -> Optional[str]:
    try:
        return date.fromisoformat(day[:10]).strftime("%a")
    except ValueError:
        return None


def week_events(
    window: dict[str, Any], levers_a: dict[str, Any], levers_b: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Every finished day as it lands, then the finished week.

    One implementation drives both routes: the streaming one forwards each event
    to the browser, and the plain one keeps the last. So the two can never
    disagree about what the week is.
    """
    channel = str(window["channel"])
    dates = [str(value) for value in window["dates"]]
    started = time.perf_counter()
    days_a: list[dict[str, Any]] = []
    days_b: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    computed = 0
    for index, day in enumerate(dates, start=1):
        record_a, ran_a = day_leg(channel, day, levers_a)
        record_b, ran_b = day_leg(channel, day, levers_b)
        computed += int(ran_a) + int(ran_b)
        record_a["date"] = day
        record_b["date"] = day
        days_a.append(record_a)
        days_b.append(record_b)
        net_a, net_b = record_a.get("revenue_net"), record_b.get("revenue_net")
        rows.append({
            "date": day,
            "weekday": _weekday(day),
            "a": _pair(record_a),
            "b": _pair(record_b),
            "delta_revenue_net": (
                None if net_a is None or net_b is None else round(float(net_b) - float(net_a), 2)
            ),
        })
        yield {
            "kind": "day",
            "index": index,
            "of": len(dates),
            "day": rows[-1],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    total = len(dates) * 2
    yield {
        "kind": "final",
        "week": {
            "a": _leg_total(days_a, levers_a, channel),
            "b": _leg_total(days_b, levers_b, channel),
            "by_day": rows,
            "scope": {
                "mode": "week",
                "channel": channel,
                "day": None,
                "dates": dates,
                "date_from": window["date_from"],
                "date_to": window["date_to"],
                "n_dates": len(dates),
                "basis": window.get("basis"),
                "days_priced": sum(1 for day in days_a if day["available"] and day["money_available"]),
                "segments": sum(day["segments"] for day in days_a),
                "runs": {"total": total, "computed": computed, "reused": total - computed},
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "day_reason": None,
            },
        },
    }


def run_week(
    window: dict[str, Any], levers_a: dict[str, Any], levers_b: dict[str, Any]
) -> dict[str, Any]:
    """The whole week, for a caller that cannot stream."""
    final: dict[str, Any] = {}
    for event in week_events(window, levers_a, levers_b):
        if event["kind"] == "final":
            final = event["week"]
    return final


def _frame(event: str, data: Any) -> str:
    """One server-sent frame. json.dumps escapes newlines, so data stays one line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/api/scenario-compare/stream", tags=["insights"])
def scenario_compare_stream(request: ScenarioCompareRequest) -> StreamingResponse:
    """The weekly comparison, one finished broadcast day at a time.

    Fourteen real optimizations take longer than anybody should watch a spinner
    for, and a spinner is also a lie about progress. Each day is emitted the
    moment both its legs are decided, with the elapsed clock beside it, so the
    first comparable day is readable in under two seconds and the week assembles
    in the open. The last frame is the identical payload the plain route returns.
    """
    from kairos_api.scenario_compare_api import prepare_week

    prepared = prepare_week(request)

    def generate() -> Iterator[str]:
        if not prepared.get("available"):
            yield _frame("error", {"available": False, "reason": prepared.get("reason")})
            return
        window = prepared["window"]
        yield _frame("window", {**window, "runs_total": len(window["dates"]) * 2})
        final: dict[str, Any] = {}
        try:
            for event in week_events(window, prepared["levers_a"], prepared["levers_b"]):
                if event["kind"] == "day":
                    yield _frame("day", event)
                else:
                    final = event["week"]
        except Exception as exc:  # pragma: no cover - data/environment dependent
            logger.exception("the streamed weekly comparison failed")
            yield _frame("error", {"available": False, "reason": f"the comparison failed: {str(exc)[:200]}"})
            return
        yield _frame("final", compare_body(final, prepared["guardrails"]))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
