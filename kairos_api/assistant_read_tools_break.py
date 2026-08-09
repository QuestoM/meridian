"""READ tool executors for the break: the object between the day and the pod.

Rule 24 of Kai's prompt names the break as one of the nine objects this product
has, and the scheduler persona is told to answer "in terms of days, segments,
breaks and pins" and to give times in the plan's own clock. No read tool reached
it. ``get_day_detail`` returns SEGMENTS carrying a break count, which is one
level up, and ``get_break_pods``/``get_pod`` read the spots inside a break, which
is one level down. The object in the middle, the one the prompt names and the
whole day board is built out of, was the one thing the scheduler could not read.

Both tools call the break routes' own functions in :mod:`kairos_api.break_api`
rather than walking the plan a second time, so the tool and the screen answer one
break the same way. Two walks of one plan are two plans on the day they disagree.
Four rules carry straight through from those routes.

**Money on a break is the optimizer's own credit at insertion.** It is never a
share of the programme divided by the break count, and nothing here re-rounds,
re-prices or re-sums it. The day's breaks add back to the day's revenue exactly
because they are the engine's own figures, passed through.

**Delivered money is a state and not a figure.** The saved plan covers November
2024 and the one daily spot file covers 2025-04-27, so no planned break has a
ledger behind it. Every break therefore reports delivered as unavailable with the
covered days named, and this module never turns that into a zero.

**The day board is a LIVE re-plan, and the committed weekly plan is a different
basis.** ``basis`` carries both: the figures this day was just re-planned to
against current settings and models, and the figures the saved weekly plan holds
for the same channel-day. Both travel, so a reader can see it when the two have
parted company rather than being handed one of them as though it were the other.

**The competitor boundary holds by construction.** ``break_store.day_plan``
builds a day only for the operator's own channel, taken from settings, so a break
id naming any other channel resolves to no day at all and answers an honest
error. ``tests/test_assistant_break_tools.py`` proves that against the real
multi-channel plan file rather than trusting this paragraph.

Split into its own module under the size cap, beside
kairos_api.assistant_read_tools_pod; kairos_api.assistant_read_tools_catalog
registers it, so the combined registry stays the only dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_LIST_ROWS = 20
# A shipped channel-day carries 80 breaks and the scheduler's question is about
# the day, so a cap below that would hide breaks from the one person whose job is
# to place every one of them. The cap is here for the pathological day.
MAX_BREAKS = 120
MAX_HOUR_BREAKS = 24


def _cap(payload: dict[str, Any], key: str, limit: int) -> None:
    """Cap ``payload[key]`` to ``limit`` rows in place, recording the overflow."""
    rows = list(payload.get(key) or [])
    payload[key] = rows[:limit]
    if len(rows) > limit:
        payload[f"{key}_total"] = len(rows)
        payload[f"{key}_omitted"] = len(rows) - limit


def _clock(seconds: Any) -> Any:
    """One offset into the broadcast day as the plan's own clock string.

    The break records the day board serves carry seconds and no clock, while the
    scheduler persona is told to give times in the plan's own clock. The break
    inspector's own renderer is called rather than a second one written here, so
    the time beside a break id in an answer is the time on the screen. The
    seconds travel too, unrounded, so nothing is lost to the rendering.
    """
    from kairos_api.break_api_detail import _clock as render

    if seconds is None:
        return None
    return render(float(seconds))


def _break_digest(record: dict[str, Any]) -> dict[str, Any]:
    """One break as a day-scan row: where it sits, what it earns, what binds it.

    Every figure is the plan's own, passed exactly as ``break_records`` computed
    it. ``delivered`` is the one field lifted off the row: it is computed once per
    day and was byte-identical on all 80 breaks of the shipped day, 500 bytes
    each, so it rides the payload once and says the same thing.
    """
    digest = {
        "break_id": record.get("break_id"),
        "segment_id": record.get("segment_id"),
        "ordinal": record.get("ordinal"),
        "breaks_in_segment": record.get("breaks_in_segment"),
        "programme": record.get("programme"),
        "genre": record.get("genre"),
        "hour": record.get("hour"),
        "start_clock": _clock(record.get("start_seconds")),
        "end_clock": _clock(record.get("end_seconds")),
        "start_seconds": record.get("start_seconds"),
        "duration_seconds": record.get("duration_seconds"),
        "offset_seconds": record.get("offset_seconds"),
        "is_gold": record.get("is_gold"),
        "projected_revenue": record.get("projected_revenue"),
        "segment_retention": record.get("segment_retention"),
        "placement_source": record.get("placement_source"),
    }
    # The pin itself rides only on the breaks that carry one. On the shipped day
    # that is none of the 80, and eighty nulls would read as eighty facts.
    if record.get("saved_placement"):
        digest["saved_placement"] = record["saved_placement"]
    return digest


def _hoist_delivered(rows: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """The delivered-money state, lifted off the rows when every row carries the same one.

    The same rule :func:`break_api_board.collapse_demo` applies to the demo
    marking, and for the same reason: a row is never described by a sentence that
    was written about a different row. When two rows disagree the block stays on
    each of them and nothing is hoisted.
    """
    distinct = {repr(sorted((row.get("delivered") or {}).items())) for row in rows}
    if len(distinct) != 1:
        return None
    shared = rows[0].get("delivered") if rows else None
    return dict(shared) if isinstance(shared, dict) else None


def _read_get_day_breaks(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from fastapi import HTTPException

    from kairos_api import break_store
    from kairos_api.break_api import plan_day

    day = str(args.get("day", "") or "").strip()
    try:
        # The route function itself, so the tool and the day board answer one day
        # the same way. A day the plan does not carry is a 404 there and an
        # honest error here, with the days that can be opened named.
        payload = dict(plan_day(day=day))
    except HTTPException as exc:
        return {"error": str(exc.detail), "day": day or None,
                "plan_days": break_store.plan_days()[:MAX_LIST_ROWS]}
    # One provenance label per tool: execute_read_tool stamps the source, and a
    # second wording of the same files would print a source with no reading.
    payload.pop("source", None)
    if not payload.get("available"):
        # No channel or no saved plan is a state of the deployment, passed
        # through with its reason in both languages and with no figure at all.
        return {key: payload.get(key) for key in
                ("available", "reason", "reason_he", "operator_channel", "day")}
    rows = list(payload.get("breaks") or [])
    on_the_day = len(rows)
    hour = args.get("hour")
    if hour not in (None, ""):
        try:
            wanted = int(hour)
        except (TypeError, ValueError):
            return {"error": f"hour must be a whole number from 0 to 23, got {hour!r}"}
        rows = [row for row in rows if int(row.get("hour", -1)) == wanted]
        payload["filtered_by"] = {"hour": wanted}
    delivered = _hoist_delivered(rows)
    if delivered is not None:
        for row in rows:
            row.pop("delivered", None)
        payload["delivered_money"] = delivered
    payload["breaks"] = [_break_digest(row) for row in rows]
    payload["breaks_on_the_day"] = on_the_day
    payload["count"] = len(payload["breaks"])
    _cap(payload, "breaks", MAX_BREAKS)
    _cap(payload, "unbound_placements", MAX_LIST_ROWS)
    # The programmes this day holds are the level above the break and
    # get_day_detail already reads them; each break names its own programme and
    # how many breaks that programme carries, so nothing here needs the 24 KB.
    payload.pop("programmes", None)
    payload["segments_tool"] = "get_day_detail reads the day's programmes, the level above the break"
    payload["contents_tool"] = "get_break_pods and get_pod read the spots inside a break, from the traffic file"
    payload["detail_tool"] = "get_break reads one break whole, by its break id"
    return payload


def _read_get_break(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from fastapi import HTTPException

    from kairos_api import break_store
    from kairos_api.break_api import break_detail

    identifier = str(args.get("break_id", "") or "").strip()
    if not identifier:
        return {"error": "provide the break id, which reads <broadcast day>|<channel>|<segment>~<ordinal>; list a day's breaks with get_day_breaks"}
    try:
        # The inspector's own route function. A break id naming a channel this
        # operator does not own resolves to no day plan and answers here exactly
        # as it answers on the wire, which is what makes the boundary one rule.
        detail = dict(break_detail(identifier))
    except HTTPException as exc:
        return {"error": str(exc.detail), "break_id": identifier,
                "plan_days": break_store.plan_days()[:MAX_LIST_ROWS]}
    guardrails = dict(detail.get("guardrails") or {})
    _cap(guardrails, "hour_breaks", MAX_HOUR_BREAKS)
    detail["guardrails"] = guardrails
    contents = dict(detail.get("contents") or {})
    if contents.get("pod"):
        # The whole pod is a second copy of what get_pod returns whole. The id
        # and the state stay, so the model can go and read it rather than being
        # handed the same spots twice under two different tool names.
        contents.pop("pod", None)
        contents["pod_tool"] = "get_pod reads this pod whole, with its spots"
    _cap(contents, "covered_days", MAX_LIST_ROWS)
    detail["contents"] = contents
    return detail


_BREAK_READ_EXECUTORS = {
    "get_day_breaks": _read_get_day_breaks,
    "get_break": _read_get_break,
}

# Provenance stamps, same vocabulary as SOURCE_BY_TOOL in assistant_read_tools:
# the dataset each figure came from, surfaced on the run trace. Both name the day
# plan rather than the weekly CSV, because a day board is re-planned live and
# saying otherwise would credit the saved file with figures it does not hold.
BREAK_SOURCE_BY_TOOL = {
    "get_day_breaks": "the day plan for one channel-day, re-planned live, with the saved weekly plan beside it",
    "get_break": "the day plan for one channel-day, one break",
}

# The schemas live beside their executors rather than in the schema module, so a
# description cannot drift from what the executor returns.
# kairos_api.assistant_tool_schemas extends its own list with these before
# READ_TOOL_NAMES freezes, so the model still sees one flat tool list.
BREAK_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_day_breaks",
        "description": (
            "Read one broadcast day of the operator's plan as breaks: every break the plan "
            "places, with its addressable break id, its programme and genre, its clock start "
            "and end and its length in seconds, its ordinal within the programme, whether it "
            "is gold, the revenue the optimizer credited it at insertion, and whether an "
            "operator placement holds it there. The day's totals, its hourly ad load against "
            "the licence limits, its compliance checks, its guardrails and the restrictions "
            "in force ride along, as does the basis, which carries both the live re-planned "
            "figures and the saved weekly plan's own figures for the same day. Delivered "
            "money is a state and never a figure. Optional hour narrows to one clock hour. "
            "Omit day for the first day the plan covers. Call this when the scheduler asks "
            "about a day's breaks, where a break sits, what an hour carries or what a break "
            "earns, then get_break for one break."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "ISO broadcast date, YYYY-MM-DD. The first planned day when omitted."},
                "hour": {"type": "integer", "description": "Only the breaks in this clock hour, 0 to 23."},
            },
        },
    },
    {
        "name": "get_break",
        "description": (
            "Read one break whole: its identity and ordinal within the programme, the "
            "programme's own window, rating, rate per point and premium, the break's clock "
            "position, length and offset, the revenue the plan credits it with together with "
            "the rating it was actually priced at and the formula behind it, the retention it "
            "costs with its credible interval, sample and confidence, its gold state, the "
            "hour it sits in with every other break in that hour, the spacing around it, the "
            "day's compliance and whether a traffic file covers its contents. Delivered money "
            "reports as unavailable with the covered days named rather than as zero. The "
            "break id reads <day>|<channel>|<segment>~<ordinal>, from get_day_breaks. Call "
            "this when the scheduler asks about one break."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"break_id": {"type": "string", "description": "The break id from get_day_breaks, for example 2024-11-01|רשת 13|000~1."}},
            "required": ["break_id"],
        },
    },
]


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge these executors and their source labels into the shared registry."""
    executors.update(_BREAK_READ_EXECUTORS)
    sources.update(BREAK_SOURCE_BY_TOOL)
