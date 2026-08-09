"""READ tool executors for the pod: the ordered spots inside one break.

Kai is told, for the traffic-operator persona, to answer in breaks, spots and
durations and to be exact about seconds. Until these two tools existed the word
pod appeared in no read tool at all, so the only thing Kai could be exact from
was the prompt's own wording, which is the dangerous half of that gap: an answer
composed from words rather than from the file reads like a measurement and is
not one.

Both tools call the pod routes' own functions in
:mod:`kairos_api.break_api_pod` rather than going near the traffic files, so the
tool and the screen answer one pod the same way, and the arithmetic is the pod's
own arithmetic. Three rules from that module carry straight through here.

**A day with no traffic file behind it is a state of the data.** It answers
``available: false`` with the reason, the days that are covered and an empty
list. Nothing here invents a pod, a count or a duration to fill it, and an
absent length stays absent instead of reading as a zero.

**The arithmetic is per-second.** Every figure is passed exactly as the pod
computed it: never re-rounded, never reformatted, never turned into a sentence.
The tool returns the data and the model does the talking.

**The competitor boundary holds by construction.** The traffic file carries no
channel column at all, so no rival channel name can reach this surface through
it; the channel printed beside a pod is the operator's own from settings, and it
says so. ``tests/test_assistant_pod_tools.py`` asserts that premise against the
files on disk rather than trusting this paragraph.

Split into its own module under the size cap, next to
kairos_api.assistant_event_pipeline and kairos_api.assistant_audience_model;
kairos_api.assistant_read_tools_catalog registers it, so the combined registry
stays the only dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_PODS = 24
MAX_LIST_ROWS = 20
# A real pod on the shipped traffic file runs to 38 spots, so a cap below that
# would hide spots from the one person whose job is to check every one of them.
# The cap is here for the pathological file, not for the working day.
MAX_POD_SPOTS = 40


def _cap(payload: dict[str, Any], key: str, limit: int) -> None:
    """Cap ``payload[key]`` to ``limit`` rows in place, recording the overflow."""
    rows = list(payload.get(key) or [])
    payload[key] = rows[:limit]
    if len(rows) > limit:
        payload[f"{key}_total"] = len(rows)
        payload[f"{key}_omitted"] = len(rows) - limit


def _arithmetic_basis() -> dict[str, str]:
    """What each pod figure measures, said once for the whole day rather than
    repeated on every pod. The sentences are the pod math module's own constants,
    read from it, so this tool cannot drift from the surface it reports."""
    from kairos_api.break_api_pod_math import (
        HEAD_GAP_BASIS,
        HEAD_GAP_BASIS_HE,
        LOAD_BASIS,
        LOAD_BASIS_HE,
        SPAN_BASIS,
        SPAN_BASIS_HE,
        UNFILLED_BASIS,
        UNFILLED_BASIS_HE,
    )

    return {
        "declared_load": LOAD_BASIS, "declared_load_he": LOAD_BASIS_HE,
        "span": SPAN_BASIS, "span_he": SPAN_BASIS_HE,
        "unfilled": UNFILLED_BASIS, "unfilled_he": UNFILLED_BASIS_HE,
        "gap_before_first_spot": HEAD_GAP_BASIS, "gap_before_first_spot_he": HEAD_GAP_BASIS_HE,
    }


def _figure(entry: Any) -> Any:
    """One arithmetic figure as its state and its seconds, and nothing else.

    The seconds are passed exactly as the pod computed them. A figure whose state
    is not real carries seconds None, and that None stays None: an absent length
    is absent, and a zero there would understate the pod by exactly the seconds
    nobody declared.
    """
    if not isinstance(entry, dict):
        return entry
    return {"state": entry.get("state"), "seconds": entry.get("seconds")}


def _pod_digest(pod: dict[str, Any]) -> dict[str, Any]:
    """One pod as a day-scan row: its identity, its arithmetic and its errors.

    A whole day of pods is 175 KB measured on the shipped file, most of it the
    same bilingual basis prose repeated on every figure, which would crowd out
    the context the answer is grounded in. So the day list carries the numbers
    and hoists the prose to the payload, and get_pod returns one pod whole.
    """
    arithmetic = pod.get("arithmetic") or {}
    positions = pod.get("positions") or {}
    order = pod.get("order") or {}
    return {
        "pod_id": pod.get("pod_id"),
        "break_start_clock": pod.get("break_start_clock"),
        "break_start_seconds": pod.get("break_start_seconds"),
        "programme": pod.get("programme"),
        "break_type": pod.get("break_type"),
        "spot_count": arithmetic.get("spot_count"),
        "spots_missing_a_length": arithmetic.get("spots_missing_a_length"),
        "declared_load": _figure(arithmetic.get("declared_load")),
        "span": _figure(arithmetic.get("span")),
        "unfilled": _figure(arithmetic.get("unfilled")),
        "gap_before_first_spot": _figure(arithmetic.get("gap_before_first_spot")),
        "gaps_between_spots": arithmetic.get("gaps_between_spots"),
        "overlaps_between_spots": arithmetic.get("overlaps_between_spots"),
        "declared_break_length": pod.get("declared_break_length"),
        "against_declared": pod.get("against_declared"),
        "copy_length_disagreements": pod.get("copy_length_disagreements"),
        "position_violation_count": positions.get("violation_count"),
        "unpositioned_spots": positions.get("unpositioned"),
        "last_held": positions.get("last_held"),
        "top_and_tail": positions.get("top_and_tail"),
        "verification_error_count": (pod.get("verification") or {}).get("count"),
        "order_state": order.get("state"),
        "locked": order.get("locked"),
    }


def _read_get_break_pods(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api.break_api_pod import list_pods

    # available, reason and path_forward pass through untouched: a day with no
    # traffic file behind it answers false with the reason and an empty list,
    # and no pod, count or duration is invented to fill it.
    payload = dict(list_pods(day=str(args.get("day", "") or "").strip()))
    # One provenance label per tool. The route words the same files a second way
    # for its own screen; execute_read_tool would keep that wording, and the run
    # trace would then print a source with no reading beside it.
    payload.pop("source", None)
    payload.pop("source_he", None)
    payload["pods"] = [_pod_digest(pod) for pod in (payload.get("pods") or [])]
    _cap(payload, "pods", MAX_PODS)
    _cap(payload, "covered_days", MAX_LIST_ROWS)
    payload["arithmetic_basis"] = _arithmetic_basis()
    payload["detail_tool"] = "get_pod reads one pod whole, with its spots"
    return payload


def _read_get_pod(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from fastapi import HTTPException

    from kairos_api.break_api_pod import covered_days, pod_detail

    identifier = str(args.get("pod_id", "") or "").strip()
    if not identifier:
        return {"error": "provide the pod id, which reads <broadcast day>~<break start clock>; list a day's pods with get_break_pods"}
    try:
        # The route function itself, so the tool and the screen answer one pod
        # the same way. A pod nothing declares is a 404 there and an honest
        # error here, never an empty pod that reads like a break with no spots.
        pod = dict(pod_detail(identifier))
    except HTTPException as exc:
        return {"error": str(exc.detail), "pod_id": identifier,
                "covered_days": covered_days()[:MAX_LIST_ROWS]}
    _cap(pod, "spots", MAX_POD_SPOTS)
    # The counts beside these lists stay the pod's own totals, so a cap never
    # hides how many violations or errors the pod actually carries.
    positions = dict(pod.get("positions") or {})
    _cap(positions, "violations", MAX_LIST_ROWS)
    pod["positions"] = positions
    verification = dict(pod.get("verification") or {})
    _cap(verification, "errors", MAX_LIST_ROWS)
    pod["verification"] = verification
    return pod


_POD_READ_EXECUTORS = {
    "get_break_pods": _read_get_break_pods,
    "get_pod": _read_get_pod,
}

# Provenance stamps, same vocabulary as SOURCE_BY_TOOL in assistant_read_tools:
# the dataset each figure came from, surfaced on the run trace. Both name the
# traffic files, because that is the only thing a pod is read from.
POD_SOURCE_BY_TOOL = {
    "get_break_pods": "daily traffic files on disk (data/daily_input), the pods of one broadcast day",
    "get_pod": "daily traffic files on disk (data/daily_input), the pod of one break",
}

# The schemas live beside their executors rather than in the schema module, so a
# description cannot drift from what the executor returns.
# kairos_api.assistant_tool_schemas extends its own list with these before
# READ_TOOL_NAMES freezes, so the model still sees one flat tool list.
POD_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_break_pods",
        "description": (
            "Read one broadcast day's pods: for every break the traffic file declares, "
            "its start clock, its programme, how many spots it holds, and the per-second "
            "arithmetic on it (declared load, span, unfilled seconds, the gap before the "
            "first spot, gaps and overlaps between spots), plus the comparison against the "
            "plan's declared break length, the copy-length disagreements, the position "
            "violations and the verification error count. A day with no traffic file behind "
            "it answers available false with the reason and an empty list; the covered days "
            "are named, so you can say which day can be read. Omit day for the first covered "
            "day. Call this when the traffic operator asks about a day's breaks, pods, spot "
            "loads, seconds or overruns, then get_pod for one break's spots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "string", "description": "ISO broadcast date, YYYY-MM-DD. The first covered day when omitted."}},
        },
    },
    {
        "name": "get_pod",
        "description": (
            "Read one pod whole: every spot in the order it airs, with its start and end "
            "clock, its declared duration in seconds, its position (1 to 5, Last, or "
            "unpositioned), advertiser, campaign, creative, house number and agency, plus "
            "the pod's own arithmetic, the check of each copy version's named length "
            "against its booked duration, the position violations and the verification "
            "error list. Durations are seconds exactly as the traffic file declares them; a "
            "spot that declares no length says so instead of reading as zero. The pod id "
            "reads <broadcast day>~<break start clock>, from get_break_pods. Call this when "
            "the traffic operator asks what is inside one break."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pod_id": {"type": "string", "description": "The pod id from get_break_pods, for example 2025-04-27~20:40:09."}},
            "required": ["pod_id"],
        },
    },
]


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge these executors and their source labels into the shared registry."""
    executors.update(_POD_READ_EXECUTORS)
    sources.update(POD_SOURCE_BY_TOOL)
