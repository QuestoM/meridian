"""How far back the record itself reaches, and what drops the rest.

Split out of ``history_api_timeline.py`` under the file-size law, and it earns
its own file for a second reason: it is the one place the product decides how
far its own evidence goes, and every surface that draws a conclusion from an
empty list depends on that one decision.

Two of the four records History merges are bounded and drop their oldest rows
without telling anybody. The request recorder keeps its newest
``activity_log.MAX_KEPT_ENTRIES`` lines. The version store keeps its newest
``version_store.MAX_VERSIONS`` restore points. The run log and the version
store's own restore audit are append-only and drop nothing.

**Measured on this deployment on 2026-08-01, five hours into a working day:**
exactly 200 manifests survived, the oldest stamped 11:01; the request recorder
held 5,227 lines, the oldest stamped 14:42; the oldest surviving entry on the
whole merged timeline was a restore of 2026-07-26. Asked for 20 July, the page
answered that nothing was recorded in those days. It could not know that, and
neither could anyone reading it.

So every read now carries where each record starts and what prunes it. A day
older than that is a day this product holds no evidence about, which is a
different answer from "nothing happened" and the only one it is entitled to.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from kairos_api import activity_log, history_api_timeline, version_store

POINTS_RETENTION = {"pruned": True, "keeps": version_store.MAX_VERSIONS,
                    "prune_at": version_store.MAX_VERSIONS, "unit": "restore_points"}

# The run log is append-only, so its start is the first run ever recorded rather
# than a retention floor, and a day before it is a day on which no run was
# recorded. That is a different sentence and a true one, so the surface is told
# which of the two this is rather than left to assume.
RUNS_RETENTION = {"pruned": False, "keeps": None, "prune_at": None, "unit": "runs"}

_DAY_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def earliest_day(values: Iterable[Any]) -> Optional[str]:
    """The oldest broadcast day among these stamps, or None when there is none.

    A stamp that does not read as a calendar day is dropped rather than sorted,
    so one malformed line cannot pull the record's stated start back to a day
    nothing was ever recorded on.
    """
    days = sorted(day for day in (history_api_timeline.broadcast_day(value) for value in values)
                  if _DAY_SHAPE.fullmatch(day))
    return days[0] if days else None


def sources(manifests: list[dict[str, Any]], activity: list[dict[str, Any]],
            runs: list[dict[str, Any]], runs_state: str, runs_available: bool,
            activity_scope: str = "all") -> dict[str, Any]:
    """The three sources: what each holds, how far back it reaches, what prunes it.

    ``starts`` and ``records`` are both properties of the store rather than of
    the reader, so both are read from the file itself and not from this caller's
    scoped slice of it: how far the evidence goes and how much of it there is do
    not change with who is looking. A source the product may not serve reports no
    start at all, because a run log it may not attribute cannot date itself
    either.

    **``records`` used to be the caller's slice, printed under the store's name.**
    Measured live on 2026-08-01 against a recorder holding 5,261 lines: an
    operator's read reported 1, a viewer's 4, and the admin's 5,261, each of them
    beside the same clause saying the recorder keeps the newest 5,000. So the
    reader's own slice moves to ``in_scope`` and travels with the rule that
    produced it, and a surface can print both without either one being a lie
    about the other.

    Only the request recorder is scoped per account, so only it carries the pair.
    Restore points are the shared operating record with no per-account scope, and
    the runs figure is the operator's own channel by the competitor boundary,
    which ``run_scope`` states in its own words rather than by a count of what
    was excluded.
    """
    return {
        "restore_points": {
            "records": len(manifests), "available": True,
            "starts": earliest_day(m.get("created_at") for m in manifests),
            "retention": POINTS_RETENTION,
        },
        "changes": {
            "records": activity_log.entry_count(), "in_scope": len(activity),
            "scope": str(activity_scope or "all"), "available": True,
            "starts": earliest_day([activity_log.first_entry_ts() or ""]),
            "retention": activity_log.RETENTION,
        },
        "runs": {
            "records": len(runs), "available": runs_available, "state": runs_state,
            "starts": earliest_day(r.get("created_at") for r in runs) if runs_available else None,
            "retention": RUNS_RETENTION,
        },
    }


def record_starts(named: dict[str, Any], entries: Iterable[dict[str, Any]]) -> Optional[str]:
    """The oldest day the merged record still holds, over every source and entry.

    The entries are read as well as the three named sources because the version
    store's own restore audit is append-only and is merged without being one of
    them, so on this deployment it is the oldest thing on the timeline: measured,
    a restore of 2026-07-26 against a request recorder starting on 2026-08-01.
    Taking the minimum means this can never claim the record starts later than
    something the page is already showing.
    """
    days = [str(source.get("starts") or "") for source in named.values()]
    days.append(earliest_day(entry.get("ts") for entry in entries) or "")
    real = sorted(day for day in days if day)
    return real[0] if real else None


def assemble(manifests: list[dict[str, Any]], activity: list[dict[str, Any]],
             runs: list[dict[str, Any]], runs_state: str, runs_available: bool,
             entries: list[dict[str, Any]],
             activity_scope: str = "all") -> tuple[dict[str, Any], Optional[str]]:
    """The sources block and the day the whole record starts on, in one call."""
    named = sources(manifests, activity, runs, runs_state, runs_available, activity_scope)
    return named, record_starts(named, entries)
