"""Refresh the competitor's schedule, and say what moved since last time.

This is what the operator actually needs before a run: not "here is a schedule"
but "here is what the rival changed since you last looked". A daily simulation
compared against a week-old competitor schedule is a simulation of a week that
is not happening.

The refresh runs at the start of the week and again before each daily run. Both
are the same call; the only difference is how much has moved.

WHAT THIS REFUSES TO DO
-----------------------
A failed pull must never look like a successful one. When the publication cannot
be reached — no session, network down, shape changed — the previous schedule
stays exactly where it is and the run is told it is STALE, with its age. It is
never silently replaced by an empty file, and today's plan is never computed
against a schedule that quietly became yesterday's while claiming to be today's.

The endpoint needs a signed-in session; it answers 401 to anyone else. That is a
credential this engine does not own, so :func:`refresh` takes a ``fetch``
callable and the caller supplies the session. Without one, the honest answer is
"not refreshed, and here is why", which the whole competitor stack already knows
how to handle: :mod:`kairos.model.future_epg` contributes exactly nothing when
the file is absent, and an uncovered date reads as unknown rather than as "no
competition".

WHY THE DIFF IS THE PRODUCT
---------------------------
A rival moving a tentpole two hours is the single most decision-relevant thing
that can happen to a plan, and it is invisible in a file that simply overwrites
itself. So every refresh is compared with the one before it and reports, per
programme: added, removed, moved (same programme, different clock) and
reshaped (same clock, different length). Placeholders the broadcaster has not
yet announced are counted apart, because "פרטים יפורסמו בהמשך" turning into a
real title is news, and one placeholder replacing another is not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from kairos.model import keshet_epg

# Titles the broadcaster publishes for a slot it has not announced yet. They are
# real rows with real clocks — the slot exists — but they carry no programme, so
# they are never treated as a change of programming.
PLACEHOLDER_TITLES = ("פרטים יפורסמו בהמשך", "פרטים בהמשך", "שידורים")


def _is_placeholder(title: str) -> bool:
    text = str(title or "").strip()
    return any(text.startswith(mark) for mark in PLACEHOLDER_TITLES)


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    """A broadcast's identity for comparison: its programme on its day.

    Deliberately NOT the clock. Keying on the clock would report a moved
    programme as one removal plus one addition, which is exactly the change a
    planner most needs to see stated as a move.
    """
    return (str(row.get("Date") or ""), str(row.get("Title") or ""))


def compare(previous: Iterable[Mapping[str, Any]],
            current: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """What the rival changed, in the terms a planner would use."""
    before = {_key(r): dict(r) for r in previous}
    after = {_key(r): dict(r) for r in current}

    added, removed, moved, reshaped, announced = [], [], [], [], []
    for key, row in after.items():
        if key in before:
            was, now = before[key], row
            if was.get("Start time") != now.get("Start time"):
                moved.append({
                    "date": key[0], "title": key[1],
                    "from": was.get("Start time"), "to": now.get("Start time"),
                })
            elif int(was.get("Duration") or 0) != int(now.get("Duration") or 0):
                reshaped.append({
                    "date": key[0], "title": key[1],
                    "from_minutes": int(was.get("Duration") or 0) // 60,
                    "to_minutes": int(now.get("Duration") or 0) // 60,
                })
            continue
        if _is_placeholder(key[1]):
            continue
        # A slot that held a placeholder at this clock and now holds a real
        # programme is an ANNOUNCEMENT, not a new broadcast appearing from
        # nowhere. That distinction is the difference between "the rival added
        # a show" and "the rival finally told us what is in a slot we knew about".
        filled = any(
            _is_placeholder(t) and d == key[0] and before[(d, t)].get("Start time") == row.get("Start time")
            for (d, t) in before
        )
        (announced if filled else added).append({
            "date": key[0], "title": key[1], "at": row.get("Start time"),
        })
    for key, row in before.items():
        if key not in after and not _is_placeholder(key[1]):
            removed.append({"date": key[0], "title": key[1], "at": row.get("Start time")})

    changes = len(added) + len(removed) + len(moved) + len(reshaped) + len(announced)
    return {
        "added": added, "removed": removed, "moved": moved,
        "reshaped": reshaped, "announced": announced,
        "changes": changes, "unchanged": len(after) - changes,
        "quiet": changes == 0,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pandas as pd

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return frame.to_dict("records")


def refresh(
    *,
    fetch: Optional[Callable[[], Any]],
    channel: str,
    target: str | Path,
    history_dir: Optional[str | Path] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Pull, convert, write, and report the difference. Never write a lie.

    ``fetch`` returns the publication payload and raises on failure. ``target``
    is where the competitor contract lives — normally
    ``data/reference/CompetitorProgrammes.csv``, which is where
    :mod:`kairos.model.future_epg` already looks.
    """
    stamp = (now or datetime.now(timezone.utc))
    target = Path(target)
    previous = _read_rows(target)
    age_note = None
    if target.exists():
        modified = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc)
        age_note = round((stamp - modified).total_seconds() / 3600.0, 1)

    if fetch is None:
        return {
            "refreshed": False,
            "reason": (
                "no way to reach the competitor schedule was supplied; the "
                "published EPG needs a signed-in session and this engine holds none"
            ),
            "kept_rows": len(previous),
            "stale_hours": age_note,
            "at": stamp.isoformat(timespec="seconds"),
        }
    try:
        payload = fetch()
        rows, status = keshet_epg.to_contract_rows(payload, channel=channel)
    except Exception as exc:  # noqa: BLE001 - a failed pull is a stated state
        return {
            "refreshed": False,
            "reason": f"the competitor schedule could not be read ({type(exc).__name__}: {exc})",
            "kept_rows": len(previous),
            "stale_hours": age_note,
            "at": stamp.isoformat(timespec="seconds"),
        }

    if not rows:
        # An empty pull is a failed pull. Writing it would erase a schedule the
        # engine had and replace it with "the rival airs nothing", which is a
        # claim no publication ever actually made.
        return {
            "refreshed": False,
            "reason": "the competitor schedule came back with no broadcasts; the previous one is kept",
            "kept_rows": len(previous),
            "stale_hours": age_note,
            "at": stamp.isoformat(timespec="seconds"),
        }

    diff = compare(previous, rows)
    keshet_epg.write_contract_csv(rows, target)
    if history_dir:
        archive = Path(history_dir) / f"CompetitorProgrammes-{stamp:%Y%m%dT%H%M%S}.csv"
        keshet_epg.write_contract_csv(rows, archive)

    return {
        "refreshed": True,
        "at": stamp.isoformat(timespec="seconds"),
        "channel": channel,
        "rows": len(rows),
        "window": [status["window_start"], status["window_end"]],
        "days": len(status["days"]),
        "first_pull": not previous,
        "changes": diff,
        "path": str(target),
    }


def headline(result: Mapping[str, Any], locale: str = "he") -> str:
    """One line an operator reads before the run. Says stale when it is."""
    if not result.get("refreshed"):
        hours = result.get("stale_hours")
        if locale == "he":
            age = f" הלוח שבידינו בן {hours} שעות." if hours is not None else " אין לוח מתחרים כלל."
            return f"לוח המתחרים לא רוענן.{age}"
        age = f" The schedule on hand is {hours}h old." if hours is not None else " There is no competitor schedule at all."
        return f"The competitor schedule was not refreshed.{age}"
    if result.get("first_pull"):
        return (f"נמשך לוח המתחרים לראשונה: {result['rows']} שידורים על פני {result['days']} ימים."
                if locale == "he" else
                f"Competitor schedule pulled for the first time: {result['rows']} broadcasts over {result['days']} days.")
    diff = result["changes"]
    if diff["quiet"]:
        return ("לוח המתחרים רוענן: אין שינוי מאז הפעם הקודמת."
                if locale == "he" else
                "Competitor schedule refreshed: nothing changed since last time.")
    parts_he, parts_en = [], []
    for key, he, en in (("added", "נוספו", "added"), ("removed", "ירדו", "removed"),
                        ("moved", "הוזזו", "moved"), ("reshaped", "שונו באורך", "changed length"),
                        ("announced", "הוכרזו", "announced")):
        if diff[key]:
            parts_he.append(f"{len(diff[key])} {he}")
            parts_en.append(f"{len(diff[key])} {en}")
    return ("לוח המתחרים רוענן: " + ", ".join(parts_he) + "."
            if locale == "he" else
            "Competitor schedule refreshed: " + ", ".join(parts_en) + ".")
