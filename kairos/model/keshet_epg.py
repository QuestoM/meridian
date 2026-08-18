"""The competitor's published schedule, pulled instead of typed.

Every competitor input this engine has ever had arrived as a file somebody put
on disk by hand. The counter-programming covariate, the competitor lineup, the
pressure signal — all of them read
``data/reference/CompetitorProgrammes.{xlsx,csv}`` and answer honestly that they
have nothing when it is absent. Which, in practice, is most of the time: nobody
re-types a rival's week.

Keshet publishes its own schedule as data. This module turns that publication
into exactly the file the existing contract already reads, so the whole
competitor stack downstream is untouched — same columns, same loader, same
honest-absence behaviour. Nothing here is a second mechanism; it is a feeder for
the one that exists.

What the publication actually carries, measured on a real capture of 127
broadcasts (70 distinct programmes):

* ``ProgramName`` always, ``EventDescription`` on 123, ``EnglishName`` on 126
* ``StartTime`` as ``DD/MM/YYYY HH:MM:SS`` and ``DurationMs``, from which the
  end clock is computed rather than trusted from a second field
* ``LiveBroadcast`` / ``RerunBroadcast`` on every record — which the programme
  classifier already understands, so a rival's repeat is not counted as a fresh
  premiere
* ``Season`` and ``Episode`` present as fields and EMPTY on all 127. They are
  not available here at any price, and only 5 of 127 descriptions mention them
  in prose. Filling them is a separate, explicit inference step
  (:mod:`kairos.model.keshet_enrich`) and never a guess made in this file.

The channel name is a parameter and not a constant, because the contract joins
the competitor schedule to history BY CHANNEL NAME: the name written here has to
be the same string the Dayparts and Spots history uses for that channel, or the
audience-strength lookup silently finds nothing. Getting that wrong produces a
schedule that loads cleanly and contributes zero, which is the worst shape a
data error can take, so the caller states it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# The contract's own column names, from kairos.data.loaders.load_programmes.
# Written out rather than imported so a change to the loader's expectations
# fails here loudly instead of producing a frame that parses into nothing.
CONTRACT_COLUMNS = ("Channel", "Title", "Date", "Start time", "End time", "Duration")

# Extra columns the contract's loader ignores and a human reading the file
# wants. They ride along because this file is also the evidence of what was
# pulled, and a schedule nobody can audit is a schedule nobody should trust.
CARRIED_COLUMNS = ("Live", "Rerun", "ProgramCode", "HouseNumber", "Description")

STAMP = "%d/%m/%Y %H:%M:%S"


class EpgShapeError(ValueError):
    """The publication did not have the shape this converter was built for."""


def programmes_of(payload: Any) -> list[dict[str, Any]]:
    """The programme records inside a capture, whatever wrapper carries them.

    The capture on disk is ``{ok, status, json: {success, data: {programs: [...]}}}``
    while a live call returns the inner object directly. Both are accepted; a
    payload with no programme list raises rather than returning an empty
    schedule, because "the rival airs nothing" and "we failed to read the file"
    must never arrive at the model looking the same.
    """
    node: Any = payload
    for key in ("json", "data"):
        if isinstance(node, Mapping) and key in node:
            node = node[key]
    if isinstance(node, Mapping) and "programs" in node:
        node = node["programs"]
    if not isinstance(node, list):
        raise EpgShapeError(
            "no programme list found in this payload; expected json.data.programs"
        )
    if node and not isinstance(node[0], Mapping):
        raise EpgShapeError("the programme list does not hold records")
    return [dict(item) for item in node]


def _end_clock(start: datetime, record: Mapping[str, Any]) -> Optional[datetime]:
    """The end of a broadcast, computed from its own duration.

    ``DisplayEndTime`` exists but is a wall clock with no date, so a programme
    running past midnight would end BEFORE it started. The duration is the only
    field that survives the day boundary, so it is the one used.
    """
    millis = record.get("DurationMs")
    if isinstance(millis, (int, float)) and millis > 0:
        return start + timedelta(milliseconds=float(millis))
    shown = str(record.get("Duration") or "").strip()
    if shown and ":" in shown:
        try:
            hours, minutes = (int(part) for part in shown.split(":")[:2])
        except ValueError:
            return None
        if hours or minutes:
            return start + timedelta(hours=hours, minutes=minutes)
    return None


def to_contract_rows(
    payload: Any,
    *,
    channel: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Publication -> the rows the CompetitorProgrammes contract expects.

    Returns ``(rows, status)``. Every record that cannot be placed in time is
    dropped and COUNTED in the status, never silently discarded and never
    invented: a broadcast with no usable start or duration is a hole in the
    rival's schedule that the operator should be able to see.
    """
    records = programmes_of(payload)
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for record in records:
        title = str(record.get("ProgramName") or "").strip()
        raw_start = str(record.get("StartTime") or record.get("Date") or "").strip()
        if not title:
            dropped.append({"reason": "no programme name", "start": raw_start})
            continue
        try:
            start = datetime.strptime(raw_start, STAMP)
        except ValueError:
            dropped.append({"reason": "unreadable start clock", "title": title,
                            "start": raw_start})
            continue
        end = _end_clock(start, record)
        if end is None:
            dropped.append({"reason": "no usable duration", "title": title,
                            "start": raw_start})
            continue
        rows.append({
            "Channel": channel,
            "Title": title,
            "Date": start.strftime("%d/%m/%Y"),
            "Start time": start.strftime("%H:%M:%S"),
            "End time": end.strftime("%H:%M:%S"),
            "Duration": int(round((end - start).total_seconds())),
            "Live": bool(record.get("LiveBroadcast")),
            "Rerun": bool(record.get("RerunBroadcast")),
            "ProgramCode": record.get("ProgramCode"),
            "HouseNumber": str(record.get("HouseNumber") or ""),
            "Description": str(record.get("EventDescription") or "").strip(),
        })
    rows.sort(key=lambda row: (row["Date"], row["Start time"]))
    days = sorted({row["Date"] for row in rows})
    status = {
        "channel": channel,
        "records_in": len(records),
        "rows_out": len(rows),
        "dropped": dropped,
        "titles": len({row["Title"] for row in rows}),
        "days": days,
        "window_start": days[0] if days else None,
        "window_end": days[-1] if days else None,
        "reruns": sum(1 for row in rows if row["Rerun"]),
        "live": sum(1 for row in rows if row["Live"]),
    }
    return rows, status


def write_contract_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write the rows where the existing competitor loader already looks.

    UTF-8 with a BOM, deliberately. Every store this product reads is opened
    with ``utf-8-sig`` after a BOM written by one writer and read by a plain
    reader renamed every frequency rule in the engine; writing the BOM keeps
    this file in the same family as the rest and safe to open in Excel, which
    is where a media planner will look at it.
    """
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(CONTRACT_COLUMNS + CARRIED_COLUMNS))
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_capture(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
