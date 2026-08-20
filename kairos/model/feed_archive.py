"""Keep every schedule this engine ever pulls, because liveness expires.

The competitor feed is the ONLY source in this repository that has ever carried
a broadcast-state label. `data/Spots.csv` and `data/Programmes.csv` hold 53,948
rated rows between them and neither has a live or repeat column, nor a live
marker in any title; the feed flags both, cleanly, on every record it publishes
(240 live, 183 recorded first runs, 281 repeats in the fortnight measured on
2026-08-19).

And the feed publishes only the NEXT fortnight. It says nothing about last
month, so a day that passes unarchived is a day whose liveness can never be
recovered at any price. The rated files will eventually cover those dates; the
labels will not be there to join to unless something kept them.

Nothing did. `keshet_refresh.refresh` has taken a ``history_dir`` since it was
written and writes a per-pull snapshot when given one; ``keshet_feed`` threads it
through as a ``--history-dir`` flag; and the daily LaunchAgent that has been
pulling every rival at 05:30 passes ``--all`` and nothing else. The capability
was complete and unreached, so every pull overwrote the one before it. This
module makes archiving what happens by DEFAULT, because a capability that needs
a flag nobody passes is a capability that does not exist.

WHAT IS KEPT, AND WHY IN THIS SHAPE

One gzipped snapshot per pull, plus an index naming every pull that ever ran.
The snapshot is the publication as it was read, so nothing downstream has to
trust a summary this module computed. Two properties matter more than size:

* **A pull that changed nothing does not duplicate the payload.** The index
  records that the pull happened and points at the snapshot it matched, so
  "we pulled and it was identical" and "we did not pull" stay different facts.
  A schedule that stops moving is the common case, not an error.
* **Nothing is ever overwritten or deleted.** A snapshot is evidence. Rewriting
  one would make the archive a claim about the past rather than a record of it.

Gaps are reported rather than smoothed. :func:`coverage` names the dates for
which no pull was ever kept, because an archive that answers "here is what I
have" without "here is what I am missing" invites exactly the reading it should
prevent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Iterable, Mapping, Optional, Sequence

# Beside the contract rather than inside it. The contract's columns are a shape
# other code depends on; the archive is a different thing with a different life.
DEFAULT_ARCHIVE = Path("data/reference/feed-archive")
INDEX_NAME = "index.json"

# The schedule is Israeli and the stamps are UTC. Pull days are reported in
# broadcast time so a gap report names mornings a person recognises.
BROADCAST_TZ = ZoneInfo("Asia/Jerusalem")

# The columns the archive exists for. A snapshot missing these is still kept
# (it is evidence either way) but the index says so, because an archive of
# schedules with no liveness in them does not answer the question it was built
# for and should not look as though it does.
LIVENESS_COLUMNS = ("Live", "Rerun")


def _slug(channel: str) -> str:
    """A filesystem-safe stand-in for a channel name, reversible by the index.

    The index carries the real name. This is only so a directory listing is
    readable and so no filesystem has to agree with us about Hebrew.
    """
    text = re.sub(r"\s+", "-", str(channel or "").strip())
    text = re.sub(r"[^\w\-]", "", text, flags=re.UNICODE)
    return text or "unnamed"


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """A content fingerprint that ignores row order but nothing else."""
    payload = sorted(
        json.dumps({str(k): ("" if v is None else str(v)) for k, v in row.items()},
                   ensure_ascii=False, sort_keys=True)
        for row in rows
    )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _root(root: Optional[str | Path]) -> Path:
    """The archive root, resolved on every call rather than frozen at import.

    A default argument of ``DEFAULT_ARCHIVE`` binds once when this module is
    first imported, so the root could never afterwards be redirected -- not by a
    test, not by an operator with a different disk.
    """
    return Path(DEFAULT_ARCHIVE if root is None else root)


def root_beside(target: str | Path) -> Path:
    """The archive that belongs to a given contract file.

    THE ARCHIVE FOLLOWS THE SCHEDULE IT ARCHIVES, and this is a correctness
    mechanism rather than a convenience. Passing the root as an argument was
    tried and it failed the same way ``--history-dir`` failed: ``refresh()``
    called ``keep()`` with no root, so every test that wrote a contract to a
    temporary directory archived its fixture into the REAL record. Fifty-two
    fabricated pulls accumulated that way in a single afternoon -- 127-row
    Keshet fixtures filed under כאן 11, under עכשיו 14, and one under a channel
    named "ק" -- in a store whose entire value is that it is not fabricated.

    Deriving the root from the target means a caller cannot forget, because
    there is nothing to remember: a schedule written somewhere is archived
    beside itself, and only the real contract path reaches the real archive.
    """
    return Path(target).parent / DEFAULT_ARCHIVE.name


def index_path(root: Optional[str | Path] = None) -> Path:
    return _root(root) / INDEX_NAME


def read_index(root: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Every pull ever kept, oldest first. An absent archive is an empty list."""
    path = index_path(root)
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        # An unreadable index is not an empty archive, and the snapshots are
        # still on disk. Saying "nothing" here would be a claim; the caller can
        # see the files. This returns nothing and never deletes.
        return []
    if not isinstance(loaded, list):
        return []
    # Sorted by WHEN THE PULL HAPPENED, not by when the line was appended. A
    # backdated entry is normal here -- this archive was seeded from a schedule
    # on disk stamped at its real pull time -- and in append order that entry
    # became "the last pull that named this broadcast", so an older statement
    # about a programme overwrote a newer one and coverage reported a last_pull
    # earlier than its first_pull.
    return sorted((e for e in loaded if isinstance(e, Mapping)),
                  key=lambda e: str(e.get("at") or ""))


def _append_index(root: Path, entry: Mapping[str, Any]) -> None:
    """Append one pull to the index, atomically, and never shrink it.

    Two failures, both found by measurement rather than by reading. ``read_index``
    answers an unparseable file with an empty list, so appending after a
    corruption wrote a ONE-entry index over the whole record: five pulls became
    one, the snapshots stayed on disk with nothing mapping them to a channel or a
    time, and ``coverage`` then read as complete. And the write was a single
    ``write_text``, so a crash or a full disk mid-write CREATES that corruption.

    So: write to a temporary file and replace, which is atomic on every
    filesystem this runs on; and refuse to write an index shorter than the one
    already there, keeping the unreadable file beside it as evidence. An archive
    that loses its own record silently is worse than one that stops appending.
    """
    entries = read_index(root)
    existing = index_path(root)
    if existing.exists() and not entries:
        # The file is there and unreadable. Do not overwrite the only copy.
        broken = existing.with_suffix(f".broken-{entry.get('at', 'unknown')}.json")
        try:
            if not broken.exists():
                broken.write_bytes(existing.read_bytes())
        except OSError:
            pass
    entries.append(dict(entry))
    root.mkdir(parents=True, exist_ok=True)
    temporary = existing.with_suffix(".writing.json")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(existing)


def keep(
    rows: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    at: Optional[datetime] = None,
    root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Keep one pull. Returns what was kept, or why nothing new was written.

    An empty pull is NOT archived. `keshet_refresh` already refuses to write one
    over the contract, on the ground that "the rival airs nothing" is a claim no
    publication ever made; archiving it would smuggle that same claim into the
    permanent record through a different door.
    """
    stamp = at or datetime.now(timezone.utc)
    root = _root(root)
    rows = [dict(r) for r in rows]
    if not rows:
        return {"kept": False, "reason": "an empty pull is not evidence of an empty schedule"}

    digest = _digest(rows)
    # Sorted as DATES. Lexicographic order on DD/MM/YYYY puts 01/09 before
    # 31/08, so a pull crossing a month boundary recorded its window backwards
    # and coverage counted two broadcast days instead of fourteen. This job runs
    # every morning on a fortnight window, so it meets a month boundary monthly.
    dates = sorted({str(r.get("Date") or "") for r in rows} - {""}, key=_as_date)
    entry: dict[str, Any] = {
        "at": stamp.isoformat(timespec="seconds"),
        "channel": str(channel),
        "rows": len(rows),
        "sha256": digest,
        "window": [dates[0], dates[-1]] if dates else [None, None],
        "days": len(dates),
        "liveness": all(col in rows[0] for col in LIVENESS_COLUMNS),
        "live": sum(1 for r in rows if str(r.get("Live")).lower() == "true"),
        "rerun": sum(1 for r in rows if str(r.get("Rerun")).lower() == "true"),
    }

    for previous in reversed(read_index(root)):
        if previous.get("sha256") == digest and previous.get("channel") == str(channel):
            # The rival did not move. That is a real observation about this
            # morning, so the pull is recorded; the bytes are not stored twice.
            entry["file"] = previous.get("file")
            entry["identical_to"] = previous.get("at")
            _append_index(root, entry)
            return {"kept": True, "unchanged_since": previous.get("at"),
                    "path": previous.get("file"), "entry": entry}

    name = f"{stamp:%Y%m%dT%H%M%S}-{_slug(channel)}.csv.gz"
    entry["file"] = name
    target = root / name
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Two pulls of one channel inside one second. Never overwrite evidence.
        name = f"{stamp:%Y%m%dT%H%M%S}-{_slug(channel)}-{digest[:8]}.csv.gz"
        entry["file"] = name
        target = root / name
    _write_snapshot(rows, target)
    _append_index(root, entry)
    return {"kept": True, "path": str(target), "entry": entry}


def _write_snapshot(rows: Sequence[Mapping[str, Any]], target: Path) -> None:
    import csv

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): row.get(k) for k in columns})


def read_snapshot(name: str, root: Optional[str | Path] = None):
    """One kept pull as a frame, or None when the file is gone."""
    import pandas as pd

    path = _root(root) / str(name)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)


def broadcasts(root: Optional[str | Path] = None, *, channel: Optional[str] = None):
    """Every broadcast ever published, with the pull that last said so.

    A schedule is a forecast until it airs, so the same broadcast appears in many
    pulls and can change between them. The LAST pull that named it is the closest
    thing the archive has to what actually happened, and that is the row kept
    here; ``first_seen`` and ``pulls`` are carried so a reader can tell a
    long-announced programme from one that appeared the night before.

    Returns None when the archive holds nothing, never an empty frame, so
    "nothing was archived" and "nothing was published" stay different answers.
    """
    import pandas as pd

    root = _root(root)
    entries = [e for e in read_index(root)
               if channel is None or str(e.get("channel")) == str(channel)]
    if not entries:
        return None
    seen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    loaded: dict[str, Any] = {}
    for entry in entries:
        name = str(entry.get("file") or "")
        if not name:
            continue
        if name not in loaded:
            loaded[name] = read_snapshot(name, root)
        frame = loaded[name]
        if frame is None:
            continue
        for row in frame.to_dict("records"):
            key = (str(row.get("Channel") or ""), str(row.get("Date") or ""),
                   str(row.get("Start time") or ""), str(row.get("Title") or ""))
            record = seen.get(key)
            if record is None:
                record = dict(row)
                record["first_seen"] = entry.get("at")
                record["pulls"] = 0
                seen[key] = record
            else:
                record.update({k: v for k, v in row.items() if k in record})
            record["last_seen"] = entry.get("at")
            record["pulls"] = int(record.get("pulls", 0)) + 1
    if not seen:
        return None
    return pd.DataFrame(list(seen.values()))


def coverage(root: Optional[str | Path] = None) -> dict[str, Any]:
    """What the archive holds and, more usefully, what it is missing.

    The missing days are the product. An archive that reports only what it has
    reads as complete, and this one is complete only from the day it started.
    """
    entries = read_index(root)
    if not entries:
        return {"pulls": 0, "channels": [], "note": "nothing has been archived yet"}
    pull_days = sorted({_broadcast_day(e.get("at")) for e in entries} - {""})
    covered: set[str] = set()
    for entry in entries:
        window = entry.get("window") or [None, None]
        if window[0] and window[1]:
            covered.update(_days_between(str(window[0]), str(window[1])))
    missing = _missing_pull_days(pull_days)
    return {
        "pulls": len(entries),
        "channels": sorted({str(e.get("channel") or "") for e in entries} - {""}),
        "first_pull": entries[0].get("at"),
        "last_pull": entries[-1].get("at"),
        "pull_days": len(pull_days),
        "days_with_no_pull": missing,
        "broadcast_days_covered": len(covered),
        "note": (
            f"{len(entries)} pulls kept since {str(entries[0].get('at'))[:10]}; "
            f"{len(missing)} day(s) in that span have no pull at all. Nothing "
            f"before that date can be labelled, at any price."
        ),
    }


def _as_date(day: str) -> tuple[int, int, int]:
    """``DD/MM/YYYY`` as a sortable triple; an unparseable day sorts last."""
    try:
        moment = datetime.strptime(day, "%d/%m/%Y")
    except ValueError:
        return (9999, 99, 99)
    return (moment.year, moment.month, moment.day)


def _broadcast_day(stamp: Any) -> str:
    """Which Israeli day a pull happened on, not which UTC day.

    The stamps are UTC and the schedule is Israeli. At UTC+3 the 05:30 job lands
    on the same date either way, but a pull run late in the evening lands on the
    NEXT UTC day, and a gap report is about mornings a person recognises. Reading
    these in UTC would invent a missing day and hide a real one.
    """
    text = str(stamp or "")
    if not text:
        return ""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text[:10]
    if moment.tzinfo is None:
        return moment.strftime("%Y-%m-%d")
    return moment.astimezone(BROADCAST_TZ).strftime("%Y-%m-%d")


def _days_between(first: str, last: str) -> set[str]:
    """The ``DD/MM/YYYY`` days from first to last inclusive, or just what parses."""
    from datetime import timedelta

    try:
        start = datetime.strptime(first, "%d/%m/%Y")
        end = datetime.strptime(last, "%d/%m/%Y")
    except ValueError:
        return {first, last} - {""}
    if end < start:
        start, end = end, start
    out: set[str] = set()
    while start <= end:
        out.add(start.strftime("%d/%m/%Y"))
        start += timedelta(days=1)
    return out


def _missing_pull_days(pull_days: Sequence[str]) -> list[str]:
    from datetime import timedelta

    if len(pull_days) < 2:
        return []
    try:
        start = datetime.strptime(pull_days[0], "%Y-%m-%d")
        end = datetime.strptime(pull_days[-1], "%Y-%m-%d")
    except ValueError:
        return []
    have = set(pull_days)
    missing: list[str] = []
    while start <= end:
        day = start.strftime("%Y-%m-%d")
        if day not in have:
            missing.append(day)
        start += timedelta(days=1)
    return missing


def keep_many(
    rows_by_channel: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    at: Optional[datetime] = None,
    root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Keep one pull per channel, reporting each separately."""
    stamp = at or datetime.now(timezone.utc)
    return {
        channel: keep(list(rows), channel=channel, at=stamp, root=root)
        for channel, rows in rows_by_channel.items()
    }
