"""Which file of a kind the engine reads, and which it keeps and never reads.

Split out of ``uploads.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule. Every function here is pure: it is handed the
resolved paths it reasons about, so ``uploads.py`` keeps sole ownership of the
writable locations and a test that relocates them still relocates everything.

Two verdicts live here and they are not the same verdict.

**Is an upload of this kind consumed at all**, which is
:func:`in_use`. It is derived from the real read paths, never from optimism: a
reference workbook that shadows a CSV, a rate card no engine code opens, a
daily file the resolver will not pick. Each false answer carries the sentence
that names the file the engine reads instead.

**Which files of this kind are on disk and read by nothing**, which is
:func:`unread_records`. Only the daily kind can hold more than one file, and
the resolver picks exactly one of them, so without this list a file an operator
had just uploaded could sit on disk named on no screen while the card reported
the file the engine does read as in use with nothing to do.

Both of those are about files that already exist. :func:`prospect` asks the same
question one moment earlier, about a candidate that has not been written, which
is the only moment at which the answer can still change what a person does.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from kairos_api import uploads_inputs, uploads_status


def in_use(
    *,
    kind: str,
    reference: Path | None,
    consumed: str | None,
    live: Path | None,
    saved_path: Path | None,
    relative: Callable[[Path], str],
) -> tuple[bool, str]:
    """Whether the engine actually consumes an upload of this kind.

    ``reference`` is the workbook that shadows this kind while it exists, from
    ``SHADOWING_REFERENCE``; ``consumed`` is the file the engine reads instead
    for a kind nothing reads here, from ``STORED_UNREAD``; ``live`` is the file
    the daily resolver currently picks; ``saved_path`` is the file an upload
    just wrote, when the question is about that file rather than about the kind.

      * Most kinds land exactly where their consumer reads (the advertiser rules
        in data/advertiser_rules.csv, the campaign flights in
        data/campaign_flights.csv), so an upload genuinely takes effect: in_use
        is True with an empty reason.

      * The daily Wally kind is read through one file out of a directory. When
        ``saved_path`` is given, in_use is True only if the resolver actually
        picks that file; otherwise the honest amber reason names the file the
        engine reads instead, so a save is never reported as an ingestion that
        will not happen.

      * The three channel-source kinds (programmes/spots/dayparts) write to flat
        data/*.csv, but the engine loaders read data/reference/*.xlsx first and
        fall back to the CSV only when that xlsx is absent. While the reference
        xlsx exists it shadows the upload: the file is stored and validated but
        the optimizer reads the xlsx. We report in_use False with the reason so
        the status never implies an ingestion that did not happen; remove the
        xlsx and the same upload becomes live.

      * A kind in STORED_UNREAD (the rate card) is saved on disk but read by NO
        engine code: the pricing engine takes its rate card from a different
        file. We report in_use False and name the file the engine really reads.
    """
    if reference is not None:
        if reference.exists():
            return (
                False,
                f"Stored but not used by the optimizer: the engine reads {relative(reference)} "
                "first and adopts this upload only when that reference file is "
                "absent, so it is currently shadowed. Remove the reference file to "
                "make this upload the live optimizer input.",
            )
        # No reference file present: the loader now falls back to this upload.
        return True, ""

    if consumed is not None:
        return (
            False,
            f"Stored, not yet read by the pricing engine: the rate card is read "
            f"from {consumed} (with the dashboard's pricing overrides), so this "
            "file is saved and validated but the optimizer does not consume it.",
        )

    if kind == "daily" and saved_path is not None:
        if live is None or live.resolve() != Path(saved_path).resolve():
            live_name = live.name if live is not None else "none"
            return (
                False,
                f"Stored but not the live daily input: the engine reads the newest "
                f"daily file by the airing date in its name, currently {live_name}. "
                "This upload is kept on disk and becomes live only if that newer "
                "file is removed.",
            )
        return True, ""

    return True, ""


def daily_rank(path: Path) -> tuple[date, float]:
    """How the daily resolver ranks one file on disk: airing date, then mtime.

    The airing date named in the filename comes first so re-uploading an OLDER
    day never displaces a newer day's plan just by having a fresher mtime, and
    a file with no date in its name falls back to the day it landed.
    """
    mtime = path.stat().st_mtime
    return (uploads_inputs.airing_date_from_name(path) or date.fromtimestamp(mtime), mtime)


def would_be_live(candidate: Path, stored: list[Path]) -> Path:
    """The daily file the resolver would pick if ``candidate`` landed right now.

    The candidate is ranked by the same key as every file on disk, under the
    airing date the name it would be stored as carries and with the mtime it
    would have, which is now. So a tie on the airing date goes to the file that
    just arrived, exactly as it would once the bytes were written, and a file
    whose date cannot win is known to lose before anything is written.
    """
    key = candidate.resolve()
    winner = candidate
    best = (uploads_inputs.airing_date_from_name(candidate) or date.today(), datetime.now().timestamp())
    for path in stored:
        if path.resolve() == key:
            continue
        rank = daily_rank(path)
        if rank > best:
            winner, best = path, rank
    return winner


def prospect(
    *,
    kind: str,
    prospective: Path,
    stored: list[Path],
    verdict: Callable[[Path | None], tuple[bool, str]],
    engine_reads: str | None,
    relative: Callable[[Path], str],
    models_dir: Path,
    root: Path,
) -> dict[str, Any]:
    """What THIS candidate file would do, answered before anything is written.

    The kind's own consequence is not this answer, and the measured gap this
    closes is exactly that substitution: a daily file whose airing date cannot
    win the resolver is stored and read by nothing, while the kind it belongs to
    IS the live input, so a door that answered for the kind told the steward the
    opposite of what committing would do. The candidate is ranked here as the
    resolver ranks it, and every field returned is about that one file.
    """
    live = would_be_live(prospective, stored) if kind == "daily" else prospective
    will_be_read, reason = verdict(live)
    # What the engine opens for this kind once this upload has landed: the
    # candidate itself when it wins, and otherwise whatever is winning now.
    reads_after = relative(prospective) if will_be_read else engine_reads
    return {
        "saves_to": relative(prospective),
        "will_be_read": will_be_read,
        "will_be_read_reason": reason,
        "engine_reads_after": reads_after,
        # What this upload takes the place of as the live input, which is
        # nothing at all when the engine will not read it.
        "replaces": engine_reads if will_be_read else None,
        "consequence": uploads_status.consequence_record(
            will_be_read,
            reads_after,
            models_dir,
            root,
            still_read=None if will_be_read else reads_after,
        ),
    }


def unread_records(
    *,
    stored: list[Path],
    live: Path,
    relative: Callable[[Path], str],
    row_count: Callable[[Path], int],
    when: Callable[[Path], str],
) -> list[dict[str, Any]]:
    """The files of this kind on disk that the engine does not read.

    ``arrived_after_live`` is the fact that turns a stored file from an archive
    into a problem: it means this file landed AFTER the one the engine reads,
    so the operator's own most recent act is not what any number rests on. The
    row count is read for the first few only, through the shared signature
    cache, and stays None beyond them rather than being guessed at.
    """
    live_key = live.resolve()
    live_mtime = live.stat().st_mtime
    records: list[dict[str, Any]] = []
    for path in stored:
        if path.resolve() == live_key:
            continue
        arrived_after = path.stat().st_mtime > live_mtime
        counted = len(records) < uploads_status.STORED_LIST_CAP
        records.append(
            {
                "filename": path.name,
                "path": relative(path),
                "rows": row_count(path) if counted else None,
                "size_bytes": int(path.stat().st_size),
                "last_modified": when(path),
                "arrived_after_live": arrived_after,
                "reason": uploads_status.stored_reason(
                    "arrived_after_the_file_that_is_read" if arrived_after else "another_day_is_read",
                    live.name,
                ),
            }
        )
    return records
