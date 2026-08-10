"""Sources, downloads: the report shelf and the source-file audit.

Moved verbatim from catalog_api.py as part of the wave-zero router split. The
five reports carry four owner departments, so the shelf itself is load bearing
and it is kept rather than dissolved. Every row's status is read from real
state: an empty plan reports empty, and the daily ledger row counts the rows
the download actually carries.

The file audit answers two questions that are not one question. A file can be
on disk and read by nothing, so every record carries its role, whether anything
reads it now, and the reason when nothing does, in both languages.

The compliance row composes the frozen plan-read verdict, the same object Rules
serves and Today prints, so the three can never disagree.

Two helper modules carry the parts that are not the route, under the file-size
cap and the ``<parent stem>_<role>.py`` naming rule: ``downloads_api_reports``
holds what each report carries and the basis it was built on, and
``downloads_api_preview`` opens the rows behind a report's row count.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from kairos_api import channel_scope, downloads_api_preview, downloads_api_reports
from kairos_api.core import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _signature,
    _summarize_schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _source_file_paths() -> list[Path]:
    """The real source files the data-quality report audits.

    Single source of truth shared with ``/api/files`` so the report's row count
    reflects the actual file set, not a magic constant.
    """
    return [
        DATA_DIR / "Dayparts.csv",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
        DATA_DIR / "rate_card_premiums.csv",
        DATA_DIR / "advertiser_rules.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        MODELS_DIR / "tv_break_posterior.pkl",
    ]


# The upload kind each audited input file belongs to, so one verdict answers
# "is the engine reading this" on both the inputs view and the file list
# instead of two surfaces deriving it separately and disagreeing.
_FILE_KINDS: dict[str, str] = {
    "data/Dayparts.csv": "dayparts",
    "data/Programmes.csv": "programmes",
    "data/Spots.csv": "spots",
    "data/rate_card_premiums.csv": "rate_card",
    "data/advertiser_rules.csv": "advertiser_rules",
}

# Why a file is not in use, in both languages, because a note rendered inside a
# Hebrew screen has to be in Hebrew. Each one names the file that is read
# instead, which is the whole point of saying it at all.
_NOTES: dict[str, dict[str, str]] = {
    "reads_instead": {
        "en": "Not in use. The engine reads {reads} instead.",
        "he": "לא בשימוש. המנוע קורא במקום זאת את {reads}.",
    },
    "no_plan": {
        "en": "No saved plan carries the plan columns yet, so this file is not in use.",
        "he": "אין עדיין תוכנית שמורה עם עמודות התוכנית, ולכן הקובץ הזה אינו בשימוש.",
    },
    "plan_fallback": {
        "en": "A fallback plan file from the earlier optimizer. The saved weekly plan is read first, so this file is not in use.",
        "he": "קובץ תוכנית חלופי מהאופטימייזר הקודם. התוכנית השבועית השמורה נקראת קודם, ולכן הקובץ הזה אינו בשימוש.",
    },
    "model_fallback": {
        "en": "A fallback model file. The plan reads the measured model version first, so this file is not in use. That version and its state are on the inputs view.",
        "he": "קובץ מודל חלופי. התוכנית קוראת קודם את גרסת המודל שנמדדה, ולכן הקובץ הזה אינו בשימוש. הגרסה ומצבה מופיעים בתצוגת הקלטים.",
    },
    "model_live": {
        "en": "The measured model version every retention figure on the plan rests on. Its name and its state are on the inputs view.",
        "he": "גרסת המודל שנמדדה, שעליה נשען כל נתון שימור בתוכנית. השם והמצב שלה מופיעים בתצוגת הקלטים.",
    },
    "inventory_live": {
        "en": "The engine read this inventory file and built {slots} demand slots from it.",
        "he": "המנוע קרא את קובץ המלאי הזה ובנה ממנו {slots} משבצות ביקוש.",
    },
    "inventory_empty": {
        "en": "The engine reads this file, but its loader produced no demand slots. The inventory placement steer is therefore inactive; this is not an absent file.",
        "he": "המנוע קורא את הקובץ הזה, אך ה-loader לא הפיק ממנו משבצות ביקוש. לכן הכוונת המיקום לפי מלאי אינה פעילה; הקובץ אינו חסר.",
    },
}

# The one file on the audit whose name is also a word the run side never
# renders. It is listed because a file audit that hides the artifact driving
# every retention number is worse than a name, and it carries no verdict, no
# coverage and no value from inside it.
MODEL_ARTIFACT = "models/tv_break_coefficients.json"
INVENTORY_INPUT = "data/Spots - inventory.csv"


def _note(code: str, **fields: str) -> dict[str, str]:
    """One note as a bilingual record, with any path it names filled in.

    A path is a left-to-right run inside a right-to-left sentence, so it is
    wrapped in a first-strong isolate. Without it the Hebrew sentence's own
    full stop is reordered to the wrong side of the path.
    """
    words = _NOTES.get(code)
    if not words:
        return {"code": "", "en": "", "he": ""}
    isolated = {key: f"⁨{value}⁩" for key, value in fields.items()}
    return {"code": code, "en": words["en"].format(**isolated), "he": words["he"].format(**isolated)}


def _plan_is_saved() -> bool:
    """True when the saved weekly plan carries the plan contract's own columns.

    The legacy ``optimization_results.csv`` only becomes the live plan when the
    saved one is absent or is missing those columns, which is the same test
    ``kairos_api.core._load_break_schedule`` applies.
    """
    try:
        frame = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv", encoding="utf-8-sig", nrows=1)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return {"predicted_revenue", "predicted_retention", "num_breaks"}.issubset(frame.columns)


_NO_NOTE = {"code": "", "en": "", "he": ""}


def _file_role(relative: str) -> tuple[str, bool, dict[str, str]]:
    """``(role, in_use, note)`` for one audited file, from its real read state.

    Three roles and no fourth: an ``input`` the engine may read, the ``plan`` a
    run produced, and a ``model`` file. A file that is not in use says why, and
    the reason names the file that is read instead.
    """
    kind = _FILE_KINDS.get(relative)
    if kind is not None:
        from kairos_api.uploads import _engine_reads, _in_use

        in_use, _ = _in_use(kind)
        if in_use:
            return "input", True, dict(_NO_NOTE)
        return "input", False, _note("reads_instead", reads=str(_engine_reads(kind) or ""))
    if relative == "output/weekly_break_schedule.csv":
        saved = _plan_is_saved()
        return "plan", saved, dict(_NO_NOTE) if saved else _note("no_plan")
    if relative == "optimization_results.csv":
        saved = _plan_is_saved()
        return "plan", not saved, _note("plan_fallback") if saved else dict(_NO_NOTE)
    if relative == "models/tv_break_posterior.pkl":
        return "model", False, _note("model_fallback")
    return "input", True, dict(_NO_NOTE)


def _daily_file() -> tuple[Path | None, str | None]:
    """The daily file the ledger is priced from, and the day it aired."""
    try:
        from kairos_api.uploads import _newest_daily, _relative
        from kairos_api.uploads_inputs import airing_date_from_name

        path = _newest_daily()
        if path is None:
            return None, None
        aired = airing_date_from_name(path)
        return Path(_relative(path)), aired.isoformat() if aired else None
    except Exception:
        logger.exception("the daily file lookup failed for the report shelf")
        return None, None


def _build_reports(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    # The compliance verdict is the frozen plan read shared with /api/compliance
    # and /api/overview, so all three quote one verdict; imported at call time so
    # the module import graph stays acyclic.
    from kairos_api.plan_read_compliance import build_compliance

    summary = _summarize_schedule(schedule)
    compliance = build_compliance(schedule, settings)
    source_files = _source_file_paths()
    present = sum(1 for path in source_files if path.exists())
    audit_updated = max(
        (path.stat().st_mtime for path in source_files if path.exists()),
        default=None,
    )
    # Daily spot ledger: the per-spot priced/dropped output of the daily pricing
    # pipeline, downloadable at /api/export/spots.csv. The row count comes from
    # actually running that pipeline over the newest daily file, so it is the
    # exact ledger the download carries; an honest 0 when no daily file exists.
    ledger_rows = 0
    ledger_status = "empty"
    try:
        from kairos_api.exporters import _load_daily_pricing

        ledger = _load_daily_pricing()
        if ledger is not None:
            ledger_rows = int(len(ledger.priced) + len(ledger.dropped) + len(ledger.frequency_dropped))
            ledger_status = "ready" if ledger_rows else "empty"
    except Exception:
        logger.exception("daily spot ledger row count failed")
        ledger_status = "attention"
    daily_path, daily_date = _daily_file()
    return {
        "reports": downloads_api_reports.build(
            schedule=schedule,
            summary=summary,
            compliance=compliance,
            # The one accessor, so the channel a report says it is scoped to is
            # the channel every other surface here scopes to.
            channel=channel_scope.operator_channel(settings),
            plan_path=OUTPUT_DIR / "weekly_break_schedule.csv",
            ledger_rows=ledger_rows,
            ledger_status=ledger_status,
            daily_path=daily_path,
            daily_date=daily_date,
            audited_files=len(source_files),
            present_files=present,
            audit_updated=datetime.fromtimestamp(audit_updated).astimezone().isoformat(timespec="seconds")
            if audit_updated
            else None,
        )
    }


@lru_cache(maxsize=16)
def _reports_cached(signature: tuple[tuple[str, int, int], ...], channel: str) -> dict[str, Any]:
    """The reports catalogue, keyed on the files AND on the operator's channel.

    The channel was not part of the key, and every count on this catalogue is
    scoped by it: how many plan rows the download serves, how many days the
    revenue report covers, what the disclosure says the file holds that is not
    yours. So an operator who set their channel kept being served the previous
    channel's catalogue until a source file happened to change underneath it.

    Found by two tests disagreeing about the same number, 1,268 against 2,540,
    where each was right about the channel it had asked for and one of them was
    reading the other's cached answer.
    """
    del signature, channel
    return _build_reports(_load_break_schedule(), _load_settings())


@router.get("/api/reports", tags=["catalog"])
def reports() -> dict[str, Any]:
    # The daily spot ledger entry counts the newest daily file's priced ledger,
    # so that file (when present) is part of the cache key.
    paths = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
    ]
    try:
        from kairos_api.uploads import _newest_daily

        newest_daily = _newest_daily()
        if newest_daily is not None:
            paths.append(newest_daily)
    except Exception:
        logger.exception("newest daily file lookup failed for the reports cache key")
    # The channel is part of the key because every count below is scoped by it.
    return _reports_cached(_signature(paths), str(channel_scope.operator_channel() or ""))


@router.get("/api/reports/{report_id}/preview", tags=["catalog"])
def report_preview(
    report_id: str,
    limit: int = Query(downloads_api_preview.DEFAULT_PREVIEW_ROWS),
) -> dict[str, Any]:
    """The rows behind a report's row count, from the source its download reads.

    Two of the five reports are streamed by this server, so their rows are
    served here. The other three are built in the browser from a live endpoint,
    so this route names that endpoint rather than re-deriving a payload it does
    not own: a second derivation is how two surfaces come to disagree.
    """
    rows = downloads_api_preview.bounded(limit)
    if report_id == "weekly-plan":
        return downloads_api_preview.weekly_plan(_load_break_schedule(), rows)
    if report_id == "daily-spots":
        return downloads_api_preview.daily_spots(rows)
    source = downloads_api_preview.CLIENT_SOURCES.get(report_id)
    if source is not None:
        raise HTTPException(
            status_code=404,
            detail=f"The rows of the {report_id} report are read from {source}, which builds the download too.",
        )
    raise HTTPException(status_code=404, detail=f"Unknown report: {report_id}")


def _file_record(path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    role, in_use, note = _file_role(relative)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        if path.exists()
        else None,
        # What this file is for, whether anything reads it right now, and the
        # reason when nothing does. Present on disk and read by the engine are
        # two different facts and a file audit that reports only the first one
        # lets a shadowed input pass for a live one.
        "role": role,
        "in_use": in_use,
        "note": note,
    }


def _also_read_paths() -> list[Path]:
    """The files the engine reads that the audited list never covered.

    Measured: four of the seven inputs are read from somewhere other than the
    file the audit lists, so a card that names what the engine reads pointed at
    a list with no such row. These are those files, kept out of the audited set
    so the source-file report keeps counting exactly what it counted before,
    and rendered beside it so no name on any card is a dead end.

    The measured model artifact is on this list too. The audited set names a
    model file the product does not read and omitted the one that drives every
    retention number, which is a file audit hiding the file that matters most.
    """
    from kairos_api import uploads

    audited = {str(path) for path in _source_file_paths()}
    candidates: list[Path] = [reference for reference in uploads.SHADOWING_REFERENCE.values()]
    candidates.append(ROOT / "config" / "optimization_weights.yaml")
    candidates.append(uploads.DATA_DIR / "campaign_flights.csv")
    candidates.append(MODELS_DIR / "tv_break_coefficients.json")
    candidates.extend([
        DATA_DIR / "Spots - inventory.csv",
        DATA_DIR / "manual_overrides.csv",
        MODELS_DIR / "audience_model.json",
        SETTINGS_PATH,
    ])
    newest_daily = uploads._newest_daily()
    if newest_daily is not None:
        candidates.append(newest_daily)
    seen: set[str] = set()
    extra: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in audited or key in seen or not path.exists():
            continue
        seen.add(key)
        extra.append(path)
    return extra


def _also_read_record(path: Path) -> dict[str, Any]:
    """One engine-read file, in the same shape as an audited one."""
    record = _file_record(path)
    is_model = path.suffix in {".pkl", ".json"} and path.parent.name == "models"
    record["role"] = "model" if is_model else "input"
    record["in_use"] = True
    normalized = record["path"].replace("\\", "/")
    if normalized == MODEL_ARTIFACT:
        record["note"] = _note("model_live")
    elif normalized == INVENTORY_INPUT:
        stat = path.stat()
        slots = _inventory_pool_size(str(path), stat.st_mtime_ns, stat.st_size)
        record["read_state"] = "read_yielding" if slots else "read_yielding_nothing"
        record["yielded_items"] = slots
        record["note"] = _note("inventory_live", slots=str(slots)) if slots else _note("inventory_empty")
    else:
        record["note"] = dict(_NO_NOTE)
    return record


@lru_cache(maxsize=8)
def _inventory_pool_size(path: str, modified_ns: int, size: int) -> int:
    """The loader's result, cached by file state rather than by path alone."""
    del modified_ns, size
    from kairos.optimize.inventory import load_inventory

    return len(load_inventory(path))


def _stored_records() -> list[dict[str, Any]]:
    """The files this product stored for an input that the engine does not read.

    Measured gap: the daily directory keeps every file ever uploaded and the
    engine reads exactly one of them, so a second daily file was on disk and on
    no screen in the whole product. A file audit that lists only what is read
    hides the file an operator is most likely to be hunting for, which is the
    one they just sent. Each record carries the reason the inputs view prints,
    so the two views quote one sentence rather than deriving two.
    """
    from kairos_api import uploads

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for meta in uploads.INPUTS:
        try:
            stored = uploads.stored_unread_files(meta["kind"])
        except Exception:
            logger.exception("the stored-file lookup failed for kind %s", meta["kind"])
            continue
        for entry in stored:
            path = ROOT / str(entry["path"])
            if str(path) in seen or not path.is_relative_to(ROOT) or not path.exists():
                continue
            seen.add(str(path))
            record = _file_record(path)
            record["role"] = "input"
            record["in_use"] = False
            record["note"] = entry["reason"]
            records.append(record)
    return records


@router.get("/api/files", tags=["catalog"])
def files() -> dict[str, Any]:
    return {
        "files": [_file_record(path) for path in _source_file_paths()],
        # The audited list is what the source-file report counts. This second
        # list is what the engine reads and the audit never named, so every
        # path a source card prints resolves to a row somewhere on this page.
        "also_read": [_also_read_record(path) for path in _also_read_paths()],
        # And this third one is the other half of that sentence: what this
        # product stored for an input and the engine does not read. Kept out of
        # both lists above because it is neither audited nor read, and a file
        # nobody reads must never be counted as one that is.
        "stored": _stored_records(),
    }
