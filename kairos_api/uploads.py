"""Upload status and ingestion endpoints for the Kairos input files.

This module owns the operator-facing data ingestion surface: it reports the
live state of every input file the optimizer depends on, and it accepts new
uploads after validating them against the expected schema. Nothing here
fabricates data; a file is only reported as present and valid when it really
exists on disk and parses with the columns the loaders require. An accepted
upload is additionally parsed with the REAL engine loader for its kind and
checked against the data contracts in :mod:`kairos.data.contracts`, so a file
that would break or silently empty the optimizer is refused at the door with
the contract's own findings, and the last validation report is surfaced on the
status endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from kairos.data import contracts
from kairos.data.loaders import (
    CHANNELS,
    DAILY_COLUMN_MAP,
    count_ambiguous_daily_dates,
    load_daily_input,
    load_dayparts,
    load_programmes,
    load_spots,
)
from kairos.optimize.pacing import load_campaigns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily_input"
REFERENCE_DIR = DATA_DIR / "reference"
BACKUP_DIR = DATA_DIR / "_backups"
# Last per-kind validation report, persisted so /status can surface it across
# restarts. Lives under output/ with the other derived artifacts, never under
# the operator's input data.
VALIDATION_REPORTS_PATH = ROOT / "output" / "upload_validation_reports.json"

# Upload size cap. The largest legitimate channel export we have measured is the
# ~50k-row Spots file at a few MB; 100 MB leaves an order of magnitude of head
# room while keeping an unbounded body from exhausting memory. The body is read
# in chunks and rejected the moment the cap is crossed, mirroring the assistant
# upload lane, with a Content-Length pre-check so an honestly-declared oversize
# body is refused before any read.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_CHUNK = 64 * 1024

# Client-facing parse failure line. The pandas exception detail (offsets, C
# parser internals) is logged server-side instead of echoed to the client.
_GENERIC_PARSE_ERROR = "The uploaded file could not be read as a CSV table. Check that it is a UTF-8 CSV export with a single header row and try again."

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

# The three channel-source kinds land as flat CSVs under data/. The engine
# loaders (kairos.data.loaders) read data/reference/*.xlsx FIRST and fall back to
# the uploaded CSV only when that xlsx is absent. So while the reference xlsx
# exists, an uploaded CSV is stored and backed up but SHADOWED: the optimizer
# reads the xlsx, not the upload. We map each shadowed kind to the reference file
# that takes precedence so the status can say so honestly instead of reporting a
# bare green "valid" that implies ingestion. Remove the reference xlsx and the
# upload becomes the live input (the loader adopts the CSV fallback).
SHADOWING_REFERENCE: dict[str, Path] = {
    "programmes": REFERENCE_DIR / "Programmes.xlsx",
    "spots": REFERENCE_DIR / "Spots.xlsx",
    "dayparts": REFERENCE_DIR / "Dayparts.xlsx",
}

# Kinds that are stored on disk but which NO engine code reads. The rate card
# uploads to data/rate_card_premiums.csv, yet the pricing engine
# (kairos.optimize.pricing.PricingModel) reads its rate card from
# config/optimization_weights.yaml, deep-merged with the dashboard's
# pricing_overrides; nothing in the optimizer, forecast, or export path opens
# data/rate_card_premiums.csv (the only other reference to it is a file-existence
# count in the data-quality report). Reporting such a kind as in_use would imply
# an ingestion that never happens, so it is reported in_use False with the real
# reason. The mapped string names the file the engine actually consumes instead.
STORED_UNREAD: dict[str, str] = {
    "rate_card": "config/optimization_weights.yaml",
}

# Required columns per kind. These are the canonical headers the loaders and
# the optimizer read; extra columns are tolerated (reported as warnings).
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "programmes": ["Title", "Channel", "Date", "Start time", "End time", "Duration"],
    "spots": ["Campaign", "Channel", "Date", "Start time", "Duration"],
    "dayparts": ["Dates", "Timebands"],
    "advertiser_rules": [
        "advertiser_id",
        "default_premium",
        "allow_positions",
        "allow_genres",
        "prime_time_only",
        "notes",
    ],
    "rate_card": ["channel", "hour_of_day", "base_rate_ils_per_sec"],
    # The pacing flight file: the exact header of the shipped seed
    # data/campaign_flights.csv, which kairos.optimize.pacing.load_campaigns reads.
    "campaign_flights": [
        "campaign_id",
        "flight_start",
        "flight_end",
        "target_impressions",
        "target_grp",
        "delivered_to_date",
        "scope_channels",
        "scope_genres",
        "scope_dayparts",
        "scope_programmes",
        "notes",
    ],
    # The daily Wally file ships with Hebrew headers; the loader maps them.
    "daily": list(DAILY_COLUMN_MAP.keys()),
}

# Per-kind presentation metadata for the dashboard.
#
# The channel provides THREE source data files (programmes, spots, dayparts);
# the optimizer also takes ONE daily operational file (the Wally ad log). The
# advertiser rules, the rate card and the campaign flights are CONFIGURATION,
# not periodic data the channel uploads, so they are grouped separately.
# Advertiser rules are also editable directly in the Advertisers screen.
INPUTS: list[dict[str, str]] = [
    {"kind": "programmes", "label_en": "Programme lineup", "label_he": "לוח תוכניות", "cadence": "weekly"},
    {"kind": "daily", "label_en": "Daily ad log (Wally)", "label_he": "קובץ פרסומות יומי", "cadence": "daily"},
    {"kind": "spots", "label_en": "Historical spots", "label_he": "תשדירים היסטוריים", "cadence": "reference"},
    {"kind": "dayparts", "label_en": "Dayparts (ratings by time)", "label_he": "חלקי יום (רייטינג לפי שעה)", "cadence": "reference"},
    {"kind": "advertiser_rules", "label_en": "Advertiser rules", "label_he": "כללי מפרסמים", "cadence": "config"},
    {"kind": "rate_card", "label_en": "Rate card", "label_he": "כרטיס תעריפים", "cadence": "config"},
    {"kind": "campaign_flights", "label_en": "Campaign flights (delivery pacing)", "label_he": "קמפיינים ויעדי אספקה (קצב)", "cadence": "config"},
]


def _relative(path: Path) -> str:
    """A repo-relative display path, or the absolute path when outside the repo."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _destination(kind: str, filename: str | None = None) -> Path:
    if kind == "programmes":
        return DATA_DIR / "Programmes.csv"
    if kind == "spots":
        return DATA_DIR / "Spots.csv"
    if kind == "dayparts":
        return DATA_DIR / "Dayparts.csv"
    if kind == "advertiser_rules":
        return DATA_DIR / "advertiser_rules.csv"
    if kind == "rate_card":
        return DATA_DIR / "rate_card_premiums.csv"
    if kind == "campaign_flights":
        return DATA_DIR / "campaign_flights.csv"
    if kind == "daily":
        name = Path(filename).name if filename else ""
        if not name or not name.lower().endswith(".csv"):
            # Stamp the LOCAL broadcast date, not UTC: an evening upload in
            # Israel must not be filed under tomorrow's (or yesterday's) date.
            name = f"Wally_{datetime.now().astimezone().strftime('%Y-%m-%d')}.csv"
        # Every stored daily file must match the Wally_*.csv pattern the engine
        # resolver globs, otherwise the upload is saved but never read. Keep the
        # operator's original stem and normalize the prefix.
        if not name.startswith("Wally_"):
            if name.lower().startswith("wally_"):
                name = "Wally_" + name[len("Wally_"):]
            else:
                name = f"Wally_{name}"
        return DAILY_DIR / name
    raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")


_FILENAME_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _airing_date_from_name(path: Path) -> date | None:
    """The latest real ISO date named in the filename, or None when there is none."""
    found: list[date] = []
    for text in _FILENAME_DATE.findall(path.name):
        try:
            found.append(date.fromisoformat(text))
        except ValueError:
            continue
    return max(found) if found else None


def _newest_daily() -> Path | None:
    """The daily file the engine reads.

    Ordered by the airing date named in the filename (``Wally_..._2025-04-27.csv``)
    so re-uploading an OLDER day never displaces a newer day's plan just by having
    a fresher mtime. A file with no date in its name falls back to its mtime date,
    and the raw mtime breaks ties.
    """
    if not DAILY_DIR.exists():
        return None
    candidates = list(DAILY_DIR.glob("Wally_*.csv"))
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[date, float]:
        mtime = path.stat().st_mtime
        airing = _airing_date_from_name(path)
        return (airing or date.fromtimestamp(mtime), mtime)

    return max(candidates, key=sort_key)


def _live_path(kind: str) -> Path | None:
    if kind == "daily":
        return _newest_daily()
    return _destination(kind)


def _engine_reads(kind: str) -> str | None:
    """The input file the engine ACTUALLY reads for this kind, honest per state.

    For a shadowed kind this is the reference xlsx while it exists and the
    uploaded CSV once it is gone; for the rate card it is the config file the
    pricing engine really consumes; for the daily kind it is the newest daily
    file by airing date (None when none exists); for everything else the upload
    destination is the live input.
    """
    reference = SHADOWING_REFERENCE.get(kind)
    if reference is not None:
        return _relative(reference) if reference.exists() else _relative(_destination(kind))
    consumed = STORED_UNREAD.get(kind)
    if consumed is not None:
        return consumed
    if kind == "daily":
        live = _newest_daily()
        return _relative(live) if live is not None else None
    return _relative(_destination(kind))


def _read_header_and_rows(path: Path) -> tuple[list[str], int, list[str]]:
    """Cheaply read the CSV header and count data rows without loading values."""
    warnings: list[str] = []
    try:
        header_frame = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        columns = [str(column) for column in header_frame.columns]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return [], 0, [f"Could not parse CSV header: {exc}"]
    try:
        # Count rows by reading a single column; falls back to full read.
        usecol = [header_frame.columns[0]] if len(header_frame.columns) else None
        counted = pd.read_csv(path, encoding="utf-8-sig", usecols=usecol)
        rows = int(len(counted))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.append(f"Could not count rows precisely: {exc}")
        rows = 0
    return columns, rows, warnings


def _in_use(kind: str, saved_path: Path | None = None) -> tuple[bool, str]:
    """Whether the engine actually consumes an upload of this kind.

    The field is derived from the real read paths, never hardcoded optimism:

      * Most kinds land exactly where their consumer reads (the advertiser rules
        in data/advertiser_rules.csv, the campaign flights in
        data/campaign_flights.csv), so an upload genuinely takes effect: in_use
        is True with an empty reason.

      * The daily Wally kind is read through :func:`_newest_daily`, which picks
        ONE file from data/daily_input/. When ``saved_path`` is given (the file
        an upload just wrote), in_use is True only if the resolver actually
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
        engine code: the pricing engine takes its rate card from a different file.
        We report in_use False and name the file the engine really consumes.
    """
    reference = SHADOWING_REFERENCE.get(kind)
    if reference is not None:
        if reference.exists():
            relative = _relative(reference)
            return (
                False,
                f"Stored but not used by the optimizer: the engine reads {relative} "
                "first and adopts this upload only when that reference file is "
                "absent, so it is currently shadowed. Remove the reference file to "
                "make this upload the live optimizer input.",
            )
        # No reference file present: the loader now falls back to this upload.
        return True, ""

    consumed = STORED_UNREAD.get(kind)
    if consumed is not None:
        return (
            False,
            f"Stored, not yet read by the pricing engine: the rate card is read "
            f"from {consumed} (with the dashboard's pricing overrides), so this "
            "file is saved and validated but the optimizer does not consume it.",
        )

    if kind == "daily" and saved_path is not None:
        live = _newest_daily()
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


def _validate_columns(kind: str, columns: list[str]) -> list[str]:
    """Return the required columns that are missing from the header.

    Extra columns are always accepted: the loaders read the columns they need
    and pass the rest through. The channel's enriched exports legitimately
    carry many additional columns (TVR, computed premiums, per-channel ratings,
    and so on), so they are never "ignored" and we never warn about them.
    Only a genuinely MISSING required column is worth flagging, because that
    would actually break the optimizer.
    """
    required = REQUIRED_COLUMNS.get(kind, [])
    present = set(columns)
    return [column for column in required if column not in present]


# ---------------------------------------------------------------------------
# Data-contract validation of uploads (the real loaders, the real validators)
# ---------------------------------------------------------------------------

_CONTRACT_VALIDATORS: dict[str, Callable[[pd.DataFrame], contracts.ValidationReport]] = {
    "programmes": contracts.validate_programmes,
    "spots": contracts.validate_spots,
    "dayparts": contracts.validate_dayparts,
    "daily": contracts.validate_daily_input,
}

# The datetime column each loader derives; a file whose rows ALL fail to parse
# a date would silently yield zero engine segments, so it is refused instead.
_LOADED_DATE_COLUMN = {"programmes": "start_dt", "spots": "air_dt", "daily": "date"}


def _load_validation_reports() -> dict[str, Any]:
    try:
        payload = json.loads(VALIDATION_REPORTS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _store_validation_report(kind: str, report_payload: dict[str, Any]) -> None:
    reports = _load_validation_reports()
    reports[kind] = report_payload
    try:
        VALIDATION_REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = VALIDATION_REPORTS_PATH.with_suffix(VALIDATION_REPORTS_PATH.suffix + ".tmp")
        tmp_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, VALIDATION_REPORTS_PATH)
    except OSError:  # pragma: no cover - the report store must never block an upload
        logger.exception("Could not persist the upload validation report for %s", kind)


def _load_with_engine_loader(kind: str, raw: bytes) -> Any:
    """Parse the uploaded bytes with the REAL engine loader for this kind.

    The bytes are written verbatim to a temporary file so the exact production
    read path runs (encoding, date conventions, column melts and renames). The
    temporary file is always removed. Returns the loader's own result: a
    DataFrame for the four contract kinds, a list of Campaign for the flights.
    """
    handle = tempfile.NamedTemporaryFile(prefix=f"kairos_upload_{kind}_", suffix=".csv", delete=False)
    try:
        handle.write(raw)
        handle.close()
        path = Path(handle.name)
        if kind == "programmes":
            return load_programmes(path)
        if kind == "spots":
            return load_spots(path)
        if kind == "dayparts":
            return load_dayparts(path)
        if kind == "daily":
            return load_daily_input(path)
        if kind == "campaign_flights":
            return load_campaigns(path)
        raise ValueError(f"no engine loader for kind {kind!r}")
    finally:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _campaign_flights_report(loaded: list[Any], raw_frame: pd.DataFrame) -> tuple[contracts.ValidationReport, int]:
    """Row-level honesty for campaign flights: what the pacing loader will read.

    A header-only file is legitimate (pacing stays an exact identity no-op), but
    a file whose rows ALL fail the loader's requirements would silently leave
    pacing inactive while looking uploaded, so that is an error. Partially
    skipped rows are surfaced as a warning with the real count.
    """
    report = contracts.ValidationReport("campaign_flights")
    data_rows = int(len(raw_frame))
    loaded_count = int(len(loaded))
    if data_rows and not loaded_count:
        report.add(
            "campaign_id",
            "no_loadable_campaigns",
            f"none of the {data_rows} row(s) carries a campaign_id, parseable flight_start/flight_end dates and a positive target, so the pacing engine would read zero campaigns from this file",
            "error",
        )
    elif loaded_count < data_rows:
        report.add(
            "campaign_id",
            "skipped_campaign_rows",
            f"{data_rows - loaded_count} of {data_rows} row(s) will be skipped by the pacing loader (missing campaign_id, flight dates, or a positive target)",
            "warning",
        )
    return report, loaded_count


def _dayparts_empty_finding(raw_frame: pd.DataFrame, report: contracts.ValidationReport) -> None:
    """Explain, in headers, why a dayparts upload melts to zero audience rows.

    The header gate only requires Dates+Timebands, but the loader melts ONLY the
    known channel columns; a file with renamed channel columns validates green
    yet yields nothing. Name the recognized set and the unrecognized headers so
    the operator can fix the export instead of chasing a silent empty model.
    """
    recognized = [c for c in CHANNELS if c in raw_frame.columns]
    unrecognized = [str(c) for c in raw_frame.columns if str(c) not in CHANNELS and str(c) not in ("Dates", "Timebands")]
    if not recognized:
        report.add(
            "channels",
            "no_recognized_channel_columns",
            f"the file yields zero audience rows because no known channel column is present; the loader recognizes {', '.join(CHANNELS)}, and the unrecognized columns found were {', '.join(unrecognized) if unrecognized else 'none'}",
            "error",
        )
    else:
        report.add(
            "<frame>",
            "no_data_rows",
            "the file yields zero audience rows: the recognized channel columns are present but carry no data rows",
            "error",
        )


def _run_contract_validation(
    kind: str, raw: bytes, raw_frame: pd.DataFrame, filename: str
) -> tuple[dict[str, Any] | None, list[str], JSONResponse | None]:
    """Parse the upload with the real loader and validate the loaded frame.

    Returns ``(report_payload, warnings, rejection)``: the JSON-safe record for
    the status endpoint (None when the kind has no loader-backed contract), the
    operator-facing warning strings, and a ready 400 response when the file
    must not replace the live input (error-severity contract violations).
    """
    if kind not in _CONTRACT_VALIDATORS and kind != "campaign_flights":
        return None, [], None

    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        loaded = _load_with_engine_loader(kind, raw)
    except Exception as exc:  # noqa: BLE001 - any loader failure is a client-input problem
        logger.warning("Upload validation: the %s loader failed on %r: %s", kind, filename, exc)
        payload = {
            "dataset": kind,
            "filename": filename,
            "checked_at": checked_at,
            "accepted": False,
            "is_valid": False,
            "errors": [_GENERIC_PARSE_ERROR],
            "warnings": [],
            "rows_loaded": 0,
        }
        return payload, [], _reject(_GENERIC_PARSE_ERROR)

    if kind == "campaign_flights":
        report, rows_loaded = _campaign_flights_report(loaded, raw_frame)
    else:
        report = _CONTRACT_VALIDATORS[kind](loaded)
        rows_loaded = int(len(loaded))
        date_column = _LOADED_DATE_COLUMN.get(kind)
        if date_column is not None and rows_loaded and date_column in loaded.columns:
            if int(loaded[date_column].notna().sum()) == 0:
                report.add(
                    date_column,
                    "no_parseable_dates",
                    f"none of the {rows_loaded} row(s) has a parseable date, so the engine would build zero segments from this file",
                    "error",
                )
        if kind == "daily":
            raw_dates = raw_frame.get("תאריך", raw_frame.get("date"))
            if raw_dates is not None:
                ambiguous = count_ambiguous_daily_dates(raw_dates)
                if ambiguous:
                    report.add(
                        "date",
                        "ambiguous_day_month",
                        f"{ambiguous} row(s) have a day/month-ambiguous slash date; the loader reads them month-first (M/D/YYYY)",
                        "warning",
                    )
        if kind == "dayparts" and rows_loaded == 0:
            _dayparts_empty_finding(raw_frame, report)

    accepted = report.is_valid
    payload = {
        "dataset": report.dataset,
        "filename": filename,
        "checked_at": checked_at,
        "accepted": accepted,
        "is_valid": report.is_valid,
        "errors": [str(v) for v in report.errors],
        "warnings": [str(v) for v in report.warnings],
        "rows_loaded": rows_loaded,
    }
    warnings = [str(v) for v in report.warnings]
    if not accepted:
        detail = (
            f"Upload rejected by the {report.dataset} data contract: "
            + "; ".join(str(v) for v in report.errors[:3])
        )
        return payload, warnings, _reject(detail, [str(v) for v in report.errors])
    return payload, warnings, None


@router.get("/status")
def upload_status() -> dict[str, Any]:
    """Report the live state of every input file the optimizer depends on."""
    reports = _load_validation_reports()
    inputs: list[dict[str, Any]] = []
    for meta in INPUTS:
        kind = meta["kind"]
        path = _live_path(kind)
        exists = bool(path and path.exists())
        in_use, in_use_reason = _in_use(kind)
        entry: dict[str, Any] = {
            "kind": kind,
            "label_en": meta["label_en"],
            "label_he": meta["label_he"],
            "cadence": meta["cadence"],
            "filename": path.name if path else _destination(kind).name,
            "path": _relative(path or _destination(kind)),
            "exists": exists,
            "rows": 0,
            "columns": [],
            "last_modified": None,
            "valid": False,
            "in_use": in_use,
            "in_use_reason": in_use_reason,
            # The file the engine ACTUALLY reads for this kind right now, so a
            # shadowed or unread upload never has to be inferred from prose.
            "engine_reads": _engine_reads(kind),
            # The last data-contract report from an upload of this kind, or
            # None when no upload has been validated yet (honest unknown).
            "last_validation": reports.get(kind),
            "warnings": [],
        }
        if exists and path is not None:
            columns, rows, read_warnings = _read_header_and_rows(path)
            missing = _validate_columns(kind, columns)
            entry["columns"] = columns
            entry["rows"] = rows
            # Local time WITH the UTC offset, so "when" is unambiguous.
            entry["last_modified"] = (
                datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
            )
            entry["valid"] = not missing
            warnings = list(read_warnings)
            if missing:
                warnings.insert(0, f"Missing required columns: {', '.join(missing)}")
            if not in_use and in_use_reason:
                # Surface the shadow state in warnings too, so a dashboard that
                # only renders warnings still stops short of a bare green badge.
                warnings.insert(0, in_use_reason)
            entry["warnings"] = warnings
        inputs.append(entry)
    return {"inputs": inputs}


def _backup(destination: Path, kind: str) -> str | None:
    if not destination.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_path = BACKUP_DIR / f"{kind}_{stamp}.csv"
    shutil.copy2(destination, backup_path)
    return _relative(backup_path)


def _reject(message: str, errors: list[str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": message, "errors": errors or [message], "valid": False},
    )


def _too_large_message() -> str:
    return f"Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit"


@router.post("/{kind}")
async def upload_file(kind: str, request: Request, file: UploadFile = File(...)) -> Any:
    """Validate and persist an uploaded CSV for the given input kind."""
    if kind not in REQUIRED_COLUMNS:
        raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")

    # Size gates: refuse an honestly-declared oversize body before reading it,
    # then stream in chunks with a hard cap so the body never has to be trusted.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + _CHUNK:
        return _reject(_too_large_message())
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        received += len(chunk)
        if received > MAX_UPLOAD_BYTES:
            return _reject(_too_large_message())
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        return _reject("Uploaded file is empty")

    # Parse with pandas to validate it is a real, readable CSV. The parser's
    # own message is logged server-side; the client gets a generic line.
    try:
        frame = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("Upload parse failed for kind %s (%r): %s", kind, file.filename, exc)
        return _reject(_GENERIC_PARSE_ERROR)

    columns = [str(column) for column in frame.columns]
    missing = _validate_columns(kind, columns)
    if missing:
        # Reject without touching the live file.
        message = f"Missing required columns for '{kind}': {', '.join(missing)}"
        return _reject(message, [f"Missing required column: {column}" for column in missing])

    # Run the kind's REAL loader and data-contract validator over the upload.
    # Error-severity violations refuse the file before it can replace the live
    # input; warnings ride along in the response and the stored report.
    report_payload, contract_warnings, rejection = _run_contract_validation(
        kind, raw, frame, str(file.filename or "")
    )
    if report_payload is not None:
        _store_validation_report(kind, report_payload)
    if rejection is not None:
        return rejection

    warnings: list[str] = list(contract_warnings)
    destination = _destination(kind, file.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backed_up = _backup(destination, kind)

    # Write the raw bytes verbatim so encoding and content are preserved exactly
    # as uploaded, via a sibling temp file and an atomic rename so a concurrent
    # engine read never sees a half-written live input. The mtime change busts
    # the loaders' mtime-keyed CSV cache automatically.
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    tmp_path.write_bytes(raw)
    os.replace(tmp_path, destination)

    if backed_up:
        warnings.append(f"Previous file backed up to {backed_up}")

    in_use, in_use_reason = _in_use(kind, saved_path=destination)
    if not in_use and in_use_reason:
        # The file saved and parsed, but the optimizer will not read it. Say so
        # in the response so the post-upload confirmation does not imply an
        # ingestion that did not happen.
        warnings.insert(0, in_use_reason)

    return {
        "kind": kind,
        "saved_path": _relative(destination),
        "rows": int(len(frame)),
        "columns": columns,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "in_use": in_use,
        "in_use_reason": in_use_reason,
        "engine_reads": _engine_reads(kind),
        "validation": report_payload,
        "warnings": warnings,
    }
