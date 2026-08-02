"""Contract validation for an uploaded file, split out of ``uploads.py``.

Split under the file-size cap, and named by the ``<parent stem>_<role>.py``
rule the package already follows (``events_access.py`` says the same thing in
its own docstring). Nothing here resolves a path or writes a file: every
function takes the bytes it is given and returns findings, so the caller keeps
every writable path in one module and a test that relocates those paths keeps
working.

The refusal is the point. An accepted upload is parsed with the REAL engine
loader for its kind and checked against :mod:`kairos.data.contracts`, so a file
that would break or silently empty the optimizer is refused before it can
replace the live input, with the contract's own findings attached.

Findings travel in two shapes on purpose. ``errors`` and ``warnings`` stay the
flat strings every existing reader already parses, and ``findings`` carries the
same violations as records (column, code, message, severity) so a surface can
render a row per finding instead of splitting a sentence.

A refusal also names the rows. A column and a count leave a steward searching a
175-row file by hand, so every violation that is about cells carries the row
numbers those cells are on, recomputed from the same frame the contract read.

A refusal never names a channel this operator does not own. It is a sentence
rendered verbatim on an operator screen, so it goes through
:mod:`kairos_api.uploads_channels` exactly like every other name this
destination prints back.

A refusal is also read in Hebrew, so every sentence this module writes itself is
written in both languages: the words live in :mod:`kairos_api.uploads_messages`
and each record carries ``message_he`` beside ``message``. A violation the frozen
contracts raised keeps its own English detail as the fallback, because the counts
and column names inside it are theirs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi.responses import JSONResponse

from kairos_api import uploads_channels, uploads_messages, uploads_replay
from kairos.data import contracts
from kairos.data.loaders import (
    CHANNELS,
    count_ambiguous_daily_dates,
    load_daily_input,
    load_dayparts,
    load_programmes,
    load_spots,
)
from kairos.optimize.pacing import load_campaigns

logger = logging.getLogger(__name__)

# Client-facing parse failure line. The pandas exception detail (offsets, C
# parser internals) is logged server-side instead of echoed to the client. The
# words are in the copy table with every other sentence this module writes, so
# the English a reader gets and the Hebrew beside it cannot drift apart.
GENERIC_PARSE_ERROR = uploads_messages.say("unreadable_file")[0]

CONTRACT_VALIDATORS: dict[str, Callable[[pd.DataFrame], contracts.ValidationReport]] = {
    "programmes": contracts.validate_programmes,
    "spots": contracts.validate_spots,
    "dayparts": contracts.validate_dayparts,
    "daily": contracts.validate_daily_input,
}

# The datetime column each loader derives. A file whose rows ALL fail to parse a
# date would silently yield zero engine segments; a file where only SOME fail is
# the realistic morning case, and those rows reach the engine dateless. Both are
# refused, with the rows named.
LOADED_DATE_COLUMN = {"programmes": "start_dt", "spots": "air_dt", "daily": "date"}

# The clock column of the daily file, and what the engine gets from it: the
# spot's daypart (kairos.export.spots) and its minute for separation
# (kairos.optimize.frequency). Measured on "99:99:99": both come back None.
DAILY_CLOCK_COLUMN = "spot_time"

# How the daily loader reads a slash date, stated in the refusal so an operator
# can tell whether 3/4 was read as the third of April or the fourth of March.
DAILY_DATE_PATTERN = "M/D/YYYY"

# The most row numbers one finding lists by name. The message carries the true
# count in words and ``rows_total`` carries it as a number, so a file whose every
# row is broken says how many without turning a refusal into a data dump.
ROW_LIST_CAP = 25

# The kinds whose loader returns exactly one row per uploaded data row, so a
# position in the loaded frame IS that row's number in the file. Measured: each
# of these reads the CSV, adds columns and resets the index without dropping a
# row. ``load_dayparts`` melts one row per channel column, so its positions are
# not file rows and no dayparts finding carries one.
ROW_ALIGNED_KINDS = frozenset({"programmes", "spots", "daily"})


# The 400 itself is shaped where its words are, so a refusal cannot leave with
# one language: ``reject`` takes a sentence, ``refuse`` takes a code.
reject = uploads_messages.reject
refuse = uploads_messages.refuse


def add_finding(report: contracts.ValidationReport, authored: dict[tuple[str, str], dict[str, Any]], column: str, code: str, severity: str, key: str | None = None, boundary: dict[str, Any] | None = None, **fields: object) -> None:
    """One violation this module raises itself, recorded in both languages.

    The report is the frozen contract's own object and it carries one detail, so
    the Hebrew half travels beside it keyed by the column and code it belongs to,
    and :func:`finding_records` puts the two back together. ``key`` is the copy
    table's entry when one code has two situations to say.

    What the sentence was made of travels with it, because the stored report
    keeps the fields and not the sentence. ``boundary`` is the raw material of
    the one sentence whose fields depend on the operator channel, and it stands
    in place of those fields rather than beside them.
    """
    english, translated = uploads_messages.say(key or code, **fields)
    report.add(column, code, english, severity)
    authored[(str(column), str(code))] = {"key": key or code, "fields": dict(fields), "boundary": boundary, "message_he": translated}


def load_reports(path: Path) -> dict[str, Any]:
    """Every stored per-kind report, or an empty record when there is none.

    The path is passed in rather than held here, so the module that owns the
    writable locations keeps owning them.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def store_report(path: Path, kind: str, report_payload: dict[str, Any]) -> None:
    """Record this kind's latest report as codes, never blocking an upload.

    What lands on disk is :func:`uploads_replay.to_store`'s form: each finding's
    code, the copy table's key and the measured fields, and never the sentence
    itself. A stored sentence freezes the operator channel configured when it was
    written, and the account that reads it back may own a different one.
    """
    reports = load_reports(path)
    reports[kind] = uploads_replay.to_store(report_payload)
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, target)
    except (OSError, TypeError, ValueError):  # pragma: no cover - never block an upload
        logger.exception("Could not persist the upload validation report for %s", kind)


def unreadable_times(values: pd.Series) -> pd.Series:
    """The rows whose clock cell the loader can read nothing from.

    Mirrors the loader's own two-step parse (the fixed ``HH:MM:SS`` first, the
    general parser for the Excel-serial leftovers), so a row marked here is a row
    the engine really gets no clock from. An empty cell is not marked: it is
    missing, not unreadable, and that is a different sentence.
    """
    text_values = values.astype(str).str.strip()
    parsed = pd.to_datetime(text_values, format="%H:%M:%S", errors="coerce")
    leftover = parsed.isna() & text_values.str.contains(":", na=False)
    if leftover.any():
        parsed = parsed.where(~leftover, pd.to_datetime(text_values.where(leftover), errors="coerce"))
    return parsed.isna() & values.notna() & text_values.ne("")


def violation_mask(violation: contracts.Violation, frame: pd.DataFrame) -> pd.Series | None:
    """The rows one violation is about, recomputed on the frame it was raised on.

    Same rule, same frame, so a listed row number is that violation's row rather
    than a guess. A violation about the header or about the frame as a whole (a
    missing column, a wrong dtype, an empty file) is about no row and returns
    None, and so does a code with no cell-level rule to re-run.
    """
    column = str(violation.field)
    code = str(violation.code)
    if code == "end_before_start" and {"start_dt", "end_dt"} <= set(frame.columns):
        both = frame["start_dt"].notna() & frame["end_dt"].notna()
        return both & (frame["end_dt"] < frame["start_dt"])
    if column not in frame.columns:
        return None
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    if code == "non_numeric_values":
        return values.notna() & numeric.isna()
    if code == "nan_values":
        return numeric.isna()
    if code == "non_positive_values":
        return numeric <= 0
    if code == "negative_values":
        return numeric < 0
    if code in ("no_parseable_dates", "unparseable_dates", "nan_channel"):
        return values.isna()
    if code == "unreadable_times":
        return unreadable_times(values)
    if code == "unknown_channel":
        return values.notna() & ~values.astype(str).isin(CHANNELS)
    return None


def row_source(kind: str, loaded: Any, raw_frame: pd.DataFrame) -> pd.DataFrame | None:
    """The frame a finding's row numbers may be read off, or None.

    Only a loader that returns exactly one row per uploaded data row can turn a
    position into a row number in the operator's file, and the equal-length check
    proves it for THIS file instead of trusting the list of kinds.
    """
    if kind not in ROW_ALIGNED_KINDS or not isinstance(loaded, pd.DataFrame):
        return None
    if len(loaded) != len(raw_frame):
        return None
    return loaded


def finding_records(report: contracts.ValidationReport, frame: pd.DataFrame | None = None, authored: dict[tuple[str, str], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every violation as a record, so a surface renders rows, not sentences.

    ``message_he`` is present on every violation this module wrote itself and
    absent on the ones the frozen contracts wrote, whose English detail carries
    counts and column names only that code can compute. A surface falls back to
    ``message``, so the missing half is an English sentence rather than a blank.

    A violation this module wrote also carries ``key`` and ``fields``, which are
    the copy table's entry and the measured numbers the two sentences were
    rendered from, and ``boundary`` when the sentence names a channel. They are
    what the stored report keeps in place of the sentence, so a reader with a
    different operator channel gets the reason rendered against their own.

    ``rows`` carries at most :data:`ROW_LIST_CAP` row numbers, counting the data
    rows of the uploaded file from 1 with the header excluded, and ``rows_total``
    how many rows the violation touches in all, so a truncated list can say so.
    Both are absent, never empty or zero, when the violation is about no row or
    when the loader does not return one row per uploaded row: a row number that
    is not the operator's row number is worse than no row number.
    """
    records: list[dict[str, Any]] = []
    for violation in report.violations:
        record: dict[str, Any] = {
            "column": str(violation.field),
            "code": str(violation.code),
            "message": str(violation.detail),
            "severity": str(violation.severity),
        }
        source = (authored or {}).get((record["column"], record["code"])) or {}
        if source.get("message_he"):
            record["message_he"] = source["message_he"]
        if source:
            record["key"] = source["key"]
            record["fields"] = source["fields"]
            if source.get("boundary") is not None:
                record["boundary"] = source["boundary"]
        mask = violation_mask(violation, frame) if frame is not None else None
        if mask is not None:
            positions = mask.fillna(False).to_numpy(dtype=bool).nonzero()[0]
            if positions.size:
                record["rows"] = [int(position) + 1 for position in positions[:ROW_LIST_CAP]]
                record["rows_total"] = int(positions.size)
        records.append(record)
    return records


def load_with_engine_loader(kind: str, raw: bytes) -> Any:
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


def campaign_flights_report(loaded: list[Any], raw_frame: pd.DataFrame, authored: dict[tuple[str, str], dict[str, Any]]) -> tuple[contracts.ValidationReport, int]:
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
        required = uploads_messages.say("campaign_requirements")
        add_finding(report, authored, "campaign_id", "no_loadable_campaigns", "error", rows=data_rows, columns=required)
    elif loaded_count < data_rows:
        fields = uploads_messages.say("campaign_fields")
        skipped = data_rows - loaded_count
        add_finding(report, authored, "campaign_id", "skipped_campaign_rows", "warning", skipped=skipped, rows=data_rows, columns=fields)
    return report, loaded_count


def dayparts_empty_finding(raw_frame: pd.DataFrame, report: contracts.ValidationReport, authored: dict[tuple[str, str], dict[str, Any]], owned: str | None = None) -> None:
    """Explain, in headers, why a dayparts upload melts to zero audience rows.

    The header gate only requires Dates+Timebands, but the loader melts ONLY the
    known channel columns; a file with renamed channel columns validates green
    yet yields nothing. Name the operator's own channel and the unrecognized
    headers, so the export can be fixed instead of a silent empty model chased.

    **Only the operator's own channel is named**, and that is the previous
    round's correction. Measured before it, with the operator channel set to
    ``רשת 13``: this message listed all four channels the loader knows, and the
    card renders a finding's message verbatim in its red refusal panel, so a
    plausibly re-exported dayparts file put three rival channel names on the
    operator's own screen. The same sentence reaches the assistant, which reads
    the stored validation report through ``get_upload_status``. The unrecognized
    headers are still listed, minus any that carry a channel this account does
    not own, and that count is stated instead.

    **How many names the loader knows is now stated**, and that is this round's
    correction. Withholding a name is not a licence to withhold the contract: a
    person cannot fix a refused export if the product will not say what it
    accepts, and this refusal only fires when the operator's own column was
    renamed too, so the accepted shape is the actionable part. A count names
    nobody, which is the same disclosure the withheld column names already take,
    so the message answers "what would you accept" with the matching rule, the
    size of the recognized set, and the one name this account may read.

    **The sentence is not what is stored**, and that is this round's correction:
    the headers this refusal may list and the count of the ones it may not go to
    disk as :func:`uploads_replay.boundary`'s record, and both the wording and
    the arithmetic are derived again on every read against the channel the
    account reading it owns then.
    """
    if [c for c in CHANNELS if c in raw_frame.columns]:
        add_finding(report, authored, "<frame>", "no_data_rows", "error")
        return
    owned = uploads_channels.owned_channel() if owned is None else str(owned or "").strip()
    bound = uploads_replay.boundary(
        [c for c in raw_frame.columns if str(c) not in CHANNELS and str(c) not in ("Dates", "Timebands")],
        owned,
    )
    key = uploads_replay.channel_key(owned)
    add_finding(report, authored, "channels", "no_recognized_channel_columns", "error", key, bound, **uploads_replay.channel_fields(bound, owned))


def run_contract_validation(
    kind: str, raw: bytes, raw_frame: pd.DataFrame, filename: str
) -> tuple[dict[str, Any] | None, list[str], JSONResponse | None]:
    """Parse the upload with the real loader and validate the loaded frame.

    Returns ``(report_payload, warnings, rejection)``: the JSON-safe record for
    the status endpoint (None when the kind has no loader-backed contract), the
    operator-facing warning strings, and a ready 400 response when the file
    must not replace the live input (error-severity contract violations).
    """
    if kind not in CONTRACT_VALIDATORS and kind != "campaign_flights":
        return None, [], None

    checked_at = datetime.now(timezone.utc).isoformat()
    authored: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        loaded = load_with_engine_loader(kind, raw)
    except Exception as exc:  # noqa: BLE001 - any loader failure is a client-input problem
        logger.warning("Upload validation: the %s loader failed on %r: %s", kind, filename, exc)
        payload = {
            "dataset": kind,
            "filename": filename,
            "checked_at": checked_at,
            "accepted": False,
            "is_valid": False,
            "errors": [GENERIC_PARSE_ERROR],
            "warnings": [],
            "findings": [uploads_messages.record("unreadable_file", "<file>", "error")],
            "rows_loaded": 0,
        }
        return payload, [], refuse("unreadable_file")

    if kind == "campaign_flights":
        report, rows_loaded = campaign_flights_report(loaded, raw_frame, authored)
    else:
        report = CONTRACT_VALIDATORS[kind](loaded)
        rows_loaded = int(len(loaded))
        date_column = LOADED_DATE_COLUMN.get(kind)
        if date_column is not None and rows_loaded and date_column in loaded.columns:
            unparseable = int(loaded[date_column].isna().sum())
            if unparseable == rows_loaded:
                add_finding(report, authored, date_column, "no_parseable_dates", "error", rows=rows_loaded)
            elif unparseable:
                add_finding(report, authored, date_column, "unparseable_dates", "error", unreadable=unparseable, rows=rows_loaded)
        if kind == "daily" and DAILY_CLOCK_COLUMN in loaded.columns:
            unreadable = unreadable_times(loaded[DAILY_CLOCK_COLUMN])
            unreadable_count = int(unreadable.sum())
            if unreadable_count:
                add_finding(report, authored, DAILY_CLOCK_COLUMN, "unreadable_times", "warning", unreadable=unreadable_count, rows=rows_loaded)
        if kind == "daily":
            raw_dates = raw_frame.get("תאריך", raw_frame.get("date"))
            if raw_dates is not None:
                ambiguous = count_ambiguous_daily_dates(raw_dates)
                if ambiguous:
                    add_finding(report, authored, "date", "ambiguous_day_month", "warning", ambiguous=ambiguous, rows=int(len(raw_dates)), pattern=DAILY_DATE_PATTERN)
        if kind == "dayparts" and rows_loaded == 0:
            dayparts_empty_finding(raw_frame, report, authored)

    accepted = report.is_valid
    payload = {
        "dataset": report.dataset,
        "filename": filename,
        "checked_at": checked_at,
        "accepted": accepted,
        "is_valid": report.is_valid,
        "errors": [str(v) for v in report.errors],
        "warnings": [str(v) for v in report.warnings],
        "findings": finding_records(report, row_source(kind, loaded, raw_frame), authored),
        "rows_loaded": rows_loaded,
    }
    warnings = [str(v) for v in report.warnings]
    if not accepted:
        # The Hebrew half of the headline reads the same three reasons, taking
        # each violation's Hebrew where this module wrote it and its English
        # detail where the frozen contract did, which is the same fallback the
        # findings take one line above.
        first = report.errors[:3]
        reasons = ("; ".join(str(v) for v in first), "; ".join((authored.get((str(v.field), str(v.code))) or {}).get("message_he") or str(v.detail) for v in first))
        detail, detail_he = uploads_messages.say("contract_refusal", dataset=report.dataset, reasons=reasons)
        return payload, warnings, reject(detail, [str(v) for v in report.errors], detail_he, payload["findings"])
    return payload, warnings, None
