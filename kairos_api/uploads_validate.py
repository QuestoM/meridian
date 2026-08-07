"""Contract validation for an uploaded file, split out of ``uploads.py``.

Split under the file-size cap, and named by the ``<parent stem>_<role>.py``
rule the package already follows. Nothing here resolves a path or writes a
file: every function takes the bytes it is given and returns findings, so the
caller keeps every writable path in one module.

The refusal is the point. An accepted upload is parsed with the REAL engine
loader for its kind and checked against :mod:`kairos.data.contracts`, so a file
that would break or silently empty the optimizer is refused before it can
replace the live input, with the contract's own findings attached.

Findings travel in two shapes on purpose. ``errors`` and ``warnings`` stay the
flat strings every existing reader already parses, and ``findings`` carries the
same violations as records so a surface can render a row per finding. A refusal
also names the rows: a column and a count leave a steward searching a 175-row
file by hand, so every violation about cells carries those cells' row numbers.

A file with a header and no data rows is the one refusal that has to be asked
about every kind rather than about a loader, so the whole of that rule, and the
two answers it has, is :mod:`kairos_api.uploads_empty`.

A refusal never names a channel this operator does not own. A header list goes
through :mod:`kairos_api.uploads_channels`; every finding's sentence goes through
:func:`kairos_api.uploads_replay.at_the_door`, so every half of it passes one
boundary for a screen and for :func:`store_report`. It never names a column the
operator's file does not have either: every violation is raised on the LOADED
frame, whose names are renamed or computed, so :func:`finding_records` resolves
each one against the candidate's own header row through
:mod:`kairos_api.uploads_columns`, once, where the record is made.

A refusal is read in two languages and neither may be the poor relation. Every
sentence this module writes itself is written in both, from
:mod:`kairos_api.uploads_messages`, and carries ``message_he`` beside
``message``. A violation the frozen contracts raised keeps its own English detail
verbatim in ``message``, where the machine record needs it, and carries the pair
a person reads in ``message_en`` and ``message_he``, authored by
:func:`finding_records` from ONE
:func:`kairos_api.uploads_messages.contract_say` call, so the locale that used to
be handed the contract's ``repr`` of a Python list cannot fall behind again."""

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

from kairos_api import uploads_channels, uploads_columns, uploads_empty, uploads_messages, uploads_replay
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
# refused, with the rows named. Every name here is the LOADER's, so it is the
# header row of the file in hand that gets named on screen, never these.
LOADED_DATE_COLUMN = {"programmes": "start_dt", "spots": "air_dt", "daily": "date"}

# The clock column of the daily file, and what the engine gets from it: the
# spot's daypart (kairos.export.spots) and its minute for separation
# (kairos.optimize.frequency). Measured on "99:99:99": both come back None.
DAILY_CLOCK_COLUMN = "spot_time"

# How the daily loader reads a slash date, stated in the refusal so an operator
# can tell whether 3/4 was read as the third of April or the fourth of March.
DAILY_DATE_PATTERN = "M/D/YYYY"

# The most row numbers one finding lists by name, so a file whose every row is
# broken says how many in words and in ``rows_total`` rather than dumping them.
ROW_LIST_CAP = 25

# The kinds whose loader returns exactly one row per uploaded data row, so a
# position in the loaded frame IS that row's number in the file. Measured: each
# reads the CSV, adds columns and resets the index without dropping a row.
# ``load_dayparts`` melts one row per channel column, so its positions are not
# file rows and no dayparts finding carries one.
ROW_ALIGNED_KINDS = frozenset({"programmes", "spots", "daily"})


# The 400 itself is shaped where its words are, so a refusal cannot leave in one
# language. ``reject`` takes a sentence and ``refuse`` takes a code.
reject = uploads_messages.reject
refuse = uploads_messages.refuse


def add_finding(report: contracts.ValidationReport, authored: dict[tuple[str, str], dict[str, Any]], column: str, code: str, severity: str, key: str | None = None, boundary: dict[str, Any] | None = None, scope: str = "", **fields: object) -> None:
    """One violation this module raises itself, recorded in both languages.

    The report is the frozen contract's own object and it carries one detail, so
    the Hebrew half travels beside it keyed by the column and code it belongs to,
    and :func:`finding_records` puts the two back together. ``key`` is the copy
    table's entry when one code has two situations to say. ``boundary`` is the
    raw material of the one sentence whose fields depend on the operator channel,
    standing in for those fields rather than beside them. ``scope`` is what a
    finding about no column is about. ``column`` is the loaded frame's own name,
    resolved to the operator's own header by :func:`finding_records`."""
    english, translated = uploads_messages.say(key or code, **fields)
    report.add(column, code, english, severity)
    authored[(str(column), str(code))] = {"key": key or code, "fields": dict(fields), "boundary": boundary, "scope": scope, "message_he": translated}


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
    code, key and measured fields, never the sentence, for that module's reason.
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
    missing, not unreadable, and that is a different sentence."""
    text_values = values.astype(str).str.strip()
    parsed = pd.to_datetime(text_values, format="%H:%M:%S", errors="coerce")
    leftover = parsed.isna() & text_values.str.contains(":", na=False)
    if leftover.any():
        parsed = parsed.where(~leftover, pd.to_datetime(text_values.where(leftover), errors="coerce"))
    return parsed.isna() & values.notna() & text_values.ne("")


def violation_mask(violation: contracts.Violation, frame: pd.DataFrame) -> pd.Series | None:
    """The rows one violation is about, recomputed on the frame it was raised on.

    Same rule, same frame, so a listed row number is that violation's row rather
    than a guess. A violation about the header or about the frame as a whole is
    about no row and returns None, and so does a code with no cell-level rule to
    re-run. The field is the loaded frame's own name, which keys this frame."""
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


def finding_records(report: contracts.ValidationReport, frame: pd.DataFrame | None = None, authored: dict[tuple[str, str], dict[str, Any]] | None = None, headers: Any = (), kind: str = "", loaded: Any = None) -> list[dict[str, Any]]:
    """Every violation as a record, so a surface renders rows, not sentences.

    ``headers`` is the candidate's own header row, and this is the one place a
    column name is resolved against it: every violation, this module's and the
    frozen contracts', is raised on the LOADED frame, whose names are renamed or
    computed and are not what the operator's export says at the top. ``kind``
    decides the one case where an absent name is still the right word.

    A violation this module wrote carries ``message_he`` beside its ``message``,
    and also ``key``, ``fields`` and ``boundary`` when the sentence names a
    channel, so a stored report can re-render it against a different reader's
    channel. One the frozen contracts wrote carries ``message_en`` and
    ``message_he``, both authored below from that code's own quantity on
    ``frame`` or, when that is None, on ``loaded`` itself (always the frame the
    contract validated), while the contract's own sentence stays theirs,
    verbatim, in ``message``. ``scope`` says what a finding about no column is
    about; ``effect`` says what a warning cost the engine.

    ``rows`` carries at most :data:`ROW_LIST_CAP` row numbers of the uploaded
    file, counted from 1, and ``rows_total`` how many in all; both stay absent
    when the loader breaks the one-row-per-row promise, gated on ``frame`` alone."""
    records: list[dict[str, Any]] = []
    for violation in report.violations:
        column = str(violation.field)
        code = str(violation.code)
        record: dict[str, Any] = {
            "column": column,
            "code": code,
            "message": str(violation.detail),
            "severity": str(violation.severity),
            **uploads_messages.effect_of(code, str(violation.severity)),
        }
        source = (authored or {}).get((column, code)) or {}
        counted_on = frame if frame is not None else (loaded if isinstance(loaded, pd.DataFrame) else None)
        mask = violation_mask(violation, counted_on) if counted_on is not None else None
        if frame is not None and mask is not None:
            positions = mask.fillna(False).to_numpy(dtype=bool).nonzero()[0]
            if positions.size:
                record["rows"] = [int(position) + 1 for position in positions[:ROW_LIST_CAP]]
                record["rows_total"] = int(positions.size)
        if source.get("message_he"):
            record["message_he"] = source["message_he"]
        elif not source:
            # Both halves of a frozen finding, from one call: distinct names for
            # unknown_channel, a value count otherwise. They arrive together or
            # not at all, so one locale cannot be left with the contract's own.
            if code == "unknown_channel":
                names = uploads_messages.unknown_channel_names(counted_on, column, mask)
                count: int | None = uploads_messages.unknown_channel_count(counted_on, column, mask)
            else:
                names, count = "", (int(mask.fillna(False).sum()) if mask is not None else None)
            english, hebrew = uploads_messages.contract_say(code, count, names)
            if english and hebrew:
                record["message_en"], record["message_he"] = english, hebrew
        if source:
            record["key"] = source["key"]
            record["fields"] = source["fields"]
            record.update({name: source[name] for name in ("boundary", "scope") if source.get(name)})
        # The rows are found by the loaded name and the person reads the header,
        # so the name is swapped for theirs once the rows are counted. A finding
        # that names no column already carries what it IS about and is left alone.
        if record["column"]:
            record["column"], scope = uploads_columns.place(record["column"], headers, kind)
            if scope:
                record["scope"] = scope
        records.append(record)
    return records


def load_with_engine_loader(kind: str, raw: bytes) -> Any:
    """Parse the uploaded bytes with the REAL engine loader for this kind.

    The bytes are written verbatim to a temporary file so the exact production
    read path runs (encoding, date conventions, column melts and renames). The
    temporary file is always removed. Returns the loader's own result: a frame
    for the four contract kinds, a list of Campaign for the flights."""
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
    skipped rows are a warning with the real count."""
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


def run_contract_validation(
    kind: str, raw: bytes, raw_frame: pd.DataFrame, filename: str
) -> tuple[dict[str, Any] | None, list[str], JSONResponse | None]:
    """Parse the upload with the real loader and validate the loaded frame.

    Returns ``(report_payload, warnings, rejection)``: the JSON-safe record for
    the status endpoint (None when there is nothing to report), the
    operator-facing warning strings, and a ready 400 response when the file
    must not replace the live input (error-severity contract violations).

    **Every kind reaches the no-rows rule, including the two that have no engine
    loader at all**, whose measured gap and two answers are
    :mod:`kairos_api.uploads_empty`'s subject. There is still no contract to run
    for those two, so a clean file of theirs reports nothing and the payload
    stays None; an empty one is looked at."""
    checked_at = datetime.now(timezone.utc).isoformat()
    authored: dict[tuple[str, str], dict[str, Any]] = {}
    loaded: Any = None
    has_loader = kind in CONTRACT_VALIDATORS or kind == "campaign_flights"
    if has_loader:
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
                "findings": [uploads_messages.record("unreadable_file", "", "error", "file")],
                "rows_loaded": 0,
            }
            return payload, [], refuse("unreadable_file")

    if kind == "campaign_flights":
        report, rows_loaded = campaign_flights_report(loaded, raw_frame, authored)
    elif not has_loader:
        # No engine loader reads this kind, so there is no contract to run and
        # the only question a file of it can still be asked is whether it
        # carries anything at all.
        report = contracts.ValidationReport(kind)
        rows_loaded = int(len(raw_frame))
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
    uploads_empty.add_when_empty(add_finding, kind, rows_loaded, raw_frame, report, authored)
    if not has_loader and not report.violations:
        # Nothing was validated and nothing is wrong, which is an honest None
        # rather than an empty report claiming a check that never ran.
        return None, [], None

    accepted = report.is_valid
    # Every list here is built from the records and not from the violations, so
    # the column an operator reads is the same word wherever it is printed: the
    # chip, the flat line and the sentence a refusal quotes cannot disagree
    # about which column of their file is wrong, and the one place that name is
    # resolved is the record. The formatting is the stored report's own.
    raw_findings = finding_records(report, row_source(kind, loaded, raw_frame), authored, raw_frame.columns, kind, loaded)
    # Swept before a rival name can leave this door, on a screen or to disk.
    findings, flat_errors, warnings = uploads_replay.at_the_door(raw_findings, uploads_channels.owned_channel())
    payload = {
        "dataset": report.dataset,
        "filename": filename,
        "checked_at": checked_at,
        "accepted": accepted,
        "is_valid": report.is_valid,
        "errors": list(flat_errors),
        "warnings": list(warnings),
        "findings": findings,
        "rows_loaded": rows_loaded,
    }
    if not accepted:
        # Both halves of the headline read the same three reasons off the same
        # three findings, one expression each, taking the sentence authored for
        # that code and falling back to the contract's own detail. The English
        # half used to quote the flat ``[error] column: code - detail`` lines
        # instead, so one locale got internal scaffolding, a bracket and a
        # Python list literal where the other got a sentence. ``errors`` still
        # carries those flat lines, unchanged, for the readers that parse them.
        first = [f for f in findings if f["severity"] == "error"][:3]
        reasons = (
            "; ".join(f.get("message_en") or f["message"] for f in first),
            "; ".join(f.get("message_he") or f["message"] for f in first),
        )
        detail, detail_he = uploads_messages.say("contract_refusal", dataset=report.dataset, reasons=reasons)
        return payload, warnings, reject(detail, list(flat_errors), detail_he, payload["findings"])
    return payload, warnings, None
