"""Sources: the state of every input a run reads, and the door they come in by.

This module owns the operator-facing ingestion surface. It reports the live
state of every input the optimizer depends on, it lets a file be checked before
it is committed, it opens the rows behind a row count, and it accepts an upload
only after validating it. Nothing here fabricates data.

**The refusal is at the door and it carries the reason.** An accepted upload is
parsed with the REAL engine loader for its kind and checked against the data
contracts in :mod:`kairos.data.contracts`, so a file that would break or
silently empty the optimizer is refused before it can replace the live input,
with the contract's own findings. ``POST /api/uploads/{kind}/check`` runs that
identical gate and writes nothing at all, so the operator sees the verdict
before committing rather than after.

**What the door says afterwards is durable, not a toast.** A file that lands on
disk and that the engine will not read is named on the status every time it is
read, with the reason and what to do about it, because a truth that lives only
in the response to one POST is gone the moment the page reloads.

Eight helper modules carry the parts that are not the door, under the file-size
cap and the ``<parent stem>_<role>.py`` naming rule: ``uploads_inputs`` declares
the seven kinds, ``uploads_validate`` holds the contract gate, ``uploads_reads``
holds the read verdicts and the prospective one, ``uploads_status`` builds the
state of every input, ``uploads_checks`` says what the door runs,
``uploads_channels`` withholds the name of any channel this operator does not
own, ``uploads_preview`` opens the rows, and ``uploads_messages`` holds every
sentence a refusal is made of, in both languages. Every writable path stays here.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from kairos_api import (
    uploads_channels,
    uploads_inputs,
    uploads_model,
    uploads_preview,
    uploads_reads,
    uploads_status,
    uploads_validate,
)
from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL, WRITE_ROLES, Wall

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily_input"
REFERENCE_DIR = DATA_DIR / "reference"
BACKUP_DIR = DATA_DIR / "_backups"
MODELS_DIR = ROOT / "models"
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
_GENERIC_PARSE_ERROR = uploads_validate.GENERIC_PARSE_ERROR

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

# Uploading is a configuration act, so affiliation is irrelevant and role is
# not: a viewer reads every state and changes none of it. The status says so
# before the click, the write route enforces it with the same sentence, and
# checking a file changes nothing so it stays open to a viewer.
UPLOAD_WALL = Wall(detail=READ_ONLY_ROLE_DETAIL, company_only=False, roles=WRITE_ROLES)

SHADOWING_REFERENCE = uploads_inputs.SHADOWING_REFERENCE
STORED_UNREAD = uploads_inputs.STORED_UNREAD
REQUIRED_COLUMNS = uploads_inputs.REQUIRED_COLUMNS
INPUTS = uploads_inputs.INPUTS


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


def _newest_daily() -> Path | None:
    """The daily file the engine reads, ranked by :func:`uploads_reads.daily_rank`."""
    if not DAILY_DIR.exists():
        return None
    candidates = list(DAILY_DIR.glob("Wally_*.csv"))
    if not candidates:
        return None
    return max(candidates, key=uploads_reads.daily_rank)


def _live_path(kind: str) -> Path | None:
    if kind == "daily":
        return _newest_daily()
    return _destination(kind)


def _stored_files(kind: str) -> list[Path]:
    """Every file of this kind on disk, most recently arrived first.

    Only the daily kind can hold more than one: it is the one input that lands
    in a directory the resolver picks from, and every other kind lands on
    exactly one path, so a second file of it cannot exist.
    """
    if kind == "daily":
        candidates = list(DAILY_DIR.glob("Wally_*.csv")) if DAILY_DIR.exists() else []
    else:
        destination = _destination(kind)
        candidates = [destination] if destination.exists() else []
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def stored_unread_files(kind: str) -> list[dict[str, Any]]:
    """The files of this kind this product stored and the engine does not read.

    The measured gap this closes: the daily directory keeps every file ever
    uploaded and :func:`_newest_daily` reads exactly one of them, so a file an
    operator had just sent could sit on disk, named on no screen, while the
    card reported the file the engine reads as in use with nothing to do.
    """
    live = _live_path(kind)
    if live is None or not live.exists():
        return []
    return uploads_reads.unread_records(
        stored=_stored_files(kind),
        live=live,
        relative=_relative,
        row_count=lambda path: uploads_status.file_shape(path, uploads_inputs.read_header_and_rows)[1],
        when=lambda path: datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
    )


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


def _in_use(kind: str, saved_path: Path | None = None, live: Path | None = None) -> tuple[bool, str]:
    """Whether the engine actually consumes an upload of this kind.

    The verdict itself is :func:`uploads_reads.in_use`, which is pure; every
    path it reasons about is resolved here, so relocating the writable
    locations still relocates everything. ``live`` overrides the file the daily
    resolver currently picks, which is how the door asks this same question
    about a candidate that has not been written yet.
    """
    return uploads_reads.in_use(
        kind=kind,
        reference=SHADOWING_REFERENCE.get(kind),
        consumed=STORED_UNREAD.get(kind),
        live=live if live is not None else (_newest_daily() if kind == "daily" else None),
        saved_path=saved_path,
        relative=_relative,
    )


def _prospect(kind: str, filename: str | None) -> dict[str, Any]:
    """The door's answer about ONE candidate file, from the real read paths.

    Every path is resolved here and the reasoning is
    :func:`uploads_reads.prospect`, so relocating the writable locations in a
    test relocates this answer with them.
    """
    prospective = _destination(kind, filename)
    return uploads_reads.prospect(
        kind=kind,
        prospective=prospective,
        stored=_stored_files(kind),
        verdict=lambda live: _in_use(kind, saved_path=prospective, live=live),
        engine_reads=_engine_reads(kind),
        relative=_relative,
        models_dir=MODELS_DIR,
        root=ROOT,
    )


def upload_status() -> dict[str, Any]:
    """Report the live state of every input the optimizer depends on.

    Kept callable with no arguments because it is the assistant's read tool as
    well as the route's body, and a read tool has no request to hand it.
    """
    return uploads_status.build(
        inputs=INPUTS,
        live_path=_live_path,
        destination=_destination,
        in_use=_in_use,
        engine_reads=_engine_reads,
        relative=_relative,
        reader=uploads_inputs.read_header_and_rows,
        missing_columns=uploads_inputs.missing_columns,
        stored_unread=stored_unread_files,
        unread_kinds=STORED_UNREAD,
        validation_reports=uploads_validate.load_reports(VALIDATION_REPORTS_PATH),
        models_dir=MODELS_DIR,
        root=ROOT,
        required_columns=REQUIRED_COLUMNS,
    )


@router.get("/status")
def upload_status_route(request: Request) -> dict[str, Any]:
    """The status, stamped with whether this account may change any of it.

    The refusal is legible before the click: a viewer session reads every
    state and gets ``can_edit`` false with the same sentence the server would
    answer a POST with, instead of an enabled control and a 403 after it.
    """
    return UPLOAD_WALL.stamp(upload_status(), request)


@router.get("/{kind}/preview")
def upload_preview(kind: str, limit: int = Query(uploads_preview.DEFAULT_PREVIEW_ROWS)) -> dict[str, Any]:
    """The first rows of this input's live file, scoped to the owned channel."""
    if kind not in REQUIRED_COLUMNS:
        raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")
    path = _live_path(kind)
    if path is None or not path.exists():
        return uploads_preview.no_file(kind)
    return uploads_preview.preview(path, kind, limit)


def _read_upload(raw: bytes, kind: str) -> tuple[pd.DataFrame | None, Any]:
    """Parse the bytes as a CSV and check the header, or return the refusal.

    Every refusal here is written by this destination, so it leaves in both
    languages through the copy table: the screen it lands on is read in Hebrew.
    """
    if not raw:
        return None, uploads_validate.refuse("empty_file")
    try:
        frame = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("Upload parse failed for kind %s: %s", kind, exc)
        return None, uploads_validate.refuse("unreadable_file")
    missing = uploads_inputs.missing_columns(kind, [str(column) for column in frame.columns])
    if missing:
        named = [f"Missing required column: {column}" for column in missing]
        return None, uploads_validate.refuse("missing_columns", named, "<header>", kind=kind, columns=", ".join(missing))
    return frame, None


async def _receive(file: UploadFile, request: Request) -> tuple[bytes | None, Any]:
    """Stream the body with a hard cap, refusing an oversize one before it lands."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + _CHUNK:
        return None, uploads_validate.refuse("too_large", megabytes=MAX_UPLOAD_BYTES // (1024 * 1024))
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        received += len(chunk)
        if received > MAX_UPLOAD_BYTES:
            return None, uploads_validate.refuse("too_large", megabytes=MAX_UPLOAD_BYTES // (1024 * 1024))
        chunks.append(chunk)
    return b"".join(chunks), None


def _backup(destination: Path, kind: str) -> str | None:
    if not destination.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_path = BACKUP_DIR / f"{kind}_{stamp}.csv"
    shutil.copy2(destination, backup_path)
    return _relative(backup_path)


@router.post("/{kind}/check")
async def check_file(kind: str, request: Request, file: UploadFile = File(...)) -> Any:
    """Run the upload's own gate over a file and write nothing at all.

    The same header check and the same data-contract validation the upload
    performs, with no backup, no replacement and no stored report, so a person
    can see the refusal and its reason before committing to it.

    The consequence is about the candidate and never about its kind. Measured:
    derived from the kind's read path alone it told a steward that an older
    daily file would replace the live input, and committing it replaced nothing.
    """
    if kind not in REQUIRED_COLUMNS:
        raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")
    raw, refusal = await _receive(file, request)
    if refusal is not None:
        return refusal
    frame, refusal = _read_upload(raw or b"", kind)
    if refusal is not None:
        return refusal
    assert frame is not None
    report_payload, warnings, rejection = uploads_validate.run_contract_validation(
        kind, raw or b"", frame, str(file.filename or "")
    )
    accepted = rejection is None
    body = {
        "kind": kind,
        "checked": True,
        "accepted": accepted,
        "filename": str(file.filename or ""),
        "rows": int(len(frame)),
        **uploads_channels.columns_record(frame.columns),
        "validation": report_payload,
        "warnings": list(warnings),
        "errors": list((report_payload or {}).get("errors") or []),
        "findings": list((report_payload or {}).get("findings") or []),
        **_prospect(kind, file.filename),
    }
    return UPLOAD_WALL.stamp(body, request)


@router.post("/{kind}")
@UPLOAD_WALL.guard()
async def upload_file(kind: str, request: Request, file: UploadFile = File(...)) -> Any:
    """Validate and persist an uploaded CSV for the given input kind.

    The guard is the lock behind the label: ``/status`` tells a viewer
    ``can_edit`` false, and this route refuses that account with the identical
    sentence instead of trusting a disabled button to hold the door.
    """
    if kind not in REQUIRED_COLUMNS:
        raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")

    # Size gates: refuse an honestly-declared oversize body before reading it,
    # then stream in chunks with a hard cap so the body never has to be trusted.
    raw, refusal = await _receive(file, request)
    if refusal is not None:
        return refusal
    # Parse with pandas to validate it is a real, readable CSV, then check the
    # header. The parser's own message is logged; the client gets a generic line.
    frame, refusal = _read_upload(raw or b"", kind)
    if refusal is not None:
        return refusal
    assert frame is not None

    # Run the kind's REAL loader and data-contract validator over the upload.
    # Error-severity violations refuse the file before it can replace the live
    # input; warnings ride along in the response and the stored report.
    report_payload, contract_warnings, rejection = uploads_validate.run_contract_validation(
        kind, raw or b"", frame, str(file.filename or "")
    )
    if report_payload is not None:
        uploads_validate.store_report(VALIDATION_REPORTS_PATH, kind, report_payload)
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
    tmp_path.write_bytes(raw or b"")
    os.replace(tmp_path, destination)

    if backed_up:
        warnings.append(f"Previous file backed up to {backed_up}")

    in_use, in_use_reason = _in_use(kind, saved_path=destination)
    if not in_use and in_use_reason:
        # The file saved and parsed, but the optimizer will not read it. Say so
        # in the response, so the confirmation implies no ingestion that did not happen.
        warnings.insert(0, in_use_reason)

    return {
        "kind": kind,
        "saved_path": _relative(destination),
        "rows": int(len(frame)),
        **uploads_channels.columns_record(frame.columns),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "in_use": in_use,
        "in_use_reason": in_use_reason,
        "engine_reads": _engine_reads(kind),
        "validation": report_payload,
        "findings": list((report_payload or {}).get("findings") or []),
        # About THIS file and never about its kind, which is what the door answered.
        "consequence": uploads_status.consequence_record(in_use, _engine_reads(kind), MODELS_DIR, ROOT, still_read=None if in_use else _engine_reads(kind)),
        "model": uploads_model.version(MODELS_DIR, ROOT),
        "warnings": warnings,
    }
