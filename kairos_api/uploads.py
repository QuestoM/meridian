"""Upload status and ingestion endpoints for the Kairos input files.

This module owns the operator-facing data ingestion surface: it reports the
live state of every input file the optimizer depends on, and it accepts new
uploads after validating them against the expected schema. Nothing here
fabricates data; a file is only reported as present and valid when it really
exists on disk and parses with the columns the loaders require.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from kairos.data.loaders import DAILY_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily_input"
BACKUP_DIR = DATA_DIR / "_backups"

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

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
    # The daily Wally file ships with Hebrew headers; the loader maps them.
    "daily": list(DAILY_COLUMN_MAP.keys()),
}

# Per-kind presentation metadata for the dashboard.
#
# The channel provides THREE source data files (programmes, spots, dayparts);
# the optimizer also takes ONE daily operational file (the Wally ad log). The
# advertiser rules and the rate card are CONFIGURATION, not periodic data the
# channel uploads, so they are grouped separately. Advertiser rules are also
# editable directly in the Advertisers screen.
INPUTS: list[dict[str, str]] = [
    {"kind": "programmes", "label_en": "Programme lineup", "label_he": "לוח תוכניות", "cadence": "weekly"},
    {"kind": "daily", "label_en": "Daily ad log (Wally)", "label_he": "קובץ פרסומות יומי", "cadence": "daily"},
    {"kind": "spots", "label_en": "Historical spots", "label_he": "תשדירים היסטוריים", "cadence": "reference"},
    {"kind": "dayparts", "label_en": "Dayparts (ratings by time)", "label_he": "חלקי יום (רייטינג לפי שעה)", "cadence": "reference"},
    {"kind": "advertiser_rules", "label_en": "Advertiser rules", "label_he": "כללי מפרסמים", "cadence": "config"},
    {"kind": "rate_card", "label_en": "Rate card", "label_he": "כרטיס תעריפים", "cadence": "config"},
]


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
    if kind == "daily":
        name = Path(filename).name if filename else ""
        if not name or not name.lower().endswith(".csv"):
            name = f"Wally_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.csv"
        return DAILY_DIR / name
    raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")


def _newest_daily() -> Path | None:
    if not DAILY_DIR.exists():
        return None
    candidates = sorted(DAILY_DIR.glob("Wally_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _live_path(kind: str) -> Path | None:
    if kind == "daily":
        return _newest_daily()
    return _destination(kind)


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


@router.get("/status")
def upload_status() -> dict[str, Any]:
    """Report the live state of every input file the optimizer depends on."""
    inputs: list[dict[str, Any]] = []
    for meta in INPUTS:
        kind = meta["kind"]
        path = _live_path(kind)
        exists = bool(path and path.exists())
        entry: dict[str, Any] = {
            "kind": kind,
            "label_en": meta["label_en"],
            "label_he": meta["label_he"],
            "cadence": meta["cadence"],
            "filename": path.name if path else _destination(kind).name,
            "path": str((path or _destination(kind)).relative_to(ROOT)).replace("\\", "/"),
            "exists": exists,
            "rows": 0,
            "columns": [],
            "last_modified": None,
            "valid": False,
            "warnings": [],
        }
        if exists and path is not None:
            columns, rows, read_warnings = _read_header_and_rows(path)
            missing = _validate_columns(kind, columns)
            entry["columns"] = columns
            entry["rows"] = rows
            entry["last_modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            entry["valid"] = not missing
            warnings = list(read_warnings)
            if missing:
                warnings.insert(0, f"Missing required columns: {', '.join(missing)}")
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
    return str(backup_path.relative_to(ROOT)).replace("\\", "/")


@router.post("/{kind}")
async def upload_file(kind: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate and persist an uploaded CSV for the given input kind."""
    if kind not in REQUIRED_COLUMNS:
        raise HTTPException(status_code=404, detail=f"Unknown input kind: {kind}")

    raw = await file.read()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"detail": "Uploaded file is empty", "errors": ["Uploaded file is empty"], "valid": False},
        )

    # Parse with pandas to validate it is a real, readable CSV.
    from io import BytesIO

    try:
        frame = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        message = f"Could not parse uploaded CSV: {exc}"
        return JSONResponse(
            status_code=400,
            content={"detail": message, "errors": [message], "valid": False},
        )

    columns = [str(column) for column in frame.columns]
    missing = _validate_columns(kind, columns)
    warnings: list[str] = []
    if missing:
        # Reject without touching the live file.
        message = f"Missing required columns for '{kind}': {', '.join(missing)}"
        return JSONResponse(
            status_code=400,
            content={
                "detail": message,
                "errors": [f"Missing required column: {column}" for column in missing],
                "valid": False,
            },
        )

    destination = _destination(kind, file.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backed_up = _backup(destination, kind)

    # Write the raw bytes verbatim so encoding and content are preserved
    # exactly as uploaded. The mtime change busts the loaders' mtime-keyed
    # CSV cache automatically.
    destination.write_bytes(raw)

    if backed_up:
        warnings.append(f"Previous file backed up to {backed_up}")

    return {
        "kind": kind,
        "saved_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(frame)),
        "columns": columns,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "warnings": warnings,
    }
