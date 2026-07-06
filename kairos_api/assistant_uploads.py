"""Agreement-file uploads for the assistant, parsed in memory and kept per user.

An operator uploads a spreadsheet (an advertiser agreement, a rate sheet). The
raw bytes are parsed IN MEMORY with pandas and then discarded; only a bounded
summary of the sheets, columns and rows is stored, strictly under the uploading
operator's own directory. The assistant reads that summary as DATA, never as
instructions. Storage relocates for tests via KAIROS_ASSISTANT_DATA_DIR, like
the rest of the action-plane state; entries older than the retention window are
pruned on access.

The per-user directory name reuses the same sanitize-plus-hash scheme the thread
store uses, so two distinct usernames can never collide and no path traversal or
dotfile is possible. GET and DELETE are keyed by the authenticated session user
only; no request parameter can reach another operator's uploads.
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from kairos_api import assistant_actions, assistant_memory

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

logger = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 300
MAX_COLS = 40
RETENTION_DAYS = 7
ALLOWED_SUFFIXES = {".xlsx", ".xls", ".csv"}
_CHUNK = 65536


def _uploads_root() -> Path:
    return assistant_actions._data_dir() / "uploads"


def _user_dir(user: str) -> Path:
    return _uploads_root() / assistant_memory._sanitize_username(user)


def _upload_path(user: str, upload_id: str) -> Path:
    # upload_id is our own hex token; sanitize defensively before any path use.
    safe = "".join(char for char in str(upload_id) if char in "0123456789abcdef")[:32]
    return _user_dir(user) / f"{safe}.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _prune(user: str) -> None:
    """Delete this user's summaries older than the retention window. Never raises."""
    directory = _user_dir(user)
    if not directory.exists():
        return
    cutoff = _now() - timedelta(days=RETENTION_DAYS)
    for path in directory.glob("*.json"):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            uploaded_at = datetime.fromisoformat(str(stored.get("uploaded_at")))
            if uploaded_at.tzinfo is None:
                uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
            if uploaded_at < cutoff:
                path.unlink()
        except Exception:  # noqa: BLE001 - a corrupt or unparseable file is dropped
            path.unlink(missing_ok=True)


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _summarize_frame(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    columns = [_stringify_cell(col) for col in list(frame.columns)[:MAX_COLS]]
    rows: list[list[str]] = []
    for _, row in frame.head(MAX_ROWS).iterrows():
        rows.append([_stringify_cell(cell) for cell in list(row)[:MAX_COLS]])
    return {"name": name, "columns": columns, "rows": rows, "total_rows": int(len(frame))}


def _parse(data: bytes, suffix: str) -> list[dict[str, Any]]:
    """Parse the raw bytes into bounded sheet summaries, or raise a 400."""
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
            frames = {"data": frame}
        else:
            frames = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
    except Exception as exc:  # noqa: BLE001 - a parse failure is an honest client error
        raise HTTPException(
            status_code=400,
            detail=f"the file could not be parsed as a spreadsheet ({type(exc).__name__})",
        ) from exc
    sheets = [_summarize_frame(str(name), frame) for name, frame in frames.items()]
    if not sheets:
        raise HTTPException(status_code=400, detail="the file carries no readable sheets")
    return sheets


async def _read_capped(file: UploadFile) -> bytes:
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"the file exceeds the {MAX_BYTES // (1024 * 1024)} MB upload limit",
            )
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="the file is empty")
    return b"".join(chunks)


def _public(summary: dict[str, Any], with_rows: bool) -> dict[str, Any]:
    sheets = [
        {
            "name": sheet.get("name"),
            "columns": sheet.get("columns", []),
            "total_rows": sheet.get("total_rows"),
            **({"rows": sheet.get("rows", [])} if with_rows else {}),
        }
        for sheet in summary.get("sheets", [])
    ]
    return {
        "upload_id": summary["upload_id"],
        "filename": summary["filename"],
        "uploaded_at": summary["uploaded_at"],
        "sheets": sheets,
        "total_rows": sum(int(sheet.get("total_rows", 0)) for sheet in summary.get("sheets", [])),
    }


# --- store helpers the read tools use ---------------------------------------------
def list_summaries(user: str | None) -> list[dict[str, Any]]:
    """Every stored summary for one user, newest first. Never raises."""
    resolved = str(user or "auth-disabled")
    _prune(resolved)
    directory = _user_dir(resolved)
    if not directory.exists():
        return []
    found: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    found.sort(key=lambda item: str(item.get("uploaded_at", "")), reverse=True)
    return found


def get_summary(user: str | None, upload_id: str) -> dict[str, Any] | None:
    """One stored summary for one user, or None. Strictly keyed by user."""
    resolved = str(user or "auth-disabled")
    _prune(resolved)
    path = _upload_path(resolved, upload_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --- endpoints --------------------------------------------------------------------
@router.post("/upload")
async def upload_agreement(http_request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Parse a spreadsheet in memory and store only a bounded summary, keyed by
    the session user. The raw bytes are discarded once parsed."""
    user = assistant_actions._actor(http_request)
    filename = str(file.filename or "").strip() or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"only .xlsx, .xls or .csv files are accepted; got {suffix or 'no extension'}",
        )
    declared = http_request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES + _CHUNK:
        raise HTTPException(
            status_code=400,
            detail=f"the file exceeds the {MAX_BYTES // (1024 * 1024)} MB upload limit",
        )
    data = await _read_capped(file)
    sheets = _parse(data, suffix)
    del data  # the raw bytes never persist

    upload_id = uuid.uuid4().hex[:12]
    summary = {
        "upload_id": upload_id,
        "filename": filename,
        "uploaded_at": _now().isoformat(),
        "sheets": sheets,
    }
    directory = _user_dir(user)
    directory.mkdir(parents=True, exist_ok=True)
    path = _upload_path(user, upload_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    _prune(user)
    return _public(summary, with_rows=False)


@router.get("/uploads")
def list_uploads(http_request: Request) -> dict[str, Any]:
    """The caller's own uploads, newest first."""
    user = assistant_actions._actor(http_request)
    uploads = [_public(summary, with_rows=False) for summary in list_summaries(user)]
    return {"uploads": uploads, "count": len(uploads), "user": user}


@router.delete("/uploads/{upload_id}")
def delete_upload(upload_id: str, http_request: Request) -> dict[str, Any]:
    """Delete one of the caller's own uploads. Another user's id is a 404."""
    user = assistant_actions._actor(http_request)
    path = _upload_path(user, upload_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no upload {upload_id!r} for this operator")
    path.unlink()
    return {"deleted": upload_id, "user": user}
