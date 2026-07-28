"""Agency records CRUD, persisted to data/agencies.csv.

The sibling of :mod:`kairos_api.advertisers`, for the agency (משרד) level: one
row per agency with contacts, commercial terms and provenance. Each operation
reads the real CSV, mutates one row, backs the file up, and writes it back
preserving column order, serialized under a module lock and written via a temp
file plus ``os.replace`` so concurrent edits cannot lose rows and readers never
see a torn CSV. Every mutation snapshots the ``agencies`` logical file into the
unified version timeline first (a safe no-op until the version store registers
that name; the timestamped backups remain the recovery path meanwhile).

Deactivation replaces deletion: a suspended agency keeps resolving on historic
spots while its conditions and rebate go inert on the pricing path. Provenance
is explicit: seeded rows are ``synthetic`` (invented contacts and terms around
an observed name), operator-created rows are ``manual``, and ``observed`` is
reserved for rows fully backfilled from real source data. The honest boundary:
agency rules and rebates touch ONLY the daily per-spot ledger (and its
reporting-only net figure); the weekly plan, retention math and QH settlement
carry no agency attribution (docs/agency-layer-design.md).
"""

from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
AGENCIES_PATH = DATA_DIR / "agencies.csv"

COLUMNS = [
    "agency_id",
    "name",
    "display_name",
    "aliases",
    "agency_type",
    "contact_name",
    "contact_role",
    "contact_phone",
    "contact_email",
    "contact2_name",
    "contact2_role",
    "contact2_phone",
    "contact2_email",
    "address_city",
    "address_street",
    "vat_id",
    "payment_terms_days",
    "rebate_percent",
    "commission_percent",
    "credit_limit_ils",
    "status",
    "onboarded_at",
    "notes",
    "data_source",
]

AGENCY_TYPES = ("מדיה מלא", "קריאייטיב", "בוטיק")
STATUSES = ("active", "suspended")
DATA_SOURCES = ("observed", "synthetic", "manual")

BOUNDARY_NOTE = (
    "Agency rules and rebates apply on the daily per-spot pricing path only: "
    "conditions compose with advertiser rules (forbid wins across levels) and "
    "rebate_percent yields a reporting-only net_revenue beside gross. The weekly "
    "break plan, retention math, quarter-hour settlement and invoicing are not "
    "affected; the weekly data carries no agency attribution."
)

router = APIRouter(prefix="/api/agencies", tags=["agencies"])

# Serializes every load-mutate-write cycle so two concurrent edits cannot drop
# each other's rows (lost update), same discipline as the advertiser stores.
_STORE_LOCK = threading.Lock()

_TEXT_FIELDS = [
    "display_name", "aliases", "agency_type", "contact_name", "contact_role",
    "contact_phone", "contact_email", "contact2_name", "contact2_role",
    "contact2_phone", "contact2_email", "address_city", "address_street",
    "vat_id", "onboarded_at", "notes",
]


class AgencyCreate(BaseModel):
    """A new agency record. ``name`` must be the exact daily-file string (or the
    string expected in future daily files), because it is the observed join key."""

    agency_id: str
    name: str
    display_name: str = ""
    aliases: str = ""
    agency_type: str = ""
    contact_name: str = ""
    contact_role: str = ""
    contact_phone: str = ""
    contact_email: str = ""
    contact2_name: str = ""
    contact2_role: str = ""
    contact2_phone: str = ""
    contact2_email: str = ""
    address_city: str = ""
    address_street: str = ""
    vat_id: str = ""
    payment_terms_days: int = 60
    rebate_percent: float = 0.0
    commission_percent: float = 0.0
    credit_limit_ils: float = 0.0
    status: str = "active"
    onboarded_at: str = ""
    notes: str = ""
    data_source: str = "manual"


class AgencyUpdate(BaseModel):
    """Editable fields for an agency. All optional for PATCH-style PUT."""

    name: Optional[str] = None
    display_name: Optional[str] = None
    aliases: Optional[str] = None
    agency_type: Optional[str] = None
    contact_name: Optional[str] = None
    contact_role: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    contact2_name: Optional[str] = None
    contact2_role: Optional[str] = None
    contact2_phone: Optional[str] = None
    contact2_email: Optional[str] = None
    address_city: Optional[str] = None
    address_street: Optional[str] = None
    vat_id: Optional[str] = None
    payment_terms_days: Optional[int] = None
    rebate_percent: Optional[float] = None
    commission_percent: Optional[float] = None
    credit_limit_ils: Optional[float] = None
    status: Optional[str] = None
    onboarded_at: Optional[str] = None
    notes: Optional[str] = None
    data_source: Optional[str] = None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _validate_percent(value: float, name: str) -> float:
    if not 0.0 <= float(value) <= 100.0:
        raise HTTPException(status_code=400, detail=f"{name} must be between 0 and 100")
    return float(value)


def _validate_nonnegative(value: float, name: str) -> float:
    if float(value) < 0:
        raise HTTPException(status_code=400, detail=f"{name} must be >= 0")
    return float(value)


def _validate_choice(value: str, allowed: tuple[str, ...], name: str, allow_blank: bool = True) -> str:
    cleaned = str(value or "").strip()
    if allow_blank and not cleaned:
        return ""
    if cleaned not in allowed:
        raise HTTPException(status_code=400, detail=f"{name} must be one of {list(allowed)}")
    return cleaned


def _load_frame() -> pd.DataFrame:
    if not AGENCIES_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(AGENCIES_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _row_to_record(row: "pd.Series[Any]") -> dict[str, Any]:
    record: dict[str, Any] = {
        "agency_id": str(row.get("agency_id", "")),
        "name": str(row.get("name", "")),
        "payment_terms_days": _coerce_int(row.get("payment_terms_days"), 0),
        "rebate_percent": round(_coerce_float(row.get("rebate_percent")), 4),
        "commission_percent": round(_coerce_float(row.get("commission_percent")), 4),
        "credit_limit_ils": round(_coerce_float(row.get("credit_limit_ils")), 2),
        "status": str(row.get("status", "active")) or "active",
        "data_source": str(row.get("data_source", "manual")) or "manual",
    }
    for column in _TEXT_FIELDS:
        record[column] = str(row.get(column, ""))
    return record


def _backup() -> None:
    if not AGENCIES_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(AGENCIES_PATH, BACKUP_DIR / f"agencies_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    """Backup, then write atomically (temp file + os.replace, like advertisers)."""
    _backup()
    AGENCIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = AGENCIES_PATH.with_name(AGENCIES_PATH.name + ".tmp")
    frame[COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, AGENCIES_PATH)


def _snapshot_before_write(request: "Request | None") -> None:
    """Version the agencies store before a manual edit writes it. A safe no-op
    until the version store registers the ``agencies`` logical name."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "agencies")


def _locate(frame: pd.DataFrame, agency_id: str) -> int:
    mask = frame["agency_id"].astype(str) == agency_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Agency '{agency_id}' not found")
    return int(frame.index[mask][0])


def _apply_validated(payload: "AgencyCreate | AgencyUpdate", partial: bool) -> dict[str, str]:
    """Validate a payload's set fields, returning CSV-ready string values."""
    data = payload.model_dump(exclude_unset=partial)
    out: dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key in {"rebate_percent", "commission_percent"}:
            out[key] = str(_validate_percent(value, key))
        elif key in {"credit_limit_ils"}:
            out[key] = str(_validate_nonnegative(value, key))
        elif key == "payment_terms_days":
            out[key] = str(int(_validate_nonnegative(value, key)))
        elif key == "agency_type":
            out[key] = _validate_choice(value, AGENCY_TYPES, "agency_type")
        elif key == "status":
            out[key] = _validate_choice(value, STATUSES, "status", allow_blank=False)
        elif key == "data_source":
            out[key] = _validate_choice(value, DATA_SOURCES, "data_source", allow_blank=False)
        else:
            out[key] = str(value)
    return out


@router.get("")
def list_agencies() -> dict[str, Any]:
    """Every agency with its conditions, overlap findings and link counts."""
    from kairos_api.agency_conditions import conditions_for, link_summary_for, overlaps_for

    frame = _load_frame()
    agencies = []
    for _, row in frame.iterrows():
        record = _row_to_record(row)
        agency_id = record["agency_id"]
        record["conditions"] = conditions_for(agency_id)
        record["overlaps"] = overlaps_for(agency_id)
        record["links"] = link_summary_for(agency_id)
        agencies.append(record)
    return {"agencies": agencies, "columns": COLUMNS, "boundary": BOUNDARY_NOTE}


@router.get("/{agency_id}")
def get_agency(agency_id: str) -> dict[str, Any]:
    from kairos_api.agency_conditions import conditions_for, link_summary_for, overlaps_for

    frame = _load_frame()
    record = _row_to_record(frame.loc[_locate(frame, agency_id)])
    record["conditions"] = conditions_for(agency_id)
    record["overlaps"] = overlaps_for(agency_id)
    record["links"] = link_summary_for(agency_id)
    record["boundary"] = BOUNDARY_NOTE
    return record


@router.post("", status_code=201)
def create_agency(payload: AgencyCreate, request: Request = None) -> dict[str, Any]:
    values = _apply_validated(payload, partial=False)
    with _STORE_LOCK:
        frame = _load_frame()
        if (frame["agency_id"].astype(str) == payload.agency_id).any():
            raise HTTPException(status_code=409, detail=f"Agency '{payload.agency_id}' already exists")
        if (frame["name"].astype(str) == payload.name).any():
            raise HTTPException(status_code=409, detail=f"An agency named '{payload.name}' already exists")
        new_row = {column: values.get(column, "") for column in COLUMNS}
        new_row["agency_id"] = payload.agency_id
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _row_to_record(frame.iloc[-1])


@router.put("/{agency_id}")
def update_agency(agency_id: str, payload: AgencyUpdate, request: Request = None) -> dict[str, Any]:
    values = _apply_validated(payload, partial=True)
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, agency_id)
        for key, value in values.items():
            if key == "name":
                clash = (frame["name"].astype(str) == value) & (frame.index != index)
                if clash.any():
                    raise HTTPException(status_code=409, detail=f"An agency named '{value}' already exists")
            frame.at[index, key] = value
        _snapshot_before_write(request)
        _write_frame(frame)
        return _row_to_record(frame.loc[index])


@router.post("/{agency_id}/deactivate")
def deactivate_agency(agency_id: str, request: Request = None) -> dict[str, Any]:
    """Suspend an agency: its conditions and rebate go inert on the pricing path,
    historic spots keep resolving to it. Reactivate with PUT status=active."""
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, agency_id)
        frame.at[index, "status"] = "suspended"
        _snapshot_before_write(request)
        _write_frame(frame)
        return _row_to_record(frame.loc[index])
