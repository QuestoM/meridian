"""Calendar events store, bundled holiday table and the model-context disclosure.

The operator-managed events store (``data/calendar_events.csv``) records special
periods (holidays, wars, sport, other) with a start date, an optional inclusive
end date (empty means open-ended), a 1..5 operator-judged intensity and an
active flag. Writes follow the store doctrine: serialized under a module lock,
written atomically (temp file plus ``os.replace``), and every manual mutation is
snapshotted first into the unified version timeline as the 'events' logical file.

The GET payload also carries two read-only blocks built ONLY from real sources:

- ``holidays``: the bundled Israeli holiday reference table
  (``kairos/config/israel_holidays.csv``), a static checked-in list the operator
  is told to verify before operational use.
- ``model_context``: what the model actually conditions on today, read from
  ``config/optimization_weights.yaml`` (the weekday pricing premiums, which are
  rate-card assertions, not measured) and from the metadata of
  ``models/tv_break_coefficients.json`` (detrend mode, the seasonal baseline
  verdict, the level-drift block, computed_at), plus the measured training
  window and the wartime disclosure: the whole 30-day window sits inside
  wartime, with the ceasefire only on 2024-11-27.

Each stored event is returned with its overlap against the coefficient training
window (``window_overlap_days``) and against the saved weekly plan's dates
(``plan_overlap_dates``), so the operator can see which plan days sit inside an
event and whether the training data ever saw that condition.

Each event also carries an operator-asserted ``price_multiplier`` (default 1.0,
validated to 0.1..5.0). It feeds the owner-gated event pricing layer
(kairos/optimize/pricing.py, activation flag ``pricing_activation.events``,
shipped OFF because turning it on moves real forecast revenue). Event RETENTION
coefficients are untouched by that layer and remain v2 only, measured behind
the held-out gate once history with real contrast exists.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVENTS_PATH = DATA_DIR / "calendar_events.csv"
HOLIDAYS_PATH = ROOT / "kairos" / "config" / "israel_holidays.csv"
WEIGHTS_CONFIG_PATH = ROOT / "config" / "optimization_weights.yaml"
COEFFICIENTS_PATH = ROOT / "models" / "tv_break_coefficients.json"

router = APIRouter(prefix="/api/events", tags=["events"])

COLUMNS = ("event_id", "name", "type", "start_date", "end_date",
           "intensity", "notes", "active", "price_multiplier")
EVENT_TYPES = ("holiday", "war", "special", "sport", "other")
# The operator-asserted price multiplier bounds (a 10x cut to a 5x surge).
PRICE_MULTIPLIER_MIN = 0.1
PRICE_MULTIPLIER_MAX = 5.0

# The measured coefficient training window (30 days of reference history) and
# the wartime facts disclosed with it. The ceasefire date is the historical
# Israel-Hezbollah ceasefire; the post-ceasefire tail count is the measured
# number of breaks starting after it, from the recon in
# docs/calendar-events-design.md (132 of 2532 measured breaks, about 3.5 days).
TRAINING_WINDOW_START = date(2024, 11, 1)
TRAINING_WINDOW_END = date(2024, 11, 30)
CEASEFIRE_DATE = "2024-11-27"
POST_CEASEFIRE_TAIL_BREAKS = 132

# Serializes every load-mutate-write cycle on the events CSV so two concurrent
# edits cannot drop each other's rows (lost update).
_STORE_LOCK = threading.Lock()


class EventCreate(BaseModel):
    """A new operator event. start_date is required; end_date empty means the
    event is open-ended (a war without a declared end)."""

    name: str
    type: str
    start_date: str
    end_date: str = ""
    intensity: int = 3
    notes: str = ""
    active: bool = True
    price_multiplier: float = 1.0


class EventUpdate(BaseModel):
    """Editable fields for an event. All optional for PATCH-style PUT."""

    name: str | None = None
    type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    intensity: int | None = None
    notes: str | None = None
    active: bool | None = None
    price_multiplier: float | None = None


# --- store ---------------------------------------------------------------------
def _load_frame() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(EVENTS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            # A legacy store predating the price_multiplier column reads as the
            # neutral 1.0 (no price effect), never as a missing value.
            frame[column] = "1.0" if column == "price_multiplier" else ""
    return frame


def _write_frame(frame: pd.DataFrame) -> None:
    """Write atomically (temp file plus os.replace) so a reader that opens the
    CSV mid-write sees either the old or the new file, never a truncated one.
    Callers hold ``_STORE_LOCK`` across load-mutate-write."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = EVENTS_PATH.with_name(EVENTS_PATH.name + ".tmp")
    frame[list(COLUMNS)].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, EVENTS_PATH)


def _snapshot_before_write(request: "Request | None") -> None:
    """Record a version of the events store before a manual edit writes it."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "events")


# --- validation ----------------------------------------------------------------
def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}")


def _validate(name: str, type_: str, start_date: str, end_date: str,
              intensity: int, price_multiplier: float) -> dict[str, str]:
    if not str(name or "").strip():
        raise HTTPException(status_code=400, detail="name is required")
    type_clean = str(type_ or "").strip().lower()
    if type_clean not in EVENT_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"type must be one of {list(EVENT_TYPES)}, got {type_!r}")
    start = _parse_date(start_date, "start_date")
    end_clean = str(end_date or "").strip()
    if end_clean:
        end = _parse_date(end_clean, "end_date")
        if end < start:
            raise HTTPException(status_code=400,
                                detail="end_date must be on or after start_date")
    if not 1 <= int(intensity) <= 5:
        raise HTTPException(status_code=400, detail="intensity must be between 1 and 5")
    try:
        multiplier = float(price_multiplier)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="price_multiplier must be a number")
    if not PRICE_MULTIPLIER_MIN <= multiplier <= PRICE_MULTIPLIER_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"price_multiplier must be between {PRICE_MULTIPLIER_MIN} and {PRICE_MULTIPLIER_MAX}")
    return {"name": str(name).strip(), "type": type_clean,
            "start_date": start.isoformat(), "end_date": end_clean,
            "intensity": str(int(intensity)),
            "price_multiplier": str(multiplier)}


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    end = str(row.get("end_date", "")).strip()
    try:
        intensity = int(str(row.get("intensity", "")) or 0)
    except ValueError:
        intensity = 0
    try:
        # Tolerant read: a legacy row without the column is the neutral 1.0.
        multiplier = float(str(row.get("price_multiplier", "")).strip() or 1.0)
    except ValueError:
        multiplier = 1.0
    return {
        "event_id": str(row.get("event_id", "")),
        "name": str(row.get("name", "")),
        "type": str(row.get("type", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": end or None,
        "intensity": intensity,
        "notes": str(row.get("notes", "")),
        "active": str(row.get("active", "")).strip().lower() == "true",
        "price_multiplier": multiplier,
    }


def _locate(frame: pd.DataFrame, event_id: str) -> int:
    mask = frame["event_id"].astype(str) == event_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"event '{event_id}' not found")
    return int(frame.index[mask][0])


# --- overlaps ------------------------------------------------------------------
def _event_span(record: dict[str, Any]) -> "tuple[date, Optional[date]] | None":
    """The event's (start, end) as dates; end None when open-ended. None when the
    stored start does not parse (a hand-edited row), so overlap stays honest-empty."""
    try:
        start = date.fromisoformat(record["start_date"])
    except (ValueError, TypeError):
        return None
    end: Optional[date] = None
    if record["end_date"]:
        try:
            end = date.fromisoformat(record["end_date"])
        except (ValueError, TypeError):
            return None
    return start, end


def _window_overlap_days(record: dict[str, Any]) -> int:
    """Inclusive day count of the event's intersection with the coefficient
    training window. An open-ended event covers the window from its start."""
    span = _event_span(record)
    if span is None:
        return 0
    start, end = span
    low = max(start, TRAINING_WINDOW_START)
    high = min(end, TRAINING_WINDOW_END) if end is not None else TRAINING_WINDOW_END
    return (high - low).days + 1 if low <= high else 0


def _plan_dates() -> list[str]:
    """The distinct dates of the saved weekly plan, read from the same guarded
    loader the dashboard uses. Empty when no real plan exists (honest absence)."""
    try:
        from kairos_api.core import _load_break_schedule

        frame = _load_break_schedule()
    except Exception:  # noqa: BLE001 - the events list must not fail on plan trouble
        return []
    if frame.empty or "date" not in frame.columns:
        return []
    dates: set[str] = set()
    for value in frame["date"].astype(str):
        try:
            dates.add(date.fromisoformat(value.strip()[:10]).isoformat())
        except ValueError:
            continue
    return sorted(dates)


def _plan_overlap_dates(record: dict[str, Any], plan_dates: list[str]) -> list[str]:
    span = _event_span(record)
    if span is None:
        return []
    start, end = span
    overlap = []
    for iso in plan_dates:
        day = date.fromisoformat(iso)
        if day >= start and (end is None or day <= end):
            overlap.append(iso)
    return overlap


# --- holidays ------------------------------------------------------------------
def _load_holidays() -> list[dict[str, Any]]:
    """The bundled holiday reference table. Comment lines (leading '#') carry the
    verify-before-use note and are skipped. Missing file returns empty."""
    if not HOLIDAYS_PATH.exists():
        return []
    with HOLIDAYS_PATH.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        rows = []
        for row in reader:
            rows.append({
                "date": str(row.get("date", "")).strip(),
                "name": str(row.get("name", "")).strip(),
                "kind": str(row.get("kind", "")).strip(),
                "is_school_holiday": str(row.get("is_school_holiday", "")).strip().lower()
                in ("true", "1", "yes"),
            })
    return rows


# --- model context -------------------------------------------------------------
def _coefficients_metadata() -> "dict[str, Any] | None":
    try:
        payload = json.loads(COEFFICIENTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def _weekday_premiums() -> "dict[str, Any]":
    """The live program pricing weekday multipliers, read from the rate-card
    config. These are operator assertions, not measured from audience history."""
    try:
        import yaml

        config = yaml.safe_load(WEIGHTS_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        raw = (config.get("premiums") or {}).get("day_of_week") or {}
    except Exception:  # noqa: BLE001 - honest absence beats a fabricated table
        return {"available": False, "reason": "config/optimization_weights.yaml could not be read",
                "values": []}
    values = [{"iso_weekday": int(key), "multiplier": float(value)}
              for key, value in sorted(raw.items(), key=lambda item: int(item[0]))]
    return {
        "available": bool(values),
        "basis": "rate-card assertion, not measured from audience history",
        "source": "config/optimization_weights.yaml premiums.day_of_week",
        "values": values,
    }


def _wartime_disclosure(metadata: "dict[str, Any] | None") -> dict[str, Any]:
    total = int(metadata.get("total_breaks_measured", 0)) if metadata else 0
    line = (
        f"The whole 30 day training window ({TRAINING_WINDOW_START.isoformat()} to "
        f"{TRAINING_WINDOW_END.isoformat()}) was measured under wartime conditions; the "
        f"ceasefire took effect only on {CEASEFIRE_DATE}, leaving a post-ceasefire tail of "
        f"{POST_CEASEFIRE_TAIL_BREAKS} of {total or 2532} measured breaks. Holiday or "
        "war-intensity retention effects claimed from this window would be fabrication; "
        "they ship only once history with real contrast exists and passes the held-out gate."
    )
    return {
        "line": line,
        "ceasefire_date": CEASEFIRE_DATE,
        "post_ceasefire_breaks": POST_CEASEFIRE_TAIL_BREAKS,
        "total_breaks_measured": total or None,
    }


def _model_context() -> dict[str, Any]:
    """What the model conditions on today, from real config and metadata only."""
    metadata = _coefficients_metadata()
    if metadata is None:
        measurement: dict[str, Any] = {
            "available": False,
            "reason": "models/tv_break_coefficients.json not found or unreadable",
        }
    else:
        measurement = {
            "available": True,
            "detrend_baseline_mode": metadata.get("detrend_baseline_mode"),
            "seasonal_baseline": {
                "recommended": metadata.get("detrend_seasonality_recommended"),
                "holdout": metadata.get("detrend_seasonality_holdout"),
                "reason": metadata.get("detrend_seasonality_reason"),
            },
            "level_drift": metadata.get("level_drift"),
            "computed_at": metadata.get("computed_at"),
        }
    return {
        "training_window": {
            "start": TRAINING_WINDOW_START.isoformat(),
            "end": TRAINING_WINDOW_END.isoformat(),
            "days": (TRAINING_WINDOW_END - TRAINING_WINDOW_START).days + 1,
            "total_breaks_measured": (metadata or {}).get("total_breaks_measured"),
        },
        "weekday_premiums": _weekday_premiums(),
        "measurement": measurement,
        "wartime_disclosure": _wartime_disclosure(metadata),
    }


# --- routes --------------------------------------------------------------------
@router.get("")
def list_events() -> dict[str, Any]:
    """All stored events (with training-window and plan overlaps), the bundled
    holiday table, and the model-context disclosure block."""
    frame = _load_frame()
    plan_dates = _plan_dates()
    events = []
    for _, row in frame.iterrows():
        record = _record(row)
        record["window_overlap_days"] = _window_overlap_days(record)
        record["plan_overlap_dates"] = _plan_overlap_dates(record, plan_dates)
        events.append(record)
    return {"events": events, "holidays": _load_holidays(), "model_context": _model_context()}


@router.post("", status_code=201)
def create_event(payload: EventCreate, request: Request = None) -> dict[str, Any]:
    validated = _validate(payload.name, payload.type, payload.start_date,
                          payload.end_date, payload.intensity,
                          payload.price_multiplier)
    new_row = {
        "event_id": uuid.uuid4().hex[:12],
        **validated,
        "notes": str(payload.notes or ""),
        "active": str(bool(payload.active)),
    }
    with _STORE_LOCK:
        frame = _load_frame()
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.iloc[-1])


@router.put("/{event_id}")
def update_event(event_id: str, payload: EventUpdate,
                 request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, event_id)
        current = _record(frame.loc[index])
        validated = _validate(
            payload.name if payload.name is not None else current["name"],
            payload.type if payload.type is not None else current["type"],
            payload.start_date if payload.start_date is not None else current["start_date"],
            payload.end_date if payload.end_date is not None else (current["end_date"] or ""),
            payload.intensity if payload.intensity is not None else current["intensity"],
            payload.price_multiplier if payload.price_multiplier is not None
            else current["price_multiplier"],
        )
        for column, value in validated.items():
            frame.at[index, column] = value
        if payload.notes is not None:
            frame.at[index, "notes"] = str(payload.notes)
        if payload.active is not None:
            frame.at[index, "active"] = str(bool(payload.active))
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.loc[index])


@router.delete("/{event_id}")
def delete_event(event_id: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, event_id)
        frame = frame.drop(index=index).reset_index(drop=True)
        _snapshot_before_write(request)
        _write_frame(frame)
    return {"deleted": event_id}
