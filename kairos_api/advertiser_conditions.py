"""Scoped advertiser-condition CRUD, persisted to data/advertiser_conditions.csv.

This is the sibling of :mod:`kairos_api.advertisers`. The baseline rules live in
``advertiser_rules.csv``; the scoped conditional rules (premium multipliers,
requirements, forbids keyed by position/genre/daypart) live here. Each operation
reads the real CSV, mutates one row, backs the file up the way advertisers.py
does, and writes it back preserving column order. Scopes are normalized through
the engine's own serializer so what is read back matches the engine's token
semantics, and ``value`` is coerced to float so the optimizer reads clean
numbers.

Nothing here is invented: an empty conditions file (header only, the seeded
state) yields no conditions, and the overlap view returns exactly what the pure
:class:`~kairos.optimize.advertiser_rules.AdvertiserRuleEngine` reports.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kairos.optimize.advertiser_rules import (
    _EFFECTS,
    _PREMIUM_MODES,
    GOLD_POSITION,
    MULTIPLIER,
    AdvertiserRuleEngine,
    _normalize_mode,
    normalize_scope,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
CONDITIONS_PATH = DATA_DIR / "advertiser_conditions.csv"

COLUMNS = [
    "advertiser_id",
    "rule_id",
    "scope_positions",
    "scope_genres",
    "scope_dayparts",
    "scope_programmes",
    "effect",
    "value",
    "mode",
    "notes",
]

router = APIRouter(prefix="/api/advertisers", tags=["advertisers"])


class ConditionCreate(BaseModel):
    """A new scoped condition.

    ``effect`` is premium / require / forbid / pressure. ``mode`` only matters for
    a premium effect; it says how to read ``value`` (multiplier / percent /
    cpp_absolute / cpp_add / cpp_discount, see the engine). ``scope_programmes``
    scopes the rule to specific show titles, like the other scope dimensions.
    """

    rule_id: str
    effect: str
    value: float = 1.0
    mode: str = MULTIPLIER
    scope_positions: str = "ANY"
    scope_genres: str = "ANY"
    scope_dayparts: str = "ANY"
    scope_programmes: str = "ANY"
    notes: str = ""


class ConditionUpdate(BaseModel):
    """Editable fields for a condition. All optional for PATCH-style PUT."""

    effect: str | None = None
    value: float | None = None
    mode: str | None = None
    scope_positions: str | None = None
    scope_genres: str | None = None
    scope_dayparts: str | None = None
    scope_programmes: str | None = None
    notes: str | None = None


def _load_frame() -> pd.DataFrame:
    if not CONDITIONS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(CONDITIONS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not CONDITIONS_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(CONDITIONS_PATH, BACKUP_DIR / f"advertiser_conditions_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    _backup()
    CONDITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(CONDITIONS_PATH, index=False, encoding="utf-8-sig")


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_to_record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {
        "advertiser_id": str(row.get("advertiser_id", "")),
        "rule_id": str(row.get("rule_id", "")),
        "effect": str(row.get("effect", "")).strip().lower(),
        "value": round(_coerce_float(row.get("value")), 6),
        "mode": _normalize_mode(row.get("mode")),
        "scope_positions": normalize_scope(row.get("scope_positions")),
        "scope_genres": normalize_scope(row.get("scope_genres")),
        "scope_dayparts": normalize_scope(row.get("scope_dayparts")),
        "scope_programmes": normalize_scope(row.get("scope_programmes")),
        "notes": str(row.get("notes", "")),
    }


def _validate_effect(effect: str) -> str:
    cleaned = str(effect or "").strip().lower()
    if cleaned not in _EFFECTS:
        raise HTTPException(status_code=400, detail=f"effect must be one of {sorted(_EFFECTS)}")
    return cleaned


def conditions_for(advertiser_id: str) -> list[dict[str, Any]]:
    """All stored condition records for one advertiser (used by the list view)."""
    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == advertiser_id
    return [_row_to_record(row) for _, row in frame[mask].iterrows()]


def overlaps_for(advertiser_id: str) -> list[dict[str, Any]]:
    """The engine's overlap findings for one advertiser, as plain dicts."""
    engine = AdvertiserRuleEngine.from_files()
    return [asdict(finding) for finding in engine.overlaps(advertiser_id)]


@router.get("/overlaps")
def list_all_overlaps() -> dict[str, Any]:
    """Every advertiser's overlap findings in one call (operator review view)."""
    engine = AdvertiserRuleEngine.from_files()
    findings: list[dict[str, Any]] = []
    for advertiser_id in engine.conditions:
        findings.extend(asdict(finding) for finding in engine.overlaps(advertiser_id))
    return {"overlaps": findings}


def _position_options() -> list[dict[str, str]]:
    """The break-position tokens a rule can scope, including the gold break.

    The model's break positions are first/middle/last (the daily path also tags an
    integer position); the premium gold break (Hebrew: ברייק זהב) is offered as an
    explicit token so an operator can scope a rule to it.
    """
    from kairos.model.spec import DEFAULT_BREAK_POSITIONS

    labels_he = {"first": "ראשון", "middle": "אמצעי", "last": "אחרון"}
    options = [
        {"key": key, "he": labels_he.get(key, key), "en": key.capitalize()}
        for key in DEFAULT_BREAK_POSITIONS
    ]
    options.append({"key": GOLD_POSITION, "he": "ברייק זהב", "en": "Gold break"})
    return options


def _genre_options() -> list[str]:
    """The real genre vocabulary from the program classifier taxonomy."""
    try:
        from kairos.data.classifier import ProgramClassifier

        categories = ProgramClassifier.from_yaml().categories
        return list(categories() if callable(categories) else categories)
    except Exception:
        return []


def _daypart_options() -> list[dict[str, Any]]:
    """The canonical Israeli-TV daypart options (bilingual, with hour windows)."""
    try:
        from kairos.data.dayparts import daypart_options

        return list(daypart_options())
    except Exception:
        return []


def _programme_options() -> list[str]:
    """Real programme titles, sorted and de-duplicated, for the multi-select.

    Sourced from the reference EPG (``Title`` column). Returns an honest empty
    list if the file is missing rather than inventing names.
    """
    try:
        from kairos.data.loaders import load_programmes

        frame = load_programmes()
        if "Title" not in frame.columns:
            return []
        titles = {str(t).strip() for t in frame["Title"].dropna() if str(t).strip()}
        return sorted(titles)
    except Exception:
        return []


@router.get("/options")
def scope_options() -> dict[str, Any]:
    """Option lists the dashboard needs to build scoped advertiser rules.

    Everything here is sourced from real engine config and reference data (genres
    from the classifier taxonomy, dayparts from the canonical taxonomy, programmes
    from the EPG, positions from the model's break-position vocabulary plus the
    gold break). ``effects`` and ``modes`` come straight from the rule engine, so
    the dashboard never invents a value the engine cannot consume.
    """
    return {
        "positions": _position_options(),
        "genres": _genre_options(),
        "dayparts": _daypart_options(),
        "programmes": _programme_options(),
        "effects": sorted(_EFFECTS),
        "modes": list(_PREMIUM_MODES),
    }


@router.get("/{advertiser_id}/conditions")
def list_conditions(advertiser_id: str) -> dict[str, Any]:
    return {"conditions": conditions_for(advertiser_id), "overlaps": overlaps_for(advertiser_id)}


@router.post("/{advertiser_id}/conditions", status_code=201)
def create_condition(advertiser_id: str, payload: ConditionCreate) -> dict[str, Any]:
    frame = _load_frame()
    duplicate = (
        (frame["advertiser_id"].astype(str) == advertiser_id)
        & (frame["rule_id"].astype(str) == payload.rule_id)
    )
    if duplicate.any():
        raise HTTPException(
            status_code=409,
            detail=f"rule '{payload.rule_id}' already exists for advertiser '{advertiser_id}'",
        )
    new_row = {
        "advertiser_id": advertiser_id,
        "rule_id": payload.rule_id,
        "effect": _validate_effect(payload.effect),
        "value": str(float(payload.value)),
        "mode": _normalize_mode(payload.mode),
        "scope_positions": normalize_scope(payload.scope_positions),
        "scope_genres": normalize_scope(payload.scope_genres),
        "scope_dayparts": normalize_scope(payload.scope_dayparts),
        "scope_programmes": normalize_scope(payload.scope_programmes),
        "notes": payload.notes,
    }
    frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
    _write_frame(frame)
    return _row_to_record(frame.iloc[-1])


def _locate(frame: pd.DataFrame, advertiser_id: str, rule_id: str) -> int:
    mask = (
        (frame["advertiser_id"].astype(str) == advertiser_id)
        & (frame["rule_id"].astype(str) == rule_id)
    )
    if not mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"rule '{rule_id}' not found for advertiser '{advertiser_id}'",
        )
    return int(frame.index[mask][0])


@router.put("/{advertiser_id}/conditions/{rule_id}")
def update_condition(advertiser_id: str, rule_id: str, payload: ConditionUpdate) -> dict[str, Any]:
    frame = _load_frame()
    index = _locate(frame, advertiser_id, rule_id)
    if payload.effect is not None:
        frame.at[index, "effect"] = _validate_effect(payload.effect)
    if payload.value is not None:
        frame.at[index, "value"] = str(float(payload.value))
    if payload.mode is not None:
        frame.at[index, "mode"] = _normalize_mode(payload.mode)
    if payload.scope_positions is not None:
        frame.at[index, "scope_positions"] = normalize_scope(payload.scope_positions)
    if payload.scope_genres is not None:
        frame.at[index, "scope_genres"] = normalize_scope(payload.scope_genres)
    if payload.scope_dayparts is not None:
        frame.at[index, "scope_dayparts"] = normalize_scope(payload.scope_dayparts)
    if payload.scope_programmes is not None:
        frame.at[index, "scope_programmes"] = normalize_scope(payload.scope_programmes)
    if payload.notes is not None:
        frame.at[index, "notes"] = payload.notes
    _write_frame(frame)
    return _row_to_record(frame.loc[index])


@router.delete("/{advertiser_id}/conditions/{rule_id}")
def delete_condition(advertiser_id: str, rule_id: str) -> dict[str, Any]:
    frame = _load_frame()
    index = _locate(frame, advertiser_id, rule_id)
    frame = frame.drop(index=index).reset_index(drop=True)
    _write_frame(frame)
    return {"deleted": rule_id, "advertiser_id": advertiser_id}
