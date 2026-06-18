"""Advertiser rules CRUD, persisted to data/advertiser_rules.csv.

Each operation reads the real CSV, mutates one row, backs the file up, and
writes it back preserving column order. Types are coerced so the optimizer
reads clean values: default_premium as float, prime_time_only as bool.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
RULES_PATH = DATA_DIR / "advertiser_rules.csv"

COLUMNS = [
    "advertiser_id",
    "default_premium",
    "allow_positions",
    "allow_genres",
    "prime_time_only",
    "notes",
]

router = APIRouter(prefix="/api/advertisers", tags=["advertisers"])


class AdvertiserUpdate(BaseModel):
    """Editable fields for an advertiser rule. All optional for PATCH-style PUT."""

    default_premium: float | None = None
    allow_positions: str | None = None
    allow_genres: str | None = None
    prime_time_only: bool | None = None
    notes: str | None = None


class AdvertiserCreate(BaseModel):
    """A new advertiser rule. advertiser_id is required."""

    advertiser_id: str
    default_premium: float = 1.0
    allow_positions: str = "ANY"
    allow_genres: str = "ANY"
    prime_time_only: bool = False
    notes: str = ""


def _coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_frame() -> pd.DataFrame:
    if not RULES_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(RULES_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _row_to_record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {
        "advertiser_id": str(row.get("advertiser_id", "")),
        "default_premium": round(_coerce_float(row.get("default_premium")), 6),
        "allow_positions": str(row.get("allow_positions", "ANY")),
        "allow_genres": str(row.get("allow_genres", "ANY")),
        "prime_time_only": _coerce_bool(row.get("prime_time_only")),
        "notes": str(row.get("notes", "")),
    }


def _backup() -> None:
    if not RULES_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(RULES_PATH, BACKUP_DIR / f"advertiser_rules_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    _backup()
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(RULES_PATH, index=False, encoding="utf-8-sig")


@router.get("")
def list_advertisers() -> dict[str, Any]:
    # Each advertiser carries its baseline fields plus its scoped conditions and
    # any overlap findings, so the dashboard's "what covers what" view is one call.
    from kairos_api.advertiser_conditions import conditions_for, overlaps_for

    frame = _load_frame()
    advertisers = []
    for _, row in frame.iterrows():
        record = _row_to_record(row)
        advertiser_id = record["advertiser_id"]
        record["conditions"] = conditions_for(advertiser_id)
        record["overlaps"] = overlaps_for(advertiser_id)
        advertisers.append(record)
    return {"advertisers": advertisers, "columns": COLUMNS}


@router.get("/stats")
def advertiser_stats() -> dict[str, Any]:
    """Per-advertiser at-a-glance stats for the management zone.

    Every figure is computed from real data: baselines from advertiser_rules.csv
    and scoped conditions from the conditions store, via the AdvertiserRuleEngine.
    ``avg_effective_premium`` is the engine's deterministic unscoped effective
    premium (baseline times every ANY-scope premium condition); it is a real
    multiplier, not an estimate.

    Revenue/profitability fields are NULL with a ``source_pending`` marker because
    real spot-revenue attribution lives only on the daily spot-pricing path. The
    ``status`` field carries the honest caveat that the WEEKLY optimizer does not
    consume advertiser rules at all; only the daily path prices against them.
    """
    from kairos.optimize.advertiser_rules import (
        FORBID,
        PREMIUM,
        PRESSURE,
        REQUIRE,
    )
    from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
    from kairos_api.advertiser_conditions import conditions_for

    engine = AdvertiserRuleEngine.from_files()
    frame = _load_frame()
    effect_keys = [PREMIUM, REQUIRE, FORBID, PRESSURE]

    advertisers: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        advertiser_id = str(row.get("advertiser_id", ""))
        baseline = _row_to_record(row)
        conditions = conditions_for(advertiser_id)
        breakdown = {key: 0 for key in effect_keys}
        for condition in conditions:
            effect = str(condition.get("effect", "")).strip().lower()
            if effect in breakdown:
                breakdown[effect] += 1
        advertisers.append(
            {
                "advertiser_id": advertiser_id,
                "rule_count": len(conditions),
                "effect_breakdown": breakdown,
                "baseline_premium": round(baseline["default_premium"], 6),
                "avg_effective_premium": round(engine.effective_premium(advertiser_id), 6),
                "has_conditions": len(conditions) > 0,
                "revenue": None,
                "profitability": None,
                "revenue_source": "source_pending",
            }
        )

    return {
        "advertisers": advertisers,
        "count": len(advertisers),
        "effect_types": effect_keys,
        "revenue_note": "Spot-revenue attribution is computed on the daily spot-pricing path only; not available in this read-only aggregate.",
        "status": "The weekly optimizer does not consume advertiser rules; only the daily spot-pricing path prices against them.",
    }


@router.put("/{advertiser_id}")
def update_advertiser(advertiser_id: str, payload: AdvertiserUpdate) -> dict[str, Any]:
    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == advertiser_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Advertiser '{advertiser_id}' not found")

    index = frame.index[mask][0]
    if payload.default_premium is not None:
        frame.at[index, "default_premium"] = str(float(payload.default_premium))
    if payload.allow_positions is not None:
        frame.at[index, "allow_positions"] = payload.allow_positions
    if payload.allow_genres is not None:
        frame.at[index, "allow_genres"] = payload.allow_genres
    if payload.prime_time_only is not None:
        frame.at[index, "prime_time_only"] = str(bool(payload.prime_time_only))
    if payload.notes is not None:
        frame.at[index, "notes"] = payload.notes

    _write_frame(frame)
    return _row_to_record(frame.loc[index])


@router.post("")
def create_advertiser(payload: AdvertiserCreate) -> dict[str, Any]:
    frame = _load_frame()
    if (frame["advertiser_id"].astype(str) == payload.advertiser_id).any():
        raise HTTPException(status_code=409, detail=f"Advertiser '{payload.advertiser_id}' already exists")

    new_row = {
        "advertiser_id": payload.advertiser_id,
        "default_premium": str(float(payload.default_premium)),
        "allow_positions": payload.allow_positions,
        "allow_genres": payload.allow_genres,
        "prime_time_only": str(bool(payload.prime_time_only)),
        "notes": payload.notes,
    }
    frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
    _write_frame(frame)
    return _row_to_record(frame.iloc[-1])


@router.delete("/{advertiser_id}")
def delete_advertiser(advertiser_id: str) -> dict[str, Any]:
    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == advertiser_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Advertiser '{advertiser_id}' not found")
    frame = frame[~mask].reset_index(drop=True)
    _write_frame(frame)
    return {"deleted": advertiser_id}
