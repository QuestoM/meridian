"""Advertiser rules CRUD, persisted to data/advertiser_rules.csv.

Each operation reads the real CSV, mutates one row, backs the file up, and
writes it back preserving column order, serialized under a module lock and
written via a temp file plus ``os.replace`` so concurrent edits cannot lose
rows and readers never see a torn CSV. Types are coerced so the optimizer
reads clean values: default_premium as float, prime_time_only as bool.

Each record also carries a display-name layer: ``display_name`` is the editable
operator-facing name stored beside the raw ``advertiser_id`` (tolerant read, a
legacy CSV without the column reads as empty), and ``name_source`` says honestly
where the shown name comes from: ``operator`` (stored), ``observed`` (the id is
a real advertiser name seen in the daily spot data) or ``unnamed`` (raw token
only; the UI prettifies it but never invents a company name).

Beside it sits the identity layer, in the shape ``data/agencies.csv`` already
uses and for the same reason: ``name`` is the advertiser this row is about and
``aliases`` is a pipe-joined list of other spellings of it. Writing either one
BINDS the row to that advertiser, so the row's premium and its conditions start
pricing that advertiser's spots; leaving both blank, which is how all 45 shipped
rows read, binds nothing and prices nothing. The read that joins the name space,
the rules and the daily money is :mod:`kairos_api.advertisers_identity`.
"""

from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
RULES_PATH = DATA_DIR / "advertiser_rules.csv"
# The observed Hebrew advertiser vocabulary from the real daily spot files, as
# persisted by the agency layer. Used only to classify a record's name source;
# it is never written from here.
OBSERVED_NAMES_PATH = DATA_DIR / "agency_advertisers.csv"

COLUMNS = [
    "advertiser_id",
    "default_premium",
    "allow_positions",
    "allow_genres",
    "prime_time_only",
    "urgency_k",
    "ahead_k",
    "notes",
    "name",
    "display_name",
    "aliases",
]

router = APIRouter(prefix="/api/advertisers", tags=["advertisers"])

# Serializes every load-mutate-write cycle on the rules CSV so two concurrent
# edits cannot drop each other's rows (lost update).
_STORE_LOCK = threading.Lock()


class AdvertiserUpdate(BaseModel):
    """Editable fields for an advertiser rule. All optional for PATCH-style PUT.

    ``urgency_k`` / ``ahead_k`` are this advertiser's delivery-pacing-strength
    defaults: how hard its campaigns lean toward inventory when behind pace
    (urgency_k) and away when over-delivered (ahead_k). Send an empty string or a
    negative value to clear the override and fall back to the channel-wide default.
    """

    default_premium: float | None = None
    allow_positions: str | None = None
    allow_genres: str | None = None
    prime_time_only: bool | None = None
    urgency_k: float | None = None
    ahead_k: float | None = None
    clear_urgency_k: bool = False
    clear_ahead_k: bool = False
    notes: str | None = None
    # The operator-facing name shown beside the raw advertiser_id. Sending an
    # empty string clears it, so the record reads as unnamed again.
    display_name: str | None = None
    # The advertiser this row is about, and other spellings of it. Setting
    # either binds the row to that advertiser on the daily pricing path; an
    # empty string clears it and the row goes back to pricing nothing.
    name: str | None = None
    aliases: str | None = None


class AdvertiserCreate(BaseModel):
    """A new advertiser rule. advertiser_id is required.

    ``urgency_k`` / ``ahead_k`` default to ``None`` (use the channel-wide pacing
    strength); set either to give this advertiser its own default.
    """

    advertiser_id: str
    default_premium: float = 1.0
    allow_positions: str = "ANY"
    allow_genres: str = "ANY"
    prime_time_only: bool = False
    urgency_k: float | None = None
    ahead_k: float | None = None
    notes: str = ""
    display_name: str = ""
    name: str = ""
    aliases: str = ""


def _coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_opt_float(value: Any) -> float | None:
    """Read an optional non-negative pacing strength; blank/invalid/negative -> None."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0.0 else None


def _load_frame() -> pd.DataFrame:
    if not RULES_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(RULES_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _observed_names() -> frozenset[str]:
    """Advertiser names observed on the real daily spot data.

    Read tolerantly from the agency links store: a missing file or a missing
    column yields an empty set, never an error, so the advertisers list keeps
    working without the agency layer.
    """
    if not OBSERVED_NAMES_PATH.exists():
        return frozenset()
    try:
        frame = pd.read_csv(OBSERVED_NAMES_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001 - a broken side file must not break the list
        return frozenset()
    if "advertiser" not in frame.columns:
        return frozenset()
    return frozenset(name.strip() for name in frame["advertiser"].astype(str) if name.strip())


def _name_source(display_name: str, advertiser_id: str, observed: frozenset[str],
                 name: str = "") -> str:
    """Classify where a record's shown name comes from, tri-state and honest.

    ``operator``: the operator stored a display name or bound this row to an
    advertiser by name. ``observed``: the raw id itself is a real advertiser
    name seen in the daily data. ``unnamed``: only a raw token exists; the UI
    prettifies it but flags it for the operator to fill.
    """
    if display_name or name:
        return "operator"
    if advertiser_id.strip() in observed:
        return "observed"
    return "unnamed"


def _row_to_record(row: "pd.Series[Any]", observed: frozenset[str] | None = None) -> dict[str, Any]:
    if observed is None:
        observed = _observed_names()
    advertiser_id = str(row.get("advertiser_id", ""))
    display_name = str(row.get("display_name", "")).strip()
    name = str(row.get("name", "")).strip()
    return {
        "advertiser_id": advertiser_id,
        "default_premium": round(_coerce_float(row.get("default_premium")), 6),
        "allow_positions": str(row.get("allow_positions", "ANY")),
        "allow_genres": str(row.get("allow_genres", "ANY")),
        "prime_time_only": _coerce_bool(row.get("prime_time_only")),
        "urgency_k": _coerce_opt_float(row.get("urgency_k")),
        "ahead_k": _coerce_opt_float(row.get("ahead_k")),
        "notes": str(row.get("notes", "")),
        "display_name": display_name,
        "name": name,
        "aliases": str(row.get("aliases", "")).strip(),
        "name_source": _name_source(display_name, advertiser_id, observed, name),
    }


def _unclaimed_name(frame: pd.DataFrame, candidate: str, advertiser_id: str) -> str:
    """The name to store, refusing one another row is already bound to.

    Two rows bound to the same advertiser would make which row prices its spots
    depend on file order, so the second binding is rejected at the door with the
    id that already holds it, rather than resolved silently.
    """
    from kairos_api.advertisers_identity import name_is_taken

    wanted = str(candidate or "").strip()
    if not wanted:
        return ""
    rows = frame.to_dict("records")
    if name_is_taken(rows, wanted, advertiser_id):
        raise HTTPException(
            status_code=409,
            detail=f"Another advertiser row is already bound to '{wanted}'",
        )
    return wanted


def _unclaimed_aliases(frame: pd.DataFrame, candidate: str, advertiser_id: str) -> str:
    """The alias cell to store, refusing any alias another row already holds."""
    from kairos_api.advertisers_identity import normalized_aliases

    cleaned = normalized_aliases(candidate)
    if not cleaned:
        return ""
    for alias in cleaned.split("|"):
        _unclaimed_name(frame, alias, advertiser_id)
    return cleaned


def _backup() -> None:
    if not RULES_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(RULES_PATH, BACKUP_DIR / f"advertiser_rules_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    """Backup, then write atomically (temp file + os.replace, like auth_store).

    A reader that opens the CSV mid-write sees either the old or the new file,
    never a truncated one. Callers hold ``_STORE_LOCK`` across load-mutate-write.
    """
    _backup()
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RULES_PATH.with_name(RULES_PATH.name + ".tmp")
    frame[COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, RULES_PATH)


def _snapshot_before_write(request: "Request | None") -> None:
    """Record a version of the advertiser rules before a manual edit writes them."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "advertisers")


@router.get("")
def list_advertisers() -> dict[str, Any]:
    # Each advertiser carries its baseline fields plus its scoped conditions and
    # any overlap findings, so the dashboard's "what covers what" view is one call.
    from kairos_api.advertiser_conditions import conditions_for, overlaps_for

    frame = _load_frame()
    observed = _observed_names()
    advertisers = []
    for _, row in frame.iterrows():
        record = _row_to_record(row, observed)
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
    observed = _observed_names()
    effect_keys = [PREMIUM, REQUIRE, FORBID, PRESSURE]

    advertisers: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        advertiser_id = str(row.get("advertiser_id", ""))
        baseline = _row_to_record(row, observed)
        conditions = conditions_for(advertiser_id)
        breakdown = {key: 0 for key in effect_keys}
        for condition in conditions:
            effect = str(condition.get("effect", "")).strip().lower()
            if effect in breakdown:
                breakdown[effect] += 1
        advertisers.append(
            {
                "advertiser_id": advertiser_id,
                "display_name": baseline["display_name"],
                "name_source": baseline["name_source"],
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


@router.get("/identity")
def advertiser_identity() -> dict[str, Any]:
    """Every advertiser as a named record, with its rules and its money.

    Declared before the ``/{advertiser_id}`` routes so "identity" is never read
    as an advertiser id. The join and every honest empty state live in
    :mod:`kairos_api.advertisers_identity`; this route only exposes them.
    """
    from kairos_api.advertisers_identity import identity_report

    return identity_report()


@router.put("/{advertiser_id}")
def update_advertiser(advertiser_id: str, payload: AdvertiserUpdate,
                      request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
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
        if payload.clear_urgency_k:
            frame.at[index, "urgency_k"] = ""
        elif payload.urgency_k is not None:
            frame.at[index, "urgency_k"] = "" if payload.urgency_k < 0 else str(float(payload.urgency_k))
        if payload.clear_ahead_k:
            frame.at[index, "ahead_k"] = ""
        elif payload.ahead_k is not None:
            frame.at[index, "ahead_k"] = "" if payload.ahead_k < 0 else str(float(payload.ahead_k))
        if payload.notes is not None:
            frame.at[index, "notes"] = payload.notes
        if payload.display_name is not None:
            frame.at[index, "display_name"] = payload.display_name.strip()
        if payload.name is not None:
            frame.at[index, "name"] = _unclaimed_name(frame, payload.name, advertiser_id)
        if payload.aliases is not None:
            frame.at[index, "aliases"] = _unclaimed_aliases(frame, payload.aliases, advertiser_id)

        _snapshot_before_write(request)
        _write_frame(frame)
        return _row_to_record(frame.loc[index])


@router.post("")
def create_advertiser(payload: AdvertiserCreate, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        if (frame["advertiser_id"].astype(str) == payload.advertiser_id).any():
            raise HTTPException(status_code=409, detail=f"Advertiser '{payload.advertiser_id}' already exists")

        new_row = {
            "advertiser_id": payload.advertiser_id,
            "default_premium": str(float(payload.default_premium)),
            "allow_positions": payload.allow_positions,
            "allow_genres": payload.allow_genres,
            "prime_time_only": str(bool(payload.prime_time_only)),
            "urgency_k": "" if payload.urgency_k is None or payload.urgency_k < 0 else str(float(payload.urgency_k)),
            "ahead_k": "" if payload.ahead_k is None or payload.ahead_k < 0 else str(float(payload.ahead_k)),
            "notes": payload.notes,
            "display_name": payload.display_name.strip(),
            "name": _unclaimed_name(frame, payload.name, payload.advertiser_id),
            "aliases": _unclaimed_aliases(frame, payload.aliases, payload.advertiser_id),
        }
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _row_to_record(frame.iloc[-1])


@router.delete("/{advertiser_id}")
def delete_advertiser(advertiser_id: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        mask = frame["advertiser_id"].astype(str) == advertiser_id
        if not mask.any():
            raise HTTPException(status_code=404, detail=f"Advertiser '{advertiser_id}' not found")
        frame = frame[~mask].reset_index(drop=True)
        _snapshot_before_write(request)
        _write_frame(frame)
    return {"deleted": advertiser_id}
