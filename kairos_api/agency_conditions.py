"""Agency advertiser links and scoped agency conditions.

Sibling of :mod:`kairos_api.agencies` (split out to keep both modules under the
project line limit). Two stores live here:

* ``data/agency_advertisers.csv``: which advertisers buy through which agency.
  Observed links are derived LIVE from the newest daily Wally file on every
  read (the file's own 'משרד / MB' column is the source of truth), so a new
  upload refreshes the map with no migration; the seeded observed rows freeze
  the measured mapping so the layer works before any upload. Manual links are
  operator-created and override observed links per advertiser.
* ``data/agency_conditions.csv``: scoped conditional rules keyed by agency_id,
  the exact condition shape the advertiser conditions use (same effects, modes
  and scope dimensions), validated through the same engine serializers. On the
  pricing path they are evaluated agency-first, forbid wins across levels.

Same store discipline as every sibling: module lock, timestamped backup, temp
file plus ``os.replace``, snapshot-before-write into the version timeline
(safe no-ops until the version store registers the new logical names). A manual
link also names its advertiser in the name space that sits beside these two.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kairos.optimize.advertiser_rules import (
    _EFFECTS, _PREMIUM_MODES, FORBID, MULTIPLIER, REQUIRE,
    _normalize_mode, normalize_scope,
)
from kairos.optimize.positions import normalize_position_scope
from kairos_api.agency_conditions_identity import register_advertiser_name
from kairos_api.condition_validation import (
    validate_effective_mode_value, validate_mode_value, validate_weekday_scope,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
LINKS_PATH = DATA_DIR / "agency_advertisers.csv"
CONDITIONS_PATH = DATA_DIR / "agency_conditions.csv"

LINK_COLUMNS = ["agency_id", "advertiser", "source", "observed_date", "notes"]
CONDITION_COLUMNS = [
    "agency_id",
    "rule_id",
    "scope_positions",
    "scope_genres",
    "scope_dayparts",
    "scope_programmes",
    "scope_weekdays",
    "effect",
    "value",
    "mode",
    "notes",
]

router = APIRouter(prefix="/api/agencies", tags=["agencies"])

_STORE_LOCK = threading.Lock()


class LinkCreate(BaseModel):
    """A manual advertiser link. Manual links override observed links."""

    advertiser: str
    notes: str = ""


class ConditionCreate(BaseModel):
    """A new scoped agency condition, the advertiser condition shape.

    ``scope_weekdays`` is ANY or comma-joined ISO weekday numbers 1..7
    (Monday=1, Saturday=6, Sunday=7).
    """

    rule_id: str
    effect: str
    value: float = 1.0
    mode: str = MULTIPLIER
    scope_positions: str = "ANY"
    scope_genres: str = "ANY"
    scope_dayparts: str = "ANY"
    scope_programmes: str = "ANY"
    scope_weekdays: str = "ANY"
    notes: str = ""


class ConditionUpdate(BaseModel):
    """Editable fields for an agency condition. All optional for PATCH-style PUT."""

    effect: Optional[str] = None
    value: Optional[float] = None
    mode: Optional[str] = None
    scope_positions: Optional[str] = None
    scope_genres: Optional[str] = None
    scope_dayparts: Optional[str] = None
    scope_programmes: Optional[str] = None
    scope_weekdays: Optional[str] = None
    notes: Optional[str] = None


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _write_csv(path: Path, frame: pd.DataFrame, columns: list[str], backup_stem: str) -> None:
    """Backup, then write atomically (temp file + os.replace)."""
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(path, BACKUP_DIR / f"{backup_stem}_{stamp}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame[columns].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _snapshot(request: "Request | None", logical: str) -> None:
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, logical)


def _agency_engine():
    """The pricing path's own agency-condition engine, so what the API reports
    (overlaps, effective rules) is exactly what prices spots."""
    from kairos.export.agency_layer import AgencyLayer

    return AgencyLayer.from_files(conditions_path=CONDITIONS_PATH).engine


def _agency_names(agency_id: str) -> set[str]:
    """The name strings that resolve to one agency: name, display_name, aliases."""
    from kairos_api.agencies import _load_frame as _load_agencies

    frame = _load_agencies()
    mask = frame["agency_id"].astype(str) == agency_id
    names: set[str] = set()
    for _, row in frame[mask].iterrows():
        names.update(
            token.strip()
            for token in [str(row.get("name", "")), str(row.get("display_name", "")),
                          *str(row.get("aliases", "")).split("|")]
            if token.strip()
        )
    return names


def observed_pairs_from_frame(daily: pd.DataFrame) -> list[dict[str, str]]:
    """(advertiser, agency name) pairs observed in one loaded daily frame."""
    if "advertiser" not in daily.columns or "agency" not in daily.columns:
        return []
    pairs = daily[["advertiser", "agency"]].astype(str).apply(lambda s: s.str.strip())
    pairs = pairs[(pairs["advertiser"] != "") & (pairs["agency"] != "")].drop_duplicates()
    return [{"advertiser": row.advertiser, "agency": row.agency}
            for row in pairs.itertuples(index=False)]


def _latest_daily_pairs() -> tuple[list[dict[str, str]], Optional[str]]:
    """Observed pairs from the newest daily file, honest empty when none exists."""
    try:
        from kairos.data.loaders import load_daily_input
        from kairos_api.uploads import _newest_daily

        path = _newest_daily()
        if path is None:
            return [], None
        return observed_pairs_from_frame(load_daily_input(path)), path.name
    except Exception:
        return [], None


def links_for(agency_id: str) -> dict[str, Any]:
    """Observed (live-derived), stored (seed + manual) and effective links."""
    names = _agency_names(agency_id)
    live_pairs, source_file = _latest_daily_pairs()
    observed_live = sorted({p["advertiser"] for p in live_pairs if p["agency"] in names})

    frame = _load_csv(LINKS_PATH, LINK_COLUMNS)
    mine = frame[frame["agency_id"].astype(str) == agency_id]
    stored_observed = sorted(set(mine[mine["source"] != "manual"]["advertiser"].astype(str)))
    manual = sorted(set(mine[mine["source"] == "manual"]["advertiser"].astype(str)))

    # Manual wins per advertiser: a manual link to ANOTHER agency removes this
    # agency's observed claim on that advertiser.
    others_manual = set(
        frame[(frame["source"] == "manual") & (frame["agency_id"].astype(str) != agency_id)]
        ["advertiser"].astype(str)
    )
    observed = sorted(set(observed_live) | set(stored_observed))
    effective = sorted((set(observed) - others_manual) | set(manual))
    return {
        "observed": observed,
        "manual": manual,
        "effective": effective,
        "observed_source_file": source_file,
    }


def link_summary_for(agency_id: str) -> dict[str, Any]:
    links = links_for(agency_id)
    return {"advertiser_count": len(links["effective"]), "manual_count": len(links["manual"])}


def conditions_for(agency_id: str) -> list[dict[str, Any]]:
    frame = _load_csv(CONDITIONS_PATH, CONDITION_COLUMNS)
    mask = frame["agency_id"].astype(str) == agency_id
    return [_condition_record(row) for _, row in frame[mask].iterrows()]


def _condition_record(row: "pd.Series[Any]") -> dict[str, Any]:
    try:
        value = round(float(row.get("value")), 6)
    except (TypeError, ValueError):
        value = 1.0
    return {
        "agency_id": str(row.get("agency_id", "")),
        "rule_id": str(row.get("rule_id", "")),
        "effect": str(row.get("effect", "")).strip().lower(),
        "value": value,
        "mode": _normalize_mode(row.get("mode")),
        "scope_positions": normalize_position_scope(row.get("scope_positions")),
        "scope_genres": normalize_scope(row.get("scope_genres")),
        "scope_dayparts": normalize_scope(row.get("scope_dayparts")),
        "scope_programmes": normalize_scope(row.get("scope_programmes")),
        "scope_weekdays": normalize_scope(row.get("scope_weekdays")),
        "notes": str(row.get("notes", "")),
    }


def overlaps_for(agency_id: str) -> list[dict[str, Any]]:
    """The engine's pairwise findings within this agency's own rules."""
    return [asdict(finding) for finding in _agency_engine().overlaps(agency_id)]


def cross_level_overlaps(agency_id: str) -> list[dict[str, Any]]:
    """Agency rules intersected with each linked advertiser's rules.

    A require/forbid pair across the two levels is a conflict (forbid wins, at
    either level); any other intersecting pair is reported as an overlap so the
    operator sees compounded coverage before it surprises a ledger.
    """
    from kairos.optimize.advertiser_rules import AdvertiserRuleEngine

    agency_rules = _agency_engine().conditions.get(agency_id, [])
    if not agency_rules:
        return []
    advertiser_engine = AdvertiserRuleEngine.from_files()
    findings: list[dict[str, Any]] = []
    for advertiser in links_for(agency_id)["effective"]:
        for advertiser_rule in advertiser_engine.conditions.get(advertiser, []):
            for agency_rule in agency_rules:
                if not agency_rule.scope_intersects(advertiser_rule):
                    continue
                effects = {agency_rule.effect, advertiser_rule.effect}
                if effects == {REQUIRE, FORBID}:
                    kind, detail = "cross_level_conflict", (
                        "a require and a forbid cover the same scope across the agency and "
                        "advertiser levels; forbid wins at either level"
                    )
                else:
                    kind, detail = "cross_level_overlap", (
                        f"agency {agency_rule.effect} and advertiser {advertiser_rule.effect} "
                        "rules cover the same scope; premiums compose multiplicatively"
                    )
                findings.append({
                    "agency_id": agency_id, "advertiser": advertiser, "kind": kind,
                    "agency_rule_id": agency_rule.rule_id,
                    "advertiser_rule_id": advertiser_rule.rule_id, "detail": detail,
                })
    return findings


def _validate_effect(effect: str) -> str:
    cleaned = str(effect or "").strip().lower()
    if cleaned not in _EFFECTS:
        raise HTTPException(status_code=400, detail=f"effect must be one of {sorted(_EFFECTS)}")
    return cleaned


def _require_agency(agency_id: str) -> None:
    from kairos_api.agencies import _load_frame as _load_agencies

    if not (_load_agencies()["agency_id"].astype(str) == agency_id).any():
        raise HTTPException(status_code=404, detail=f"Agency '{agency_id}' not found")


@router.get("/{agency_id}/advertisers")
def list_links(agency_id: str) -> dict[str, Any]:
    _require_agency(agency_id)
    return links_for(agency_id)


@router.post("/{agency_id}/advertisers", status_code=201)
def create_link(agency_id: str, payload: LinkCreate, request: Request = None) -> dict[str, Any]:
    _require_agency(agency_id)
    advertiser = payload.advertiser.strip()
    if not advertiser:
        raise HTTPException(status_code=400, detail="advertiser is required")
    with _STORE_LOCK:
        frame = _load_csv(LINKS_PATH, LINK_COLUMNS)
        manual = frame[(frame["source"] == "manual") & (frame["advertiser"].astype(str) == advertiser)]
        if not manual.empty:
            holder = str(manual.iloc[0]["agency_id"])
            raise HTTPException(status_code=409, detail=(
                f"'{advertiser}' already has a manual link to agency '{holder}'; remove it first"))
        new_row = {"agency_id": agency_id, "advertiser": advertiser, "source": "manual",
                   "observed_date": "", "notes": payload.notes}
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot(request, "agency_links")
        _write_csv(LINKS_PATH, frame, LINK_COLUMNS, "agency_advertisers")
    identity = register_advertiser_name(advertiser, store_path=LINKS_PATH)
    return {"linked": advertiser, "agency_id": agency_id, "source": "manual", "identity": identity}


@router.delete("/{agency_id}/advertisers/{advertiser}")
def delete_link(agency_id: str, advertiser: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_csv(LINKS_PATH, LINK_COLUMNS)
        mask = (
            (frame["agency_id"].astype(str) == agency_id)
            & (frame["advertiser"].astype(str) == advertiser)
            & (frame["source"] == "manual")
        )
        if not mask.any():
            raise HTTPException(status_code=404, detail=(
                f"no manual link from '{advertiser}' to agency '{agency_id}'"))
        frame = frame[~mask].reset_index(drop=True)
        _snapshot(request, "agency_links")
        _write_csv(LINKS_PATH, frame, LINK_COLUMNS, "agency_advertisers")
    return {"unlinked": advertiser, "agency_id": agency_id}


@router.get("/{agency_id}/conditions")
def list_conditions(agency_id: str) -> dict[str, Any]:
    _require_agency(agency_id)
    return {
        "conditions": conditions_for(agency_id),
        "overlaps": overlaps_for(agency_id),
        "cross_level": cross_level_overlaps(agency_id),
    }


@router.post("/{agency_id}/conditions", status_code=201)
def create_condition(agency_id: str, payload: ConditionCreate,
                     request: Request = None) -> dict[str, Any]:
    _require_agency(agency_id)
    mode = _normalize_mode(payload.mode)
    validate_mode_value(mode, payload.value)
    new_row = {
        "agency_id": agency_id,
        "rule_id": payload.rule_id,
        "effect": _validate_effect(payload.effect),
        "value": str(float(payload.value)),
        "mode": mode,
        "scope_positions": normalize_position_scope(payload.scope_positions),
        "scope_genres": normalize_scope(payload.scope_genres),
        "scope_dayparts": normalize_scope(payload.scope_dayparts),
        "scope_programmes": normalize_scope(payload.scope_programmes),
        "scope_weekdays": validate_weekday_scope(payload.scope_weekdays),
        "notes": payload.notes,
    }
    with _STORE_LOCK:
        frame = _load_csv(CONDITIONS_PATH, CONDITION_COLUMNS)
        duplicate = (
            (frame["agency_id"].astype(str) == agency_id)
            & (frame["rule_id"].astype(str) == payload.rule_id)
        )
        if duplicate.any():
            raise HTTPException(
                status_code=409,
                detail=f"rule '{payload.rule_id}' already exists for agency '{agency_id}'",
            )
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot(request, "agency_conditions")
        _write_csv(CONDITIONS_PATH, frame, CONDITION_COLUMNS, "agency_conditions")
        return _condition_record(frame.iloc[-1])


def _locate_condition(frame: pd.DataFrame, agency_id: str, rule_id: str) -> int:
    mask = (
        (frame["agency_id"].astype(str) == agency_id)
        & (frame["rule_id"].astype(str) == rule_id)
    )
    if not mask.any():
        raise HTTPException(
            status_code=404, detail=f"rule '{rule_id}' not found for agency '{agency_id}'"
        )
    return int(frame.index[mask][0])


@router.put("/{agency_id}/conditions/{rule_id}")
def update_condition(agency_id: str, rule_id: str, payload: ConditionUpdate,
                     request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_csv(CONDITIONS_PATH, CONDITION_COLUMNS)
        index = _locate_condition(frame, agency_id, rule_id)
        effective_mode = _normalize_mode(
            payload.mode if payload.mode is not None else frame.at[index, "mode"]
        )
        validate_effective_mode_value(effective_mode, payload.value, frame.at[index, "value"])
        if payload.effect is not None:
            frame.at[index, "effect"] = _validate_effect(payload.effect)
        if payload.value is not None:
            frame.at[index, "value"] = str(float(payload.value))
        if payload.mode is not None:
            frame.at[index, "mode"] = effective_mode
        for scope in ("scope_positions", "scope_genres", "scope_dayparts", "scope_programmes"):
            value = getattr(payload, scope)
            if value is not None:
                # Positions carry their own vocabulary (1 to 5 and L); every
                # other scope dimension stays free text.
                reader = normalize_position_scope if scope == "scope_positions" else normalize_scope
                frame.at[index, scope] = reader(value)
        if payload.scope_weekdays is not None:
            frame.at[index, "scope_weekdays"] = validate_weekday_scope(payload.scope_weekdays)
        if payload.notes is not None:
            frame.at[index, "notes"] = payload.notes
        _snapshot(request, "agency_conditions")
        _write_csv(CONDITIONS_PATH, frame, CONDITION_COLUMNS, "agency_conditions")
        return _condition_record(frame.loc[index])


@router.delete("/{agency_id}/conditions/{rule_id}")
def delete_condition(agency_id: str, rule_id: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_csv(CONDITIONS_PATH, CONDITION_COLUMNS)
        index = _locate_condition(frame, agency_id, rule_id)
        frame = frame.drop(index=index).reset_index(drop=True)
        _snapshot(request, "agency_conditions")
        _write_csv(CONDITIONS_PATH, frame, CONDITION_COLUMNS, "agency_conditions")
    return {"deleted": rule_id, "agency_id": agency_id}
