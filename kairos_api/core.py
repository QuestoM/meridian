"""Shared kernel for the Kairos API: settings, paths, cached loaders.

The API is a modular monolith: one FastAPI process composed of domain routers
(overrides, pricing, uploads, constraints, phase B, recompute, ...) that all
share this kernel. Everything here moved verbatim from server.py so the names
and cache objects stay identical; server.py re-exports them for compatibility.
Keep this module dependency-light and side-effect free: it owns the engine
availability probe, the operator settings contract, the file loaders with
their mtime-keyed caches, and the small shared response helpers (records,
series, signature, schedule summary), nothing else.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

from kairos.optimize.guardrails import Guardrails

logger = logging.getLogger(__name__)

# The real optimization engine is imported defensively: if its dependencies are
# absent the rest of the API still boots, and the engine-backed endpoints report
# that honestly instead of crashing.
try:
    from dataclasses import asdict as _asdict

    from kairos.data.loaders import CHANNELS as KAIROS_CHANNELS
    from kairos.export.schedule import build_weekly_schedule, write_weekly_schedule
    from kairos.optimize.pricing import OptimizerAssumptions, PricingModel
    from kairos.service import guardrails_from_settings, optimize_day_plan, run_scenario

    _ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover - engine optional at import time
    _ENGINE_AVAILABLE = False
    _asdict = None
    KAIROS_CHANNELS = ()
    build_weekly_schedule = None
    write_weekly_schedule = None
    OptimizerAssumptions = None
    PricingModel = None
    guardrails_from_settings = None
    optimize_day_plan = None
    run_scenario = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
SETTINGS_PATH = DATA_DIR / "kairos_settings.json"


class KairosSettings(BaseModel):
    """Operational controls for market, regulatory, and UX behavior.

    These values are deliberately configurable because Israeli TV rules,
    internal sales policy, and customer contracts can change.
    """

    profile_name: str = "Israel commercial TV"
    locale: Literal["he", "en"] = "he"
    direction: Literal["rtl", "ltr"] = "rtl"
    chart_direction: Literal["ltr"] = "ltr"
    timezone: str = "Asia/Jerusalem"
    currency: str = "ILS"
    effective_date: str = "2026-06-14"
    regulatory_source_url: str = "https://www.rashut2.org.il/"
    # The single most important lever: how the optimizer balances ad revenue
    # against viewer retention. 0 protects retention only (places no breaks),
    # 100 chases revenue only (fills to the guardrails); 60 is a revenue-leaning
    # balance. Persisted here so the operator's choice drives the saved weekly
    # schedule, the frontier, and the forecasts, not just a transient simulation.
    revenue_weight: int = Field(default=60, ge=0, le=100)
    max_ad_minutes_per_hour: float = Field(default=12.0, ge=0, le=60)
    max_breaks_per_hour: int = Field(default=4, ge=1, le=20)
    min_break_spacing_minutes: int = Field(default=7, ge=0, le=120)
    min_retention_floor: float = Field(default=0.72, ge=0, le=1)
    risk_lambda: float = Field(default=0.0, ge=0, le=1)
    max_daily_ad_minutes: int = Field(default=160, ge=0, le=1440)
    protected_program_types: list[str] = Field(default_factory=lambda: ["News", "Kids", "Children"])
    protected_program_max_ad_minutes_per_hour: float = Field(default=8.0, ge=0, le=60)
    sponsorships_enabled: bool = True
    gold_breaks_enabled: bool = True
    gold_breaks_max_per_day: int = Field(default=3, ge=0, le=50)
    require_manual_approval: bool = True
    notes: str = "Configurable baseline. Validate with current counsel and broadcaster policy before production use."
    # The operator is the client and owns exactly one channel. All placement
    # constraints are scoped to this channel automatically; the resolver never
    # touches another channel's breaks. Empty string = not yet configured
    # (constraints match any channel, an honest no-op until the operator picks one).
    operator_channel: str = ""
    # Delivery pacing: steer placement toward campaigns that are BEHIND their
    # flight pace and away from campaigns that are AHEAD (over-delivered). This is
    # a placement-bias signal only; it never changes charged revenue. It is also an
    # exact identity no-op until real campaign rows land in campaign_flights.csv, so
    # the defaults are safe. pacing_reference_date pins "today" for the pace math
    # (empty = use the run's own date); urgency_k/ahead_k set how hard a behind/ahead
    # campaign is pushed; urgency_max caps the boost and weight_floor floors the
    # over-delivery penalty so a slot is de-prioritized but never forbidden.
    pacing_enabled: bool = True
    pacing_reference_date: str = ""
    pacing_urgency_k: float = Field(default=1.0, ge=0, le=10)
    pacing_urgency_max: float = Field(default=2.0, ge=1, le=10)
    pacing_ahead_k: float = Field(default=1.0, ge=0, le=10)
    pacing_weight_floor: float = Field(default=0.5, ge=0, le=1)
    pacing_epsilon: float = Field(default=0.05, ge=0.001, le=1)
    # How the optimizer scores a plan. 'blend' (default) maximizes the unitless
    # revenue-vs-retention convex blend, the shipped behavior. 'revenue_net'
    # maximizes revenue minus the retention cost priced in ILS (lost baseline_tvr
    # valued at the real CPP), so it drops breaks whose retention cost outweighs
    # their revenue. Off by default: switching to revenue_net moves the saved plan
    # (fewer breaks, higher retention, lower gross but higher net), so it is a
    # deliberate operator choice made on the optimizer page, never silent.
    objective_mode: Literal["blend", "revenue_net"] = "blend"
    # Pricing hierarchy overrides: the operator's dashboard edits to the rate card, in
    # the same nested shape as config/optimization_weights.yaml (base_price_per_second_
    # per_tvr_point, premiums.{program_type,day_of_week,position_in_break,ad_type,show},
    # pricing_activation.{position,ad_type,show}). Deep-merged onto the YAML defaults by
    # PricingModel.from_config, so an empty dict is an exact identity to the shipped rate
    # card: the optimizer, dashboard and export are unchanged until the operator edits a
    # value. Default-OFF activation keeps revenue unchanged until the operator opts in.
    # See docs/pricing-hierarchy-design.md and the /api/pricing endpoints.
    pricing_overrides: dict[str, Any] = Field(default_factory=dict)


def _safe_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise HTTPException(status_code=400, detail="Path is outside project root")
    return path


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if not kwargs:
        stat = path.stat()
        return _read_csv_cached(str(path), stat.st_mtime_ns, stat.st_size).copy()
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


@lru_cache(maxsize=64)
def _read_csv_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    return pd.read_csv(Path(path), encoding="utf-8-sig")


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _load_settings() -> KairosSettings:
    if not SETTINGS_PATH.exists():
        return KairosSettings()
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            return KairosSettings(**json.load(handle))
    except (OSError, ValueError, TypeError):
        return KairosSettings()


def _save_settings(settings: KairosSettings) -> KairosSettings:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_model_dump(settings), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return settings


def _settings_to_guardrails(settings: KairosSettings) -> Guardrails:
    return Guardrails(
        max_ad_seconds_per_hour=settings.max_ad_minutes_per_hour * 60,
        max_breaks_per_hour=settings.max_breaks_per_hour,
        min_break_spacing_seconds=settings.min_break_spacing_minutes * 60,
        min_retention_floor=settings.min_retention_floor,
        max_daily_ad_seconds=settings.max_daily_ad_minutes * 60,
        protected_program_types=tuple(settings.protected_program_types),
        protected_max_ad_seconds_per_hour=settings.protected_program_max_ad_minutes_per_hour * 60,
        gold_breaks_max_per_day=settings.gold_breaks_max_per_day,
    )


def _reference_today(settings: KairosSettings) -> date:
    """The reference date for delivery-pacing math.

    Prefers the explicit ``pacing_reference_date`` (the operator's pinned "today"),
    falls back to the profile's ``effective_date``, and finally to the real current
    date. Pure-string parsing so a malformed value degrades to ``date.today()``
    rather than raising. The pacing math is identity until campaign rows land, so an
    imperfect date is harmless until the operator uploads real flights.
    """
    for text in (settings.pacing_reference_date, settings.effective_date):
        head = str(text or "").strip().split(" ")[0].split("T")[0]
        try:
            return date.fromisoformat(head)
        except ValueError:
            continue
    return date.today()


def _pacing_call_kwargs() -> dict[str, Any]:
    """Saved-settings ``today`` + ``settings`` to forward to the optimizer service.

    Centralizes how every scenario/plan call threads the pacing reference date and
    the dashboard pacing knobs, so the over-delivery steer is consistent across the
    scenario slider, the frontier, the weekly plan and the day plan.
    """
    saved = _load_settings()
    return {"today": _reference_today(saved), "settings": _model_dump(saved)}


def _percent(value: Any) -> float:
    numeric = _safe_number(value, 0.0)
    if numeric <= 1.5:
        return numeric * 100
    return numeric


def _ratio(value: Any) -> float:
    numeric = _safe_number(value, 0.0)
    if numeric > 1.5:
        return numeric / 100
    return numeric


def _money(value: Any) -> float:
    return round(_safe_number(value, 0.0), 2)


def _time_to_seconds(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parts = [int(float(part)) for part in text.split(":")[:3]]
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return 0.0
        return float(parsed.hour * 3600 + parsed.minute * 60 + parsed.second)
    if len(parts) == 1:
        return float(parts[0] * 3600)
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    else:
        hour, minute, second = parts
    return float(hour * 3600 + minute * 60 + second)


def _load_break_schedule() -> pd.DataFrame:
    # Only accept a candidate that carries the weekly-plan contract columns. The
    # legacy optimization_results.csv is a Spots-shaped artifact from the older
    # optimizer (no predicted_revenue / predicted_retention / num_breaks), so
    # loading it would make every builder substitute placeholder zeros and fake a
    # plan. Guarding the schema keeps the API honest: when no real plan exists we
    # return empty and the endpoints report "run the optimizer" rather than zeros.
    required = {"predicted_revenue", "predicted_retention", "num_breaks"}
    candidates = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
    ]
    for path in candidates:
        frame = _read_csv(path)
        if not frame.empty and required.issubset(frame.columns):
            return frame
    return pd.DataFrame()


@lru_cache(maxsize=4)
def _load_programmes_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Parse the programmes xlsx once per (path, mtime, size). The reference
    parse is seconds-slow on the real file, and several builders load it per
    request, so memoize on the file signature and hand back a copy."""
    del mtime_ns, size
    from kairos.data.loaders import load_programmes as _lp
    return _lp(Path(path))


def _load_programmes() -> pd.DataFrame:
    """Load EPG from the authoritative reference xlsx; fall back to legacy CSV."""
    xlsx = DATA_DIR / "reference" / "Programmes.xlsx"
    if xlsx.exists() and _ENGINE_AVAILABLE:
        try:
            stat = xlsx.stat()
            return _load_programmes_cached(str(xlsx), stat.st_mtime_ns, stat.st_size).copy()
        except Exception:
            logger.exception("reference xlsx load failed, falling back to legacy CSV")
    return _read_csv(DATA_DIR / "Programmes.csv")


@lru_cache(maxsize=4)
def _load_spots_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Parse the spots xlsx once per (path, mtime, size). The reference parse is
    tens-of-seconds slow on the real 50k-row file (date combination), and the
    overview, inventory, and campaigns builders each load it per request, so
    memoize on the file signature and hand back a copy."""
    del mtime_ns, size
    from kairos.data.loaders import load_spots as _ls
    return _ls(Path(path))


def _load_spots() -> pd.DataFrame:
    """Load spots from the authoritative reference xlsx; fall back to legacy CSV."""
    xlsx = DATA_DIR / "reference" / "Spots.xlsx"
    if xlsx.exists() and _ENGINE_AVAILABLE:
        try:
            stat = xlsx.stat()
            return _load_spots_cached(str(xlsx), stat.st_mtime_ns, stat.st_size).copy()
        except Exception:
            logger.exception("reference xlsx load failed, falling back to legacy CSV")
    return _read_csv(DATA_DIR / "Spots.csv")


def _signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    result = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            result.append((str(path), 0, 0))
    return tuple(result)


def _records(frame: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.head(limit).replace({pd.NA: None}).where(pd.notna(frame.head(limit)), None)
    return cleaned.to_dict("records")


def _series(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    """Return frame[name] if present, else a Series of `default` aligned to frame.

    DataFrame.get(name, default) returns the bare scalar default when the column
    is missing, which then has no .fillna/.astype and raises AttributeError. This
    guarantees a Series so the builders degrade to honest zeros on a restructured
    CSV (for example a Spots export without a revenue_ils column) instead of
    crashing the endpoint into a 500.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


# Retention shortfall (in percentage points below the configured floor) that the
# honest schedule risk score treats as the worst case (reads as 100). A 30-point
# shortfall (for example floor 72%, realised 42%) is already a severe plan, so it
# anchors the top of the scale; smaller shortfalls scale linearly toward 0.
_RISK_FULL_SHORTFALL = 30.0


def _risk_from_retention(average_retention_percent: float, floor_percent: float) -> float:
    """Honest schedule risk: the measured average-retention shortfall below the
    operator's configured floor, on a 0-100 scale.

    The earlier formula added ``total_breaks * 0.8``; over a whole schedule the
    break count is in the hundreds, so that term saturated the score to 100 for
    every channel regardless of the real plan. This version is sourced entirely
    from quantities the optimizer actually produces: the realised average
    retention (``average_retention_percent``) and the operator-configured
    ``min_retention_floor``. At or above the floor the risk is 0; a shortfall of
    ``_RISK_FULL_SHORTFALL`` retention points or more reads as 100. Nothing is
    fabricated and the score no longer saturates.
    """
    shortfall = max(0.0, floor_percent - average_retention_percent)
    return round(max(0.0, min(100.0, shortfall / _RISK_FULL_SHORTFALL * 100.0)), 1)


def _summarize_schedule(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        # No saved schedule yet (fresh deploy, or post-upload pre-recompute). The
        # break/second counts are honestly zero, but revenue, retention and risk
        # are unknown, not measured lows. Report them as null so the dashboard
        # renders an honest "-" rather than a confident "Low risk / 0% / 0" that
        # no computation produced. The frontend guards each on null (formatCurrency
        # and formatPercent return "-", and the risk metric is gated on === null).
        return {
            "total_breaks": 0,
            "total_ad_seconds": 0,
            "projected_revenue": None,
            "average_retention": None,
            "risk_score": None,
        }

    num_breaks = pd.to_numeric(schedule.get("num_breaks", 1), errors="coerce").fillna(1)
    break_time = pd.to_numeric(
        schedule.get("total_break_time", schedule.get("break_length", 0)),
        errors="coerce",
    ).fillna(0)
    revenue = pd.to_numeric(
        schedule.get("predicted_revenue", schedule.get("revenue_ils", 0)),
        errors="coerce",
    ).fillna(0)
    retention = pd.to_numeric(schedule.get("predicted_retention", 0), errors="coerce")
    retention = retention[retention > 0]
    avg_retention = retention.mean() if not retention.empty else 0.0
    total_breaks = int(num_breaks.sum())
    avg_retention_pct = round(_percent(avg_retention), 1)
    # Honest risk score: the measured average-retention shortfall below the
    # operator's configured floor (see _risk_from_retention). Sourced from the
    # optimizer plan and operator settings, not the old saturating break-count
    # formula. Falls back to the guardrail default floor when settings are absent.
    floor_percent = round(_load_settings().min_retention_floor * 100, 1)
    risk_score = _risk_from_retention(avg_retention_pct, floor_percent)

    return {
        "total_breaks": total_breaks,
        "total_ad_seconds": int(break_time.sum()),
        "projected_revenue": _money(revenue.sum()),
        "average_retention": avg_retention_pct,
        "risk_score": risk_score,
    }


def _augment_segment_ids(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return the schedule with a populated segment_id and a boolean is_gold column.

    The weekly CSV carries both (kairos.export.schedule). This restores them when an
    older CSV predates the columns: segment_id is rebuilt as
    ``f"{date}|{channel}|{index:03d}"`` with index the row's position within its
    channel-day, exactly the build-order key the exporter writes and the override /
    constraint engines key their target_id on. Nothing is fabricated; a segment with
    no gold break reads False.

    Shared across the dashboard reads (segment list, segment detail, recommendations,
    compliance geometry) and the assistant context, so it lives in the kernel: a pure
    pandas transform with no engine or filesystem dependency.
    """
    frame = schedule.copy()
    if "segment_id" in frame.columns:
        sid = frame["segment_id"].astype(str).str.strip()
    else:
        sid = pd.Series([""] * len(frame), index=frame.index)
    blank = sid == ""
    if bool(blank.any()) and {"date", "channel"}.issubset(frame.columns):
        order = frame.groupby(["date", "channel"], sort=False).cumcount()
        rebuilt = (
            frame["date"].astype(str).str.strip() + "|"
            + frame["channel"].astype(str).str.strip() + "|"
            + order.map(lambda i: f"{int(i):03d}")
        )
        sid = sid.where(~blank, rebuilt)
    frame["segment_id"] = sid
    if "is_gold" in frame.columns:
        frame["is_gold"] = frame["is_gold"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes", "y"}
        )
    else:
        frame["is_gold"] = False
    return frame


def _row_anchor(row: Any) -> dict[str, str]:
    """The semantic anchor for a schedule row: the trio the override store records
    beside the build-order segment_id so a re-ingest cannot silently rebind."""
    return {
        "date": str(row.get("date", "")).strip(),
        "start_clock": str(row.get("start_time", "")).strip(),
        "program": str(row.get("program_type", "")).strip(),
    }
