"""Catalog read endpoints: inventory, break library, campaigns, forecasts,
reports, source files, and the measured-impact summary.

Thin domain router over the shared kernel (:mod:`kairos_api.core`); the builders
and their mtime-signature caches moved verbatim from server.py, which re-exports
the names so existing references keep working. Behavior is unchanged. The
reports builder composes the compliance verdict owned by the server module,
imported at call time so the module import graph stays acyclic.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter

from kairos_api.core import (
    DATA_DIR, MODELS_DIR, OUTPUT_DIR, ROOT, SETTINGS_PATH, KairosSettings,
    _load_break_schedule, _load_settings, _load_spots, _model_dump, _money,
    _percent, _records, _reference_today, _safe_number, _series, _signature,
    _summarize_schedule, run_scenario,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])


def _segment_key(channel_name: str) -> tuple[str, str, str] | None:
    parts = str(channel_name or "").split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def _weighted_impact_rows(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        segment = str(item.get(key) or "")
        coefficient = _safe_number(item.get("coefficient"), math.nan)
        if not segment or not math.isfinite(coefficient):
            continue
        grouped.setdefault(segment, []).append(item)

    rows: list[dict[str, Any]] = []
    for segment, values in grouped.items():
        total_weight = 0
        weighted_coefficient = 0.0
        weighted_raw = 0.0
        ci_low: list[float] = []
        ci_high: list[float] = []
        for item in values:
            sample_count = max(1, int(_safe_number(item.get("n"), 1)))
            coefficient = _safe_number(item.get("coefficient"), 0.0)
            raw_delta = _safe_number(item.get("raw_delta"), coefficient)
            weighted_coefficient += coefficient * sample_count
            weighted_raw += raw_delta * sample_count
            total_weight += sample_count
            low = _safe_number(item.get("ci_low"), math.nan)
            high = _safe_number(item.get("ci_high"), math.nan)
            if math.isfinite(low):
                ci_low.append(low)
            if math.isfinite(high):
                ci_high.append(high)
        if total_weight <= 0:
            continue
        rows.append(
            {
                "segment": segment,
                "average_coefficient": round(weighted_coefficient / total_weight, 6),
                "average_raw_delta": round(weighted_raw / total_weight, 6),
                "sample_count": total_weight,
                "channel_count": len(values),
                "ci_low": round(min(ci_low), 6) if ci_low else None,
                "ci_high": round(max(ci_high), 6) if ci_high else None,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["average_coefficient"])), reverse=True)


def _pooling_note(metadata: dict[str, Any]) -> str | None:
    """Honest disclosure that the per-cell retention effects collapse toward one
    pooled constant. Empirical Bayes shrinks the programme-type x position x length
    cells because the between-cell variance sits far below the within-cell variance,
    so the cells share almost all of their signal. Numbers are read straight from
    the coefficient artifact metadata, never hand-set."""
    tau2 = _safe_number(metadata.get("between_cell_variance_tau2"), math.nan)
    within = _safe_number(metadata.get("pooled_within_variance"), math.nan)
    if not math.isfinite(tau2) or not math.isfinite(within) or within <= 0:
        return None
    cells = int(_safe_number(metadata.get("channels"), 0)) or None
    method = str(metadata.get("pooling_method") or "empirical_bayes").replace("_", " ")
    cell_phrase = f"{cells} " if cells else ""
    return (
        f"The {cell_phrase}(programme type x position x length) cells pool to approximately "
        f"one shared constant under {method}: between-cell variance tau^2 = {tau2:.2e} sits "
        f"far below within-cell variance {within:.3f}, so the per-cell effects collapse toward "
        f"a single pooled value."
    )


def _load_measured_impact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": "legacy_csv",
            "pooling_note": None,
            "program_type": [],
            "position": [],
            "length": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "source": "legacy_csv",
            "pooling_note": None,
            "program_type": [],
            "position": [],
            "length": [],
        }

    details = payload.get("detail", {})
    items: list[dict[str, Any]] = []
    for name, raw in details.items():
        if not isinstance(raw, dict):
            continue
        segment = _segment_key(str(raw.get("channel_name") or name))
        if not segment:
            continue
        program_type, position, length = segment
        items.append(
            {
                **raw,
                "program_type": program_type,
                "position": position,
                "length": length,
            }
        )

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "source": payload.get("method") or "measured_coefficients",
        "metadata": metadata,
        "pooling_note": _pooling_note(metadata),
        "program_type": _weighted_impact_rows(items, "program_type"),
        "position": _weighted_impact_rows(items, "position"),
        "length": _weighted_impact_rows(items, "length"),
    }


def _build_inventory(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        return {
            "summary": {"spots": 0, "revenue": None, "seconds": 0},
            "revenue_available": False,
            "by_channel": [],
            "by_hour": [],
        }

    frame = spots.copy()
    # Revenue is reported only when the spots source actually carries it. The
    # reference airings export has no revenue column, so fabricating a zero
    # would misstate a real quantity; report an honest unavailable instead.
    has_revenue = "revenue_ils" in frame.columns
    frame["revenue_ils"] = pd.to_numeric(_series(frame, "revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(_series(frame, "Duration", 0), errors="coerce").fillna(0)
    frame["hour_of_day"] = pd.to_numeric(_series(frame, "hour_of_day", 0), errors="coerce").fillna(0).astype(int)
    frame["target"] = _series(frame, "is_target_channel", False).astype(str).str.lower().isin(["true", "1", "yes"])
    valid_hours = frame[(frame["hour_of_day"] >= 0) & (frame["hour_of_day"] <= 23)]

    sort_key = "revenue" if has_revenue else "seconds"
    by_channel = (
        frame.groupby("Channel", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"), target_spots=("target", "sum"))
        .reset_index()
        .sort_values(sort_key, ascending=False)
        .head(12)
    )
    by_hour = (
        valid_hours.groupby("hour_of_day", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"))
        .reset_index()
        .sort_values("hour_of_day")
    )
    if not has_revenue:
        by_channel["revenue"] = None
        by_hour["revenue"] = None

    return {
        "summary": {
            "spots": int(len(frame)),
            "revenue": _money(frame["revenue_ils"].sum()) if has_revenue else None,
            "seconds": int(frame["Duration"].sum()),
        },
        "revenue_available": has_revenue,
        "by_channel": _records(by_channel),
        "by_hour": _records(by_hour, 24),
    }


def _build_campaigns(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        return {"campaigns": []}

    frame = spots.copy()
    # Revenue is reported only when the spots source actually carries it (the
    # reference airings export does not); otherwise the rollup ranks by spot
    # volume and reports revenue as unavailable rather than a fabricated zero.
    has_revenue = "revenue_ils" in frame.columns
    frame["revenue_ils"] = pd.to_numeric(_series(frame, "revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(_series(frame, "Duration", 0), errors="coerce").fillna(0)
    # The restructured Spots export may omit the identity/grouping columns. Backfill
    # any that are missing with honest neutral defaults so the rollup degrades to a
    # single bucket instead of crashing the endpoint (KeyError) into a 500.
    frame["Campaign"] = _series(frame, "Campaign", "Unknown campaign")
    frame["advertiser_id"] = _series(frame, "advertiser_id", "")
    frame["Channel"] = _series(frame, "Channel", "")
    frame["Date"] = _series(frame, "Date", "")
    grouped = (
        frame.groupby(["Campaign", "advertiser_id"], dropna=False)
        .agg(
            spots=("Campaign", "count"),
            seconds=("Duration", "sum"),
            revenue=("revenue_ils", "sum"),
            channels=("Channel", "nunique"),
            last_airing=("Date", "max"),
        )
        .reset_index()
        .sort_values("revenue" if has_revenue else "spots", ascending=False)
        .head(50)
    )
    if not has_revenue:
        grouped["revenue"] = None
    return {"campaigns": _records(grouped), "revenue_available": has_revenue}


def _build_break_library(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"breaks": []}

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    frame["total_break_time"] = pd.to_numeric(frame.get("total_break_time", 0), errors="coerce").fillna(0)
    frame["priority"] = frame["predicted_revenue"] * frame["predicted_retention"].clip(lower=0.1)
    frame = frame.sort_values("priority", ascending=False).head(80)
    floor_percent = _load_settings().min_retention_floor * 100
    frame["status"] = frame["predicted_retention"].map(lambda value: "at_risk" if _percent(value) < floor_percent else "ready")
    return {"breaks": _records(frame)}


def _build_forecasts(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    if schedule.empty:
        return {"by_day": [], "scenarios": []}

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    by_day = (
        frame.groupby("day", dropna=False)
        .agg(revenue=("predicted_revenue", "sum"), retention=("predicted_retention", "mean"), breaks=("num_breaks", "sum"))
        .reset_index()
    )
    return {"by_day": _records(by_day), "scenarios": _build_forecast_scenarios(settings)}


def _build_forecast_scenarios(settings: KairosSettings) -> list[dict[str, Any]]:
    """Three named what-if points, each a REAL optimization at a different revenue
    weight under the operator's saved guardrails (not a percentage nudge off the
    current plan). 'Retention guardrail' leans retention-first, 'Revenue priority'
    leans revenue-first, 'Balanced' uses the saved weight; each value comes from
    :func:`kairos.service.run_scenario`. Empty (honest) when no plan computes."""
    saved_weight = settings.revenue_weight
    named = [
        ("Retention guardrail", "ריסון לטובת צפייה", 20),
        ("Balanced", "מאוזן", saved_weight),
        ("Revenue priority", "עדיפות להכנסה", 90),
    ]
    scenarios: list[dict[str, Any]] = []
    for name, name_he, weight in named:
        try:
            payload = run_scenario(
                revenue_weight=weight,
                retention_floor=settings.min_retention_floor,
                max_breaks_per_hour=settings.max_breaks_per_hour,
                risk_lambda=settings.risk_lambda,
                today=_reference_today(settings),
                settings=_model_dump(settings),
            )
        except Exception:
            logger.exception("forecast scenario '%s' failed at revenue_weight=%s", name, weight)
            continue
        summary = payload.get("summary", {})
        revenue = summary.get("projected_revenue")
        retention = summary.get("average_retention")
        if revenue is None or retention is None:
            continue
        scenarios.append(
            {
                "name": name,
                "name_he": name_he,
                "revenue_weight": weight,
                "revenue": round(_safe_number(revenue), 2),
                "retention": round(_safe_number(retention), 1),
            }
        )
    return scenarios


def _source_file_paths() -> list[Path]:
    """The real source files the data-quality report audits.

    Single source of truth shared with ``/api/files`` so the report's row count
    reflects the actual file set, not a magic constant.
    """
    return [
        DATA_DIR / "Dayparts.csv",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
        DATA_DIR / "rate_card_premiums.csv",
        DATA_DIR / "advertiser_rules.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        MODELS_DIR / "tv_break_posterior.pkl",
    ]


def _build_reports(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    # The compliance verdict is composed by the server module, which owns the plan
    # guardrail geometry shared with /api/compliance and /api/overview; imported at
    # call time so the module import graph stays acyclic.
    from kairos_api.server import _build_compliance

    summary = _summarize_schedule(schedule)
    compliance = _build_compliance(schedule, settings)
    source_files = _source_file_paths()
    present = sum(1 for path in source_files if path.exists())
    # Status is sourced from the real plan state, not a fixed "ready". An empty
    # schedule (no plan run yet) reports "empty" so the operator sees the honest
    # state instead of a green light backed by zero rows.
    plan_rows = int(len(schedule))
    revenue_rows = int(summary["total_breaks"])
    return {
        "reports": [
            {"id": "weekly-plan", "title": "Weekly traffic plan", "status": "ready" if plan_rows else "empty", "rows": plan_rows, "owner": "Traffic"},
            {"id": "compliance", "title": "Compliance and guardrails", "status": compliance["status"], "rows": len(compliance["checks"]), "owner": "Legal / Ops"},
            {"id": "revenue", "title": "Revenue forecast", "status": "ready" if revenue_rows else "empty", "rows": revenue_rows, "owner": "Revenue"},
            {"id": "data-quality", "title": "Source file audit", "status": "ready" if present == len(source_files) else "attention", "rows": present, "owner": "Data"},
        ]
    }


@lru_cache(maxsize=16)
def _inventory_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_inventory(_load_spots())


@lru_cache(maxsize=16)
def _break_library_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_break_library(_load_break_schedule())


@lru_cache(maxsize=16)
def _campaigns_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_campaigns(_load_spots())


@lru_cache(maxsize=16)
def _forecasts_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_forecasts(_load_break_schedule(), _load_settings())


@lru_cache(maxsize=16)
def _reports_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_reports(_load_break_schedule(), _load_settings())


@lru_cache(maxsize=16)
def _impact_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    summary = _load_measured_impact_summary(MODELS_DIR / "tv_break_coefficients.json")
    # Weekly level drift of the coefficient measurement base, measured at
    # rebuild time and carried in the artifact metadata (see
    # kairos.model.drift_monitor and docs/model-validation/
    # uncertainty-calibration.md finding 4). Echoed here for the Data page;
    # when the artifact predates the monitor (or carries no metadata) the
    # block is an honest "unavailable", never a fabricated verdict.
    metadata = summary.get("metadata")
    drift = metadata.get("level_drift") if isinstance(metadata, dict) else None
    if not isinstance(drift, dict) or not drift:
        drift = {
            "status": "unavailable",
            "reason": (
                "the coefficients artifact carries no level-drift measurement; "
                "rebuild the measured coefficients to compute it"
            ),
        }
    return {
        "coefficient_impacts": summary,
        "drift": drift,
    }


@router.get("/api/impact")
def impact() -> dict[str, Any]:
    return _impact_cached(
        _signature([MODELS_DIR / "tv_break_coefficients.json"])
    )


@router.get("/api/inventory")
def inventory() -> dict[str, Any]:
    return _inventory_cached(_signature([DATA_DIR / "Spots.csv"]))


@router.get("/api/break-library")
def break_library() -> dict[str, Any]:
    return _break_library_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@router.get("/api/campaigns")
def campaigns() -> dict[str, Any]:
    return _campaigns_cached(_signature([DATA_DIR / "Spots.csv"]))


@router.get("/api/forecasts")
def forecasts() -> dict[str, Any]:
    return _forecasts_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@router.get("/api/reports")
def reports() -> dict[str, Any]:
    return _reports_cached(
        _signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv", DATA_DIR / "Programmes.csv", SETTINGS_PATH])
    )


@router.get("/api/files")
def files() -> dict[str, Any]:
    paths = _source_file_paths()
    return {
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if path.exists()
                else None,
            }
            for path in paths
        ]
    }
