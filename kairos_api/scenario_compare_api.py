"""Plan, week, compare: the daily forecast rows and the two-scenario A/B.

Moved verbatim from insights_api.py (the A/B) and catalog_api.py (the forecast
rows and the named scenarios) as part of the wave-zero router split. They travel
together because they are one surface today: the A/B control lives on the
Forecasts page.

Both legs of the A/B and every named forecast are real optimizer runs under the
operator's saved guardrails, scoped to the owned channel-day through the shared
selector, so no what-if revenue point is ever a competitor's. The objective
reported is the optimizer's convex-blend score under its own name, never
relabelled as a revenue net of retention cost.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from kairos_api.core import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _records,
    _reference_today,
    _safe_number,
    _signature,
    run_scenario,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ScenarioCompareRequest(BaseModel):
    """A what-if A/B: two revenue weights under shared (optional) guardrails.

    ``weight_a``/``weight_b`` are the 0..100 revenue-vs-retention levers. The three
    guardrails are optional; when omitted they fall back to the operator's saved
    settings so the comparison reflects the real plan baseline, not an arbitrary
    default. Both legs run the genuine optimizer; nothing here is synthesized.
    """

    weight_a: int = Field(ge=0, le=100)
    weight_b: int = Field(ge=0, le=100)
    retention_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_breaks_per_hour: Optional[int] = Field(default=None, ge=1, le=20)
    risk_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _scenario_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparable fields from a run_scenario payload.

    ``objective`` is the optimizer's convex-blend score (a weighted blend of
    revenue and retention, NOT a literal revenue-minus-cost subtraction), so it is
    reported under its own name and never relabeled as revenue_net.
    """
    summary = payload.get("summary", {})
    return {
        "revenue_weight": payload.get("controls", {}).get("revenue_weight"),
        "projected_revenue": summary.get("projected_revenue"),
        "average_retention": summary.get("average_retention"),
        "total_breaks": summary.get("total_breaks"),
        "total_ad_seconds": summary.get("total_ad_seconds"),
        "objective": summary.get("objective"),
        "compliant": summary.get("compliant"),
        "channel": payload.get("channel"),
        "day": payload.get("day"),
    }


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> Optional[float]:
    """b - a for a numeric summary field, or None when either side is missing."""
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return round(float(bv) - float(av), 4)


def _build_scenario_compare(request: ScenarioCompareRequest) -> dict[str, Any]:
    server = _server()
    if not server._ENGINE_AVAILABLE:
        return {"available": False, "reason": "Optimization engine unavailable."}

    saved = server._load_settings()
    floor = request.retention_floor if request.retention_floor is not None else saved.min_retention_floor
    max_bph = request.max_breaks_per_hour if request.max_breaks_per_hour is not None else saved.max_breaks_per_hour
    risk = request.risk_lambda if request.risk_lambda is not None else saved.risk_lambda

    from kairos.service import run_scenario

    # The full saved settings and pacing reference date, threaded into both legs
    # exactly as /api/optimizer-plan and the frontier do (server._pacing_call_kwargs),
    # so the A/B baseline honours every operator guardrail, the pricing overrides and
    # the operator channel scope instead of silently falling back to engine defaults.
    # The scenario overrides (floor/max_bph/risk) still apply on top of this base,
    # so the scenario-control semantics are unchanged.
    reference_today = server._reference_today(saved)
    settings_map = server._model_dump(saved)
    # Scope both A/B legs to the operator's owned channel-day (the shared selector
    # the scenario slider and frontier use), so the comparison is the owned
    # channel's, never the source's first channel-day (a competitor).
    from kairos_api.plan_read_scope import owned_scope

    channel, day = owned_scope(saved)

    def _run(weight: int) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=weight,
            retention_floor=floor,
            max_breaks_per_hour=max_bph,
            risk_lambda=risk,
            today=reference_today,
            settings=settings_map,
            channel=channel,
            day=day,
        )

    try:
        payload_a = _run(request.weight_a)
        payload_b = _run(request.weight_b)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        return {"available": False, "reason": f"Optimizer run failed: {str(exc)[:200]}"}

    a = _scenario_summary(payload_a)
    b = _scenario_summary(payload_b)
    return {
        "available": True,
        "guardrails": {"retention_floor": floor, "max_breaks_per_hour": max_bph, "risk_lambda": risk},
        "a": a,
        "b": b,
        "delta": {
            "revenue": _delta(a, b, "projected_revenue"),
            "retention": _delta(a, b, "average_retention"),
            "breaks": _delta(a, b, "total_breaks"),
            "ad_seconds": _delta(a, b, "total_ad_seconds"),
            "objective": _delta(a, b, "objective"),
            "revenue_net": None,
        },
        "revenue_net_note": (
            "A literal revenue-net-of-retention figure is not a summary field of run_scenario; "
            "the optimizer exposes a convex-blend objective instead, reported under 'objective'."
        ),
    }


def _build_forecasts(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    """Daily revenue/retention forecast rows from the saved plan, per real date.

    The saved plan spans a whole month across every channel, so the old
    weekday-of-week grouping summed roughly four channels times four-plus
    calendar dates into each "day" row, quoting whole-market multi-week money as
    a day's forecast. This groups by the REAL calendar date, scoped to the
    operator's channel (competitor rows inform the retention model, never the
    operator's money); each row keeps the weekday under the existing ``day`` key
    and adds ``date``. Retention per date is TVR-weighted on the plan's own
    ``baseline_tvr`` so 0-break filler rows stop diluting it; when no weight is
    available the mean is kept and the basis says so. ``by_day_basis`` discloses
    scope and grouping honestly.
    """
    def _basis(scope: str | None, n_dates: int, retention_basis: str | None, grouped_by: str | None) -> dict[str, Any]:
        return {
            "scope_channel": scope,
            "n_dates": n_dates,
            "retention_basis": retention_basis,
            "grouped_by": grouped_by,
        }

    if schedule.empty:
        return {"by_day": [], "scenarios": [], "by_day_basis": _basis(None, 0, None, None)}

    frame = schedule.copy()
    owned = str(settings.operator_channel or "").strip()
    scope_channel: str | None = None
    if owned and "channel" in frame.columns:
        frame = frame[frame["channel"].astype(str).str.strip() == owned]
        scope_channel = owned
    if frame.empty:
        # An owned channel is configured but the saved plan carries no rows for
        # it: honest empty rows, never another channel's money.
        return {
            "by_day": [],
            "scenarios": _build_forecast_scenarios(settings),
            "by_day_basis": _basis(scope_channel, 0, None, None),
        }

    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0)
    group_column = "date" if "date" in frame.columns else "day"
    if "baseline_tvr" in frame.columns:
        weights = pd.to_numeric(frame["baseline_tvr"], errors="coerce")
        weights = weights.where(weights > 0)
    else:
        weights = pd.Series(float("nan"), index=frame.index)
    frame["_tvr_weight"] = weights
    frame["_retention_weighted"] = frame["predicted_retention"] * frame["_tvr_weight"]

    grouped = (
        frame.groupby(group_column, dropna=False)
        .agg(
            revenue=("predicted_revenue", "sum"),
            retention_mean=("predicted_retention", "mean"),
            breaks=("num_breaks", "sum"),
            _weighted_sum=("_retention_weighted", "sum"),
            _weight_total=("_tvr_weight", "sum"),
        )
        .reset_index()
        .sort_values(group_column)
    )
    weighted_rows = grouped["_weight_total"] > 0
    grouped["retention"] = (grouped["_weighted_sum"] / grouped["_weight_total"]).where(
        weighted_rows, grouped["retention_mean"]
    )
    if bool(weighted_rows.all()):
        retention_basis = "tvr_weighted"
    elif bool(weighted_rows.any()):
        retention_basis = "mixed"
    else:
        retention_basis = "unweighted_mean"

    if group_column == "date":
        if "day" in frame.columns:
            weekday_map = frame.groupby("date")["day"].first()
            grouped["day"] = grouped["date"].map(weekday_map)
        else:
            grouped["day"] = pd.to_datetime(grouped["date"], errors="coerce").dt.strftime("%a")
    by_day = grouped[[column for column in ("day", "date", "revenue", "retention", "breaks") if column in grouped.columns]]
    n_dates = int(frame[group_column].astype(str).nunique())
    return {
        "by_day": _records(by_day),
        "scenarios": _build_forecast_scenarios(settings),
        "by_day_basis": _basis(scope_channel, n_dates, retention_basis, group_column),
    }


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
    # Scope every named forecast to the operator's owned channel-day (the shared
    # selector the scenario slider and frontier use), so these what-if revenue
    # points are the owned channel's forecast, never the source's first
    # channel-day (a competitor). Imported at call time to keep the module graph
    # acyclic.
    from kairos_api.plan_read_scope import owned_scope

    channel, day = owned_scope(settings)
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
                channel=channel,
                day=day,
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


@lru_cache(maxsize=16)
def _forecasts_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_forecasts(_load_break_schedule(), _load_settings())


@router.post("/api/scenario-compare", tags=["insights"])
def scenario_compare(request: ScenarioCompareRequest) -> dict[str, Any]:
    return _build_scenario_compare(request)


@router.get("/api/forecasts", tags=["catalog"])
def forecasts() -> dict[str, Any]:
    # The named scenarios re-run the optimizer under the saved settings over the
    # EPG, so the settings file and the Programmes source belong in the cache
    # key; without them a settings edit or an EPG re-ingest kept serving the
    # stale cached forecast.
    return _forecasts_cached(_signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        SETTINGS_PATH,
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
    ]))
