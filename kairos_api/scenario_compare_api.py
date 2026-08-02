"""Plan, week, compare: the daily forecast rows and the two-scenario A/B.

Moved verbatim from insights_api.py (the A/B) and catalog_api.py (the forecast
rows and the named scenarios) as part of the wave-zero router split. They travel
together because they are one surface today: the A/B control lives on the
Forecasts page.

Both legs of the A/B and every named forecast are real optimizer runs under the
operator's saved guardrails, scoped to the owned channel-day through the shared
selector, so no what-if revenue point is ever a competitor's.

**Net after retention cost is the quantity this surface exists for, and it is now
computed.** JS-2 defines the planner's comparison on revenue net of retention
cost, and the panel printed "Not exposed" because the optimizer's own summary
carries a convex-blend objective rather than a subtraction. The subtraction does
exist: :func:`kairos_api.plan_read_frontier.scenario_plan_money` prices any
scenario plan on the engine's per-break retention-cost model, which is the same
basis the committed plan's yield-per-second money uses. Each leg is priced with
it, so both legs report gross, retention cost and net on one shared basis, and
the objective keeps its own name beside them instead of standing in for a figure
it is not.

**Both legs carry every lever, because the revenue weight alone moves nothing.**
Measured on ``רשת 13 / 2024-11-11``: weight 60 and weight 85 both return
1,414,695.20 in revenue, 95.0 percent retention, 80 breaks and 9,600 ad seconds,
and only the blended score differs. That is the engine being consistent, not a
defect: at a fixed retention floor the refined optimum is nearly invariant to the
weight, which the frontier module documents. The floor, the hourly break cap, the
risk aversion and the engine focus do move the plan, so a comparison that only
offered the weight could not answer the planner's question. When two legs come
back identical the payload says so and names which levers were equal.

**And both legs run the plan's own week.** JS-2's comparison is of next week, and
running it on one representative broadcast day put two different quantities under
one label on one destination: the goal strip's week beside a single day's money.
``scope`` now defaults to ``week`` and the run is 14 real optimizations over the
same seven dates the goal strip reports, in :mod:`scenario_compare_api_week`.
``scope: "day"`` still runs the single representative day and the payload says
which of the two happened, in ``scope.mode``, so no figure is ever read against a
window it was not computed on.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter

# ``_delta`` and ``_scenario_summary`` are re-exported by the frozen wave-zero
# layer ``insights_api``, which binds them by name from this module, so they stay
# bound here even where this module no longer calls them itself.
from kairos_api.scenario_compare_api_money import (  # noqa: F401
    _delta,
    _identical_note,
    _priced,
    _resolve_levers,
    _scenario_summary,
    compare_body,
)
from kairos_api.scenario_compare_levers import ScenarioCompareRequest, ScenarioLevers  # noqa: F401
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


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _resolved(request: ScenarioCompareRequest) -> dict[str, Any]:
    """The saved settings, the shared guardrails and both legs' levers.

    One resolution serves the plain route, the streaming route and the weekly
    runner, so the three cannot disagree about what a request asked for.
    """
    server = _server()
    if not server._ENGINE_AVAILABLE:
        return {"available": False, "reason": "Optimization engine unavailable."}
    saved = server._load_settings()
    floor = request.retention_floor if request.retention_floor is not None else saved.min_retention_floor
    max_bph = request.max_breaks_per_hour if request.max_breaks_per_hour is not None else saved.max_breaks_per_hour
    risk = request.risk_lambda if request.risk_lambda is not None else saved.risk_lambda
    shared = {
        "retention_floor": floor,
        "max_breaks_per_hour": max_bph,
        "risk_lambda": risk,
        "objective_mode": str(getattr(saved, "objective_mode", "blend") or "blend"),
    }
    return {
        "available": True,
        "saved": saved,
        "guardrails": {"retention_floor": floor, "max_breaks_per_hour": max_bph, "risk_lambda": risk},
        "levers_a": _resolve_levers(request.a, request.weight_a, shared),
        "levers_b": _resolve_levers(request.b, request.weight_b, shared),
    }


def prepare_week(request: ScenarioCompareRequest) -> dict[str, Any]:
    """What the streaming route needs before it opens the response: the levers,
    the guardrails and the plan's own week, or the honest reason there is none."""
    from kairos_api import scenario_compare_api_week as week

    resolved = _resolved(request)
    if not resolved.get("available"):
        return resolved
    window = week.plan_week_window(resolved["saved"])
    if not window.get("available"):
        return {"available": False, "reason": window.get("reason"), "window": window}
    return {
        "available": True,
        "window": window,
        "levers_a": resolved["levers_a"],
        "levers_b": resolved["levers_b"],
        "guardrails": resolved["guardrails"],
    }


def _build_scenario_compare(request: ScenarioCompareRequest) -> dict[str, Any]:
    from kairos_api import scenario_compare_api_week as week

    resolved = _resolved(request)
    if not resolved.get("available"):
        return resolved
    saved = resolved["saved"]
    levers_a = resolved["levers_a"]
    levers_b = resolved["levers_b"]

    # The plan's own week is the default window, because it is the window the
    # goal strip, the supply panel and the published plan all report. A caller
    # that asks for one day, or a plan that has no week yet, falls through to the
    # single representative day below and the payload says which one it was.
    day_reason: Optional[str] = None
    if request.scope == "week":
        window = week.plan_week_window(saved)
        if window.get("available"):
            return compare_body(
                week.run_week(window, levers_a, levers_b), resolved["guardrails"]
            )
        day_reason = str(window.get("reason") or "")

    from kairos.service import run_scenario

    # The full saved settings and pacing reference date, threaded into both legs
    # exactly as /api/optimizer-plan and the frontier do (server._pacing_call_kwargs),
    # so the A/B baseline honours every operator guardrail, the pricing overrides and
    # the operator channel scope instead of silently falling back to engine defaults.
    # The scenario overrides (floor/max_bph/risk/focus) still apply on top of this
    # base, so the scenario-control semantics are unchanged.
    server = _server()
    reference_today = server._reference_today(saved)
    settings_map = server._model_dump(saved)
    # Scope both A/B legs to the operator's owned channel-day (the shared selector
    # the scenario slider and frontier use), so the comparison is the owned
    # channel's, never the source's first channel-day (a competitor).
    from kairos_api.plan_read_scope import owned_scope

    channel, day = owned_scope(saved)

    def _run(levers: dict[str, Any]) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=levers["revenue_weight"],
            retention_floor=levers["retention_floor"],
            max_breaks_per_hour=levers["max_breaks_per_hour"],
            risk_lambda=levers["risk_lambda"],
            objective_mode=levers["objective_mode"],
            today=reference_today,
            settings=settings_map,
            channel=channel,
            day=day,
        )

    started = time.perf_counter()
    try:
        payload_a = _run(levers_a)
        payload_b = _run(levers_b)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        return {"available": False, "reason": f"Optimizer run failed: {str(exc)[:200]}"}

    # The engine's own segments for the scoped channel-day, rebuilt once and
    # priced against both legs, so the two nets are on one basis. Measured at 15
    # to 105 ms against a 2.2 s response, so this is not what the planner waits
    # for.
    segments: list[Any] = []
    segment_reason: Optional[str] = None
    if channel and day:
        try:
            from kairos_api.core import _plan_segment_index

            segments = list(_plan_segment_index(((channel, str(day)),), settings_map).values())
        except Exception as exc:  # pragma: no cover - data/environment dependent
            logger.exception("segment rebuild for the A/B money failed")
            segment_reason = f"segment rebuild failed: {str(exc)[:160]}"
    else:
        segment_reason = "no operator channel-day is in scope, so retention cost cannot be priced"

    a = _priced(_scenario_summary(payload_a, levers_a), payload_a, segments, levers_a["risk_lambda"])
    b = _priced(_scenario_summary(payload_b, levers_b), payload_b, segments, levers_b["risk_lambda"])
    for leg in (a, b):
        if segment_reason and not leg.get("money_available"):
            leg["money_reason"] = segment_reason
    return compare_body(
        {
            "a": a,
            "b": b,
            "by_day": None,
            # The single-day window, declared as plainly as the weekly one. A
            # reader who sees this mode knows the money on screen is one
            # broadcast day and not the week, and knows why.
            "scope": {
                "mode": "day",
                "channel": channel,
                "day": day,
                "dates": [str(day)] if day else [],
                "date_from": day,
                "date_to": day,
                "n_dates": 1 if day else 0,
                "basis": "representative_day",
                # A day counts as priced when its money actually priced, on the
                # same reading the weekly runner uses. Rebuilt segments alone are
                # not a price: the pricer still refuses when the plan and the
                # segments no longer join, and a count that said one there would
                # contradict the money_available beside it.
                "days_priced": 1 if a.get("money_available") else 0,
                "segments": len(segments),
                "runs": {"total": 2, "computed": 2, "reused": 0},
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "day_reason": day_reason,
            },
        },
        resolved["guardrails"],
    )


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


# The streamed comparison rides this module's registration rather than appending
# another stanza to server.py: it is the same comparison, on the same body,
# delivered a day at a time, and one mount keeps the OpenAPI diff readable.
from kairos_api.scenario_compare_api_week import router as _week_router  # noqa: E402

router.include_router(_week_router)
