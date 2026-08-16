"""Scenario and optimizer-plan endpoints: the dashboard simulation controls,
the saved-settings optimizer plan, the one-day optimize call, and the
parameters surface.

Thin domain router over the shared kernel (:mod:`kairos_api.core`). The request
models, builders and the scenario cache moved verbatim from server.py as part
of the modular-monolith carve-up; behavior is unchanged and server.py
re-exports the moved names so existing references keep working.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos.optimize.inventory import InventoryInputError, load_inventory
from kairos_api.core import (
    DATA_DIR,
    KAIROS_CHANNELS,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    OptimizerAssumptions,
    _ENGINE_AVAILABLE,
    _asdict,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _pacing_call_kwargs,
    _reference_today,
    _risk_from_retention,
    _signature,
    _summarize_schedule,
    guardrails_from_settings,
    optimize_day_plan,
    run_scenario,
)

# The single frontier background machine and its net-focused bundle live in
# dashboard_api; importing the accessor (dashboard_api never imports this
# module, so there is no cycle) keeps exactly one state/lock/thread instance.
# _owned_scope is the shared owned-channel/representative-day selector the
# frontier uses; the scenario and optimizer-plan surfaces reuse it so every
# operator-facing preview optimizes the owned channel, never a competitor day.
from kairos_api.dashboard_api import _frontier_state, _owned_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scenario"])


def _require_authoritative_inventory() -> None:
    """Refuse a present all-invalid inventory source before serving a preview.

    A missing source is still the optimizer's documented neutral signal. Only a
    file with rows that cannot produce a single slot raises.
    """
    load_inventory(require_usable=True)


def _scenario_data_signature() -> tuple[tuple[str, int, int], ...]:
    """Files whose valid contents can move a cached scenario result."""
    return _signature([
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots - inventory.csv",
        ROOT / "config" / "optimization_weights.yaml",
        MODELS_DIR / "tv_break_posterior.pkl",
        MODELS_DIR / "tv_break_coefficients.json",
    ])


class ScenarioRequest(BaseModel):
    """Lightweight scenario controls used by the dashboard simulation."""

    revenue_weight: int = Field(default=60, ge=0, le=100)
    retention_floor: float = Field(default=0.72, ge=0.0, le=1.0)
    max_breaks_per_hour: int = Field(default=3, ge=1, le=12)
    # How conservatively to value an uncertain retention cost: 0 uses the point
    # estimate (today's behavior), 1 uses the worst plausible cost in the interval.
    risk_lambda: float = Field(default=0.0, ge=0.0, le=1.0)


class OptimizePlanRequest(BaseModel):
    """Controls for a real, in-process optimization of one channel-day."""

    channel: str | None = Field(default=None)
    day: str | None = Field(default=None)
    revenue_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # When None, the saved settings' risk_lambda applies; set it to override the
    # uncertainty preference for this run only.
    risk_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    # When set, the day's real daily plan (the Wally csv) drives the decision
    # instead of the Programmes EPG; channel and day are read from the file.
    daily_input: str | None = Field(default=None)


def _build_optimizer_plan(request: ScenarioRequest | None = None) -> dict[str, Any]:
    _require_authoritative_inventory()
    saved = _load_settings()
    if request is None:
        # The default plan is the operator's SAVED decision, not a static default:
        # it honors the persisted revenue/retention balance, floor, and risk.
        request = ScenarioRequest(
            revenue_weight=saved.revenue_weight,
            retention_floor=saved.min_retention_floor,
            max_breaks_per_hour=saved.max_breaks_per_hour,
            risk_lambda=saved.risk_lambda,
        )
    if not _ENGINE_AVAILABLE:
        return {
            "summary": {
                **_summarize_schedule(_load_break_schedule()),
                "is_compliant": False,
            },
            "controls": _model_dump(request),
            "engine": "unavailable",
        }
    # Scope the preview to the operator's owned channel on its representative
    # broadcast day (the same shared selector the frontier uses), so the plan is
    # the owned channel's forecast and never the first channel-day in the source
    # (a competitor day). With no owned channel configured, channel/day stay None
    # and run_scenario keeps its documented whole-source default.
    channel, day = _owned_scope(saved)
    payload = run_scenario(
        revenue_weight=request.revenue_weight,
        retention_floor=request.retention_floor,
        max_breaks_per_hour=request.max_breaks_per_hour,
        risk_lambda=request.risk_lambda,
        channel=channel,
        day=day,
        require_usable_inventory=True,
        **_pacing_call_kwargs(),
    )
    summary = payload.setdefault("summary", {})
    summary["is_compliant"] = bool(summary.get("is_compliant", summary.get("compliant", False)))
    return payload


@lru_cache(maxsize=128)
def _scenario_cached(
    revenue_weight: int, retention_floor: float, max_breaks_per_hour: int, risk_lambda: float = 0.0,
    pacing_today: str = "", settings_json: str = "", channel: str = "", day: str = "",
    data_signature: tuple[tuple[str, int, int], ...] = (),
) -> dict[str, Any]:
    # The full saved settings (guardrails, pricing and pacing) are threaded in as
    # a JSON string, exactly as /api/optimizer-plan threads them, so the scenario
    # preview honours every operator setting instead of silently dropping the
    # guardrails and pricing to defaults. Both the settings JSON and the reference
    # date are part of the cache key, so any settings edit invalidates the cached
    # scenario honestly. The scenario controls (revenue_weight, retention_floor,
    # max_breaks_per_hour, risk_lambda) remain the explicit per-request overrides.
    # channel/day are the owned-channel scope from the shared selector: they pin
    # the preview to the operator's channel-day rather than the source's first
    # channel-day (a competitor), and are part of the cache key so a data change
    # that shifts the representative day invalidates the cached scenario honestly.
    del data_signature  # cache key only
    today = None
    if pacing_today:
        try:
            today = date.fromisoformat(pacing_today)
        except ValueError:
            today = None
    settings = json.loads(settings_json) if settings_json else None
    result = run_scenario(
        revenue_weight=revenue_weight,
        retention_floor=retention_floor,
        max_breaks_per_hour=max_breaks_per_hour,
        risk_lambda=risk_lambda,
        today=today,
        settings=settings,
        channel=channel or None,
        day=day or None,
        require_usable_inventory=True,
    )
    summary = result["summary"]
    return {
        "summary": {
            "total_breaks": summary["total_breaks"],
            "total_ad_seconds": summary["total_ad_seconds"],
            "projected_revenue": summary["projected_revenue"],
            "average_retention": summary["average_retention"],
            "risk_score": _risk_from_retention(
                summary["average_retention"], round(retention_floor * 100, 1)
            ),
        },
        "controls": result["controls"],
        "guardrails": result["guardrails"],
        "channel": result["channel"],
        "day": result["day"],
        "compliant": summary["compliant"],
        "engine": "kairos",
    }


@lru_cache(maxsize=8)
def _optimizer_plan_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature  # cache key only
    return _build_optimizer_plan()


@router.get("/api/optimizer-plan")
def optimizer_plan() -> dict[str, Any]:
    # Memoized on the settings+data signature like the sibling reads: the GET
    # side is the saved decision re-read, so re-running a full optimization per
    # request bought nothing. Any settings edit, EPG/plan re-ingest, rate-card
    # change or model rebuild changes the signature and recomputes honestly.
    try:
        _require_authoritative_inventory()
        return _optimizer_plan_cached(_signature([
            OUTPUT_DIR / "weekly_break_schedule.csv",
            DATA_DIR / "reference" / "Programmes.xlsx",
            DATA_DIR / "Programmes.csv",
            DATA_DIR / "Spots - inventory.csv",
            SETTINGS_PATH,
            ROOT / "config" / "optimization_weights.yaml",
            MODELS_DIR / "tv_break_posterior.pkl",
            MODELS_DIR / "tv_break_coefficients.json",
        ]))
    except InventoryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/optimizer-plan")
def create_optimizer_plan(request: ScenarioRequest) -> dict[str, Any]:
    try:
        _require_authoritative_inventory()
        return _build_optimizer_plan(request)
    except InventoryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _scenario_unavailable(request: ScenarioRequest, reason: str, detail: str | None = None) -> dict[str, Any]:
    """The honest no-numbers scenario response.

    The old fallback substituted the saved whole-month, all-channel CSV summary,
    shaped exactly like a one-day simulation result, so a transient engine
    failure quietly showed month-scale money on the day slider. Keys stay
    identical to the real summary; every value is null and the reason is named.
    """
    payload: dict[str, Any] = {
        "summary": {
            "total_breaks": None,
            "total_ad_seconds": None,
            "projected_revenue": None,
            "average_retention": None,
            "risk_score": None,
        },
        "controls": _model_dump(request),
        "engine": "unavailable",
        "reason": reason,
    }
    if detail:
        payload["detail"] = detail
    return payload


@router.post("/api/scenario")
def scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Run a real optimization for the scenario controls (no placeholder math).

    When the engine or its data is unavailable the summary is null with the
    reason named, never the saved whole-plan summary dressed up as a day result.
    """
    if _ENGINE_AVAILABLE:
        try:
            _require_authoritative_inventory()
            pacing = _pacing_call_kwargs()
            channel, day = _owned_scope(_load_settings())
            return _scenario_cached(
                request.revenue_weight, request.retention_floor, request.max_breaks_per_hour,
                request.risk_lambda,
                pacing_today=pacing["today"].isoformat(),
                settings_json=json.dumps(pacing["settings"], sort_keys=True, ensure_ascii=False),
                channel=channel or "",
                day=day or "",
                data_signature=_scenario_data_signature(),
            )
        except Exception as exc:  # pragma: no cover - data/environment dependent
            return _scenario_unavailable(
                request,
                "the scenario optimization failed, so no numbers are shown for these controls",
                detail=str(exc)[:300],
            )
    return _scenario_unavailable(
        request, "the optimization engine is unavailable, so no scenario numbers exist"
    )


def _warm_scenario() -> dict[str, Any]:
    """Prime the scenario-preview cache on the operator's saved decision.

    Mirrors the /api/scenario request path (same owned-channel scope selector and
    the same cache key) so the first slider read finds the cache hot. Runs on the
    single startup warm-up thread beside the frontier sweep; it never spawns a
    thread of its own. Returns the warmed payload so the caller can log or ignore.
    """
    if not _ENGINE_AVAILABLE:
        return {"engine": "unavailable"}
    _require_authoritative_inventory()
    saved = _load_settings()
    pacing = _pacing_call_kwargs()
    channel, day = _owned_scope(saved)
    return _scenario_cached(
        saved.revenue_weight, saved.min_retention_floor, saved.max_breaks_per_hour,
        saved.risk_lambda,
        pacing_today=pacing["today"].isoformat(),
        settings_json=json.dumps(pacing["settings"], sort_keys=True, ensure_ascii=False),
        channel=channel or "",
        day=day or "",
        data_signature=_scenario_data_signature(),
    )


@router.post("/api/optimal-plan")
def optimize_plan(request: OptimizePlanRequest) -> dict[str, Any]:
    """Serve a real optimal break plan, driven by the saved settings.

    Runs the optimization engine in process, using the live KairosSettings as
    guardrails, so the dashboard's settings page controls the optimizer directly.
    """
    if not _ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optimization engine is unavailable")
    settings = _load_settings()
    risk = request.risk_lambda if request.risk_lambda is not None else getattr(settings, "risk_lambda", 0.0)
    # Default to the operator's owned channel-day (the shared scope selector) when
    # the caller pins neither, so an unparameterized call optimizes the owned
    # channel rather than the source's first channel-day (a competitor). An
    # explicit channel/day from the request always wins; when both are omitted and
    # the owned channel resolves, its representative day keeps the run interactive.
    scope_channel, scope_day = _owned_scope(settings)
    # Competitor boundary: the operator owns exactly one channel, so an explicit
    # channel request may only ever be that owned channel. A request for any
    # other channel is refused rather than projecting revenue for a channel the
    # operator does not own. When no owned channel is configured there is no
    # boundary to enforce, so an explicit channel is accepted.
    owned = str(settings.operator_channel or "").strip()
    if request.channel and owned and request.channel != owned:
        raise HTTPException(
            status_code=400,
            detail="Only the operator's own channel can be optimized",
        )
    channel = request.channel or scope_channel or None
    day = request.day
    if day is None and request.channel is None and channel == scope_channel:
        day = scope_day
    try:
        return optimize_day_plan(
            channel=channel,
            day=day,
            revenue_weight=request.revenue_weight,
            risk_lambda=risk,
            daily_input_path=request.daily_input,
            settings=_model_dump(settings),
            today=_reference_today(settings),
            require_usable_inventory=True,
        )
    except InventoryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Reference data not found: {exc}")
    except Exception as exc:  # pragma: no cover - data/environment dependent
        raise HTTPException(status_code=503, detail=f"Optimization failed: {exc}")


@router.get("/api/optimizer/net-comparison")
def optimizer_net_comparison() -> dict[str, Any]:
    """The saved-plan objective versus a net-focused plan, computed not quoted.

    Both sides come from the SAME scenario runner on the frontier's owned scope
    under the saved guardrails, refined: 'current' at the saved
    revenue_weight/objective_mode and 'net_focused' under
    objective_mode='revenue_net'. Each side is priced with the per-break
    retention-cost model, so gross minus retention_cost equals net on both
    sides and the deltas (net_focused minus current) are internally consistent.
    While the shared background sweep is computing this reports
    status='computing' with no numbers; when the scope or pricing cannot
    produce an honest comparison it reports status='unavailable' with the
    reason. Nothing is fabricated.
    """
    settings = _load_settings()
    _points, bundle, status = _frontier_state(settings, None)
    channel = (bundle or {}).get("channel") or (settings.operator_channel or None)
    day = (bundle or {}).get("day")
    scope_text = f" ({channel}, {day})" if channel and day else ""
    basis = (
        "Both sides are the same refined scenario-runner optimization of the owned "
        f"channel's representative broadcast day{scope_text} under the saved guardrails, "
        "with retention cost priced per break from the measured coefficients; a modeled "
        "forecast, not the saved weekly plan total."
    )
    response: dict[str, Any] = {
        "status": "unavailable",
        "basis": basis,
        "current": None,
        "net_focused": None,
        "delta": None,
    }
    if status == "computing":
        response["status"] = "computing"
        return response
    if status == "no_channel":
        response["reason"] = (
            "No operator channel is configured; pick your channel in settings first."
        )
        return response
    if not bundle or not bundle.get("comparison_available"):
        response["reason"] = str((bundle or {}).get("reason") or "Comparison could not be computed.")
        return response
    current = {key: bundle["current"][key] for key in ("gross", "retention_cost", "net", "breaks")}
    net_focused = {key: bundle["net_focused"][key] for key in ("gross", "retention_cost", "net", "breaks")}
    response["status"] = "ready"
    response["current"] = current
    response["net_focused"] = net_focused
    response["delta"] = {
        "gross": round(net_focused["gross"] - current["gross"], 2),
        "retention_cost": round(net_focused["retention_cost"] - current["retention_cost"], 2),
        "net": round(net_focused["net"] - current["net"], 2),
        "breaks": int(net_focused["breaks"] - current["breaks"]),
    }
    return response
# The parameters surface rides this module's registration rather than appending a
# second stanza to server.py: it is the same domain and one mount keeps the
# append-only region's OpenAPI diff readable. The function is re-exported so any
# existing reference to scenario_api.parameters still resolves.
from kairos_api.scenario_api_parameters import parameters  # noqa: E402,F401
from kairos_api.scenario_api_parameters import router as _parameters_router  # noqa: E402

router.include_router(_parameters_router)
