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
from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kairos_api.core import (
    KAIROS_CHANNELS,
    MODELS_DIR,
    ROOT,
    OptimizerAssumptions,
    PricingModel,
    _ENGINE_AVAILABLE,
    _asdict,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _pacing_call_kwargs,
    _reference_today,
    _risk_from_retention,
    _summarize_schedule,
    guardrails_from_settings,
    optimize_day_plan,
    run_scenario,
)

# The single frontier background machine and its net-focused bundle live in
# dashboard_api; importing the accessor (dashboard_api never imports this
# module, so there is no cycle) keeps exactly one state/lock/thread instance.
from kairos_api.dashboard_api import _frontier_state

router = APIRouter(tags=["scenario"])


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
    if request is None:
        # The default plan is the operator's SAVED decision, not a static default:
        # it honors the persisted revenue/retention balance, floor, and risk.
        saved = _load_settings()
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
    payload = run_scenario(
        revenue_weight=request.revenue_weight,
        retention_floor=request.retention_floor,
        max_breaks_per_hour=request.max_breaks_per_hour,
        risk_lambda=request.risk_lambda,
        **_pacing_call_kwargs(),
    )
    summary = payload.setdefault("summary", {})
    summary["is_compliant"] = bool(summary.get("is_compliant", summary.get("compliant", False)))
    return payload


@lru_cache(maxsize=128)
def _scenario_cached(
    revenue_weight: int, retention_floor: float, max_breaks_per_hour: int, risk_lambda: float = 0.0,
    pacing_today: str = "", settings_json: str = "",
) -> dict[str, Any]:
    # The full saved settings (guardrails, pricing and pacing) are threaded in as
    # a JSON string, exactly as /api/optimizer-plan threads them, so the scenario
    # preview honours every operator setting instead of silently dropping the
    # guardrails and pricing to defaults. Both the settings JSON and the reference
    # date are part of the cache key, so any settings edit invalidates the cached
    # scenario honestly. The scenario controls (revenue_weight, retention_floor,
    # max_breaks_per_hour, risk_lambda) remain the explicit per-request overrides.
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


@router.get("/api/optimizer-plan")
def optimizer_plan() -> dict[str, Any]:
    return _build_optimizer_plan()


@router.post("/api/optimizer-plan")
def create_optimizer_plan(request: ScenarioRequest) -> dict[str, Any]:
    return _build_optimizer_plan(request)


@router.post("/api/scenario")
def scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Run a real optimization for the scenario controls (no placeholder math).

    Falls back to the stored schedule summary only if the engine or its data is
    unavailable, reporting that honestly instead of inventing numbers.
    """
    if _ENGINE_AVAILABLE:
        try:
            pacing = _pacing_call_kwargs()
            return _scenario_cached(
                request.revenue_weight, request.retention_floor, request.max_breaks_per_hour,
                request.risk_lambda,
                pacing_today=pacing["today"].isoformat(),
                settings_json=json.dumps(pacing["settings"], sort_keys=True, ensure_ascii=False),
            )
        except Exception as exc:  # pragma: no cover - data/environment dependent
            return {
                "summary": _summarize_schedule(_load_break_schedule()),
                "controls": _model_dump(request),
                "engine": "unavailable",
                "detail": str(exc)[:300],
            }
    return {
        "summary": _summarize_schedule(_load_break_schedule()),
        "controls": _model_dump(request),
        "engine": "unavailable",
    }


@router.post("/api/optimize-plan")
def optimize_plan(request: OptimizePlanRequest) -> dict[str, Any]:
    """Serve a real optimal break plan, driven by the saved settings.

    Runs the optimization engine in process, using the live KairosSettings as
    guardrails, so the dashboard's settings page controls the optimizer directly.
    """
    if not _ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optimization engine is unavailable")
    settings = _load_settings()
    risk = request.risk_lambda if request.risk_lambda is not None else getattr(settings, "risk_lambda", 0.0)
    try:
        return optimize_day_plan(
            channel=request.channel,
            day=request.day,
            revenue_weight=request.revenue_weight,
            risk_lambda=risk,
            daily_input_path=request.daily_input,
            settings=_model_dump(settings),
            today=_reference_today(settings),
        )
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


@router.get("/api/parameters")
def parameters() -> dict[str, Any]:
    """Every adjustable parameter the optimizer uses, in one place.

    Surfaces the guardrails (derived from the saved settings), the declared
    optimizer assumptions, the pricing model, and the known channels, so the
    dashboard can show and edit each one.
    """
    settings = _load_settings()
    payload: dict[str, Any] = {"settings": _model_dump(settings)}
    if not _ENGINE_AVAILABLE:
        payload["engine"] = "unavailable"
        return payload
    payload["guardrails"] = _asdict(guardrails_from_settings(_model_dump(settings)))
    payload["assumptions"] = _asdict(OptimizerAssumptions())
    payload["channels"] = list(KAIROS_CHANNELS)
    payload["operator_channel"] = settings.operator_channel
    # Honest flag: when no channel is selected the competitor-boundary filter is
    # inactive (constraints match any channel). The dashboard uses this to warn
    # the operator so they know to visit OperatorChannelPanel and pick a channel.
    payload["operator_channel_unset"] = not bool(settings.operator_channel)
    # available_channels drives the operator-channel picker. Derive it from the
    # real loaded EPG (the same channel_options the constraint engine uses) so the
    # picker can never drift from the channel ids the optimizer actually schedules
    # on. Fall back to the canonical channel constant only if the EPG is missing.
    try:
        from kairos_api._constraint_options import channel_options as _channel_options

        _data_channels = _channel_options()
    except Exception:
        _data_channels = []
    payload["available_channels"] = _data_channels or list(KAIROS_CHANNELS)
    try:
        pricing = PricingModel.from_yaml()
        payload["pricing"] = {
            "base_price_per_second_per_tvr_point": pricing.base_price,
            "program_type_premiums": pricing.program_type_premiums,
            "ad_type_premiums": pricing.ad_type_premiums,
            "position_premiums": {str(k): v for k, v in pricing.position_premiums.items()},
            "day_of_week_premiums": {str(k): v for k, v in pricing.day_of_week_premiums.items()},
        }
    except Exception as exc:  # pragma: no cover - config dependent
        payload["pricing"] = {"error": str(exc)[:200]}
    # Honest freshness of the measured retention coefficients: re-hash the source
    # files the coefficients were computed from and report fresh/stale/unknown so
    # the dashboard can warn when the data has moved on from the stored deltas.
    try:
        from kairos.model.freshness import coefficient_freshness
        from kairos.model.measure import read_coefficients_metadata

        metadata = read_coefficients_metadata(MODELS_DIR / "tv_break_coefficients.json")
        payload["coefficient_freshness"] = coefficient_freshness(metadata, root=ROOT)
        # Surface the self-activating first-break retention lever from the measured
        # coefficients metadata so the dashboard can show, honestly, when a show's
        # first break is charged extra retention cost. Off (multiplier 1.0) when the
        # gate found no real first-break contrast.
        payload["first_break_active"] = bool(metadata.get("first_break_active", False))
        try:
            payload["first_break_multiplier"] = float(metadata.get("first_break_multiplier", 1.0) or 1.0)
        except (TypeError, ValueError):
            payload["first_break_multiplier"] = 1.0
    except Exception as exc:  # pragma: no cover - defensive, never blocks parameters
        payload["coefficient_freshness"] = {
            "status": "unknown",
            "computed_at": None,
            "changed_files": [],
            "reason": f"freshness check unavailable: {str(exc)[:160]}",
        }
        payload["first_break_active"] = False
        payload["first_break_multiplier"] = 1.0
    return payload
