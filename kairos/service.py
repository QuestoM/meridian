"""Service layer: one call from real data to a serialisable optimization result.

This is the seam the API calls. It wraps the whole engine (load -> classify ->
price -> optimize) into a single function that returns plain dicts, and it maps
the dashboard's settings (expressed in minutes) onto the engine's Guardrails
(expressed in seconds). With this in place the endpoints can serve real,
computed plans, and every adjustable number is echoed back so the dashboard can
display and edit it instead of relying on placeholder math.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from kairos.data import ProgramClassifier
from kairos.data.ai_classifier import CachedClassifier, load_ai_overrides
from kairos.data.loaders import REFERENCE_DIR, load_daily_input, load_programmes
from kairos.data.transform import (
    build_segments_from_daily_input,
    build_segments_from_programmes,
)
from kairos.model.impact import ImpactModel, load_impact_model
from kairos.observability.run_log import build_run_record, write_run_log
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.demand import build_demand_weights
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.objective import clamp
from kairos.optimize.optimizer import OptimizationResult, PlacementPin, optimize_breaks
from kairos.optimize.overrides import OverrideSet
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0

ROOT = Path(__file__).resolve().parents[1]
# Operator overrides (pins / forces / forbids / gold), loaded when present so a
# manual hand-fix reaches the recompute. Absent file means no overrides.
DEFAULT_OVERRIDES_PATH = ROOT / "data" / "manual_overrides.csv"
# The trained Meridian posterior, when present. load_impact_model falls back to
# the declared assumption coefficient when this file or Meridian is absent.
DEFAULT_IMPACT_MODEL_PATH = ROOT / "models" / "tv_break_posterior.pkl"
# Trusted AI genres for titles the rule-based classifier left as "Other",
# written by scripts/classify_unclassified.py. Absent or empty means no AI
# overrides, and classification stays purely rule-based (no fabrication).
AI_CLASSIFICATIONS_PATH = ROOT / "models" / "ai_program_classifications.json"


def _build_classifier() -> ProgramClassifier:
    """The rule-based classifier, wrapped with any trusted AI genres on disk."""
    classifier = ProgramClassifier.from_yaml()
    overrides = load_ai_overrides(AI_CLASSIFICATIONS_PATH)
    return CachedClassifier(classifier, overrides) if overrides else classifier


def guardrails_from_settings(settings: Mapping[str, Any]) -> Guardrails:
    """Map the dashboard settings (minutes) onto engine Guardrails (seconds).

    Missing keys fall back to the Guardrails defaults, so a partial settings
    object still produces a valid, fully populated guardrail set.
    """
    base = Guardrails()
    settings = settings or {}

    def minutes(key: str, default_seconds: float) -> float:
        value = settings.get(key)
        return float(value) * SECONDS_PER_MINUTE if value is not None else default_seconds

    protected = settings.get("protected_program_types")
    return Guardrails(
        max_ad_seconds_per_hour=minutes("max_ad_minutes_per_hour", base.max_ad_seconds_per_hour),
        max_breaks_per_hour=int(settings.get("max_breaks_per_hour", base.max_breaks_per_hour)),
        min_break_spacing_seconds=minutes("min_break_spacing_minutes", base.min_break_spacing_seconds),
        min_retention_floor=float(settings.get("min_retention_floor", base.min_retention_floor)),
        max_daily_ad_seconds=minutes("max_daily_ad_minutes", base.max_daily_ad_seconds),
        protected_program_types=tuple(protected) if protected else base.protected_program_types,
        protected_max_ad_seconds_per_hour=minutes(
            "protected_program_max_ad_minutes_per_hour", base.protected_max_ad_seconds_per_hour
        ),
        gold_breaks_max_per_day=int(settings.get("gold_breaks_max_per_day", base.gold_breaks_max_per_day)),
    )


def _hhmm(start_seconds: float) -> str:
    total_minutes = int(start_seconds // SECONDS_PER_MINUTE)
    return f"{(total_minutes // 60) % 24:02d}:{total_minutes % 60:02d}"


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a retention coefficient for display, preserving an honest ``None``.

    A missing credible bound stays ``None`` rather than collapsing to 0.0, so the
    dashboard can tell "no interval known" apart from "interval is zero".
    """
    return None if value is None else round(value, digits)


def result_to_dict(
    result: OptimizationResult,
    *,
    channel: Optional[str],
    day: Optional[str],
) -> dict[str, Any]:
    """Serialise an OptimizationResult into the dashboard-facing shape."""
    placements = [
        {
            "segment_id": p.segment_id,
            "channel": p.channel,
            "day": p.day,
            "hour": p.hour,
            "start_time": _hhmm(p.start_seconds),
            "duration_seconds": round(p.duration_seconds, 1),
            "program_type": p.program_type,
            "position_in_segment": p.position_in_segment,
            "retention_percent": round(p.retention * 100, 1),
            "revenue": round(p.revenue, 2),
            "is_gold": p.is_gold,
        }
        for p in result.placements
    ]
    return {
        "channel": channel,
        "day": day,
        "summary": {
            "total_breaks": result.total_breaks,
            "total_ad_seconds": int(round(sum(p.duration_seconds for p in result.placements))),
            "projected_revenue": round(result.total_revenue, 2),
            "average_retention": round(result.aggregate_retention * 100, 1),
            "objective": round(result.objective, 4),
            "compliant": result.is_compliant,
        },
        "placements": placements,
        "segments": [
            {
                "segment_id": s.segment_id,
                "num_breaks": s.num_breaks,
                "retention_percent": round(s.retention * 100, 1),
                "revenue": round(s.revenue, 2),
                "retention_cost": {
                    "point": _round_or_none(s.retention_cost_point),
                    "used": _round_or_none(s.retention_cost_used),
                    "ci_low": _round_or_none(s.retention_cost_ci_low),
                    "ci_high": _round_or_none(s.retention_cost_ci_high),
                    "n": s.retention_cost_n,
                    "confidence": s.retention_confidence,
                },
            }
            for s in result.segments
        ],
        "violations": [
            {"code": v.code, "scope": v.scope, "observed": v.observed, "limit": v.limit, "detail": v.detail}
            for v in result.violations
        ],
        "decisions": [
            {
                "segment_id": d.segment_id,
                "break_index": d.break_index,
                "marginal_objective_gain": round(d.marginal_objective_gain, 4),
                "marginal_revenue": round(d.marginal_revenue, 2),
                "retention_after_percent": round(d.retention_after * 100, 1),
            }
            for d in result.decisions
        ],
        "weights": {
            "revenue_weight": result.revenue_weight,
            "revenue_scale": round(result.revenue_scale, 2),
            "risk_lambda": result.risk_lambda,
        },
    }


def optimize_day_plan(
    *,
    channel: Optional[str] = None,
    day: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    revenue_weight: Optional[float] = None,
    risk_lambda: Optional[float] = None,
    assumptions: Optional[OptimizerAssumptions] = None,
    pricing: Optional[PricingModel] = None,
    programmes: Optional[pd.DataFrame] = None,
    programmes_path: Optional[str] = None,
    daily_input_path: Optional[str] = None,
    impact_model: Optional[ImpactModel] = None,
    overrides: Optional[OverrideSet] = None,
    placement_pins: Optional[Mapping[str, Any]] = None,
    log_run: bool = True,
    run_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full real pipeline and return a serialisable plan.

    ``channel`` and ``day`` (``YYYY-MM-DD``) narrow the grid; omitting both
    optimises every channel-day in the source. ``settings`` is the dashboard's
    KairosSettings (used for guardrails); ``revenue_weight`` overrides the
    assumptions default. ``impact_model`` supplies the retention coefficient;
    when omitted it is loaded from the trained posterior if present, else the
    declared assumption. The returned dict echoes the guardrails and assumptions
    used, so every adjustable number is visible to the caller. When ``log_run``
    is set, the run is appended to ``output/run_log.jsonl`` for audit, stamped
    with ``run_id`` and ``created_at`` (generated here when not supplied).
    """
    pricing = pricing or PricingModel.from_yaml()
    assumptions = assumptions or OptimizerAssumptions()
    guardrails = guardrails_from_settings(settings) if settings else Guardrails()
    weight = revenue_weight if revenue_weight is not None else assumptions.revenue_weight
    risk = risk_lambda if risk_lambda is not None else assumptions.risk_lambda
    if impact_model is None:
        impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    overrides = overrides if overrides is not None else _load_default_overrides()
    classifier = _build_classifier()

    if daily_input_path is not None:
        # Drive the decision from the real daily plan (the Wally csv): the
        # optimizer places breaks on the day's actual programme lineup.
        daily = load_daily_input(daily_input_path)
        segments = build_segments_from_daily_input(
            daily, classifier, pricing, assumptions=assumptions, impact_model=impact_model,
        )
        if segments:
            channel = channel or segments[0].channel
            day = day or segments[0].day
    else:
        if programmes is None:
            programmes = load_programmes(programmes_path)
        # The real decision is per channel per day. With neither given, default to
        # the first channel-day in the source so a plain call stays a single day.
        if channel is None and day is None:
            channel, day = _first_channel_day(programmes)
        segments = build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=assumptions, impact_model=impact_model, channel=channel, day=day,
        )
    # Demand weights: always computed, self-neutralizing when the advertiser
    # CSVs carry no matching rules (every weight is 1.0, output byte-identical).
    demand_engine = AdvertiserRuleEngine.from_files()
    demand_weights = build_demand_weights(segments, demand_engine)
    result = optimize_breaks(
        segments, guardrails, revenue_weight=weight, risk_lambda=risk,
        overrides=overrides, placement_pins=placement_pins,
        demand_weights=demand_weights,
    )

    payload = result_to_dict(result, channel=channel, day=day)
    payload["guardrails"] = asdict(guardrails)
    payload["assumptions"] = asdict(assumptions)
    payload["segment_count"] = len(segments)
    payload["impact_source"] = impact_model.source

    if log_run:
        # Provenance: hash the programmes file when one fed the run. With a frame
        # passed directly there is no file, so the default reference path is the
        # honest source. run_id and created_at come from here, the app layer.
        source = daily_input_path or programmes_path or (REFERENCE_DIR / "Programmes.xlsx")
        record = build_run_record(
            run_id=run_id or uuid.uuid4().hex,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            channel=channel,
            day=day,
            source_paths={"programmes": source},
            guardrails=asdict(guardrails),
            assumptions=asdict(assumptions),
            summary=payload["summary"],
            segment_count=len(segments),
        )
        write_run_log(record)
    return payload


def run_scenario(
    *,
    revenue_weight: float,
    retention_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float = 0.0,
    channel: Optional[str] = None,
    day: Optional[str] = None,
    programmes: Optional[pd.DataFrame] = None,
    programmes_path: Optional[str] = None,
    overrides: Optional[OverrideSet] = None,
    placement_pins: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run a real optimization for the dashboard scenario slider.

    ``revenue_weight`` is the dashboard's 0..100 control; it is scaled to the
    optimizer's [0, 1]. ``retention_floor`` and ``max_breaks_per_hour`` override
    the matching guardrails. The plan is computed on the chosen channel-day (or
    the first one in the source when none is given) to stay responsive.
    """
    weight = clamp(float(revenue_weight) / 100.0, 0.0, 1.0)
    if programmes is None:
        programmes = load_programmes(programmes_path)
    if channel is None and day is None:
        channel, day = _first_channel_day(programmes)

    guardrails = Guardrails(
        min_retention_floor=float(retention_floor),
        max_breaks_per_hour=int(max_breaks_per_hour),
    )
    pricing = PricingModel.from_yaml()
    assumptions = OptimizerAssumptions()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    overrides = overrides if overrides is not None else _load_default_overrides()
    classifier = _build_classifier()
    segments = build_segments_from_programmes(
        programmes, classifier, pricing,
        assumptions=assumptions, impact_model=impact_model, channel=channel, day=day,
    )
    # Demand weights: always computed, self-neutralizing when no rules match.
    demand_engine = AdvertiserRuleEngine.from_files()
    demand_weights = build_demand_weights(segments, demand_engine)
    result = optimize_breaks(
        segments, guardrails, revenue_weight=weight, risk_lambda=risk_lambda,
        overrides=overrides, placement_pins=placement_pins,
        demand_weights=demand_weights,
    )

    payload = result_to_dict(result, channel=channel, day=day)
    payload["controls"] = {
        "revenue_weight": revenue_weight,
        "retention_floor": retention_floor,
        "max_breaks_per_hour": max_breaks_per_hour,
        "risk_lambda": risk_lambda,
    }
    payload["guardrails"] = asdict(guardrails)
    return payload


def _load_default_overrides() -> Optional[OverrideSet]:
    """Load the operator overrides CSV when present, else None (no overrides)."""
    if DEFAULT_OVERRIDES_PATH.exists():
        return OverrideSet.from_csv(DEFAULT_OVERRIDES_PATH)
    return None


def _first_channel_day(programmes: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    valid = programmes[programmes.get("start_dt").notna()] if "start_dt" in programmes.columns else programmes
    if valid.empty:
        return None, None
    row = valid.sort_values("start_dt").iloc[0]
    return str(row["Channel"]), row["start_dt"].strftime("%Y-%m-%d")
