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
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from kairos.data import ProgramClassifier
from kairos.data.ai_classifier import CachedClassifier, load_ai_overrides
from kairos.data.dayparts import daypart_for_hour
from kairos.data.loaders import REFERENCE_DIR, load_daily_input, load_programmes
from kairos.data.transform import (
    build_segments_from_daily_input,
    build_segments_from_programmes,
)
from kairos.model.freshness import coefficient_freshness
from kairos.model.impact import ImpactModel, load_impact_model
from kairos.model.measure import read_coefficients_metadata
from kairos.observability.run_log import build_run_record, write_run_log
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.demand import build_demand_weights
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.inventory import build_inventory_weights, load_inventory
from kairos.optimize.objective import clamp
from kairos.optimize.pacing import (
    URGENCY_EPSILON,
    URGENCY_K,
    URGENCY_K_AHEAD,
    URGENCY_U_MAX,
    URGENCY_U_MIN,
    build_pacing_weights,
    load_campaigns,
)
from kairos.optimize.optimizer import OptimizationResult, PlacementPin, optimize_breaks
from kairos.optimize.overrides import OverrideSet
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel, pricing_from_settings

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0

ROOT = Path(__file__).resolve().parents[1]
# Operator overrides (pins / forces / forbids / gold), loaded when present so a
# manual hand-fix reaches the recompute. Absent file means no overrides.
DEFAULT_OVERRIDES_PATH = ROOT / "data" / "manual_overrides.csv"
# The trained Meridian posterior, when present. load_impact_model falls back to
# the declared assumption coefficient when this file or Meridian is absent.
DEFAULT_IMPACT_MODEL_PATH = ROOT / "models" / "tv_break_posterior.pkl"
# The measured per-cell retention coefficients. Its metadata carries the
# freshness stamp (computed_at + source_fingerprints) used to detect stale deltas.
DEFAULT_COEFFICIENTS_PATH = ROOT / "models" / "tv_break_coefficients.json"
# Trusted AI genres for titles the rule-based classifier left as "Other",
# written by scripts/classify_unclassified.py. Absent or empty means no AI
# overrides, and classification stays purely rule-based (no fabrication).
AI_CLASSIFICATIONS_PATH = ROOT / "models" / "ai_program_classifications.json"


def _build_classifier() -> ProgramClassifier:
    """The rule-based classifier, wrapped with any trusted AI genres on disk."""
    classifier = ProgramClassifier.from_yaml()
    overrides = load_ai_overrides(AI_CLASSIFICATIONS_PATH)
    return CachedClassifier(classifier, overrides) if overrides else classifier


def _coefficient_freshness_block(impact_model: ImpactModel) -> dict[str, Any]:
    """Honest freshness verdict for the retention coefficients the plan used.

    Only the measured JSON carries source fingerprints, so freshness can be
    verified only when the impact source is "measured": we read its metadata and
    compare the fingerprinted source files against disk now. For any other source
    (the declared assumption, or a trained posterior) there is no fingerprinted
    measured JSON to check, so the verdict is an honest "unknown" that names the
    source, never a fabricated "fresh".
    """
    if impact_model.source == "measured":
        metadata = read_coefficients_metadata(DEFAULT_COEFFICIENTS_PATH)
        return coefficient_freshness(metadata, root=ROOT)
    return {
        "status": "unknown",
        "computed_at": None,
        "changed_files": [],
        "reason": (
            f"Coefficients are not the measured JSON (impact source is "
            f"'{impact_model.source}'); freshness is not applicable."
        ),
    }


def _apply_first_break_multiplier(assumptions: OptimizerAssumptions) -> OptimizerAssumptions:
    """Fold the measured first-break multiplier from the coefficients JSON into
    the assumptions, identically to the export path (export/schedule.py).

    The build pipeline persists ``first_break_multiplier`` (1.0 when the gate did
    not earn a value, > 1.0 when a show's first interruption measurably sheds more
    audience). The CSV export already reads it; the live service path did not, so
    once the self-activating gate ships a value above 1.0 the scenario/day-plan
    numbers would silently disagree with the exported plan. We read the same JSON
    here so both paths charge the first break the same measured cost. A missing or
    1.0 value (the current state) leaves the assumptions unchanged, so default
    behaviour and reported revenue are byte-identical. An explicit operator
    override above 1.0 is respected and never lowered.
    """
    metadata = read_coefficients_metadata(DEFAULT_COEFFICIENTS_PATH)
    measured = metadata.get("first_break_multiplier")
    try:
        measured_value = float(measured) if measured is not None else 1.0
    except (TypeError, ValueError):
        measured_value = 1.0
    chosen = max(assumptions.first_break_multiplier, measured_value)
    if chosen == assumptions.first_break_multiplier:
        return assumptions
    return replace(assumptions, first_break_multiplier=chosen)


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


def _parse_pacing_date(value: Any) -> Optional[date]:
    """Parse a YYYY-MM-DD reference date from settings, or None when absent/invalid."""
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0].split("T")[0]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _pacing_knobs_from_settings(
    settings: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Translate the dashboard's pacing settings into pacing-urgency knobs.

    Returns ``None`` when no settings are supplied, so a call with no settings
    keeps the module-default pacing behavior (identity until campaign data lands).
    Otherwise returns a dict carrying the ``enabled`` flag, the optional reference
    date, and the five urgency knobs, each falling back to the module default when
    the matching field is absent. ``pacing_weight_floor`` maps to the over-delivery
    floor ``u_min`` (the lowest a de-prioritized slot's weight may fall).
    """
    if not settings:
        return None

    def _num(key: str, default: float) -> float:
        try:
            return float(settings.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": bool(settings.get("pacing_enabled", True)),
        "reference_date": settings.get("pacing_reference_date"),
        "k": _num("pacing_urgency_k", URGENCY_K),
        "u_max": _num("pacing_urgency_max", URGENCY_U_MAX),
        "k_ahead": _num("pacing_ahead_k", URGENCY_K_AHEAD),
        "u_min": _num("pacing_weight_floor", URGENCY_U_MIN),
        "epsilon": _num("pacing_epsilon", URGENCY_EPSILON),
    }


def _assemble_demand_weights(
    segments: list,
    *,
    today: Optional[date] = None,
    pacing_knobs: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Build the optimizer's per-segment placement weights from all three signals.

    Folds advertiser demand (always computed), inventory awareness, and delivery
    pacing into one weight map via :func:`build_demand_weights`. Each signal is an
    identity no-op until its data lands, so with no advertiser rules, no inventory
    file, and a header-only ``campaign_flights.csv`` every weight is exactly 1.0 and
    the optimizer output is byte-identical to a run with no weights at all.

    Pacing fires only when (a) real campaign rows exist, (b) pacing is enabled (the
    dashboard default; disabled only when settings say so), and (c) a reference date
    is available, either the explicit ``pacing_reference_date`` setting or the caller
    ``today``. Missing any of these leaves pacing as ``None`` (1.0 everywhere), the
    honest conservative choice. Knob overrides from ``pacing_knobs`` feed straight
    into :func:`build_pacing_weights`; per-campaign overrides inside the CSV still
    take precedence over these channel-wide defaults.
    """
    segments = list(segments)
    engine = AdvertiserRuleEngine.from_files()
    inventory_weights = build_inventory_weights(segments, load_inventory())

    pacing_weights: Optional[dict[str, float]] = None
    enabled = True if pacing_knobs is None else bool(pacing_knobs.get("enabled", True))
    campaigns = load_campaigns() if enabled else []
    if campaigns:
        reference = today
        if pacing_knobs is not None:
            override = _parse_pacing_date(pacing_knobs.get("reference_date"))
            if override is not None:
                reference = override
        if reference is not None:
            daypart_of = {
                seg.segment_id: daypart_for_hour(seg.hour) for seg in segments
            }
            knob_kwargs: dict[str, float] = {}
            if pacing_knobs is not None:
                for key in ("k", "u_max", "k_ahead", "u_min", "epsilon"):
                    value = pacing_knobs.get(key)
                    if value is not None:
                        knob_kwargs[key] = value
            pacing_weights = build_pacing_weights(
                segments, campaigns, reference, daypart_of=daypart_of,
                advertiser_k_of=engine.pacing_overrides(), **knob_kwargs
            )

    return build_demand_weights(
        segments, engine,
        inventory_weights=inventory_weights,
        pacing_weights=pacing_weights,
    )


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
    today: Optional[date] = None,
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
    pricing = pricing_from_settings(settings, pricing)
    assumptions = _apply_first_break_multiplier(assumptions or OptimizerAssumptions())
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
    # Demand weights: advertiser demand always computed, folded with the
    # inventory-awareness and pacing-urgency signals. Each is identity (1.0) until
    # its data lands, so with no inventory file and no campaign rows the output is
    # byte-identical to a run with no weights at all.
    demand_weights = _assemble_demand_weights(
        segments, today=today, pacing_knobs=_pacing_knobs_from_settings(settings),
    )
    # Honor the operator's stored placement constraints, wired identically to the
    # CSV recompute path so the live plan and the saved schedule agree. No-op when
    # no constraints file exists (the current state).
    merged_pins, merged_overrides = _constraint_inputs(
        segments, overrides, placement_pins, operator_channel=_operator_channel(settings),
    )
    result = optimize_breaks(
        segments, guardrails, revenue_weight=weight, risk_lambda=risk,
        overrides=merged_overrides, placement_pins=merged_pins,
        demand_weights=demand_weights,
    )

    payload = result_to_dict(result, channel=channel, day=day)
    payload["guardrails"] = asdict(guardrails)
    payload["assumptions"] = asdict(assumptions)
    payload["segment_count"] = len(segments)
    payload["impact_source"] = impact_model.source
    payload["coefficient_freshness"] = _coefficient_freshness_block(impact_model)

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
    refine: bool = True,
    today: Optional[date] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run a real optimization for the dashboard scenario slider.

    ``revenue_weight`` is the dashboard's 0..100 control; it is scaled to the
    optimizer's [0, 1]. ``retention_floor`` and ``max_breaks_per_hour`` override
    the matching guardrails. The plan is computed on the chosen channel-day (or
    the first one in the source when none is given) to stay responsive.

    ``refine`` (default ``True``) runs the F1 local-search refiner after greedy
    for the committed plan. Set ``refine=False`` for the pure-greedy optimum,
    which is much faster on large multi-day scopes: the exploratory revenue
    frontier uses it so a whole-channel sweep stays interactive (greedy is the
    same real optimizer, just without the per-group local-search polish).
    """
    weight = clamp(float(revenue_weight) / 100.0, 0.0, 1.0)
    if programmes is None:
        programmes = load_programmes(programmes_path)
    if channel is None and day is None:
        channel, day = _first_channel_day(programmes)

    # Start from the operator's saved guardrails so the scenario/frontier path
    # honours every setting (ad-minutes, spacing, daily load, protected types,
    # gold cap), not only the two the scenario slider controls. retention_floor
    # and max_breaks_per_hour are the explicit scenario overrides on top.
    base_guardrails = guardrails_from_settings(settings) if settings else Guardrails()
    guardrails = replace(
        base_guardrails,
        min_retention_floor=float(retention_floor),
        max_breaks_per_hour=int(max_breaks_per_hour),
    )
    pricing = pricing_from_settings(settings)
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    overrides = overrides if overrides is not None else _load_default_overrides()
    classifier = _build_classifier()
    segments = build_segments_from_programmes(
        programmes, classifier, pricing,
        assumptions=assumptions, impact_model=impact_model, channel=channel, day=day,
    )
    # Demand weights: advertiser demand, inventory awareness and delivery pacing,
    # folded together and self-neutralizing when no rules, inventory or campaigns
    # are present (every weight 1.0, byte-identical to a run with no weights).
    demand_weights = _assemble_demand_weights(
        segments, today=today, pacing_knobs=_pacing_knobs_from_settings(settings),
    )
    # Honor the operator's stored placement constraints, wired identically to the
    # CSV recompute path so the frontier / scenario plan is the achievable optimum
    # under the operator's rules, not an unconstrained one. No-op with no file.
    merged_pins, merged_overrides = _constraint_inputs(
        segments, overrides, placement_pins, operator_channel=_operator_channel(settings),
    )
    result = optimize_breaks(
        segments, guardrails, revenue_weight=weight, risk_lambda=risk_lambda,
        overrides=merged_overrides, placement_pins=merged_pins,
        demand_weights=demand_weights, refine=refine,
    )

    payload = result_to_dict(result, channel=channel, day=day)
    payload["controls"] = {
        "revenue_weight": revenue_weight,
        "retention_floor": retention_floor,
        "max_breaks_per_hour": max_breaks_per_hour,
        "risk_lambda": risk_lambda,
    }
    payload["guardrails"] = asdict(guardrails)
    payload["impact_source"] = impact_model.source
    payload["coefficient_freshness"] = _coefficient_freshness_block(impact_model)
    return payload


def _load_default_overrides() -> Optional[OverrideSet]:
    """Load the operator overrides CSV when present, else None (no overrides)."""
    if DEFAULT_OVERRIDES_PATH.exists():
        return OverrideSet.from_csv(DEFAULT_OVERRIDES_PATH)
    return None


def _constraint_inputs(
    segments: list,
    overrides: Optional[OverrideSet],
    placement_pins: Optional[Mapping[str, Any]],
    *,
    operator_channel: str = "",
) -> tuple[Optional[Mapping[str, Any]], Optional[OverrideSet]]:
    """Fold the operator's stored placement constraints into (pins, overrides).

    This is the SAME wiring the CSV recompute path uses
    (:func:`kairos.export.schedule.build_weekly_schedule` via its
    ``_load_constraints`` + ``_constraint_inputs`` + ``merged_pins`` steps), so the
    live scenario / day-plan numbers agree with the saved schedule for the same
    channel-day and inputs. The stored constraints are loaded from the default
    ``data/kairos_constraints.csv`` (only when it exists), resolved against THIS
    run's ``segments`` with the operator's own channel, and merged on top of the
    caller's ``overrides`` and ``placement_pins``.

    The merge mirrors the export path exactly: caller placement pins win on a
    segment-id collision (``{**day_pins, **caller_pins}``), and constraint count
    pins / forbids are appended after the caller's overrides in one OverrideSet.
    With no constraints file (the current deployment state) this returns
    ``(placement_pins, overrides)`` untouched, so behaviour is byte-identical to
    before and no revenue moves.
    """
    from kairos.optimize.constraints_store import (
        DEFAULT_CONSTRAINTS_PATH,
        count_pins_to_overrides,
        load_constraints,
        resolve_constraints,
    )

    constraints = load_constraints(DEFAULT_CONSTRAINTS_PATH) if DEFAULT_CONSTRAINTS_PATH.exists() else []
    if not constraints:
        return placement_pins, overrides

    day_pins, count_pins, forbids, _ = resolve_constraints(
        segments, constraints, operator_channel=operator_channel,
    )
    constraint_overrides = count_pins_to_overrides(count_pins, forbids)
    if overrides is not None:
        constraint_overrides = OverrideSet(
            overrides=list(overrides.overrides) + list(constraint_overrides.overrides),
        )
    merged_pins = {**day_pins, **(placement_pins or {})}
    return (merged_pins or None), constraint_overrides


def _operator_channel(settings: Optional[Mapping[str, Any]]) -> str:
    """The operator's own channel from settings, or '' (no channel filter).

    Sourced identically to the export path
    (:func:`kairos.export.schedule.build_weekly_schedule`), so constraints scope to
    the same channel in both engines, honouring the competitor-information
    boundary. Empty when no settings or no key, the honest no-op.
    """
    if not settings:
        return ""
    return str(settings.get("operator_channel", "") or "")


def _first_channel_day(programmes: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    valid = programmes[programmes.get("start_dt").notna()] if "start_dt" in programmes.columns else programmes
    if valid.empty:
        return None, None
    row = valid.sort_values("start_dt").iloc[0]
    return str(row["Channel"]), row["start_dt"].strftime("%Y-%m-%d")
