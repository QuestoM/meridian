"""Persist a real optimized weekly break schedule for the dashboard.

The dashboard renders its schedule canvas, break-operations board and
recommendation list from ``output/weekly_break_schedule.csv``. When that file is
missing the API falls back to placeholder revenue and retention numbers. This
module removes that fallback in practice: it runs the real engine across every
channel-day in the source and writes one row per programme segment with the
columns the dashboard consumes, so every number on those screens is computed,
not invented.

Each row is one programme segment. ``num_breaks`` is the optimizer's decision,
``predicted_revenue`` and ``predicted_retention`` come straight from the
:class:`~kairos.optimize.optimizer.SegmentPlan`, ``base_rate`` is the segment's
effective per-second rate (CPP times premium) and ``break_type`` is derived from
the break length. Nothing on the row is fabricated: a programme the optimizer
left without breaks earns zero and keeps its full baseline retention.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import ImpactModel, load_impact_model
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.day_core import _optimize_one_day
from kairos.optimize.guardrails import Guardrails
# build_inventory_weights and build_pacing_weights are no longer called in this
# module's body: the demand fold lives in kairos.service._assemble_demand_weights,
# which every optimize path reaches through _optimize_one_day. They stay imported
# because the demand-assembly equivalence test patches them on this module as the
# fold's per-module seam; load_inventory / load_campaigns are still called directly
# to load the demand resources once before the channel-day loop.
from kairos.optimize.inventory import build_inventory_weights, load_inventory  # noqa: F401
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.overrides import OverrideSet
from kairos.optimize.pacing import build_pacing_weights, load_campaigns  # noqa: F401
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel, pricing_from_settings
from kairos.service import (
    _apply_first_break_multiplier,
    _build_classifier,
    _pacing_knobs_from_settings,
    guardrails_from_settings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "output" / "weekly_break_schedule.csv"
# The trained Meridian posterior, when present. Loaded once and threaded into
# every channel-day so the exported schedule uses the measured per-channel
# coefficients, matching what the live service returns. Falls back honestly to
# the declared assumption when the file or Meridian is absent.
DEFAULT_IMPACT_MODEL_PATH = ROOT / "models" / "tv_break_posterior.pkl"

SECONDS_PER_MINUTE = 60

# The measured first-break multiplier is folded into the assumptions by the shared
# kairos.service._apply_first_break_multiplier, imported above so the export and the
# live paths charge the show's first break the same measured extra cost.

# Column order the dashboard's schedule readers expect (extra columns are
# ignored by them, missing ones trigger the placeholder fallback we are killing).
COLUMNS = [
    "channel",
    "date",
    "day",
    "program_type",
    "start_time",
    "num_breaks",
    "break_length",
    "total_break_time",
    "predicted_revenue",
    "predicted_retention",
    "position",
    "break_type",
    "base_rate",
    # Risk-adjusted retention columns: the value the optimizer actually decided
    # with under the active risk_lambda (equals predicted_retention when
    # risk_lambda=0 or no CI is available), plus the per-segment credible
    # interval and confidence that justify it. Empty (None -> blank in CSV)
    # when the segment has no measured CI, so nothing is fabricated.
    "retention_used",
    "retention_ci_low",
    "retention_ci_high",
    "retention_n",
    "retention_confidence",
    # Row identity: the optimizer's own segment key, so every schedule row is
    # addressable on the plane the override and constraint engines consume (they
    # key target_id on this exact segment_id). Additive and non-fabricated: it is
    # the same key used to join plans to segments below, restored as a column.
    "segment_id",
    # Whether the optimizer marked this segment's breaks gold. Sourced honestly
    # from the plan's placements (each BreakPlacement carries is_gold, set from
    # the segment / override gold flag); a segment with no gold break reads False.
    # No per-segment objective_contribution column is emitted: SegmentPlan does
    # not carry one (the objective is only defined at group level via
    # _group_objective_contribution, and the greedy decisions are invalidated by
    # the refiner), so a real value is not available without an engine change.
    "is_gold",
]


def _clock(start_seconds: float) -> str:
    total_minutes = int(start_seconds // SECONDS_PER_MINUTE)
    return f"{(total_minutes // 60) % 24:02d}:{total_minutes % 60:02d}"


def _weekday_abbrev(day: str) -> str:
    """Map a ``YYYY-MM-DD`` date to the dashboard's day key (``Mon`` .. ``Sun``)."""
    return pd.Timestamp(day).strftime("%a")


def _break_type(break_seconds: float) -> str:
    """Label a break by its length, the way the operations board groups them."""
    if break_seconds < 90:
        return "short"
    if break_seconds < 180:
        return "medium"
    return "long"


def _channel_days(programmes: pd.DataFrame) -> list[tuple[str, str]]:
    if "start_dt" not in programmes.columns:
        raise ValueError("programmes frame must have a start_dt column (use load_programmes)")
    valid = programmes[programmes["start_dt"].notna()]
    if valid.empty:
        return []
    pairs = (
        valid.assign(_day=valid["start_dt"].dt.strftime("%Y-%m-%d"))[["Channel", "_day"]]
        .drop_duplicates()
        .sort_values(["_day", "Channel"])
    )
    return [(str(channel), str(day)) for channel, day in pairs.itertuples(index=False, name=None)]


def _load_constraints(constraints_path: Optional[str | Path]):
    """Load scoped placement constraints, honoring the default file.

    An explicit path is always loaded. With no path, the default
    ``data/kairos_constraints.csv`` is loaded only when it exists, so a deployment
    that never created one gets an empty list and unchanged behaviour.
    """
    from kairos.optimize.constraints_store import DEFAULT_CONSTRAINTS_PATH, load_constraints

    if constraints_path is not None:
        return load_constraints(constraints_path)
    if DEFAULT_CONSTRAINTS_PATH.exists():
        return load_constraints(DEFAULT_CONSTRAINTS_PATH)
    return []


def build_weekly_schedule(
    programmes: Optional[pd.DataFrame] = None,
    *,
    programmes_path: Optional[str] = None,
    pricing: Optional[PricingModel] = None,
    assumptions: Optional[OptimizerAssumptions] = None,
    settings: Optional[Mapping[str, Any]] = None,
    revenue_weight: Optional[float] = None,
    risk_lambda: float = 0.0,
    classifier: Optional[ProgramClassifier] = None,
    impact_model: Optional[ImpactModel] = None,
    overrides: Optional[OverrideSet] = None,
    placement_pins: Optional[Mapping[str, Any]] = None,
    constraints_path: Optional[str | Path] = None,
    operator_channel: str = "",
    today: Optional[date] = None,
) -> pd.DataFrame:
    """Optimise every channel-day and return one schedule row per segment.

    ``settings`` are the dashboard's KairosSettings (mapped onto guardrails);
    ``revenue_weight`` overrides the assumptions default. ``impact_model`` supplies
    the per-channel retention coefficient; when omitted it is loaded from the
    trained posterior if present, else the declared assumption (so the exported
    schedule matches the live service). The frame is sorted by day then channel so
    the output is deterministic.

    ``constraints_path`` points at a scoped placement-constraint CSV
    (:mod:`kairos.optimize.constraints_store`); when set, or when the default
    ``data/kairos_constraints.csv`` exists, each channel-day's segments are matched
    against the stored constraints and the resolved placement pins / count pins /
    forbids are passed into that channel-day's optimize_breaks call, so the
    exported num_breaks honors the operator's scoped rules. With no file it is
    unchanged, so the path is fully backward compatible.

    ``operator_channel`` is the operator's own channel name (from
    KairosSettings.operator_channel). Constraints are automatically scoped to this
    channel, honoring the competitor-information boundary: the operator constrains
    only their own channel's breaks. Empty string -> no channel filter (matches any
    channel, the honest no-op before the operator has picked one).

    ``today`` is the reference date for the delivery-pacing urgency signal
    (:mod:`kairos.optimize.pacing`); it is passed in so the math is deterministic
    (no clock read inside the engine). When omitted, the pacing signal stays inert
    even if campaign data is present, so behaviour is unchanged. The inventory and
    pacing signals fold into demand_weights and steer placement only; with no
    inventory file and no campaign rows the schedule is byte-identical to today.
    """
    pricing = pricing_from_settings(settings, pricing)
    assumptions = assumptions or OptimizerAssumptions()
    # Default to the SAME classifier seam the live service uses: the YAML classifier
    # wrapped with any trusted AI genres on disk. With the AI-overrides file absent
    # (today) this equals ``ProgramClassifier.from_yaml()``, so the export is a no-op.
    classifier = classifier or _build_classifier()
    guardrails = guardrails_from_settings(settings) if settings else Guardrails()
    weight = revenue_weight if revenue_weight is not None else assumptions.revenue_weight
    # If operator_channel was not given explicitly, read it from settings dict.
    if not operator_channel and settings:
        operator_channel = str(settings.get("operator_channel", "") or "")
    if impact_model is None:
        impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    # Fold the measured first-break multiplier (when the gate shipped one) into the
    # assumptions, so the show's first break is charged its measured extra cost.
    # Off (1.0) when the coefficients file has no value, so behaviour is unchanged.
    assumptions = _apply_first_break_multiplier(assumptions)
    if programmes is None:
        programmes = load_programmes(programmes_path)
    if overrides is None and OverrideSet.from_csv().overrides:
        overrides = OverrideSet.from_csv()

    # Load scoped placement constraints once. When the caller gives an explicit
    # path use it; otherwise fall back to the default file only when it exists, so
    # a deployment with no constraints file behaves exactly as before.
    constraints = _load_constraints(constraints_path)

    # Build the advertiser rule engine once for the whole schedule. Self-neutralizing:
    # when the CSVs carry no matching rules every weight is 1.0 (identity).
    demand_engine = AdvertiserRuleEngine.from_files()
    # Load the two coupled placement signals once. Both are identity no-ops until
    # the owner uploads data: an empty inventory pool and an empty campaign list
    # leave every weight at 1.0, so the schedule is byte-identical to today.
    inventory_pool = load_inventory()
    campaigns = load_campaigns()
    # Pacing urgency needs a reference date. We pass it in (deterministic, no
    # datetime.now in the math); a caller that omits it gets none, so even with
    # campaign data present the pacing signal stays inert until a date is given.
    pacing_today = today
    # Pacing knobs from the dashboard settings: the enable flag, the optional
    # reference-date override, and the five urgency knobs. This is the SAME seam
    # the live optimize_day_plan path uses, so the weekly CSV and the dashboard
    # simulation agree once campaign flights land. None when no settings, which
    # keeps the module-default pacing behaviour (identity until campaign data). The
    # enable gate and the reference-date override are applied inside the shared
    # demand fold, so campaigns and pacing_today are passed through untouched here.
    pacing_knobs = _pacing_knobs_from_settings(settings)

    rows: list[dict[str, Any]] = []
    for channel, day in _channel_days(programmes):
        segments = build_segments_from_programmes(
            programmes, classifier, pricing, assumptions=assumptions,
            impact_model=impact_model, channel=channel, day=day,
        )
        if not segments:
            continue
        # One channel-day through the shared core, the same seam the live day plan
        # and scenario slider use. It folds the demand signal (advertiser demand,
        # inventory awareness, delivery pacing; each identity until its data lands)
        # from the resources loaded ONCE above, and resolves the operator's scoped
        # constraints against THIS day's segments, merging the explicit
        # ``placement_pins`` on top. ``optimize_fn`` is this module's optimize_breaks
        # so the demand-assembly equivalence test can observe the folded weights.
        result = _optimize_one_day(
            segments,
            guardrails=guardrails,
            revenue_weight=weight,
            risk_lambda=risk_lambda,
            demand_engine=demand_engine,
            inventory_pool=inventory_pool,
            campaigns=campaigns,
            pacing_today=pacing_today,
            pacing_knobs=pacing_knobs,
            constraints=constraints,
            overrides=overrides,
            placement_pins=placement_pins,
            operator_channel=operator_channel,
            optimize_fn=optimize_breaks,
        )
        plans = {plan.segment_id: plan for plan in result.segments}
        for segment in segments:
            plan = plans.get(segment.segment_id)
            num_breaks = plan.num_breaks if plan else 0
            # A 0-break segment keeps its baseline retention and earns nothing.
            retention = plan.retention if plan else segment.retention_baseline
            revenue = plan.revenue if plan else 0.0

            # Risk-adjusted retention fields: surface the per-segment uncertainty
            # the optimizer used so the weekly CSV carries the full risk decision.
            # ``plan.retention`` is already the risk-adjusted value (computed with
            # the conservative coefficient when risk_lambda > 0 and a CI exists).
            # The CI columns translate the per-break coefficient interval into
            # retention bounds at ``num_breaks`` breaks, so the reader can see the
            # pessimistic and optimistic retention the decision rests on. All four
            # auxiliary fields are None (blank in CSV) when the segment has no
            # measured CI, keeping the export honest.
            if plan is not None:
                ret_ci_low: Optional[float] = None
                ret_ci_high: Optional[float] = None
                if (
                    plan.retention_cost_ci_low is not None
                    and plan.retention_cost_ci_high is not None
                    and num_breaks > 0
                ):
                    from kairos.optimize.objective import clamp, predicted_retention as _pred_ret
                    ret_ci_low = round(
                        _pred_ret(
                            segment.retention_baseline,
                            plan.retention_cost_ci_low,
                            num_breaks,
                        ),
                        4,
                    )
                    ret_ci_high = round(
                        _pred_ret(
                            segment.retention_baseline,
                            plan.retention_cost_ci_high,
                            num_breaks,
                        ),
                        4,
                    )
                retention_n: Optional[int] = plan.retention_cost_n if plan.retention_cost_n else None
                retention_confidence: Optional[str] = plan.retention_confidence or None
            else:
                ret_ci_low = None
                ret_ci_high = None
                retention_n = None
                retention_confidence = None

            rows.append(
                {
                    "channel": segment.channel,
                    "date": segment.day,
                    "day": _weekday_abbrev(segment.day),
                    "program_type": segment.program_type,
                    "start_time": _clock(segment.start_seconds),
                    "num_breaks": num_breaks,
                    "break_length": round(segment.break_length_seconds, 1),
                    "total_break_time": round(num_breaks * segment.break_length_seconds, 1),
                    "predicted_revenue": round(revenue, 2),
                    "predicted_retention": round(retention, 4),
                    "position": "middle",
                    "break_type": _break_type(segment.break_length_seconds),
                    "base_rate": round(segment.cpp * segment.premium, 4),
                    "retention_used": round(retention, 4),
                    "retention_ci_low": ret_ci_low,
                    "retention_ci_high": ret_ci_high,
                    "retention_n": retention_n,
                    "retention_confidence": retention_confidence,
                    "segment_id": segment.segment_id,
                    # Gold is a per-break flag on the plan's placements; a segment
                    # is gold when the optimizer emitted a gold break for it. False
                    # (no gold break, honest) for a 0-break or non-gold segment.
                    "is_gold": bool(plan is not None and any(p.is_gold for p in plan.placements)),
                }
            )

    return pd.DataFrame(rows, columns=COLUMNS)


def write_weekly_schedule(
    path: Optional[str | Path] = None,
    *,
    frame: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> Path:
    """Build (unless ``frame`` is supplied) and write the schedule CSV.

    Returns the path written. Extra keyword arguments are forwarded to
    :func:`build_weekly_schedule`.
    """
    if frame is None:
        frame = build_weekly_schedule(**kwargs)
    target = Path(path) if path is not None else DEFAULT_OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8")
    # Stamp a freshness sidecar next to the CSV so the dashboard can tell honestly
    # whether this saved schedule still matches its inputs (settings, constraints,
    # overrides, coefficients, data). Imported lazily to avoid an import cycle, and
    # guarded so a meta failure never breaks the CSV write, which is the critical
    # path: a missing sidecar simply reads as freshness "unknown" downstream.
    try:
        from kairos.export.schedule_freshness import write_schedule_meta

        write_schedule_meta(target, ROOT)
    except Exception:  # pragma: no cover - meta is best-effort, never blocks export
        logger.warning("Could not write schedule freshness sidecar for %s.", target, exc_info=True)
    return target
