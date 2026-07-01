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
from typing import Any, Callable, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
# _clock is re-exported (unused here) because kairos_api.overrides documents its
# anchor clock as matching kairos.export.schedule._clock; the implementation and
# the shared row construction now live in kairos.export.incremental.
from kairos.export.incremental import (  # noqa: F401
    _break_type,
    _clock,
    _weekday_abbrev,
    incremental_weekly_frame,
    resolve_commit_overrides,
    rows_from_result as _rows_from_result,
    unmatched_anchor_reports,
)
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
    only_days: Optional[list[tuple[str, str]]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    existing_csv: Optional[str | Path] = None,
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

    ``only_days`` (incremental recompute) re-optimizes ONLY the listed
    ``(channel, date)`` pairs and merges the fresh rows into the saved CSV at
    ``existing_csv`` (default: the module's output path), preserving every other
    day's rows verbatim and the full build's row ordering, so the merged result
    is byte-comparable to a full rebuild. Callers are responsible for deriving
    the list from :func:`kairos.export.incremental.classify_change`; a
    hand-picked list can leave a stale day looking current. When the saved CSV
    is missing, has a stale schema, or any other precondition fails, this falls
    back to a FULL build (the honest escape hatch, logged). The merged frame is
    all-string (the CSV's own text form); a full build returns typed columns.

    ``progress_cb(done, total)`` fires once per completed channel-day, in both
    full and incremental runs (after an incremental fallback it restarts with
    the full run's total).

    Overrides whose stored semantic anchor no longer matches their segment are
    SKIPPED at commit time (same guard as the /api/overrides/effect preview);
    the skipped list is logged and returned on ``frame.attrs["skipped_overrides"]``.
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

    skipped_overrides: list[dict[str, Any]] = []
    seen_segment_ids: set[str] = set()

    def _segments_for(channel: str, day: str) -> list:
        return build_segments_from_programmes(
            programmes, classifier, pricing, assumptions=assumptions,
            impact_model=impact_model, channel=channel, day=day,
        )

    def _day_rows(channel: str, day: str) -> list[dict[str, Any]]:
        segments = _segments_for(channel, day)
        if not segments:
            return []
        # Anchor guard at COMMIT time, the same semantics as the /effect preview:
        # a stale-anchored override is dropped from this day's set and reported,
        # a blank-anchor legacy override still binds. With no overrides loaded
        # (the common case) this passes None through untouched.
        day_overrides = resolve_commit_overrides(
            overrides, segments,
            seen_segment_ids=seen_segment_ids, skipped=skipped_overrides,
        )
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
            overrides=day_overrides,
            placement_pins=placement_pins,
            operator_channel=operator_channel,
            optimize_fn=optimize_breaks,
        )
        return _rows_from_result(segments, result)

    pairs = _channel_days(programmes)
    frame: Optional[pd.DataFrame] = None
    if only_days is not None:
        requested = list(dict.fromkeys((str(c), str(d)) for c, d in only_days))
        frame = incremental_weekly_frame(
            pairs=pairs,
            requested=requested,
            existing_csv_path=Path(existing_csv) if existing_csv is not None else DEFAULT_OUTPUT_PATH,
            columns=COLUMNS,
            day_rows=_day_rows,
            has_segments=lambda channel, day: bool(_segments_for(channel, day)),
            progress_cb=progress_cb,
        )
        if frame is None:
            # Honest escape hatch: a precondition failed, so rebuild everything.
            # Reset the accumulators a partial incremental pass may have filled.
            skipped_overrides.clear()
            seen_segment_ids.clear()

    if frame is None:
        rows: list[dict[str, Any]] = []
        total = len(pairs)
        for done, (channel, day) in enumerate(pairs, start=1):
            rows.extend(_day_rows(channel, day))
            if progress_cb is not None:
                progress_cb(done, total)
        # A full run covered every channel-day, so an active anchored override
        # that matched none of them points at nothing in the current schedule.
        skipped_overrides.extend(unmatched_anchor_reports(overrides, seen_segment_ids))
        frame = pd.DataFrame(rows, columns=COLUMNS)

    if skipped_overrides:
        logger.warning(
            "Weekly schedule commit skipped %d stale-anchored override(s): %s",
            len(skipped_overrides),
            ", ".join(str(entry.get("override_id", "?")) for entry in skipped_overrides),
        )
    # Surfaced on the frame itself (no new API surface): callers that persist or
    # summarize this build can read frame.attrs["skipped_overrides"].
    frame.attrs["skipped_overrides"] = skipped_overrides
    return frame


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
