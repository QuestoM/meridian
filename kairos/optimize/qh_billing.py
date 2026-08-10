"""Round-quarter-hour settlement billing: the market's rating currency, computable.

Market CPP settlement values a spot's rating at the average TVR of the ROUND
quarter hour (:00/:15/:30/:45) in which the spot airs, content minutes included
(owner-stated rule, confirmed in the plan data: planned_tvr steps exactly at
quarter-hour boundaries, even mid-break; see docs/quarter-hour-billing.md and
analysis/quarter-hour/settlement/VERDICT.md). The engine's own basis bills a
break at its programme's rating times realised retention. These are different
currencies: the settlement window average is diluted only by the break's own
share of the window, so it usually sits ABOVE the in-break audience and the two
bases diverge most for long or straddling breaks.

This module makes the settlement currency computable for a placed schedule:

  * :func:`billed_points` assigns every break's seconds to their round
    quarter-hour windows, computes each window's average TVR from the segment
    lineup (content at the segment baseline, break seconds diluted by the
    MEASURED median in-break dip for that break length), and bills each break at
    its windows' averages. Shared windows (2+ breaks in one quarter hour, 60.3
    percent of real breaks) are priced jointly by construction: every co-window
    break dips the same average.
  * :func:`restate_on_billed_points` restates an
    :class:`~kairos.optimize._types.OptimizationResult`'s revenue onto that
    billed basis, leaving break counts, placements and retention untouched. It
    changes the REPORTED currency of the schedule the optimizer already chose;
    it does not steer placement (the measured boundary-placement lever is small
    for modal break lengths; see the Design section of the doc).

Activation follows the owner-gated pricing-layer pattern
(:mod:`kairos.optimize.pricing`): the ``qh_settlement`` flag under
``pricing_activation`` defaults OFF, and :func:`maybe_restate` is an exact
identity (the same result object) while it is off. A requested activation is
refused unless both configuration and every billed segment explicitly carry
Jewish-household, overnight+1 rating provenance from the same named source;
switching the revenue basis MOVES REAL REPORTED REVENUE and an unlabeled TVR
must never be silently promoted to the market currency.

Measured inputs: the in-break dip fractions are the Nov-2024 median dips by
length bin from analysis/quarter-hour/settlement/settlement_results.json
(5,747 measured breaks, key ``dip_frac_by_len_bin``). They are measured
medians, not fitted coefficients, and are the honest first expression of the
mechanic per the owner directive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Optional, Sequence

from kairos.optimize._types import BreakPlacement, OptimizationResult, ProgramSegment, SegmentPlan
from kairos.optimize.objective import break_revenue, weighted_objective

QH_WINDOW_SECONDS = 900.0
REQUIRED_AUDIENCE_BASIS = "jewish_households"
REQUIRED_RATING_VINTAGE = "overnight_plus_1"


class QHSettlementConfigurationError(ValueError):
    """The owner requested settlement without provable market-currency input."""

# Median in-break audience dip as a fraction of the surrounding content level,
# by break length, measured on 5,747 real Nov-2024 breaks
# (analysis/quarter-hour/settlement/settlement_results.json, dip_frac_by_len_bin).
# Each row is (max length in seconds exclusive, dip fraction). The 3-4m bin
# (0.0677) sitting above 4-6m (0.0620) is the measured reality, kept as is.
MEASURED_DIP_BY_LENGTH: tuple[tuple[float, float], ...] = (
    (60.0, 0.0377),
    (120.0, 0.0434),
    (180.0, 0.0548),
    (240.0, 0.0677),
    (360.0, 0.0620),
    (540.0, 0.0908),
    (float("inf"), 0.2040),
)

DipFn = Callable[[float], float]


def measured_dip_fraction(duration_seconds: float) -> float:
    """The measured median in-break dip for a break of this length.

    Looks the duration up in :data:`MEASURED_DIP_BY_LENGTH`. Non-positive
    durations return 0.0 (no audience to dip).
    """
    if duration_seconds <= 0:
        return 0.0
    for max_seconds, dip in MEASURED_DIP_BY_LENGTH:
        if duration_seconds < max_seconds:
            return dip
    return MEASURED_DIP_BY_LENGTH[-1][1]


def window_start_of(t_seconds: float) -> float:
    """The round quarter-hour window (start second from midnight) containing ``t``."""
    return (t_seconds // QH_WINDOW_SECONDS) * QH_WINDOW_SECONDS


def break_window_spans(start_seconds: float, duration_seconds: float) -> list[tuple[float, float]]:
    """Split a break into (window_start, seconds_in_window) spans.

    This is the per-spot rule at break granularity: every second of the break
    belongs to exactly one round quarter hour, and a straddling break spreads
    its seconds across two (or more) settlement windows.
    """
    if duration_seconds <= 0:
        return []
    spans: list[tuple[float, float]] = []
    end = start_seconds + duration_seconds
    window = window_start_of(start_seconds)
    while window < end:
        seconds = min(end, window + QH_WINDOW_SECONDS) - max(start_seconds, window)
        if seconds > 0:
            spans.append((window, seconds))
        window += QH_WINDOW_SECONDS
    return spans


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


@dataclass(frozen=True)
class WindowBill:
    """One round quarter-hour settlement window and its modeled average TVR.

    ``covered_seconds`` is how much of the 900-second window the segment lineup
    actually covers; the average is over covered seconds only (the model has no
    TVR for off-schedule time, and pretending it does would fabricate audience).
    """

    channel: str
    day: str
    window_start_seconds: float
    covered_seconds: float
    break_seconds: float
    average_tvr: float


@dataclass(frozen=True)
class BreakBill:
    """One break's rating under both currencies: the engine's and the market's."""

    segment_id: str
    position_in_segment: int
    start_seconds: float
    duration_seconds: float
    billed_tvr: float          # duration-weighted mean of the windows' average TVR
    engine_revenue: float      # what the engine credited this break (its own basis)
    billed_revenue: float      # cpp * billed_tvr * duration_units * premium


@dataclass(frozen=True)
class QHBillingReport:
    """The settlement-currency view of a placed schedule."""

    windows: tuple[WindowBill, ...]
    breaks: tuple[BreakBill, ...]
    engine_revenue: float
    billed_revenue: float


def _window_stats(
    window: float,
    segs: Sequence[ProgramSegment],
    placed: Sequence[tuple[BreakPlacement, ProgramSegment]],
    dip_fn: DipFn,
) -> tuple[float, float, float]:
    """(covered_seconds, break_seconds, average_tvr) of one settlement window.

    Content seconds carry their segment's ``baseline_tvr``; break seconds carry
    ``baseline_tvr * (1 - dip(length))`` with the measured median dip. A break's
    dip applies only where it overlaps its own segment's span (a break laid
    outside its segment has no content level to dilute). With 2+ breaks in the
    window every dip lands in the same average, which is the shared-window
    coupling the settlement lane measured on 60.3 percent of real breaks.
    """
    w0, w1 = window, window + QH_WINDOW_SECONDS
    covered = 0.0
    integral = 0.0
    for seg in segs:
        seconds = _overlap(seg.start_seconds, seg.start_seconds + seg.duration_seconds, w0, w1)
        covered += seconds
        integral += seg.baseline_tvr * seconds
    break_seconds = 0.0
    for placement, seg in placed:
        b0 = placement.start_seconds
        b1 = b0 + placement.duration_seconds
        break_seconds += _overlap(b0, b1, w0, w1)
        s0 = seg.start_seconds
        s1 = s0 + seg.duration_seconds
        dip_span = _overlap(max(b0, s0), min(b1, s1), w0, w1)
        if dip_span > 0:
            integral -= dip_fn(placement.duration_seconds) * seg.baseline_tvr * dip_span
    average = integral / covered if covered > 0 else 0.0
    return covered, break_seconds, average


def billed_points(
    segments: Iterable[ProgramSegment],
    placements: Iterable[BreakPlacement],
    *,
    dip_fn: Optional[DipFn] = None,
) -> QHBillingReport:
    """Bill every placed break at its round quarter-hour window averages.

    ``segments`` supply the content lineup (baseline TVR, span, cpp, premium);
    ``placements`` are the breaks the optimizer placed. Each break's rating is
    the duration-weighted mean of its settlement windows' average TVR, and its
    billed revenue reuses the engine's own price stack
    (:func:`~kairos.optimize.objective.break_revenue`) with only the rating
    basis swapped. A break in a window with zero lineup coverage (nothing to
    average) falls back to its own diluted level rather than fabricating a
    window figure.
    """
    dip = dip_fn or measured_dip_fraction
    seg_by_id = {s.segment_id: s for s in segments}
    by_day: dict[tuple[str, str], list[tuple[BreakPlacement, ProgramSegment]]] = {}
    for placement in placements:
        seg = seg_by_id.get(placement.segment_id)
        if seg is None:
            raise ValueError(f"placement references unknown segment {placement.segment_id!r}")
        by_day.setdefault((placement.channel, placement.day), []).append((placement, seg))

    window_bills: list[WindowBill] = []
    break_bills: list[BreakBill] = []
    engine_total = 0.0
    billed_total = 0.0
    for (channel, day), placed in sorted(by_day.items()):
        day_segs = [s for s in seg_by_id.values() if s.channel == channel and s.day == day]
        averages: dict[float, tuple[float, float, float]] = {}
        for placement, _seg in placed:
            for window, _seconds in break_window_spans(placement.start_seconds, placement.duration_seconds):
                if window not in averages:
                    averages[window] = _window_stats(window, day_segs, placed, dip)
        for window in sorted(averages):
            covered, break_seconds, average = averages[window]
            window_bills.append(WindowBill(
                channel=channel, day=day, window_start_seconds=window,
                covered_seconds=covered, break_seconds=break_seconds, average_tvr=average,
            ))
        for placement, seg in placed:
            weighted = 0.0
            for window, seconds in break_window_spans(placement.start_seconds, placement.duration_seconds):
                covered, _breaks, average = averages[window]
                if covered <= 0:
                    average = seg.baseline_tvr * (1.0 - dip(placement.duration_seconds))
                weighted += average * seconds
            billed_tvr = weighted / placement.duration_seconds if placement.duration_seconds > 0 else 0.0
            billed_rev = break_revenue(
                billed_tvr, placement.duration_seconds, seg.cpp,
                unit_seconds=seg.unit_seconds, premium=seg.premium,
            )
            break_bills.append(BreakBill(
                segment_id=placement.segment_id,
                position_in_segment=placement.position_in_segment,
                start_seconds=placement.start_seconds,
                duration_seconds=placement.duration_seconds,
                billed_tvr=billed_tvr,
                engine_revenue=placement.revenue,
                billed_revenue=billed_rev,
            ))
            engine_total += placement.revenue
            billed_total += billed_rev
    return QHBillingReport(
        windows=tuple(window_bills),
        breaks=tuple(break_bills),
        engine_revenue=engine_total,
        billed_revenue=billed_total,
    )


def restate_on_billed_points(
    result: OptimizationResult,
    segments: Iterable[ProgramSegment],
    *,
    dip_fn: Optional[DipFn] = None,
) -> OptimizationResult:
    """Restate a finished schedule's revenue onto the settlement currency.

    Break counts, positions, retention, guardrail verdicts and the greedy
    decision trace are untouched: this changes what the schedule is WORTH under
    round-quarter-hour billing, not what the optimizer chose. The objective is
    recomputed from the restated revenue with the result's own weight and scale
    (the scale stays the engine-basis normaliser, so the retention term's weight
    is unchanged and the two objective values are directly comparable).
    """
    segment_list = list(segments)
    report = billed_points(segment_list, result.placements, dip_fn=dip_fn)
    billed_by_key = {(b.segment_id, b.position_in_segment): b.billed_revenue for b in report.breaks}
    new_plans: list[SegmentPlan] = []
    new_flat: list[BreakPlacement] = []
    for plan in result.segments:
        new_placements = tuple(
            replace(p, revenue=billed_by_key[(p.segment_id, p.position_in_segment)])
            for p in plan.placements
        )
        new_flat.extend(new_placements)
        new_plans.append(replace(
            plan,
            revenue=sum(p.revenue for p in new_placements),
            placements=new_placements,
        ))
    total = sum(p.revenue for p in new_flat)
    objective = weighted_objective(
        total, result.aggregate_retention,
        revenue_weight=result.revenue_weight, revenue_scale=result.revenue_scale,
    )
    return replace(
        result,
        segments=tuple(new_plans),
        placements=tuple(new_flat),
        total_revenue=total,
        objective=objective,
        revenue_basis="round_quarter_hour_rating_points",
        rating_audience_basis=segment_list[0].rating_audience_basis,
        rating_vintage=segment_list[0].rating_vintage,
        rating_source=segment_list[0].rating_source,
    )


def qh_settlement_enabled(pricing: Any) -> bool:
    """Whether the requested settlement has a complete config provenance.

    Dataset provenance is validated separately by :func:`maybe_restate`, since
    it lives on the segments. This helper never reports an incomplete flag as
    effectively enabled.
    """
    if not bool(getattr(pricing, "enable_qh_settlement", False)):
        return False
    return (
        str(getattr(pricing, "qh_audience_basis", "")).strip().lower()
        == REQUIRED_AUDIENCE_BASIS
        and str(getattr(pricing, "qh_rating_vintage", "")).strip().lower()
        == REQUIRED_RATING_VINTAGE
        and bool(str(getattr(pricing, "qh_rating_source", "")).strip())
    )


def validate_qh_settlement_provenance(
    pricing: Any, segments: Sequence[ProgramSegment]
) -> None:
    problems: list[str] = []
    configured_basis = str(getattr(pricing, "qh_audience_basis", "")).strip()
    configured_vintage = str(getattr(pricing, "qh_rating_vintage", "")).strip()
    configured_source = str(getattr(pricing, "qh_rating_source", "")).strip()
    if configured_basis.lower() != REQUIRED_AUDIENCE_BASIS:
        problems.append(f"qh_audience_basis must be {REQUIRED_AUDIENCE_BASIS!r}")
    if configured_vintage.lower() != REQUIRED_RATING_VINTAGE:
        problems.append(f"qh_rating_vintage must be {REQUIRED_RATING_VINTAGE!r}")
    if not configured_source:
        problems.append("qh_rating_source must identify the ingested rating source")
    if not segments:
        problems.append("no billed segments carry rating provenance")
    for segment in segments:
        if str(segment.rating_audience_basis).strip().lower() != REQUIRED_AUDIENCE_BASIS:
            problems.append(f"segment {segment.segment_id!r} lacks Jewish-household rating provenance")
            break
        if str(segment.rating_vintage).strip().lower() != REQUIRED_RATING_VINTAGE:
            problems.append(f"segment {segment.segment_id!r} lacks overnight+1 rating provenance")
            break
        if not str(segment.rating_source).strip():
            problems.append(f"segment {segment.segment_id!r} lacks a rating source")
            break
        if str(segment.rating_source).strip() != configured_source:
            problems.append(
                f"segment {segment.segment_id!r} rating source does not match qh_rating_source"
            )
            break
    if problems:
        raise QHSettlementConfigurationError(
            "QH settlement is requested but its rating currency is not proven: "
            + "; ".join(problems)
        )


def maybe_restate(
    result: OptimizationResult,
    segments: Iterable[ProgramSegment],
    pricing: Any,
    *,
    dip_fn: Optional[DipFn] = None,
) -> OptimizationResult:
    """Apply the settlement restatement only when the owner-gated flag is on.

    With the flag off (the default) this returns ``result`` itself, the very
    same object, so every shipped path is byte-identical until the owner
    activates ``pricing_activation.qh_settlement``.
    """
    if not bool(getattr(pricing, "enable_qh_settlement", False)):
        return result
    segment_list = list(segments)
    validate_qh_settlement_provenance(pricing, segment_list)
    return restate_on_billed_points(result, segment_list, dip_fn=dip_fn)
