"""Public and internal dataclass types for the Kairos optimizer.

All types that the optimizer, refiner, and external callers share are
defined here so optimizer.py and refiner.py do not need to import each
other for type definitions alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kairos.optimize.guardrails import Violation
from kairos.optimize.objective import STANDARD_UNIT_SECONDS

DEFAULT_BREAK_LENGTH_SECONDS = 120.0  # a two-minute break, a common unit


@dataclass(frozen=True)
class ProgramSegment:
    """One programme the optimizer may load with breaks.

    ``start_seconds`` is measured from midnight, so a break's clock hour is
    ``start_seconds // 3600``. ``impact_coefficient`` is the retention change per
    break (normally negative) and defaults to zero so a caller without a fitted
    impact model still gets a revenue-only allocation.

    ``impact_ci_low`` / ``impact_ci_high`` are the credible interval on that
    per-break coefficient when the impact model supplies one (both ``None`` means
    only the point is known, so the optimizer treats the cost as certain).
    ``impact_n`` is how many real breaks the estimate rests on and
    ``impact_confidence`` is its high / medium / low label, both carried purely so
    the plan can report how trustworthy each segment's retention cost is.
    """

    segment_id: str
    channel: str
    day: str
    start_seconds: float
    duration_seconds: float
    program_type: str
    baseline_tvr: float                 # rating points with no breaks
    cpp: float                          # cost per rating point (daypart-adjusted)
    impact_coefficient: float = 0.0     # retention delta per break, usually <= 0
    retention_baseline: float = 1.0
    premium: float = 1.0                # position / daypart premium
    is_gold: bool = False
    max_breaks: int = 4
    break_length_seconds: float = DEFAULT_BREAK_LENGTH_SECONDS
    unit_seconds: float = STANDARD_UNIT_SECONDS   # the duration ``cpp`` is quoted per
    impact_ci_low: Optional[float] = None         # credible interval on the coefficient
    impact_ci_high: Optional[float] = None
    impact_n: int = 0                             # real breaks behind the estimate
    impact_confidence: str = "low"                # high / medium / low label
    program_title: str = ""                       # programme title, for cross-date matching
    first_break_multiplier: float = 1.0           # extra retention cost on the show's first break

    @property
    def hour(self) -> int:
        return int(self.start_seconds // 3600.0)

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: duration_seconds must be positive")
        if self.baseline_tvr < 0:
            raise ValueError(f"segment {self.segment_id}: baseline_tvr must be non-negative")
        if self.cpp < 0:
            raise ValueError(f"segment {self.segment_id}: cpp must be non-negative")
        if self.premium < 0:
            raise ValueError(f"segment {self.segment_id}: premium must be non-negative")
        if self.max_breaks < 0:
            raise ValueError(f"segment {self.segment_id}: max_breaks must be non-negative")
        if self.break_length_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: break_length_seconds must be positive")
        if self.unit_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: unit_seconds must be positive")
        if not 0.0 <= self.retention_baseline <= 1.0:
            raise ValueError(f"segment {self.segment_id}: retention_baseline must be in [0, 1]")
        if self.first_break_multiplier < 1.0:
            raise ValueError(f"segment {self.segment_id}: first_break_multiplier must be >= 1.0")


@dataclass(frozen=True)
class PlacementPin:
    """One explicit break the operator pinned onto a segment.

    ``offset_seconds`` is measured from the segment's start, so the break's
    absolute clock position is ``segment.start_seconds + offset_seconds``.
    ``duration_seconds`` is this break's own length (breaks in one segment may
    differ in length). ``is_gold`` marks just this break gold, on top of any
    segment-level or override gold flag. Pinned breaks are honored as a HARD
    constraint by every optimizer tier: the segment's count is fixed at the
    number of pins and the breaks are emitted at exactly these positions.
    """

    offset_seconds: float
    duration_seconds: float
    is_gold: bool = False


@dataclass(frozen=True)
class BreakPlacement:
    """A single break the optimizer placed, with the value credited to it."""

    segment_id: str
    channel: str
    day: str
    hour: int
    start_seconds: float
    duration_seconds: float
    program_type: str
    position_in_segment: int       # 1-based order within the segment
    retention: float               # realised retention of the segment it sits in
    revenue: float                 # marginal revenue credited at insertion
    is_gold: bool


@dataclass(frozen=True)
class SegmentPlan:
    """The optimizer's decision for one segment.

    The ``retention_cost_*`` fields make the retention side of the decision
    auditable: ``retention_cost_point`` is the impact model's point estimate of the
    per-break retention drop, ``retention_cost_used`` is the (possibly more
    conservative) value the optimizer actually decided with after applying
    ``risk_lambda``, ``retention_cost_ci_low`` / ``retention_cost_ci_high`` is the
    credible interval (``None`` when only a point is known), ``retention_cost_n`` is
    the number of real breaks behind it, and ``retention_confidence`` is its
    high / medium / low label. They let the dashboard show not just how many breaks
    a segment carries but how trustworthy the cost driving that count was.
    """

    segment_id: str
    num_breaks: int
    retention: float
    revenue: float
    placements: tuple[BreakPlacement, ...]
    retention_cost_point: float = 0.0
    retention_cost_used: float = 0.0
    retention_cost_ci_low: Optional[float] = None
    retention_cost_ci_high: Optional[float] = None
    retention_cost_n: int = 0
    retention_confidence: str = "low"


@dataclass(frozen=True)
class Decision:
    """One greedy step, kept so the schedule can explain itself."""

    segment_id: str
    break_index: int                   # the break number added (1-based)
    marginal_objective_gain: float
    marginal_revenue: float
    retention_after: float


@dataclass(frozen=True)
class RejectedOverride:
    """One operator override the optimizer could not honor, with why.

    Honesty surface: an override is rejected (and kept OUT of the plan) when
    obeying it would breach a hard guardrail, for example a force that exceeds the
    segment's ``max_breaks`` or a pin that breaks the spacing guardrail. The
    operator sees exactly which override was dropped and the reason, so nothing is
    silently bent or silently ignored.
    """

    segment_id: str
    kind: str                          # pin / force (forbid and gold cannot be infeasible)
    requested: int                     # the break count the override asked for
    reason: str


@dataclass(frozen=True)
class OptimizationResult:
    segments: tuple[SegmentPlan, ...]
    placements: tuple[BreakPlacement, ...]     # every break, flat
    total_revenue: float
    aggregate_retention: float
    objective: float
    violations: tuple[Violation, ...]          # empty when compliant
    revenue_weight: float
    revenue_scale: float
    decisions: tuple[Decision, ...]
    rejected_overrides: tuple[RejectedOverride, ...] = ()
    risk_lambda: float = 0.0                    # uncertainty preference applied to costs

    @property
    def total_breaks(self) -> int:
        return len(self.placements)

    @property
    def is_compliant(self) -> bool:
        return not self.violations
