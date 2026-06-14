"""Kairos optimization: revenue and retention economics, guardrails, optimizer."""

from kairos.optimize.guardrails import (
    Break,
    Guardrails,
    Violation,
    evaluate,
    is_compliant,
)
from kairos.optimize.objective import (
    break_revenue,
    clamp,
    fixed_revenue,
    predicted_retention,
    retention_adjusted_revenue,
    weighted_objective,
)
from kairos.optimize.optimizer import (
    BreakPlacement,
    Decision,
    OptimizationResult,
    ProgramSegment,
    SegmentPlan,
    optimize_breaks,
)

__all__ = [
    "Break",
    "BreakPlacement",
    "Decision",
    "Guardrails",
    "OptimizationResult",
    "ProgramSegment",
    "SegmentPlan",
    "Violation",
    "break_revenue",
    "clamp",
    "evaluate",
    "fixed_revenue",
    "is_compliant",
    "optimize_breaks",
    "predicted_retention",
    "retention_adjusted_revenue",
    "weighted_objective",
]
