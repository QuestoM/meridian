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
from kairos.optimize.agreements import (
    AdvertiserAgreement,
    AgreementConstraint,
    AgreementViolation,
    agreement_violations,
    load_agreements,
)
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel

__all__ = [
    "AdvertiserAgreement",
    "AgreementConstraint",
    "AgreementViolation",
    "Break",
    "BreakPlacement",
    "Decision",
    "Guardrails",
    "agreement_violations",
    "load_agreements",
    "OptimizationResult",
    "OptimizerAssumptions",
    "PricingModel",
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
