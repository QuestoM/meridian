"""Kairos optimization: revenue and retention economics, guardrails, optimizer."""

from kairos.optimize.guardrails import (
    AirtimeCaps,
    Break,
    DayFractionAdCap,
    Guardrails,
    Violation,
    WindowAdCap,
    airtime_caps_from_mapping,
    cap_state,
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
    "AirtimeCaps",
    "Break",
    "BreakPlacement",
    "DayFractionAdCap",
    "Decision",
    "Guardrails",
    "WindowAdCap",
    "agreement_violations",
    "airtime_caps_from_mapping",
    "cap_state",
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
