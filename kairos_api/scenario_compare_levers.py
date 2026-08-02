"""The request and the per-leg lever model for the week's A/B comparison.

Its own module so the router, the money helper and the weekly runner can all
import them without a cycle: the two routes that serve the comparison, the plain
one and the streaming one, live in different modules and take the same body.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# The five levers a leg can set. The revenue weight is the one the panel has
# always offered and the one that, measured, moves the plan least; the other
# four are why the comparison can now answer the planner's question.
LEVER_FIELDS = ("revenue_weight", "retention_floor", "max_breaks_per_hour", "risk_lambda", "objective_mode")


class ScenarioLevers(BaseModel):
    """One leg's levers. Every field omitted falls back to the shared level.

    The fallback chain is deliberate and is what keeps an old caller working:
    a per-leg value wins, then the request-level guardrail, then the operator's
    saved setting. So a request that names only ``weight_a`` and ``weight_b``
    still compares two weights under the saved plan baseline, exactly as before.
    """

    revenue_weight: Optional[int] = Field(default=None, ge=0, le=100)
    retention_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_breaks_per_hour: Optional[int] = Field(default=None, ge=1, le=20)
    risk_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    objective_mode: Optional[str] = Field(default=None, pattern="^(blend|revenue_net)$")


class ScenarioCompareRequest(BaseModel):
    """A what-if A/B: two full lever sets under shared (optional) guardrails.

    ``weight_a``/``weight_b`` are the 0..100 revenue-vs-retention levers and stay
    required, so the shape a caller already sends is unchanged. ``a`` and ``b``
    carry the rest of each leg's levers when a planner wants to compare on
    something the weight cannot move, which measurement says is most of what
    matters. The three request-level guardrails are optional; when omitted they
    fall back to the operator's saved settings so the comparison reflects the
    real plan baseline, not an arbitrary default. Both legs run the genuine
    optimizer; nothing here is synthesized.

    ``scope`` is the window each leg is optimized over. It defaults to ``week``,
    the plan's own week, because that is the window the rest of the destination
    reports and the window JS-2's comparison is defined on. ``day`` runs the
    single representative broadcast day, which is what this route did before and
    what a caller who wants one fast answer can still ask for.
    """

    weight_a: int = Field(ge=0, le=100)
    weight_b: int = Field(ge=0, le=100)
    retention_floor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_breaks_per_hour: Optional[int] = Field(default=None, ge=1, le=20)
    risk_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    a: Optional[ScenarioLevers] = Field(default=None)
    b: Optional[ScenarioLevers] = Field(default=None)
    scope: str = Field(default="week", pattern="^(week|day)$")
