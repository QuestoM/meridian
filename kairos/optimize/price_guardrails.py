"""Final-CPP guardrails for a composed slot price.

A composed price (base x layers x overrides) can land somewhere the operator did
not intend: below the channel floor, above a ceiling, under the cost basis, or at
an explicit zero (a promo). These guardrails inspect a finished
:class:`~kairos.optimize.pricing.PriceBreakdown` and return named, human-readable
warnings; they never silently clamp the price. The dashboard surfaces each
warning inline so a wrong price is visible before it ships (Law 9: nothing hidden,
nothing fabricated).

Config lives under ``guardrails`` in the operator's pricing overrides (and the
YAML rate card), so every bound is dashboard-tunable:

    guardrails:
      floor_cpp: 0          # final CPP must be >= this (0 disables)
      ceiling_cpp: 0        # final CPP must be <= this (0 disables)
      cost_cpp: 0           # warn when final CPP < cost basis (0 disables)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kairos.optimize.pricing import PriceBreakdown


@dataclass(frozen=True)
class GuardrailWarning:
    """One guardrail breach: a code, the bound it crossed, and a plain message."""

    code: str
    bound: float
    message: str


@dataclass(frozen=True)
class Guardrails:
    """Operator-configured final-CPP bounds. A zero bound disables that check."""

    floor_cpp: float = 0.0
    ceiling_cpp: float = 0.0
    cost_cpp: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> "Guardrails":
        """Read the ``guardrails`` block from a pricing-overrides mapping (or model)."""
        raw: dict[str, Any] = {}
        if isinstance(config, dict):
            raw = config.get("guardrails") or {}
        elif config is not None:
            raw = getattr(config, "guardrails", None) or {}
        return cls(
            floor_cpp=float(raw.get("floor_cpp", 0.0) or 0.0),
            ceiling_cpp=float(raw.get("ceiling_cpp", 0.0) or 0.0),
            cost_cpp=float(raw.get("cost_cpp", 0.0) or 0.0),
        )

    def check(self, breakdown: PriceBreakdown) -> list[GuardrailWarning]:
        """Return the warnings a composed price triggers, in priority order.

        ``explicit_zero`` fires whenever the final CPP is exactly 0 (a promo or a
        zeroing override), so a free spot is always a deliberate, surfaced choice.
        The floor/ceiling/below-cost checks fire only when their bound is enabled.
        """
        final = breakdown.final_cpp
        warnings: list[GuardrailWarning] = []
        if final == 0.0:
            warnings.append(GuardrailWarning(
                "explicit_zero", 0.0,
                "final CPP is zero (a promo or a zeroing override); confirm this is intended",
            ))
        if self.floor_cpp > 0 and final < self.floor_cpp:
            warnings.append(GuardrailWarning(
                "below_floor", self.floor_cpp,
                f"final CPP {final:.2f} is below the floor {self.floor_cpp:.2f}",
            ))
        if self.ceiling_cpp > 0 and final > self.ceiling_cpp:
            warnings.append(GuardrailWarning(
                "above_ceiling", self.ceiling_cpp,
                f"final CPP {final:.2f} is above the ceiling {self.ceiling_cpp:.2f}",
            ))
        if self.cost_cpp > 0 and 0.0 < final < self.cost_cpp:
            warnings.append(GuardrailWarning(
                "below_cost", self.cost_cpp,
                f"final CPP {final:.2f} is below the cost basis {self.cost_cpp:.2f}",
            ))
        return warnings


__all__ = ["GuardrailWarning", "Guardrails"]
