"""Config-driven pricing and the optimizer's declared assumptions.

Two things live here, both surfaced as editable configuration so the dashboard
can expose every adjustable number:

  * ``PricingModel`` is a typed view over ``config/optimization_weights.yaml``:
    the CPP base price and the premium tables (program type, ad type, position in
    break, day of week). It only looks values up, with safe fallbacks, so the
    revenue math stays in :mod:`kairos.optimize.objective`.

  * ``OptimizerAssumptions`` holds the retention-side numbers the Meridian impact
    model has not estimated yet (the per-break retention drop, the baseline, the
    default break length and count). These are explicit, named defaults the owner
    can override. They are assumptions, not fabricated measurements, and are
    reported as such so nothing pretends to be a fitted result.

Israeli TV ad pricing is Cost Per (rating) Point: revenue scales with the rating
the break delivers, the seconds it runs, and a stack of multiplicative premiums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = ROOT / "config" / "optimization_weights.yaml"

_OTHER = "Other"
_MIDDLE_KEY = "default_middle"
_LAST_KEY = "last"


@dataclass(frozen=True)
class PriceLayer:
    """One named multiplicative layer in a slot price, with its provenance.

    ``source`` records where the multiplier came from (``rate_card`` for a
    configured premium, ``override`` for a per-advertiser or per-campaign rule),
    so a pricing surface can label every line honestly (Law 9).
    """

    name: str
    multiplier: float
    source: str = "rate_card"


@dataclass(frozen=True)
class PriceBreakdown:
    """A composed slot price: base CPP times a stack of named premium layers.

    ``final_cpp`` is ``base_cpp`` times the product of every layer multiplier.
    ``layers`` keeps each named layer and its source, so every number traces back
    to base x named layers (no opaque aggregate). This is the single composition
    primitive the optimizer, dashboard and spot export are meant to converge on
    (see docs/pricing-hierarchy-design.md). Until the position, ad-type and
    override layers are switched on, a breakdown carries only the program and day
    layers, so ``total_premium`` equals the legacy :meth:`PricingModel.segment_premium`.
    """

    base_cpp: float
    layers: tuple[PriceLayer, ...] = ()

    @property
    def total_premium(self) -> float:
        premium = 1.0
        for layer in self.layers:
            premium *= layer.multiplier
        return premium

    @property
    def final_cpp(self) -> float:
        return self.base_cpp * self.total_premium


@dataclass(frozen=True)
class OptimizerAssumptions:
    """Retention-side values not yet estimated by the Meridian impact model.

    Each is a declared default, overridable by the owner. ``revenue_weight`` is
    the revenue-versus-retention balance the optimizer maximises (1.0 = revenue
    only, 0.0 = retention only). ``risk_lambda`` is the uncertainty preference the
    optimizer applies to a break's retention cost when that cost carries a credible
    interval: 0.0 values the break at the point estimate (the default, no change in
    behavior), 1.0 values it at the worst plausible cost in the interval, and values
    in between apply a partial variance penalty. It only bites where the impact model
    actually supplies an interval; a bare point estimate is unaffected.
    """

    retention_baseline: float = 1.0
    retention_impact_per_break: float = -0.03   # per-break drop, until Meridian is trained
    default_break_length_seconds: float = 120.0
    default_max_breaks: int = 4
    revenue_weight: float = 0.5
    risk_lambda: float = 0.0   # how conservatively to value an uncertain retention cost
    # Extra retention cost charged to the FIRST break of a programme (the show's
    # first interruption), as a multiplier on the per-break coefficient. Measured
    # from the real airings; 1.0 is OFF (the show's first break costs the same as
    # any later break) and is the safe default until the measurement earns a value.
    first_break_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.retention_baseline <= 1.0:
            raise ValueError("retention_baseline must be in [0, 1]")
        if self.retention_impact_per_break > 0:
            raise ValueError("retention_impact_per_break should be <= 0 (breaks do not raise retention)")
        if self.default_break_length_seconds <= 0:
            raise ValueError("default_break_length_seconds must be positive")
        if self.default_max_breaks < 0:
            raise ValueError("default_max_breaks must be non-negative")
        if not 0.0 <= self.revenue_weight <= 1.0:
            raise ValueError("revenue_weight must be in [0, 1]")
        if not 0.0 <= self.risk_lambda <= 1.0:
            raise ValueError("risk_lambda must be in [0, 1]")
        if self.first_break_multiplier < 1.0:
            raise ValueError("first_break_multiplier must be >= 1.0 (an adjustment only adds cost)")


@dataclass(frozen=True)
class PricingModel:
    """A typed, fallback-safe view over the optimization-weights config."""

    base_price_per_second_per_tvr_point: float
    program_type_premiums: dict[str, float] = field(default_factory=dict)
    ad_type_premiums: dict[str, float] = field(default_factory=dict)
    position_premiums: dict[Any, float] = field(default_factory=dict)
    day_of_week_premiums: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_price_per_second_per_tvr_point < 0:
            raise ValueError("base_price_per_second_per_tvr_point must be non-negative")
        for table_name in ("program_type_premiums", "ad_type_premiums", "day_of_week_premiums"):
            for key, value in getattr(self, table_name).items():
                if value < 0:
                    raise ValueError(f"{table_name}[{key!r}] must be non-negative")

    @classmethod
    def from_weights(cls, weights: dict[str, Any]) -> "PricingModel":
        premiums = (weights or {}).get("premiums") or {}
        day_raw = premiums.get("day_of_week") or {}
        return cls(
            base_price_per_second_per_tvr_point=float(
                (weights or {}).get("base_price_per_second_per_tvr_point", 0.0)
            ),
            program_type_premiums={str(k): float(v) for k, v in (premiums.get("program_type") or {}).items()},
            ad_type_premiums={str(k): float(v) for k, v in (premiums.get("ad_type") or {}).items()},
            position_premiums=dict(premiums.get("position_in_break") or {}),
            day_of_week_premiums={int(k): float(v) for k, v in day_raw.items()},
        )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PricingModel":
        path = Path(path) if path else DEFAULT_WEIGHTS_PATH
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_weights(yaml.safe_load(handle) or {})

    @property
    def base_price(self) -> float:
        return self.base_price_per_second_per_tvr_point

    def program_premium(self, pricing_class: str) -> float:
        """Premium for a pricing class (News / PrimeShow1 / PrimeShow2 / Other).

        Falls back to the configured ``Other`` value, then to 1.0, so an unknown
        class never silently zeroes revenue.
        """
        table = self.program_type_premiums
        if pricing_class in table:
            return table[pricing_class]
        return table.get(_OTHER, 1.0)

    def ad_type_premium(self, ad_type: str) -> float:
        return self.ad_type_premiums.get(ad_type, 1.0)

    def day_premium(self, weekday_iso: int) -> float:
        """Premium for an ISO weekday (1 = Monday ... 7 = Sunday)."""
        return self.day_of_week_premiums.get(weekday_iso, 1.0)

    def position_premium(self, position: int, break_size: int) -> float:
        """Premium for a 1-based ad position within a break of ``break_size`` ads.

        Positions 1, 2 and 3 use their explicit premiums. A last position beyond
        the third uses the ``last`` premium. Anything else uses ``default_middle``.
        """
        if position < 1:
            raise ValueError("position must be >= 1")
        table = self.position_premiums
        if position in table:
            return float(table[position])
        if position == break_size and position > 3:
            return float(table.get(_LAST_KEY, 1.0))
        return float(table.get(_MIDDLE_KEY, 1.0))

    def segment_premium(self, *, pricing_class: str, weekday_iso: int) -> float:
        """The premium that applies to a whole break segment: program class x day.

        Position and ad-type premiums vary per ad inside the break, so they are
        applied separately when an individual spot is priced.
        """
        return self.program_premium(pricing_class) * self.day_premium(weekday_iso)

    def price_slot(
        self,
        *,
        pricing_class: str,
        weekday_iso: int,
        position: int | None = None,
        break_size: int | None = None,
        ad_type: str | None = None,
        base_cpp: float | None = None,
        enable_position: bool = False,
        enable_ad_type: bool = False,
    ) -> PriceBreakdown:
        """Compose a slot price as base CPP times named, traceable premium layers.

        By default only the program-class and day layers are active, so the
        returned ``total_premium`` equals :meth:`segment_premium` exactly: the same
        number the optimizer and dashboard already produce, now with a per-layer
        breakdown that names every premium and its source. ``base_cpp`` defaults to
        the configured channel base; pass a value for a per-advertiser negotiated
        base.

        The position and ad-type layers are wired but OFF by default. Their
        configured multipliers are not 1.0 (a first-position premium, a zero promo
        rate), so switching them on is a real revenue change that the owner approves
        deliberately (``enable_position`` / ``enable_ad_type``), never a silent one.
        This keeps the composition primitive identity-preserving until the staged
        pricing-hierarchy rollout activates each layer.
        """
        base = self.base_price if base_cpp is None else float(base_cpp)
        layers: list[PriceLayer] = [
            PriceLayer("program", self.program_premium(pricing_class)),
            PriceLayer("day", self.day_premium(weekday_iso)),
        ]
        if enable_position and position is not None and break_size is not None:
            layers.append(PriceLayer("position", self.position_premium(position, break_size)))
        if enable_ad_type and ad_type is not None:
            layers.append(PriceLayer("ad_type", self.ad_type_premium(ad_type)))
        return PriceBreakdown(base_cpp=base, layers=tuple(layers))
