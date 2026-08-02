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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# Re-exported so callers keep one pricing import surface; the store readers
# live in kairos.optimize.event_pricing (monkeypatch DEFAULT_EVENTS_PATH there).
from kairos.optimize.event_pricing import (  # noqa: F401
    DEFAULT_EVENTS_PATH,
    EVENT_OPEN_HORIZON_DAYS,
    load_event_day_multipliers,
    load_price_events,
)
from kairos.optimize.positions import (
    canonical_token,
    parse_preferred,
    premium_token,
    resolve_preferred,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = ROOT / "config" / "optimization_weights.yaml"

_OTHER = "Other"


def pricing_from_settings(
    settings: Any = None, pricing: "PricingModel | None" = None
) -> "PricingModel":
    """Build the live PricingModel, honoring the operator's saved dashboard edits.

    The single seam the revenue path uses so a dashboard pricing edit reaches the
    optimizer, the dashboard forecast and the spot export without per-call-site
    plumbing. An explicit ``pricing`` wins (test or override). Otherwise the
    operator's ``pricing_overrides`` are read from ``settings`` (a KairosSettings
    model or a plain mapping) and merged onto the YAML rate card. No settings or
    empty overrides is an exact identity to the YAML, so the rate card is unchanged
    until the operator edits it in the dashboard.
    """
    if pricing is not None:
        return pricing
    overrides: dict[str, Any] = {}
    if settings is not None:
        raw = getattr(settings, "pricing_overrides", None)
        if raw is None and hasattr(settings, "get"):
            raw = settings.get("pricing_overrides")
        if isinstance(raw, dict):
            overrides = raw
    model = PricingModel.from_config(overrides)
    # The event-date map is read from the events store ONLY when the operator has
    # activated the events layer (pricing_activation.events). With the flag off
    # (the shipped default) the store is never read, so every revenue number is
    # exactly what it was before the layer existed.
    if model.enable_events:
        model = replace(model, event_day_multipliers=load_event_day_multipliers())
    return model


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` onto a copy of ``base`` (overrides win).

    Nested dicts merge key by key so an operator can override a single premium (for
    example position 2) without resupplying the whole table. Non-dict values replace.
    """
    merged = dict(base)
    for key, value in (overrides or {}).items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


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
    # Per-show premium keyed on the literal programme title (decision 2026-06-20:
    # title string for v1). Distinct from the program_type class, stacks on top of it.
    show_premiums: dict[str, float] = field(default_factory=dict)
    # Per-layer activation. A layer is multiplied into the live price only when its flag
    # is on. Position and ad-type default OFF because their configured multipliers are not
    # 1.0, so turning them on is a deliberate, dashboard-driven revenue change that the
    # operator sees in the tester, never a silent restatement (docs/pricing-hierarchy-*).
    enable_position: bool = False
    enable_ad_type: bool = False
    enable_show: bool = False
    # Round-quarter-hour settlement restatement (kairos/optimize/qh_billing.py):
    # when on, a finished schedule's revenue is restated onto the market's
    # round-window billed-points basis. OFF by default because it moves real
    # reported revenue (docs/quarter-hour-billing.md, Design section).
    enable_qh_settlement: bool = False
    # Event-date price multipliers (day-iso -> multiplier), built by
    # pricing_from_settings from the ACTIVE calendar events the operator stored
    # with a non-1.0 price_multiplier. Overlapping events compose multiplicatively.
    # These are operator assertions (like the day-of-week premiums), not
    # measurements; retention coefficients are untouched by this layer. Gated on
    # enable_events, which ships OFF because activating it moves real forecast
    # revenue.
    event_day_multipliers: dict[str, float] = field(default_factory=dict)
    enable_events: bool = False
    # Which of the six positions count as PREFERRED. The trade sets this per
    # client and per agreement, so it is configuration and never a constant, and
    # it is tri-state: None means unset, and a preferred-position percentage may
    # not be computed at all until somebody configures one (an unlabelled or
    # guessed percentage is worse than no percentage, docs/media-domain-from-the-
    # trade.md). Both ship unset, so nothing computes one today.
    preferred_positions_default: frozenset[str] | None = None
    preferred_positions_by_advertiser: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_price_per_second_per_tvr_point < 0:
            raise ValueError("base_price_per_second_per_tvr_point must be non-negative")
        for table_name in ("program_type_premiums", "ad_type_premiums", "day_of_week_premiums", "show_premiums", "event_day_multipliers"):
            for key, value in getattr(self, table_name).items():
                if value < 0:
                    raise ValueError(f"{table_name}[{key!r}] must be non-negative")

    @classmethod
    def from_weights(cls, weights: dict[str, Any]) -> "PricingModel":
        premiums = (weights or {}).get("premiums") or {}
        day_raw = premiums.get("day_of_week") or {}
        activation = (weights or {}).get("pricing_activation") or {}
        preferred = (weights or {}).get("preferred_positions") or {}
        by_advertiser = {}
        for key, raw in (preferred.get("per_advertiser") or {}).items():
            parsed = parse_preferred(raw)
            if parsed is not None:
                by_advertiser[str(key)] = parsed
        return cls(
            base_price_per_second_per_tvr_point=float(
                (weights or {}).get("base_price_per_second_per_tvr_point", 0.0)
            ),
            program_type_premiums={str(k): float(v) for k, v in (premiums.get("program_type") or {}).items()},
            ad_type_premiums={str(k): float(v) for k, v in (premiums.get("ad_type") or {}).items()},
            # Canonical string tokens, so "2", 2, "last" and "L" are one key and
            # a dashboard edit (JSON object keys are always strings) can never
            # silently no-op against a differently typed rate-card key.
            position_premiums={
                canonical_token(k): float(v)
                for k, v in (premiums.get("position_in_break") or {}).items()
                if canonical_token(k) is not None
            },
            day_of_week_premiums={int(k): float(v) for k, v in day_raw.items()},
            show_premiums={str(k): float(v) for k, v in (premiums.get("show") or {}).items()},
            enable_position=bool(activation.get("position", False)),
            enable_ad_type=bool(activation.get("ad_type", False)),
            enable_show=bool(activation.get("show", False)),
            enable_qh_settlement=bool(activation.get("qh_settlement", False)),
            enable_events=bool(activation.get("events", False)),
            preferred_positions_default=parse_preferred(preferred.get("channel_default")),
            preferred_positions_by_advertiser=by_advertiser,
        )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PricingModel":
        path = Path(path) if path else DEFAULT_WEIGHTS_PATH
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_weights(yaml.safe_load(handle) or {})

    @classmethod
    def from_config(
        cls, overrides: dict[str, Any] | None = None, path: str | Path | None = None
    ) -> "PricingModel":
        """Load the YAML rate card, then deep-merge the operator's dashboard overrides.

        ``overrides`` carries exactly what the operator edits in the dashboard (a base
        price, per-table premium edits, per-show premiums, and the per-layer activation
        flags), in the same nested shape as the YAML. An empty or absent override is an
        exact identity to :meth:`from_yaml`, so the rate card is unchanged until the
        operator touches it. This is the single constructor the live revenue path uses so
        a dashboard edit reaches the optimizer, the dashboard and the spot export alike.
        """
        path = Path(path) if path else DEFAULT_WEIGHTS_PATH
        with open(path, "r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        return cls.from_weights(_deep_merge(base, overrides or {}))

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

    def show_premium(self, title: str | None) -> float:
        """Premium for a specific programme title (for example Big Brother).

        Unknown or missing titles return 1.0 (no effect), so a show without a
        configured premium never zeroes revenue.
        """
        if not title:
            return 1.0
        return self.show_premiums.get(title, 1.0)

    def day_premium(self, weekday_iso: int) -> float:
        """Premium for an ISO weekday (1 = Monday ... 7 = Sunday)."""
        return self.day_of_week_premiums.get(weekday_iso, 1.0)

    def event_premium(self, day: str | None) -> float:
        """The operator-asserted event multiplier for a broadcast date.

        ``day`` is a YYYY-MM-DD string. A date no active event covers (or a
        missing/unparseable date) returns the neutral 1.0, so a schedule outside
        every event is never touched. Overlapping events already composed
        multiplicatively when the map was built.
        """
        if not day:
            return 1.0
        return self.event_day_multipliers.get(str(day)[:10], 1.0)

    def position_key(self, position: int, break_size: int | None) -> str:
        """Which rate-card key prices this spot: an ordinal, ``L``, or the middle.

        The trade's six positions are 1 to 5 and L, where L is LAST and is its
        own position rather than the fifth ordinal (docs/media-domain-from-the-
        trade.md). A spot can hold two of them at once, so pricing has to choose:
        an ordinal the operator has priced explicitly wins, otherwise the tail of
        the break is L, otherwise the middle default. Ordinals 4 and 5 ship
        unpriced, which is a no-op, so every spot prices exactly as it did before
        they became addressable.
        """
        return premium_token(position, break_size, self.position_premiums.keys())

    def position_premium(self, position: int, break_size: int) -> float:
        """Premium for a 1-based ad position within a break of ``break_size`` ads.

        The multiplier behind :meth:`position_key`. An unpriced key is 1.0, so an
        ordinal nobody has set never changes a price.
        """
        return float(self.position_premiums.get(self.position_key(position, break_size), 1.0))

    def preferred_positions(
        self, advertiser: str | None = None, agreement: Any = None
    ) -> tuple[frozenset[str] | None, str]:
        """The preferred-position set that applies, and the scope it came from.

        Most specific wins: the agreement, then the client, then the channel
        default. ``(None, "unset")`` when nobody has configured one, which is the
        shipped state and which forbids computing a percentage at all.
        """
        return resolve_preferred(
            agreement=agreement,
            per_advertiser=self.preferred_positions_by_advertiser,
            advertiser=advertiser,
            channel_default=self.preferred_positions_default,
        )

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
        show: str | None = None,
        position: int | None = None,
        break_size: int | None = None,
        ad_type: str | None = None,
        day: str | None = None,
        base_cpp: float | None = None,
        enable_show: bool | None = None,
        enable_position: bool | None = None,
        enable_ad_type: bool | None = None,
        enable_events: bool | None = None,
    ) -> PriceBreakdown:
        """Compose a slot price as base CPP times named, traceable premium layers.

        Canonical layer order: program, day, show, position, ad-type. By default
        (every activation flag off, the engine's shipped state) only the program-class
        and day layers are active, so the returned ``total_premium`` equals
        :meth:`segment_premium` exactly: the same number the optimizer and dashboard
        already produce, now with a per-layer breakdown that names every premium and
        its source. ``base_cpp`` defaults to the configured channel base; pass a value
        for a per-advertiser negotiated base.

        The show, position, ad-type and event layers are wired but default OFF (their
        configured multipliers are not 1.0), so switching them on is a deliberate,
        dashboard-driven revenue change the operator sees, never a silent one. Each
        ``enable_*`` argument defaults to the model-level flag (set from the operator's
        saved pricing config); pass an explicit bool to force a single call. ``day``
        (YYYY-MM-DD) opts the slot into the event-date layer: when the events layer is
        active and an active stored event covers the date with a non-1.0 multiplier,
        an ``event`` layer is appended (an operator assertion, not a measurement).
        """
        base = self.base_price if base_cpp is None else float(base_cpp)
        use_show = self.enable_show if enable_show is None else enable_show
        use_position = self.enable_position if enable_position is None else enable_position
        use_ad_type = self.enable_ad_type if enable_ad_type is None else enable_ad_type
        use_events = self.enable_events if enable_events is None else enable_events
        layers: list[PriceLayer] = [
            PriceLayer("program", self.program_premium(pricing_class)),
            PriceLayer("day", self.day_premium(weekday_iso)),
        ]
        if use_show and show:
            layers.append(PriceLayer("show", self.show_premium(show)))
        if use_position and position is not None and break_size is not None:
            layers.append(PriceLayer("position", self.position_premium(position, break_size)))
        if use_ad_type and ad_type is not None:
            layers.append(PriceLayer("ad_type", self.ad_type_premium(ad_type)))
        if use_events and day:
            multiplier = self.event_premium(day)
            if multiplier != 1.0:
                layers.append(PriceLayer("event", multiplier, source="operator_event"))
        return PriceBreakdown(base_cpp=base, layers=tuple(layers))
