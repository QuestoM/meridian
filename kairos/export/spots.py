"""Per-spot pricing for the daily Wally ad log, honoring advertiser rules.

This is the one pricing path where advertiser rules legitimately apply. The
weekly break-count optimizer decides break COUNTS per programme segment and
never attributes a break to an advertiser or a position, so it cannot consume
per-advertiser rules. The daily Wally file, by contrast, carries an advertiser,
a campaign and a position per individual spot (see
:data:`kairos.data.loaders.DAILY_COLUMN_MAP`), so it is here that
:class:`~kairos.optimize.advertiser_rules.AdvertiserRuleEngine` is wired in:

  * each priced spot's revenue is multiplied by ``effective_premium`` for its
    advertiser, position, genre and daypart, and
  * a spot that fails ``is_allowed`` is dropped from the priced output and
    recorded in a separate dropped list with the reason, so nothing is silently
    lost or silently kept.

ONE HONEST SENTENCE (for the module and the dashboard status text): advertiser
rules now take effect on the per-spot daily pricing/export path
(:func:`price_daily_spots`); the weekly break-count optimizer does not yet honor
them, because it does not attribute breaks to advertisers.

Spot revenue uses the same CPP math as the rest of the engine
(:func:`kairos.optimize.objective.break_revenue`): a spot's planned break rating
times its duration in 30-second units times the channel base price, then the
advertiser premium. A fixed-price (FIX) spot earns its stated price when one is
present; with no price it falls back to CPP so it is never zeroed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_daily_input
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.objective import break_revenue, fixed_revenue
from kairos.optimize.pricing import PricingModel

# The daily Wally file is a single channel and carries no daypart column; we
# derive a coarse daypart from the spot clock so prime-time-only baselines and
# daypart-scoped conditions have something real to match.
_PRIME_START_HOUR = 20
_PRIME_END_HOUR = 23  # inclusive


def _daypart_for_hour(hour: Optional[int]) -> Optional[str]:
    """Map a spot's clock hour to a coarse daypart token, or None when unknown.

    ``prime`` covers 20:00..23:59 (matching the prime-time pricing window used
    elsewhere in the engine); ``daytime`` covers 06:00..19:59; ``overnight``
    covers the rest. A None hour yields None so nothing is guessed.
    """
    if hour is None:
        return None
    if _PRIME_START_HOUR <= hour <= _PRIME_END_HOUR:
        return "prime"
    if 6 <= hour < _PRIME_START_HOUR:
        return "daytime"
    return "overnight"


def _hour_from_time(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


@dataclass(frozen=True)
class PricedSpot:
    """One priced daily spot, with the advertiser premium that was applied."""

    advertiser: str
    campaign: str
    program: str
    position: Optional[int]
    genre: str
    daypart: Optional[str]
    duration_seconds: float
    planned_tvr: float
    pricing_type: str
    premium: float
    revenue: float


@dataclass(frozen=True)
class DroppedSpot:
    """One daily spot dropped because its advertiser rule forbids the placement."""

    advertiser: str
    campaign: str
    program: str
    position: Optional[int]
    genre: str
    daypart: Optional[str]
    reason: str


@dataclass(frozen=True)
class DailyPricingResult:
    """The outcome of pricing one daily Wally file under the advertiser rules."""

    priced: list[PricedSpot] = field(default_factory=list)
    dropped: list[DroppedSpot] = field(default_factory=list)

    @property
    def total_revenue(self) -> float:
        return round(sum(spot.revenue for spot in self.priced), 2)

    @property
    def status_text(self) -> str:
        """One honest sentence for the dashboard about what now honors the rules."""
        return (
            "Advertiser rules are applied on the per-spot daily pricing path "
            f"({len(self.priced)} spots priced, {len(self.dropped)} dropped by a rule); "
            "the weekly break-count optimizer does not yet honor advertiser rules."
        )


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def price_daily_spots(
    daily: pd.DataFrame,
    *,
    engine: Optional[AdvertiserRuleEngine] = None,
    pricing: Optional[PricingModel] = None,
    classifier: Optional[ProgramClassifier] = None,
) -> DailyPricingResult:
    """Price every spot in a loaded daily Wally frame under the advertiser rules.

    ``daily`` must carry the canonical columns produced by
    :func:`kairos.data.loaders.load_daily_input` (advertiser, campaign, program,
    position_in_break, planned_tvr, duration_sec, pricing_type, spot_time). Each
    spot is classified for its genre, assigned a coarse daypart from its clock,
    priced with CPP math times the advertiser's effective premium, and either
    kept (allowed) or dropped (forbidden by a rule), never both.
    """
    engine = engine or AdvertiserRuleEngine.from_files()
    pricing = pricing or PricingModel.from_yaml()
    classifier = classifier or ProgramClassifier.from_yaml()

    priced: list[PricedSpot] = []
    dropped: list[DroppedSpot] = []

    for row in daily.itertuples(index=False):
        advertiser = str(getattr(row, "advertiser", "") or "")
        if not advertiser:
            continue
        program = str(getattr(row, "program", "") or "")
        genre = classifier.classify(program).category
        position = _coerce_int(getattr(row, "position_in_break", None))
        daypart = _daypart_for_hour(_hour_from_time(getattr(row, "spot_time", None)))

        decision = engine.allow_decision(
            advertiser, position=position, genre=genre, daypart=daypart
        )
        campaign = str(getattr(row, "campaign", "") or "")
        if not decision.allowed:
            dropped.append(DroppedSpot(
                advertiser=advertiser, campaign=campaign, program=program,
                position=position, genre=genre, daypart=daypart, reason=decision.reason,
            ))
            continue

        premium = engine.effective_premium(
            advertiser, position=position, genre=genre, daypart=daypart
        )
        duration = _coerce_float(getattr(row, "duration_sec", None))
        planned_tvr = _coerce_float(getattr(row, "planned_tvr", None))
        pricing_type = str(getattr(row, "pricing_type", "") or "").strip().upper()
        stated_price = _coerce_float(getattr(row, "price", None))

        if pricing_type == "FIX" and stated_price > 0:
            revenue = fixed_revenue(stated_price) * premium
        else:
            revenue = break_revenue(
                planned_tvr, duration, pricing.base_price, premium=premium,
            )

        priced.append(PricedSpot(
            advertiser=advertiser, campaign=campaign, program=program,
            position=position, genre=genre, daypart=daypart,
            duration_seconds=round(duration, 1), planned_tvr=planned_tvr,
            pricing_type=pricing_type or "CPP", premium=round(premium, 6),
            revenue=round(revenue, 2),
        ))

    return DailyPricingResult(priced=priced, dropped=dropped)


def price_daily_file(
    path: str | Path,
    *,
    engine: Optional[AdvertiserRuleEngine] = None,
    pricing: Optional[PricingModel] = None,
    classifier: Optional[ProgramClassifier] = None,
) -> DailyPricingResult:
    """Load a daily Wally csv from ``path`` and price it under the advertiser rules."""
    daily = load_daily_input(path)
    return price_daily_spots(daily, engine=engine, pricing=pricing, classifier=classifier)
