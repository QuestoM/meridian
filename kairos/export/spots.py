"""Per-spot pricing for the daily Wally ad log, honoring advertiser rules.

This is the one pricing path where advertiser rules legitimately apply. The
weekly break-count optimizer decides break COUNTS per programme segment and
never attributes a break to an advertiser or a position, so it cannot consume
per-advertiser rules. The daily Wally file, by contrast, carries an advertiser,
a campaign and a position per individual spot (see
:data:`kairos.data.loaders.DAILY_COLUMN_MAP`), so it is here that
:class:`~kairos.optimize.advertiser_rules.AdvertiserRuleEngine` is wired in:

  * each priced spot's revenue is multiplied by ``effective_premium`` for its
    advertiser, position, genre, daypart and programme (so a programme-scoped
    rule bites here, on the path that actually prices the spot),
  * each priced spot also carries ``placement_value`` (the same spot valued with
    the placement-preference/pressure multiplier), so a steer is visible without
    inflating the charged revenue, and
  * a spot that fails ``is_allowed`` is dropped from the priced output and
    recorded in a separate dropped list with the reason, so nothing is silently
    lost or silently kept.

ONE HONEST SENTENCE (for the module and the dashboard status text): advertiser
rules now take effect on the per-spot daily pricing/export path
(:func:`price_daily_spots`); the weekly break-count optimizer does not yet honor
them, because it does not attribute breaks to advertisers.

Spot revenue uses the same CPP math as the rest of the engine
(:func:`kairos.optimize.objective.break_revenue`) on the same per-second basis:
a spot's planned break rating times its duration in seconds times the
per-second channel base price (``unit_seconds=1.0``, exactly how the weekly
segments are priced in :mod:`kairos.data.transform`), then the advertiser
premium. A fixed-price (FIX) spot earns its stated price when one is present;
with no price it falls back to CPP so it is never zeroed silently.

Agency layer (:class:`AgencyLayer`): the daily file also carries an agency per
spot, so agency-scoped conditions are evaluated here, agency-first then
advertiser, forbid wins across both levels, premiums compose multiplicatively.
An agency's ``rebate_percent`` yields a reporting-only ``net_revenue`` beside
gross; gross is unchanged and nothing is invoiced (no invoicing exists). The
weekly plan, retention math and QH settlement carry no agency attribution and
are untouched (see docs/agency-layer-design.md).
"""

from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.dayparts import daypart_for_hour
from kairos.data.loaders import load_daily_input
from kairos.export.agency_layer import AgencyLayer, AgencyTerms  # noqa: F401 - AgencyTerms re-exported
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize._frequency_rules import FrequencyRuleSet, load_frequency_rules
from kairos.optimize.frequency import (
    FrequencyDrop,
    SpotView,
    _minute_of_day,
    enforce_spots,
)
from kairos.optimize.objective import break_revenue, fixed_revenue
from kairos.optimize.overrides import OverrideSet
from kairos.optimize.pricing import PricingModel

# The daily Wally file is a single channel and carries no daypart column; the
# daypart comes from the spot clock via the one canonical taxonomy
# (kairos.data.dayparts.daypart_for_hour) so a daypart-scoped rule means the
# same minutes here as in the weekly plan and the training data. A None or
# invalid hour yields None (no guess).


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
    """One priced daily spot, with the advertiser premium that was applied.

    ``revenue`` is the REAL money (baseline premium times matching premium-effect
    rules only). ``placement_value`` is the same spot valued with the placement
    multiplier (premium times any matching placement-preference/pressure rules);
    it is what the optimizer ranks on, never charged. When no pressure rule matches
    the spot, ``placement_value == revenue`` (no steer); a positive pressure rule
    makes ``placement_value > revenue``, showing the bias without inflating revenue.
    """

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
    placement_value: float
    ad: str = ""
    break_id: str = ""
    # Agency layer: the resolved agency name ("" when unresolved), the agency-level
    # premium factor already composed into ``premium``/``revenue`` (1.0 when no
    # agency condition matched), and the reporting-only net figure:
    # net_revenue = revenue x (1 - rebate_percent/100). Gross ``revenue`` is
    # unchanged by the rebate; nothing is invoiced.
    agency: str = ""
    agency_premium: float = 1.0
    rebate_percent: float = 0.0
    net_revenue: float = 0.0


@dataclass(frozen=True)
class DroppedSpot:
    """One daily spot dropped because an advertiser or agency rule forbids it."""

    advertiser: str
    campaign: str
    program: str
    position: Optional[int]
    genre: str
    daypart: Optional[str]
    reason: str
    agency: str = ""


def _spot_id(advertiser: str, campaign: str, date: Any, position: Optional[int]) -> str:
    """The daily spot identifier a spot-scope override targets.

    The format is ``advertiser|campaign|date|position`` (position blank when
    unknown), which is what :meth:`OverrideSet.spot_overrides` keys on. It is the
    daily-level analogue of the segment id the weekly optimizer uses.
    """
    date_text = str(date or "").strip()[:10]
    position_text = "" if position is None else str(position)
    return f"{advertiser}|{campaign}|{date_text}|{position_text}"


@dataclass(frozen=True)
class DailyPricingResult:
    """The outcome of pricing one daily Wally file under the advertiser rules."""

    priced: list[PricedSpot] = field(default_factory=list)
    dropped: list[DroppedSpot] = field(default_factory=list)
    frequency_dropped: list[FrequencyDrop] = field(default_factory=list)

    @property
    def total_revenue(self) -> float:
        return round(sum(spot.revenue for spot in self.priced), 2)

    @property
    def total_net_revenue(self) -> float:
        """Gross minus agency rebates, reporting only. Equals total_revenue when
        no priced spot resolved to an active agency with a nonzero rebate."""
        return round(sum(spot.net_revenue for spot in self.priced), 2)

    @property
    def total_placement_value(self) -> float:
        """The placement-ranked total (with pressure). Equals total_revenue when no
        placement-preference rule matched any priced spot; it is never charged."""
        return round(sum(spot.placement_value for spot in self.priced), 2)

    @property
    def status_text(self) -> str:
        """One honest sentence for the dashboard about what now honors the rules."""
        return (
            "Advertiser rules are applied on the per-spot daily pricing path "
            f"({len(self.priced)} spots priced, {len(self.dropped)} dropped by a rule, "
            f"{len(self.frequency_dropped)} dropped by a frequency/separation rule); "
            "the weekly break-count optimizer does not yet honor these, because it "
            "does not attribute breaks to advertisers."
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
    overrides: Optional[OverrideSet] = None,
    frequency: Optional[FrequencyRuleSet] = None,
    agency_layer: Optional[AgencyLayer] = None,
) -> DailyPricingResult:
    """Price every spot in a loaded daily Wally frame under the advertiser rules.

    ``daily`` must carry the canonical columns produced by
    :func:`kairos.data.loaders.load_daily_input` (advertiser, campaign, program,
    position_in_break, planned_tvr, duration_sec, pricing_type, spot_time). Each
    spot is classified for its genre, assigned a coarse daypart from its clock,
    priced with CPP math times the advertiser's effective premium, and either
    kept (allowed) or dropped (forbidden by a rule), never both.

    ``overrides`` honors operator spot overrides here, where they genuinely bite:

      * a ``lock`` spot passes through untouched and is NEVER dropped by an
        advertiser rule, so a hand-pinned spot keeps its placement and price.
      * a ``move`` spot is re-tagged before pricing: its position (and, where
        given, its coarse daypart) are set to the override's target, then the
        advertiser rules are applied to the re-tagged spot. The daily path tags
        position and daypart but cannot re-place a spot at a clock time it never
        owned, so a move re-tags what it can and the rest is recorded honestly in
        the spot's ``move`` intent rather than fabricated.

    ``frequency`` adds ad-repetition and competitive-separation enforcement over
    the priced spots, IN PRICED ORDER. When None the shipped CSV is loaded; pass
    an EMPTY :class:`FrequencyRuleSet` to disable enforcement entirely (the spot
    log is then identical to the no-frequency output, the identity case). A spot
    removed by a frequency rule is recorded in ``frequency_dropped`` with its
    reason, never silently lost. This enforcement is advertiser-vs-advertiser
    WITHIN the owned channel only; it never touches the competitor-channel
    boundary, and the weekly count optimizer (no attribution) is untouched.

    ``agency_layer`` (loaded from the shipped stores when None) is evaluated
    agency-first: an agency forbid drops the spot before the advertiser rules
    run (forbid wins across levels; a locked spot is never dropped by either),
    agency premiums compose multiplicatively with the advertiser premium, and
    ``rebate_percent`` yields the reporting-only ``net_revenue``. With no agency
    conditions and rebate 0 the gross output is byte-identical to a run with no
    agency layer at all.
    """
    engine = engine or AdvertiserRuleEngine.from_files()
    pricing = pricing or PricingModel.from_yaml()
    classifier = classifier or ProgramClassifier.from_yaml()
    frequency = frequency if frequency is not None else load_frequency_rules()
    agency_layer = agency_layer if agency_layer is not None else AgencyLayer.from_files()
    spot_overrides = overrides.spot_overrides() if overrides is not None else {}

    priced: list[PricedSpot] = []
    dropped: list[DroppedSpot] = []
    spot_clocks: list[Optional[float]] = []

    for row in daily.itertuples(index=False):
        advertiser = str(getattr(row, "advertiser", "") or "")
        if not advertiser:
            continue
        program = str(getattr(row, "program", "") or "")
        genre = classifier.classify(program).category
        position = _coerce_int(getattr(row, "position_in_break", None))
        daypart = daypart_for_hour(_hour_from_time(getattr(row, "spot_time", None)))
        campaign = str(getattr(row, "campaign", "") or "")
        ad = str(getattr(row, "creative", "") or "")
        break_id = str(getattr(row, "break_start", "") or "")

        override = spot_overrides.get(
            _spot_id(advertiser, campaign, getattr(row, "date", None), position)
        )
        locked = bool(override and override.get("lock"))
        move = override.get("move") if override else None
        if move:
            # Re-tag the spot to the move target before pricing. Only position and
            # daypart can be re-tagged on the daily path; a clock-time move cannot
            # be realized here and is left to the recorded intent.
            new_position = _coerce_int(move.get("position"))
            if new_position is not None:
                position = new_position
            if move.get("daypart"):
                daypart = str(move["daypart"]).strip().lower()

        # Agency level first (forbid wins across levels; suspended terms are inert).
        agency_terms = agency_layer.resolve(getattr(row, "agency", ""), advertiser)
        agency_name = agency_terms.name if agency_terms else str(getattr(row, "agency", "") or "").strip()
        agency_key = agency_terms.agency_id if agency_terms and agency_terms.active else None
        rebate = agency_terms.rebate_percent if agency_key else 0.0
        agency_premium, agency_placement = 1.0, 1.0
        if agency_key:
            agency_decision = agency_layer.engine.allow_decision(
                agency_key, position=position, genre=genre, daypart=daypart, programme=program,
            )
            if not agency_decision.allowed and not locked:
                dropped.append(DroppedSpot(
                    advertiser=advertiser, campaign=campaign, program=program,
                    position=position, genre=genre, daypart=daypart,
                    reason=f"agency {agency_name}: {agency_decision.reason}", agency=agency_name,
                ))
                continue
            agency_premium = agency_layer.engine.effective_premium(
                agency_key, position=position, genre=genre, daypart=daypart,
                programme=program, base_cpp=pricing.base_price,
            )
            agency_placement = agency_layer.engine.placement_multiplier(
                agency_key, position=position, genre=genre, daypart=daypart,
                programme=program, base_cpp=pricing.base_price,
            )

        decision = engine.allow_decision(
            advertiser, position=position, genre=genre, daypart=daypart, programme=program,
        )
        if not decision.allowed and not locked:
            dropped.append(DroppedSpot(
                advertiser=advertiser, campaign=campaign, program=program,
                position=position, genre=genre, daypart=daypart, reason=decision.reason,
                agency=agency_name,
            ))
            continue

        premium = engine.effective_premium(
            advertiser, position=position, genre=genre, daypart=daypart,
            programme=program, base_cpp=pricing.base_price,
        ) * agency_premium
        placement_premium = engine.placement_multiplier(
            advertiser, position=position, genre=genre, daypart=daypart,
            programme=program, base_cpp=pricing.base_price,
        ) * agency_placement
        duration = _coerce_float(getattr(row, "duration_sec", None))
        planned_tvr = _coerce_float(getattr(row, "planned_tvr", None))
        pricing_type = str(getattr(row, "pricing_type", "") or "").strip().upper()
        stated_price = _coerce_float(getattr(row, "price", None))

        if pricing_type == "FIX" and stated_price > 0:
            base_value = fixed_revenue(stated_price)
            revenue = base_value * premium
            placement_value = base_value * placement_premium
        else:
            # base_price is quoted per second per TVR point (config yaml and the
            # weekly segments both say so), so the CPP math must run on the
            # per-second basis. break_revenue's default unit is 30 seconds,
            # which priced every daily spot at 1/30th of the engine's own basis.
            revenue = break_revenue(
                planned_tvr, duration, pricing.base_price, unit_seconds=1.0, premium=premium,
            )
            placement_value = break_revenue(
                planned_tvr, duration, pricing.base_price, unit_seconds=1.0, premium=placement_premium,
            )

        priced.append(PricedSpot(
            advertiser=advertiser, campaign=campaign, program=program,
            position=position, genre=genre, daypart=daypart,
            duration_seconds=round(duration, 1), planned_tvr=planned_tvr,
            pricing_type=pricing_type or "CPP", premium=round(premium, 6),
            revenue=round(revenue, 2), placement_value=round(placement_value, 2),
            ad=ad or campaign, break_id=break_id,
            agency=agency_name, agency_premium=round(agency_premium, 6),
            rebate_percent=round(rebate, 4),
            net_revenue=round(revenue * (1.0 - rebate / 100.0), 2),
        ))
        spot_clocks.append(_minute_of_day(getattr(row, "spot_time", None)))

    priced, freq_dropped = _apply_frequency(priced, spot_clocks, frequency)
    return DailyPricingResult(
        priced=priced, dropped=dropped, frequency_dropped=freq_dropped,
    )


def _apply_frequency(
    priced: list[PricedSpot],
    spot_clocks: list[Optional[float]],
    frequency: FrequencyRuleSet,
) -> tuple[list[PricedSpot], list[FrequencyDrop]]:
    """Run the deterministic frequency/separation pass over the priced spots.

    Returns the surviving priced spots (in their original priced order) and the
    spots removed by a rule, each with an explicit reason. With no enabled rule
    this is the identity: every spot survives, nothing is dropped.
    """
    if not any(r.enabled for r in frequency.rules):
        return priced, []
    views = [
        SpotView(
            key=index,
            advertiser=spot.advertiser,
            campaign=spot.campaign,
            ad=spot.ad or spot.campaign,
            break_id=spot.break_id,
            position=spot.position,
            minute=spot_clocks[index] if index < len(spot_clocks) else None,
        )
        for index, spot in enumerate(priced)
    ]
    outcome = enforce_spots(views, frequency)
    kept = {key for key in outcome.kept}
    survivors = [spot for index, spot in enumerate(priced) if index in kept]
    return survivors, outcome.dropped


def price_daily_file(
    path: str | Path,
    *,
    engine: Optional[AdvertiserRuleEngine] = None,
    pricing: Optional[PricingModel] = None,
    classifier: Optional[ProgramClassifier] = None,
    overrides: Optional[OverrideSet] = None,
    frequency: Optional[FrequencyRuleSet] = None,
    agency_layer: Optional[AgencyLayer] = None,
) -> DailyPricingResult:
    """Load a daily Wally csv from ``path`` and price it under the advertiser
    and agency rules."""
    daily = load_daily_input(path)
    return price_daily_spots(
        daily, engine=engine, pricing=pricing, classifier=classifier,
        overrides=overrides, frequency=frequency, agency_layer=agency_layer,
    )
