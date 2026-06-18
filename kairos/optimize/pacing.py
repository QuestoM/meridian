"""Delivery-pacing urgency demand weights for the optimizer (Signal 2).

A campaign is a flight: an advertiser books a target (impressions or GRP) to be
delivered between a start and an end date. Some campaigns run ahead of plan,
some fall behind. A campaign that is 85% through its flight but only 30%
delivered is at risk of under-delivery, and the channel wants the optimizer to
LEAN breaks toward the inventory that campaign targets so it can catch up. That
lean is this module's "pacing urgency" signal.

Honesty contract (identical to the advertiser_conditions header-only seed)
--------------------------------------------------------------------------
Urgency enters ONLY through ``optimize_breaks(demand_weights=...)``, scaled into
the ranking by ``max(1.0, weight)`` and never charged. So:

  * With no campaign data present :func:`load_campaigns` returns ``[]`` and
    :func:`build_pacing_weights` returns ``{}`` -> every weight defaults to 1.0
    -> the schedule and total_revenue are BYTE-IDENTICAL to today.
  * A campaign can never be charged more because it is behind pace; urgency only
    changes WHERE the next break prefers to go.

Pure and deterministic: the urgency math reads a reference ``today`` that the
CALLER passes in. This module never calls datetime.now or random, so the same
inputs always produce the same weights.

Urgency formula
---------------
For a campaign with flight ``[start, end]``, target ``T`` and delivered ``D``,
and a reference date ``today``:

    elapsed_frac   = clamp((today - start) / (end - start), 0, 1)
    delivered_frac = clamp(D / T, 0, 1)
    behind         = max(0, elapsed_frac - delivered_frac)
    u = clamp(1 + K * behind / max(EPSILON, 1 - elapsed_frac), 1.0, U_MAX)

The ``(1 - elapsed_frac)`` denominator makes urgency rise sharply as the flight
runs out: the same shortfall is mild early and acute near the end. K, U_MAX and
EPSILON are explicit, documented, editable knobs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from kairos.optimize.optimizer import ProgramSegment

ROOT = Path(__file__).resolve().parents[2]
# Header-only seed by default, exactly like advertiser_conditions.csv: the file
# exists with a header and no rows, so the signal is wired but inert until the
# owner uploads real campaign flights.
DEFAULT_CAMPAIGNS_PATH = ROOT / "data" / "campaign_flights.csv"

# Urgency knobs. K is how hard a unit of "behind-ness" pushes; U_MAX caps the
# lift so one desperate campaign cannot dominate the schedule; EPSILON floors the
# (1 - elapsed_frac) denominator so a flight on its very last day does not divide
# by zero. All three are documented and editable.
URGENCY_K = 1.0
URGENCY_U_MAX = 2.0
URGENCY_EPSILON = 0.05


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Campaign:
    """One advertiser campaign flight read from campaign_flights.csv.

    ``flight_start`` / ``flight_end`` are ISO ``YYYY-MM-DD`` dates. ``target`` is
    the booked target_impressions or target_grp (either, both are a positive
    delivery target); ``delivered`` is delivered_to_date in the same unit.

    ``channels`` / ``genres`` / ``dayparts`` are the segment scope this campaign's
    spots target; an empty set means "any" in that dimension. ``programmes`` is an
    optional programme-title scope. These let urgency map onto the right segments.
    """

    campaign_id: str
    flight_start: str
    flight_end: str
    target: float
    delivered: float = 0.0
    channels: frozenset[str] = frozenset()
    genres: frozenset[str] = frozenset()
    dayparts: frozenset[str] = frozenset()
    programmes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AtRiskCampaign:
    """A campaign projected to finish under target, from the make-good helper."""

    campaign_id: str
    elapsed_frac: float
    delivered_frac: float
    projected_frac: float       # projected end-of-flight delivered / target
    projected_shortfall: float  # max(0, 1 - projected_frac), the gap to make good


def _parse_iso(value: object) -> Optional[date]:
    """Parse a YYYY-MM-DD (or leading ISO datetime) into a date, or None."""
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0].split("T")[0]
    parts = head.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None
    return None


def _tokens(value: object) -> frozenset[str]:
    """Split a scope cell into a lowercased token set; empty cell -> empty set."""
    text = str(value or "").strip()
    if not text:
        return frozenset()
    raw = text.replace(";", ",").replace("|", ",")
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def load_campaigns(path: Optional[str | Path] = None) -> list[Campaign]:
    """Read campaign flights from CSV; return ``[]`` when the file is header-only.

    A missing file or a file with only a header yields an empty list, so the
    pacing signal is a pure identity no-op until real campaign rows land. Rows
    missing an id, either flight date, or a positive target are skipped so a
    malformed line never invents urgency.
    """
    target = Path(path) if path is not None else DEFAULT_CAMPAIGNS_PATH
    if not target.exists():
        return []
    out: list[Campaign] = []
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for row in reader:
            campaign_id = str(row.get("campaign_id", "") or "").strip()
            start = _parse_iso(row.get("flight_start"))
            end = _parse_iso(row.get("flight_end"))
            target_val = _to_float(
                row.get("target_impressions") or row.get("target_grp") or row.get("target"),
                0.0,
            )
            if not campaign_id or start is None or end is None or target_val <= 0:
                continue
            out.append(Campaign(
                campaign_id=campaign_id,
                flight_start=start.isoformat(),
                flight_end=end.isoformat(),
                target=target_val,
                delivered=max(0.0, _to_float(row.get("delivered_to_date"), 0.0)),
                channels=_tokens(row.get("scope_channels")),
                genres=_tokens(row.get("scope_genres")),
                dayparts=_tokens(row.get("scope_dayparts")),
                programmes=_tokens(row.get("scope_programmes")),
            ))
    return out


def elapsed_fraction(campaign: Campaign, today: date) -> float:
    """Fraction of the flight elapsed at ``today``, clamped to [0, 1].

    A zero- or negative-length flight (end <= start) collapses to 1.0 on/after
    the start (the flight is effectively over), 0.0 strictly before it.
    """
    start = _parse_iso(campaign.flight_start)
    end = _parse_iso(campaign.flight_end)
    if start is None or end is None:
        return 0.0
    span = (end - start).days
    if span <= 0:
        return 1.0 if today >= start else 0.0
    return _clamp((today - start).days / span, 0.0, 1.0)


def delivered_fraction(campaign: Campaign) -> float:
    """Fraction of target delivered to date, clamped to [0, 1]."""
    if campaign.target <= 0:
        return 1.0
    return _clamp(campaign.delivered / campaign.target, 0.0, 1.0)


def campaign_urgency(
    campaign: Campaign,
    today: date,
    *,
    k: float = URGENCY_K,
    u_max: float = URGENCY_U_MAX,
    epsilon: float = URGENCY_EPSILON,
) -> float:
    """Pacing urgency for one campaign at ``today``, in [1.0, u_max].

        elapsed = clamp((today - start) / (end - start), 0, 1)
        delivered = clamp(delivered / target, 0, 1)
        u = clamp(1 + k * max(0, elapsed - delivered) / max(epsilon, 1 - elapsed), 1, u_max)

    A not-started, on-pace, or fully-delivered campaign returns exactly 1.0 (no
    steer). A behind-and-late campaign returns a high weight, bounded by u_max.
    """
    elapsed = elapsed_fraction(campaign, today)
    delivered = delivered_fraction(campaign)
    behind = max(0.0, elapsed - delivered)
    if behind <= 0.0:
        return 1.0
    denom = max(epsilon, 1.0 - elapsed)
    return _clamp(1.0 + k * behind / denom, 1.0, u_max)


def _campaign_matches_segment(
    campaign: Campaign,
    *,
    channel: str,
    genre: str,
    daypart: Optional[str],
    programme: Optional[str],
) -> bool:
    """True when a segment falls inside a campaign's targeting scope.

    An empty scope set in a dimension means "any", the honest conservative
    default. Matching is case-insensitive on lowercased tokens.
    """
    if campaign.channels and channel.strip().lower() not in campaign.channels:
        return False
    if campaign.genres and genre.strip().lower() not in campaign.genres:
        return False
    if campaign.dayparts:
        if daypart is None or daypart.strip().lower() not in campaign.dayparts:
            return False
    if campaign.programmes:
        if programme is None or programme.strip().lower() not in campaign.programmes:
            return False
    return True


def build_pacing_weights(
    segments: Iterable[ProgramSegment],
    campaigns: Optional[Sequence[Campaign]],
    today: date,
    *,
    daypart_of: Optional[Mapping[str, Optional[str]]] = None,
    k: float = URGENCY_K,
    u_max: float = URGENCY_U_MAX,
    epsilon: float = URGENCY_EPSILON,
) -> dict[str, float]:
    """Per-segment pacing-urgency weights (>= 1.0) keyed by segment_id.

    Each segment's weight is the MAX urgency over the campaigns whose scope it
    falls inside (the most urgent campaign pressing on a slot sets the lean). A
    segment matched by no behind-pace campaign keeps weight 1.0.

    ``campaigns`` of ``None`` or ``[]`` (the header-only seed case) yields every
    weight at 1.0, a pure identity no-op. ``today`` is supplied by the caller so
    the math is deterministic; ``daypart_of`` optionally maps segment_id to its
    daypart key for daypart-scoped campaigns (absent -> daypart treated as None).
    """
    # Materialize once: segments may be a one-shot iterator, and we walk it twice.
    seg_list = list(segments)
    weights: dict[str, float] = {seg.segment_id: 1.0 for seg in seg_list}
    if not campaigns:
        return weights
    daypart_of = daypart_of or {}
    for segment in seg_list:
        best = 1.0
        for campaign in campaigns:
            if not _campaign_matches_segment(
                campaign,
                channel=segment.channel,
                genre=segment.program_type,
                daypart=daypart_of.get(segment.segment_id),
                programme=segment.program_title or None,
            ):
                continue
            u = campaign_urgency(campaign, today, k=k, u_max=u_max, epsilon=epsilon)
            if u > best:
                best = u
        weights[segment.segment_id] = best
    return weights


def project_make_goods(
    campaigns: Optional[Sequence[Campaign]],
    today: date,
    *,
    epsilon: float = URGENCY_EPSILON,
) -> list[AtRiskCampaign]:
    """Project end-of-flight delivery and flag campaigns finishing under target.

    Linear projection from current pace::

        projected_frac = clamp(delivered_frac / max(epsilon, elapsed_frac), 0, ...)

    A campaign is at risk when ``projected_frac < 1`` (it is on track to finish
    short). The returned list carries each at-risk campaign and its projected
    shortfall ``max(0, 1 - projected_frac)``. Sorted worst-first so a dashboard
    alert can show the most exposed campaigns at the top.

    Data-pending: ``campaigns`` of ``None``/``[]`` returns ``[]``. Not-started
    campaigns (elapsed_frac == 0) are excluded: there is no pace to project yet.
    Pure: ``today`` is passed in, no clock is read here.
    """
    if not campaigns:
        return []
    at_risk: list[AtRiskCampaign] = []
    for campaign in campaigns:
        elapsed = elapsed_fraction(campaign, today)
        if elapsed <= 0.0:
            continue  # not started: nothing to project from
        delivered = delivered_fraction(campaign)
        projected = delivered / max(epsilon, elapsed)
        if projected >= 1.0:
            continue  # on track to finish at or above target
        at_risk.append(AtRiskCampaign(
            campaign_id=campaign.campaign_id,
            elapsed_frac=elapsed,
            delivered_frac=delivered,
            projected_frac=projected,
            projected_shortfall=max(0.0, 1.0 - projected),
        ))
    at_risk.sort(key=lambda c: c.projected_shortfall, reverse=True)
    return at_risk
