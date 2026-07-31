"""The daily spot ledger, read once and grouped, for every surface that needs it.

The product prices exactly one daily file through exactly one pipeline
(:func:`kairos.export.spots.price_daily_file`, composed for the API in
``kairos_api.exporters._load_daily_pricing``). The Agencies page already reports
that pipeline's totals; per-advertiser money had no reader at all, so an
advertiser could carry a name and still show ``revenue: null``.

This module adds the grouping and nothing else. It runs no pricing of its own,
so a figure read from here can never disagree with the same figure read from the
export or from the agency summary: there is one composition, and this calls it.

What it reports per advertiser, all from the same run:

  * ``gross``, the priced revenue, and ``net``, gross after agency rebates,
    which is reporting only and is the same net the agency summary shows.
  * ``spots``, the number of priced spots.
  * ``dropped_by_rule`` and ``dropped_by_frequency``, so an advertiser whose
    spots were removed is visible instead of merely absent. The shipped
    frequency rule drops a third of the day, and money that is not there for a
    stated reason is not the same as money that is zero.

Honesty rules. With no daily file, or a pipeline that raises, every figure is
``None`` and ``available`` is false with the reason; nothing is estimated and no
advertiser is invented. Advertiser keys are the strings the daily file itself
carries, unresolved and unfolded, because resolution belongs to the caller that
knows the name space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The grouped shape every caller reads. Kept as plain data so a route can return
# it directly and a test can compare it without a client.


@dataclass(frozen=True)
class AdvertiserMoney:
    """One advertiser's share of one daily ledger."""

    advertiser: str
    gross: float = 0.0
    net: float = 0.0
    spots: int = 0
    dropped_by_rule: int = 0
    dropped_by_frequency: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "advertiser": self.advertiser,
            "gross": round(self.gross, 2),
            "net": round(self.net, 2),
            "spots": self.spots,
            "dropped_by_rule": self.dropped_by_rule,
            "dropped_by_frequency": self.dropped_by_frequency,
        }


@dataclass(frozen=True)
class LedgerRead:
    """One reading of the daily ledger, grouped by advertiser.

    ``available`` false means there was nothing to read and ``reason`` says why;
    every total is then ``None`` and ``by_advertiser`` is empty.
    """

    available: bool = False
    reason: str = ""
    basis: Optional[str] = None
    gross: Optional[float] = None
    net: Optional[float] = None
    spots: Optional[int] = None
    dropped_by_rule: Optional[int] = None
    dropped_by_frequency: Optional[int] = None
    by_advertiser: dict[str, AdvertiserMoney] = field(default_factory=dict)

    def for_advertiser(self, advertiser: str) -> Optional[AdvertiserMoney]:
        return self.by_advertiser.get(advertiser)

    def totals_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "basis": self.basis,
            "gross": self.gross,
            "net": self.net,
            "spots": self.spots,
            "dropped_by_rule": self.dropped_by_rule,
            "dropped_by_frequency": self.dropped_by_frequency,
        }


_NO_DAILY_FILE = "No daily spot file on disk, so there is no ledger to read."
_PIPELINE_FAILED = "The daily pricing pipeline could not run, so no figure is reported."


def read_ledger() -> LedgerRead:
    """Price the newest daily file once and group its money by advertiser.

    Uses the API's own composition of the pricing pipeline, the same one the
    spots export and the agency summary use, so every figure here is the same
    figure those produce. Any failure to reach it is reported as unavailable
    rather than worked around, because a second composition would be a second
    set of numbers.
    """
    try:
        from kairos_api.exporters import _load_daily_pricing

        result = _load_daily_pricing()
    except Exception:  # noqa: BLE001 - an unreadable ledger is a state, not a crash
        return LedgerRead(reason=_PIPELINE_FAILED)
    if result is None:
        return LedgerRead(reason=_NO_DAILY_FILE)
    return group_result(result, basis=_basis_name())


def _basis_name() -> Optional[str]:
    """The daily file the totals came from, or None when it cannot be named."""
    try:
        from kairos_api.uploads import _newest_daily

        path = _newest_daily()
    except Exception:  # noqa: BLE001 - naming the basis must never fail the read
        return None
    return path.name if path is not None else None


def group_result(result: Any, *, basis: Optional[str] = None) -> LedgerRead:
    """Group one already-priced :class:`DailyPricingResult` by advertiser."""
    gross: dict[str, float] = {}
    net: dict[str, float] = {}
    spots: dict[str, int] = {}
    rule_dropped: dict[str, int] = {}
    frequency_dropped: dict[str, int] = {}
    for spot in result.priced:
        key = spot.advertiser
        gross[key] = gross.get(key, 0.0) + spot.revenue
        net[key] = net.get(key, 0.0) + spot.net_revenue
        spots[key] = spots.get(key, 0) + 1
    for spot in result.dropped:
        rule_dropped[spot.advertiser] = rule_dropped.get(spot.advertiser, 0) + 1
    for spot in result.frequency_dropped:
        frequency_dropped[spot.advertiser] = frequency_dropped.get(spot.advertiser, 0) + 1

    by_advertiser = {
        key: AdvertiserMoney(
            advertiser=key,
            gross=round(gross.get(key, 0.0), 2),
            net=round(net.get(key, 0.0), 2),
            spots=spots.get(key, 0),
            dropped_by_rule=rule_dropped.get(key, 0),
            dropped_by_frequency=frequency_dropped.get(key, 0),
        )
        for key in sorted(set(gross) | set(rule_dropped) | set(frequency_dropped))
    }
    return LedgerRead(
        available=True,
        reason="",
        basis=basis,
        gross=result.total_revenue,
        net=result.total_net_revenue,
        spots=len(result.priced),
        dropped_by_rule=len(result.dropped),
        dropped_by_frequency=len(result.frequency_dropped),
        by_advertiser=by_advertiser,
    )
