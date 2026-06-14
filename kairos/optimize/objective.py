"""Revenue and retention primitives for the Kairos optimizer.

These are deliberately small, pure functions with no hidden constants, so the
optimizer's economics are transparent and fully testable. They replace the
placeholder math (for example ``viewing_points * 45000``) that the API used
when no real schedule was available.

Israeli TV ad pricing is Cost Per (rating) Point, CPP: a 30-second spot worth
one rating point costs one CPP unit, scaled by daypart and position premiums.
Sponsorships are usually priced at a fixed amount (FIX) rather than CPP.
"""

from __future__ import annotations

STANDARD_UNIT_SECONDS = 30.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))


def break_revenue(
    rating_points: float,
    duration_seconds: float,
    cpp: float,
    *,
    unit_seconds: float = STANDARD_UNIT_SECONDS,
    premium: float = 1.0,
) -> float:
    """Revenue of a single CPP-priced break, in the currency of ``cpp``.

    revenue = cpp * rating_points * (duration_seconds / unit_seconds) * premium
    """
    if rating_points < 0 or duration_seconds < 0 or cpp < 0:
        raise ValueError("rating_points, duration_seconds and cpp must be non-negative")
    if unit_seconds <= 0:
        raise ValueError("unit_seconds must be positive")
    if premium < 0:
        raise ValueError("premium must be non-negative")
    units = duration_seconds / unit_seconds
    return cpp * rating_points * units * premium


def fixed_revenue(price: float) -> float:
    """Revenue of a fixed-price (FIX / sponsorship) placement."""
    if price < 0:
        raise ValueError("price must be non-negative")
    return float(price)


def predicted_retention(
    baseline: float,
    impact_coefficient: float,
    num_breaks: float,
    *,
    floor: float = 0.2,
    ceiling: float = 1.0,
) -> float:
    """Predicted viewer retention for a segment carrying ``num_breaks`` breaks.

    retention = clamp(baseline + impact_coefficient * num_breaks, floor, ceiling)

    ``impact_coefficient`` comes from the trained impact model and is typically
    negative (more or longer breaks reduce retention).
    """
    if num_breaks < 0:
        raise ValueError("num_breaks must be non-negative")
    return clamp(baseline + impact_coefficient * num_breaks, floor, ceiling)


def retention_adjusted_revenue(revenue: float, retention: float) -> float:
    """Revenue actually realised after the retention-driven audience change."""
    if revenue < 0:
        raise ValueError("revenue must be non-negative")
    return revenue * clamp(retention, 0.0, 1.0)


def weighted_objective(
    revenue: float,
    retention: float,
    *,
    revenue_weight: float,
    revenue_scale: float,
) -> float:
    """Single scalar the optimizer maximises, balancing revenue and retention.

    ``revenue_weight`` is in [0, 1]: 1.0 cares only about revenue, 0.0 only about
    retention. Revenue is normalised by ``revenue_scale`` (a reference revenue
    level, for example the maximum achievable) so the two terms are comparable.
    Returns a value in [0, 1].
    """
    if not 0.0 <= revenue_weight <= 1.0:
        raise ValueError("revenue_weight must be in [0, 1]")
    if revenue_scale <= 0:
        raise ValueError("revenue_scale must be positive")
    if revenue < 0:
        raise ValueError("revenue must be non-negative")
    revenue_term = clamp(revenue / revenue_scale, 0.0, 1.0)
    retention_term = clamp(retention, 0.0, 1.0)
    return revenue_weight * revenue_term + (1.0 - revenue_weight) * retention_term
