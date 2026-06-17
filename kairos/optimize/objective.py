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


def conservative_impact(
    point: float,
    ci_low: float,
    ci_high: float,
    *,
    risk_lambda: float = 0.0,
) -> float:
    """Risk-adjust a per-break retention coefficient for an uncertainty-aware decision.

    The optimizer is choosing whether a break's marginal revenue beats its
    retention cost. When that cost is uncertain (a wide credible interval, a thin
    cell), valuing the break at the bare point estimate ignores the chance the true
    cost is far worse. This returns a conservative (more pessimistic) retention
    coefficient so the decision is robust to estimation error.

    Decision-theory rationale: with an interval ``[ci_low, ci_high]`` on the
    retention delta (which is <= 0, so the MORE negative end is the worse, more
    damaging case), the robust choice values the break at the worst plausible cost
    in that band. ``risk_lambda`` scales how far toward that worst case we go:

      * ``risk_lambda = 0.0`` returns the point estimate exactly. This is the
        default, so nothing in the optimizer changes unless a risk preference is
        opted into.
      * ``risk_lambda = 1.0`` returns the lower credible bound (``min(ci_low,
        ci_high, point)``) -- the full upper-quantile-of-the-damage robust value.
      * ``0 < risk_lambda < 1`` returns ``point - risk_lambda * half_width``, a
        partial variance penalty between the two, where ``half_width`` is half the
        interval width.

    The result is clamped non-positive (a break cannot be valued as raising
    retention) and never returns a value more optimistic than the point estimate.
    A non-finite interval degrades to the point estimate, never to a fabricated
    cost.
    """
    if risk_lambda < 0:
        raise ValueError("risk_lambda must be non-negative")
    if risk_lambda == 0.0:
        return min(0.0, point)
    # NaN guard runs on each raw bound first: a non-finite interval must degrade to
    # the point estimate before any clamping. Each value is tested on its own via the
    # self-inequality trick, because min(0.0, nan) silently erases the NaN and
    # min(nan, x) depends on argument order, so a combined min() can miss it.
    if ci_low != ci_low or ci_high != ci_high or point != point:  # NaN guard
        return min(0.0, point)
    # The coefficient domain is non-positive (a break cannot raise retention), so
    # the credible interval on it must be non-positive too. Clamp both bounds to
    # <= 0 so a positive (gain-side) upper bound cannot inflate the variance penalty
    # below, and the interval stays consistent with the clamped point estimate.
    ci_low = min(0.0, ci_low)
    ci_high = min(0.0, ci_high)
    low = min(ci_low, ci_high)
    # The worst plausible cost is the most damaging (most negative) of the two
    # credible bounds and the point itself, so a point below its own interval is
    # never ignored.
    worst = min(low, point)
    if risk_lambda >= 1.0:
        return min(0.0, worst)
    half_width = abs(ci_high - ci_low) / 2.0
    penalized = point - risk_lambda * half_width
    # Never more optimistic than the point estimate, never below the worst bound
    # at full risk aversion, and never positive.
    conservative = max(worst, min(point, penalized))
    return min(0.0, conservative)


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
