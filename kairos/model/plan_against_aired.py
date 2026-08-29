"""How a plan's advertising load compares with what the channel actually airs.

A plan's revenue figure is read as money on the table. It is only that if the
airtime behind it is airtime the channel would really sell, and measured on the
real month it is not close: the shipped plan schedules 2.70 times the commercial
pods the operator channel aired and 1.70 times the seconds, so its 34.71M is the
value of a substantially larger and different inventory, not an improvement on
the one that exists.

None of that is visible anywhere. The plan publishes its own ad seconds, and the
number it should be read against was never computed, because nothing read the
broadcaster's own pod numbering until :mod:`kairos.data.as_aired`. A caveat that
lives one screen away from a figure is not disclosure; this exists so the
comparison can sit at the number itself.

This module states the comparison and refuses to editorialise it. It says how
much airtime the plan carries, how much the channel aired over the same days,
and the ratio. It does not adjust revenue, does not call a plan wrong, and does
not guess when there is no as-run source to compare against -- an absent
measurement reads as unknown, never as parity.

The operator's own channel only. What a competitor airs is out of bounds here as
everywhere else, and the channel argument is required rather than defaulted so
that boundary is impossible to cross by forgetting.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

from kairos.data.as_aired import identify_aired_pods


def compare_plan_to_aired(
    *,
    plan_pods: int,
    plan_ad_seconds: float,
    spots: Optional[pd.DataFrame],
    channel: str,
    days: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """The plan's advertising load beside the load the channel actually aired.

    ``days`` restricts the as-run side to the plan's own days, so a month of
    plan is never compared against a year of broadcast. Passing ``None`` uses
    every day the source carries and says so in ``aired_days``.

    Returns ``comparable: False`` with a reason whenever there is no as-run
    source, no pods in it, or nothing to compare -- the honest answer when the
    baseline is missing, and the one thing this must never fabricate.
    """
    unknown = {
        "comparable": False,
        "plan_pods": int(plan_pods),
        "plan_ad_seconds": round(float(plan_ad_seconds), 1),
        "aired_pods": None,
        "aired_ad_seconds": None,
        "pod_ratio": None,
        "ad_seconds_ratio": None,
        "aired_days": 0,
        "channel": channel,
    }
    if spots is None or getattr(spots, "empty", True):
        return {**unknown, "reason": "no as-run source is connected, so there is nothing to compare against"}

    pods = identify_aired_pods(spots, channel=channel)
    if pods.empty:
        return {**unknown, "reason": f"the as-run source carries no commercial pods for {channel}"}

    if days is not None:
        wanted = {str(day) for day in days}
        pods = pods[pods["day"].isin(wanted)]
        if pods.empty:
            return {**unknown, "reason": "the as-run source covers none of the days this plan covers"}

    aired_seconds = float(pd.to_numeric(pods["seconds"], errors="coerce").fillna(0).sum())
    aired_pods = int(len(pods))
    if aired_seconds <= 0:
        return {**unknown, "reason": "the as-run source reports no advertising seconds on these days"}

    return {
        "comparable": True,
        "reason": None,
        "channel": channel,
        "plan_pods": int(plan_pods),
        "plan_ad_seconds": round(float(plan_ad_seconds), 1),
        "aired_pods": aired_pods,
        "aired_ad_seconds": round(aired_seconds, 1),
        "aired_days": int(pods["day"].nunique()),
        "pod_ratio": round(plan_pods / aired_pods, 3) if aired_pods else None,
        "ad_seconds_ratio": round(float(plan_ad_seconds) / aired_seconds, 3),
    }


def disclosure_sentence(comparison: dict[str, Any], locale: str = "he") -> Optional[str]:
    """One sentence to sit beside the revenue figure, or ``None`` when unknown.

    Deliberately a statement of fact with no verdict attached: the operator is
    told what the plan assumes about its own ad load, and draws the conclusion.
    Returning ``None`` when the comparison could not be made keeps an unknown
    baseline from being narrated as agreement.
    """
    if not comparison.get("comparable"):
        return None
    ratio = comparison.get("ad_seconds_ratio")
    if ratio is None:
        return None
    plan_hours = comparison["plan_ad_seconds"] / 3600.0
    aired_hours = comparison["aired_ad_seconds"] / 3600.0
    if locale == "he":
        return (
            f"התוכנית מתזמנת {plan_hours:,.1f} שעות פרסום על פני {comparison['aired_days']} ימים, "
            f"מול {aired_hours:,.1f} שעות שהערוץ שידר באותם ימים, פי {ratio:.2f}. "
            f"ההכנסה שמוצגת היא של נפח הפרסום הזה, לא תוספת על מה שמשודר היום."
        )
    return (
        f"This plan schedules {plan_hours:,.1f} hours of advertising across "
        f"{comparison['aired_days']} days against the {aired_hours:,.1f} hours the channel aired "
        f"on the same days, {ratio:.2f} times as much. The revenue shown is the value of that "
        f"volume, not an increment on what airs today."
    )
