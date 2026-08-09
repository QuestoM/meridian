"""Preparation layer for the exact interval-sweep DP (:mod:`kairos.optimize.dp_refine`).

Everything the sweep needs decided BEFORE the first stage runs: which constraints
it cannot represent, which break counts each segment is allowed to take, which
segments carry gold, when a segment stops being able to interact with a later one,
and the per-segment additive objective terms. Split out of ``dp_refine`` so the
sweep module stays a readable single responsibility.

The operator constraint set (:mod:`kairos.optimize._override_logic`) reaches the
optimizer as four side maps: per-segment ``floors``, per-segment ``caps``,
``gold_by_id``, and ``placements``. Three of the four are representable in the DP
without a new state dimension and are honored EXACTLY here:

  * an override floor is the smallest allowed count, an override cap the largest,
    so both are a filter on the per-segment allowed set;
  * a gold mark is a per-segment boolean the daily gold budget already carries as a
    state dimension, so it only has to be read from ``gold_by_id`` rather than from
    the frozen segment's own ``is_gold``.

Placement pins are the one that is NOT representable: a pin carries its own
per-break duration, and the sweep's daily ad-seconds budget is an integer count of
uniform-length breaks (``max_daily_ad_seconds // break_length_seconds``). Mixed
durations turn that budget into a 0/1 knapsack over seconds, which is a different
state space, so a pinned channel-day still falls back to greedy+F1 with a named
reason rather than being answered approximately.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Guardrails
from kairos.optimize._segment_math import (
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)
from kairos.optimize._types import ProgramSegment

_EPSILON = 1e-9

OBJECTIVE_BLEND = "blend"
OBJECTIVE_REVENUE_NET = "revenue_net"

# Default open-depth guard. The exact DP's worst-case state count is exponential in
# the simultaneously-open depth; measured across all 120 real channel-days the max
# open depth is 13 (Kan 11 2024-11-09, ``analysis/dp-exactness/verify``), so the
# guard defaults ABOVE 13 to keep the real corpus on the exact path. A day deeper
# than the guard falls back to greedy+F1 rather than risk a runtime blowup.
DEFAULT_MAX_OPEN_DEPTH = 14

# Per-group compute budgets, the honest backstop behind the depth guard: depth is a
# proxy, and an adversarial in-guard day (many long overlapping segments) can still
# blow the state table or the clock. Measured across all 120 real channel-days
# (2026-07-17): the worst day peaks at 40,485 pruned states and 0.70 wall seconds
# (the depth-13 Kan 11 2024-11-09), so 200k states / 5.0 seconds keep the whole real
# corpus on the exact path with about 5x headroom while an adversarial day degrades
# to the labeled greedy+F1 fallback instead of stalling.
DEFAULT_MAX_STATES = 200_000
DEFAULT_WALL_BUDGET_SECONDS = 5.0


class DPBudgetExceeded(Exception):
    """The sweep exhausted its per-group state or wall budget (honest fallback).

    ``code`` is the stable histogram key (``state_budget`` / ``wall_budget``);
    ``str(exc)`` carries the measured detail for the human-readable reason.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _all_finite(group: Sequence[ProgramSegment]) -> bool:
    """Every numeric input the DP reads is finite (no NaN or inf reaches the sweep)."""
    for s in group:
        for value in (
            s.impact_coefficient, s.baseline_tvr, s.cpp, s.premium, s.unit_seconds,
            s.retention_baseline, s.start_seconds, s.duration_seconds,
            s.break_length_seconds, float(s.first_break_multiplier),
        ):
            if not math.isfinite(value):
                return False
    return True


def _blocking_constraint(
    group: Sequence[ProgramSegment],
    placements: Optional[Mapping[str, Sequence]],
) -> tuple[str, str]:
    """The first constraint off the DP's free path as ``(code, message)``, or two empties.

    Only placement pins remain: floors, caps and gold marks are represented exactly
    (see the module docstring), so a channel-day carrying any of those three now
    takes the exact path instead of surrendering the whole day. A pin's own break
    duration is what the count-based daily ad-seconds budget cannot express.
    """
    placements = placements or {}
    for s in group:
        if placements.get(s.segment_id):
            return "placement_pins", "placement pins present"
    return "", ""


def _effective_gold_flags(
    group: Sequence[ProgramSegment], gold_by_id: Optional[Mapping[str, bool]]
) -> list[bool]:
    """Per-segment gold as the guardrail sees it: the segment's own flag OR the override.

    :func:`kairos.optimize._segment_math._group_breaks` marks a break gold when
    ``segment.is_gold or gold_by_id[segment_id]``, and
    :func:`kairos.optimize.guardrails.check_gold_breaks` counts exactly those. The
    sweep's daily gold budget must be charged off the same union, otherwise an
    operator's gold mark on a non-gold segment is invisible to the budget and the
    DP can propose a plan the engine's own check rejects.
    """
    gold_by_id = gold_by_id or {}
    return [bool(s.is_gold) or bool(gold_by_id.get(s.segment_id, False)) for s in group]


def _window_ends(group: Sequence[ProgramSegment], guardrails: Guardrails, bl: float) -> list:
    """When each segment's breaks can no longer interact with a later start.

    A segment stays OPEN until the later of its last possible break plus the minimum
    spacing and the ceiling of the clock hour that last break can reach: past both,
    no future break can share a spacing pair or an hour with it.

    This closes the three LOCAL guardrails (spacing, breaks per hour, ad seconds per
    hour) and nothing else. The two DAILY guardrails (ad seconds per channel-day,
    gold breaks per channel-day) couple every segment in the day no matter how far
    apart they sit, which is why the sweep carries them as explicit state dimensions
    and why this boundary is NOT an independence cut the day could be split on.
    """
    ends = []
    for s in group:
        tail = s.start_seconds + s.duration_seconds + bl / 2.0
        spacing_end = tail + guardrails.min_break_spacing_seconds
        hour_ceiling = (int(tail // 3600.0) + 1) * 3600.0
        ends.append(max(spacing_end, hour_ceiling))
    return ends


def _max_open_depth(group: Sequence[ProgramSegment], window_end: Sequence[float]) -> int:
    """Largest set of segments simultaneously open at any later segment's start."""
    depth = 0
    for s in group:
        live = sum(
            1 for j, t in enumerate(group)
            if t.start_seconds <= s.start_seconds and window_end[j] > s.start_seconds
        )
        depth = max(depth, live)
    return depth


def _allowed_break_counts(
    group: Sequence[ProgramSegment],
    guardrails: Guardrails,
    floors: Optional[Mapping[str, int]] = None,
    caps: Optional[Mapping[str, int]] = None,
) -> list[list[int]]:
    """Per-segment ALLOWED break counts under overrides, the retention floor and own spacing.

    An override floor and an override cap bound the range the DP may explore, which
    is exactly how the greedy allocator reads them (it seeds ``state`` at the floors
    and refuses to add a break at the cap). With no overrides the bounds are
    ``0..max_breaks`` and this is identical to the pre-override scan.

    Inside those bounds every k is tested independently rather than stopping at the
    first failing k: with a POSITIVE impact coefficient retention RISES in k, so a
    below-floor count can be followed by an above-floor one and a first-failure
    break would silently understate the cap. k = 0 (no breaks emitted, nothing to
    check) is always allowed when the override floor permits it. The DP explores
    exactly these counts, so a mid-range k that breaches the retention floor or its
    own consecutive spacing is never proposed. With the usual non-positive
    coefficients both feasibilities are monotone in k, the allowed set is the prefix
    0..cap, and this scan reproduces the old first-failure cap exactly.

    A segment can come back with an EMPTY allowed set (an override floor above every
    feasible count); the caller turns that into a labeled fallback rather than
    guessing a count the operator did not ask for.
    """
    floors = floors or {}
    caps = caps or {}
    allowed: list[list[int]] = []
    for s in group:
        low = max(0, int(floors.get(s.segment_id, 0)))
        high = min(int(s.max_breaks), int(caps.get(s.segment_id, s.max_breaks)))
        ks: list[int] = []
        for k in range(low, high + 1):
            if k == 0:
                ks.append(0)
                continue
            if _segment_retention(s, k) < guardrails.min_retention_floor:
                continue
            breaks = _segment_break_objects(s, k)
            if any(
                cur.start_seconds - (prev.start_seconds + prev.duration_seconds)
                < guardrails.min_break_spacing_seconds
                for prev, cur in zip(breaks, breaks[1:])
            ):
                continue
            ks.append(k)
        allowed.append(ks)
    return allowed


def _contributions(
    group: Sequence[ProgramSegment],
    revenue_scale: float,
    total_tvr: float,
    kmax: Sequence[int],
    *,
    objective_mode: str,
    revenue_weight: float,
    net_of: Optional[Callable[[ProgramSegment, int], float]],
) -> list:
    """Per-segment additive objective term for k in 0..kmax[i].

    Blend mirrors :func:`_group_objective_contribution` exactly (the unclamped
    additive share ``revenue_weight * revenue / revenue_scale + (1 -
    revenue_weight) * tvr-weighted retention / total_tvr``), so the DP maximises the
    same scalar the shipped scorer measures. Revenue-net sums the caller's own
    per-segment net primitive. Both use the global normalisers passed in, not
    per-group ones, so the argmax matches the shipped objective.
    """
    if objective_mode == OBJECTIVE_REVENUE_NET:
        return [[net_of(s, k) for k in range(kmax[i] + 1)]
                for i, s in enumerate(group)]
    out = []
    for i, s in enumerate(group):
        row = []
        for k in range(kmax[i] + 1):
            rev = revenue_weight * _segment_revenue(s, k) / revenue_scale
            ret = ((1.0 - revenue_weight) * s.baseline_tvr * _segment_retention(s, k)
                   / total_tvr if total_tvr > _EPSILON else 0.0)
            row.append(rev + ret)
        out.append(row)
    return out
