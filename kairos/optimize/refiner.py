"""F1 refiner: recover the revenue greedy leaves on the table per channel-day.

The greedy allocator in :mod:`kairos.optimize.optimizer` adds one break at a
time and stops when no single break improves the objective. That is suboptimal
because :func:`~kairos.optimize.optimizer._segment_break_objects` re-spaces a
segment's breaks at ``duration / (k + 1)``, so break POSITIONS move when a
segment's count changes and feasibility (spacing, hourly cap) is non-monotone in
the break count. The per-hour cap also COUPLES segments that share an hour. So a
coordinated multi-segment move (lower segment A, raise segment B) can be both
feasible and better, yet unreachable one break at a time through an infeasible
intermediate.

The weighted objective is a SUM of per-segment contributions and every guardrail
is scoped to one channel-day or finer, so channel-days are INDEPENDENT and the
global optimum is the sum of each channel-day's own optimum. This module refines
ONE channel-day group at a time, seeded from the greedy counts, and is tiered:

  * EXACT enumeration when the break-count box is small enough to be provably
    optimal (synthetic / tiny groups, the brute-force test oracle);
  * guardrail-aware LOCAL SEARCH otherwise (real channel-days hold dozens of
    programmes, so ``5 ** segments`` is astronomical): single-coordinate ascent
    then time-adjacent pairwise 2-opt, climbing to a local (not proven-global)
    optimum, so the recovered gain is a lower bound.

Every adopted move stays guardrail-compliant and STRICTLY improves the group's
objective contribution, so the refined plan never regresses below greedy. The
caller (:func:`~kairos.optimize.optimizer.optimize_breaks`) keeps greedy as the
warm start, adopts the refined counts only where they strictly beat greedy, and
rebuilds that group's placements / totals / decision trace from the real breaks.

The compliance and objective math is NOT forked here: this module calls the
optimizer's own primitives (``_group_breaks`` for the guardrail geometry and
``_group_objective_contribution`` for the score), so the refiner and greedy
decide against byte-identical economics and limits.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Optional, Sequence

from kairos.optimize.guardrails import Break, Guardrails, is_compliant
from kairos.optimize._types import Decision, PlacementPin, ProgramSegment
from kairos.optimize._segment_math import (
    _EPSILON,
    _group_breaks,
    _group_objective_contribution,
    _marginal_revenue,
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)

# Tier threshold: at or below this many break-count vectors the group is
# enumerated for a provably global optimum; above it the group is refined by
# guardrail-aware local search seeded from greedy. Kept small so only tiny /
# synthetic groups (and the brute-force test oracle) take the exact path; a real
# channel-day holds dozens of segments, so 5 ** segments is far larger than this.
_MAX_EXACT_COMBOS = 4096

# Production local search is kept cheap enough for the weekly 120-channel-day
# export: pairwise 2-opt only over a small window of TIME-ADJACENT segments (the
# only ones that can share an hour or spacing window), and a capped pass count.
_PAIRWISE_WINDOW = 6
_MAX_PASSES = 20


def _enumerate_group_exact(
    group: list[ProgramSegment],
    ranges: list[range],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> dict[str, int]:
    """Globally optimal compliant break counts by exhaustive enumeration.

    Used only for groups small enough to enumerate (``<= _MAX_EXACT_COMBOS``
    break-count vectors). Returns the counts that maximise the group's objective
    contribution among all guardrail-compliant vectors in the box. The all-floors
    vector is always compliant (verified upstream), so a result always exists.
    """
    placements = placements or {}
    best_counts: dict[str, int] = {s.segment_id: r.start for s, r in zip(group, ranges)}
    best_contribution = float("-inf")
    for vector in product(*ranges):
        counts = {segment.segment_id: k for segment, k in zip(group, vector)}
        if not is_compliant(_group_breaks(group, counts, gold_by_id, placements), guardrails):
            continue
        contribution, _, _ = _group_objective_contribution(
            group, counts,
            revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
            placements=placements,
        )
        if contribution > best_contribution + _EPSILON:
            best_contribution = contribution
            best_counts = counts
    return best_counts


def _local_search_group(
    group: list[ProgramSegment],
    seed_counts: dict[str, int],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> dict[str, int]:
    """Refine a large group's break counts by guardrail-aware local search.

    Real channel-days hold dozens of programmes, so the box of break-count
    vectors (``5 ** segments``) is far too large to enumerate. This climbs from
    the greedy ``seed_counts`` using moves that the one-break-at-a-time greedy
    search cannot reach because re-spacing makes feasibility non-monotone:

      * single-coordinate: set one segment to any feasible count, others fixed;
      * pairwise (2-opt): jointly set two TIME-ADJACENT segments, which is where
        the hourly-cap and spacing guardrails actually couple segments.

    Every adopted move is guardrail-compliant and strictly improves the group's
    objective contribution, so the result never regresses below greedy. It climbs
    to a local optimum, not a proven global one, so it is a lower bound on the
    achievable gain. Pairwise is restricted to a window of time-adjacent segments
    (the only ones that share an hour) to keep the search linear in segment count.
    A segment whose floor equals its cap (a pin / forbid / count override) is
    fixed, so the search never touches it.
    """
    placements = placements or {}
    order = sorted(range(len(group)), key=lambda i: group[i].start_seconds)
    window = _PAIRWISE_WINDOW
    counts = dict(seed_counts)

    # Precompute each segment's per-count contribution to the group objective. The
    # objective is separable across segments (revenue is a sum, retention is a
    # tvr-weighted sum, both over the same global constants), so the group
    # contribution is the sum of these and a single segment's move changes it by
    # exactly its own term delta. This lets the search GATE on the cheap objective
    # delta before the expensive (full group-rebuild) compliance check, which is
    # what makes the 120-channel-day weekly export land in minutes, not tens.
    rev_w = revenue_weight / revenue_scale
    ret_w = (1.0 - revenue_weight) / total_tvr if total_tvr > _EPSILON else 0.0
    term: dict[str, dict[int, float]] = {}
    for segment in group:
        sid = segment.segment_id
        pins = placements.get(sid)
        cell: dict[int, float] = {}
        for k in range(floors[sid], caps[sid] + 1):
            revenue = _segment_revenue(segment, k, pins)
            retention = _segment_retention(segment, k)
            cell[k] = rev_w * revenue + ret_w * segment.baseline_tvr * retention
        term[sid] = cell

    seg_by_id = {s.segment_id: s for s in group}

    # Cache each segment's break geometry per count, so a single- or two-segment
    # move only rebuilds the changed segments and reuses the rest. _group_breaks
    # rebuilds every segment, which dominates the cost on an 80-segment real day.
    geom: dict[str, dict[int, list[Break]]] = {s.segment_id: {} for s in group}

    def segment_breaks(sid: str, k: int) -> list[Break]:
        cache = geom[sid]
        cached = cache.get(k)
        if cached is None:
            cached = _segment_break_objects(
                seg_by_id[sid], k,
                is_gold=gold_by_id.get(sid, False), pins=placements.get(sid),
            )
            cache[k] = cached
        return cached

    # The flat break list for the current counts, kept in sync as moves are
    # adopted; trials swap in only the changed segments before the compliance check.
    cur_breaks: dict[str, list[Break]] = {s.segment_id: segment_breaks(s.segment_id, counts[s.segment_id]) for s in group}

    # Incremental guardrail aggregates for the CURRENT adopted counts, so a trial
    # move can be REJECTED by a localized check on just the hours and day it
    # touches, instead of rebuilding and fully re-evaluating the whole day's flat
    # break list on every trial. These mirror kairos.optimize.guardrails exactly:
    # per (channel, day, hour) break count, ad seconds and protected-break count;
    # per (channel, day) ad seconds and gold count. The localized check only ever
    # REJECTS (each branch is a sound necessary condition for compliance); anything
    # it does not reject falls through to the authoritative is_compliant below, so
    # the accepted set and the final plan are identical to a full re-evaluate on
    # every trial. Break and gold counts are integer-exact; the two ad-seconds
    # aggregates are compared with a tiny tolerance so floating-point accumulation
    # can never turn a compliant trial into a wrong reject (a knife-edge trial
    # simply pays for the authoritative check).
    protected_types = {p.lower() for p in guardrails.protected_program_types}
    seg_protected = {s.segment_id: str(s.program_type).lower() in protected_types for s in group}
    seconds_tolerance = 1e-6
    hour_count: dict[tuple, int] = defaultdict(int)
    hour_seconds: dict[tuple, float] = defaultdict(float)
    hour_protected: dict[tuple, int] = defaultdict(int)
    day_seconds: dict[tuple, float] = defaultdict(float)
    day_gold: dict[tuple, int] = defaultdict(int)

    def _apply(sid: str, breaks: list[Break], sign: int) -> None:
        protected = seg_protected[sid]
        for b in breaks:
            hkey = (b.channel, b.day, b.hour)
            dkey = (b.channel, b.day)
            hour_count[hkey] += sign
            hour_seconds[hkey] += sign * b.duration_seconds
            if protected:
                hour_protected[hkey] += sign
            day_seconds[dkey] += sign * b.duration_seconds
            if b.is_gold:
                day_gold[dkey] += sign

    for s in group:
        _apply(s.segment_id, cur_breaks[s.segment_id], 1)

    def _localized_reject(changes: dict[str, int]) -> bool:
        """True only when the trial provably breaches an aggregate guardrail.

        Evaluates ONLY the hours and day the changed segments touch, reusing the
        maintained aggregates for the untouched remainder. Every branch is a
        necessary condition for compliance, so a True here is always a real
        violation the full is_compliant would also reject; spacing (the one
        guardrail a localized delta cannot settle) is left to that full check.
        """
        d_count: dict[tuple, int] = defaultdict(int)
        d_seconds: dict[tuple, float] = defaultdict(float)
        d_protected: dict[tuple, int] = defaultdict(int)
        d_day_seconds: dict[tuple, float] = defaultdict(float)
        d_day_gold: dict[tuple, int] = defaultdict(int)

        def accumulate(breaks: list[Break], protected: bool, sign: int) -> None:
            for b in breaks:
                hkey = (b.channel, b.day, b.hour)
                dkey = (b.channel, b.day)
                d_count[hkey] += sign
                d_seconds[hkey] += sign * b.duration_seconds
                if protected:
                    d_protected[hkey] += sign
                d_day_seconds[dkey] += sign * b.duration_seconds
                if b.is_gold:
                    d_day_gold[dkey] += sign

        for sid, k in changes.items():
            new_breaks = segment_breaks(sid, k)
            for b in new_breaks:
                if b.retention < guardrails.min_retention_floor:
                    return True
            protected = seg_protected[sid]
            accumulate(cur_breaks[sid], protected, -1)
            accumulate(new_breaks, protected, 1)

        for hkey, delta in d_count.items():
            if delta and hour_count[hkey] + delta > guardrails.max_breaks_per_hour:
                return True
        for hkey in set(d_seconds) | set(d_protected):
            seconds = hour_seconds[hkey] + d_seconds.get(hkey, 0.0)
            protected = hour_protected[hkey] + d_protected.get(hkey, 0) > 0
            limit = (guardrails.protected_max_ad_seconds_per_hour if protected
                     else guardrails.max_ad_seconds_per_hour)
            if seconds - seconds_tolerance > limit:
                return True
        for dkey, delta in d_day_seconds.items():
            if day_seconds[dkey] + delta - seconds_tolerance > guardrails.max_daily_ad_seconds:
                return True
        for dkey, delta in d_day_gold.items():
            if delta and day_gold[dkey] + delta > guardrails.gold_breaks_max_per_day:
                return True
        return False

    def compliant_with(changes: dict[str, int]) -> bool:
        # Cheap localized reject first; only trials it cannot rule out pay for the
        # authoritative full-list evaluation, which stays the sole gate on adoption.
        if _localized_reject(changes):
            return False
        flat: list[Break] = []
        for s in group:
            sid = s.segment_id
            flat.extend(segment_breaks(sid, changes[sid]) if sid in changes else cur_breaks[sid])
        return is_compliant(flat, guardrails)

    def adopt(changes: dict[str, int]) -> None:
        for sid, k in changes.items():
            _apply(sid, cur_breaks[sid], -1)
            new_breaks = segment_breaks(sid, k)
            _apply(sid, new_breaks, 1)
            counts[sid] = k
            cur_breaks[sid] = new_breaks

    for _ in range(_MAX_PASSES):
        improved = False
        for segment in group:
            sid = segment.segment_id
            cur = counts[sid]
            base_term = term[sid][cur]
            for k in range(floors[sid], caps[sid] + 1):
                if k == cur:
                    continue
                # Cheap separable gate: skip any move that cannot improve before
                # paying for the (incremental) compliance rebuild.
                if term[sid][k] <= base_term + _EPSILON:
                    continue
                if compliant_with({sid: k}):
                    adopt({sid: k})
                    cur, base_term, improved = k, term[sid][k], True
        for a, pos in enumerate(order):
            sa = group[pos].segment_id
            for b in order[a + 1:a + 1 + window]:
                sb = group[b].segment_id
                cur_pair = term[sa][counts[sa]] + term[sb][counts[sb]]
                for ka in range(floors[sa], caps[sa] + 1):
                    for kb in range(floors[sb], caps[sb] + 1):
                        if ka == counts[sa] and kb == counts[sb]:
                            continue
                        if term[sa][ka] + term[sb][kb] <= cur_pair + _EPSILON:
                            continue
                        if compliant_with({sa: ka, sb: kb}):
                            adopt({sa: ka, sb: kb})
                            cur_pair = term[sa][ka] + term[sb][kb]
                            improved = True
        if not improved:
            break
    return counts


def optimize_group(
    group: list[ProgramSegment],
    seed_counts: dict[str, int],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> dict[str, int]:
    """Best compliant break counts for one channel-day.

    Enumerates exactly when the box of break-count vectors is small enough to be
    provably optimal (``<= _MAX_EXACT_COMBOS``); otherwise climbs from the greedy
    ``seed_counts`` by local search (large real channel-days, where exact
    enumeration is intractable). Greedy is suboptimal because re-spacing breaks at
    ``duration / (k + 1)`` makes feasibility non-monotone in the break count, so
    greedy cannot reach a feasible-and-better allocation through an infeasible
    intermediate one. The returned counts are guaranteed compliant and never worse
    than ``seed_counts`` on the group's objective contribution.
    """
    ranges = [range(floors[s.segment_id], caps[s.segment_id] + 1) for s in group]
    combos = 1
    for r in ranges:
        combos *= len(r)
    if combos <= _MAX_EXACT_COMBOS:
        return _enumerate_group_exact(
            group, ranges, gold_by_id, guardrails,
            revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
            placements=placements,
        )
    return _local_search_group(
        group, seed_counts, floors, caps, gold_by_id, guardrails,
        revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
        placements=placements,
    )


def replay_group_decisions(
    group: list[ProgramSegment],
    target_counts: dict[str, int],
    floors: dict[str, int],
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> list[Decision]:
    """Reconstruct a marginal-gain-ordered decision trace reaching the target.

    Adds breaks one at a time, each step taking the segment whose next break
    gains the most on the group's objective contribution, until every segment
    reaches its target count. This is the explanation for an already-chosen
    optimum, so intermediate states need not be guardrail-feasible (that is the
    very reason greedy could not reach this optimum on its own). Because the
    objective is separable, a single break only moves its own group's
    contribution, so the contribution delta here is the same marginal gain the
    greedy loop records globally.
    """
    placements = placements or {}
    counts = {s.segment_id: floors[s.segment_id] for s in group}
    base, _, _ = _group_objective_contribution(
        group, counts,
        revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
        placements=placements,
    )
    seg_by_id = {s.segment_id: s for s in group}
    decisions: list[Decision] = []
    pending = sum(target_counts[s.segment_id] - counts[s.segment_id] for s in group)
    for _ in range(pending):
        best_gain: Optional[float] = None
        best_id: Optional[str] = None
        for segment in group:
            sid = segment.segment_id
            if counts[sid] >= target_counts[sid]:
                continue
            trial = dict(counts)
            trial[sid] += 1
            contribution, _, _ = _group_objective_contribution(
                group, trial,
                revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
                placements=placements,
            )
            gain = contribution - base
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_id = sid
        if best_id is None:
            break
        counts[best_id] += 1
        segment = seg_by_id[best_id]
        k = counts[best_id]
        decisions.append(Decision(
            segment_id=best_id,
            break_index=k,
            marginal_objective_gain=best_gain if best_gain is not None else 0.0,
            marginal_revenue=_marginal_revenue(segment, k, placements.get(best_id)),
            retention_after=_segment_retention(segment, k),
        ))
        base += best_gain if best_gain is not None else 0.0
    return decisions
