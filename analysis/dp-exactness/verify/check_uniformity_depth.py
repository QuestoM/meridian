"""Independent re-check of two measured preconditions in the dp-exactness claim.

1. Uniform break_length_seconds per channel-day (the daily-budget state B is
   exact only under uniformity). Reports any day with more than one distinct
   break length among its segments.
2. Interaction-window open-depth per segment, recomputed from scratch with the
   claim's own definition: window_end[i] = max(end_i + bl/2 + 420,
   next-hour-ceiling(end_i + bl/2)); depth of segment j = number of earlier
   segments i < j with window_end[i] > start_j. Claim: max 6 over all real
   segments, <= 2 for about 92 percent.

Usage: PYTHONPATH=<repo>:<repo>/analysis/dp-exactness python check_uniformity_depth.py
"""
from __future__ import annotations

from collections import Counter

from dp_prototype import _load_groups
from kairos.optimize.guardrails import Guardrails

GR = Guardrails()


def main():
    depth_counter = Counter()
    max_depth = 0
    max_day = None
    nonuniform_days = 0
    n_days = 0
    n_segs = 0
    for channel, day, segs in _load_groups():
        n_days += 1
        n_segs += len(segs)
        lengths = Counter(s.break_length_seconds for s in segs)
        if len(lengths) != 1:
            nonuniform_days += 1
            print(f"NON-UNIFORM break lengths {channel} {day}: {dict(lengths)}")
        window_end = []
        for s in segs:
            bl = s.break_length_seconds
            last = s.start_seconds + s.duration_seconds + bl / 2.0
            we = max(last + GR.min_break_spacing_seconds,
                     (int(last // 3600.0) + 1) * 3600.0)
            window_end.append(we)
        for j, s in enumerate(segs):
            d = sum(1 for i in range(j) if window_end[i] > s.start_seconds)
            depth_counter[d] += 1
            if d > max_depth:
                max_depth = d
                max_day = (channel, day)
    print(f"days={n_days} segments={n_segs} nonuniform_break_length_days={nonuniform_days}")
    total = sum(depth_counter.values())
    cum = 0
    for d in sorted(depth_counter):
        cum += depth_counter[d]
        print(f"depth {d}: {depth_counter[d]} ({100.0 * depth_counter[d] / total:.1f}%) "
              f"cum<= {100.0 * cum / total:.1f}%")
    print(f"max open-depth: {max_depth} at {max_day}")


if __name__ == "__main__":
    main()
