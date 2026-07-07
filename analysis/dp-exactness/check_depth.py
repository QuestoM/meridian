"""Measure segment time-overlap depth on real data.

The DP order is segment start time. Exactness under overlap needs a joint
state over all OPEN segments (started, but whose break-interaction window has
not closed). This measures, per channel-day, the maximum and mean number of
segments whose interaction windows overlap a later segment's start:

  interaction window of segment i = [start_i, end_i + bl/2 + spacing_slack]

where end_i = start_i + duration_i, bl = break_length_seconds, and
spacing_slack = 420s (a break ending inside 420s of a later break start still
couples through the spacing guardrail). Depth at segment j = number of i < j
with window_i extending past start_j. The joint DP state carries the undecided
k of every such open segment, so cost multiplies by (max_breaks+1)^depth.

Read-only. Prints depth stats.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, _build_classifier  # noqa: E402
from kairos.model.impact import load_impact_model  # noqa: E402
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings  # noqa: E402
from kairos.service import _apply_first_break_multiplier  # noqa: E402

SPACING = 420.0


def main() -> None:
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(None, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()
    channels = sorted(set(programmes["Channel"].dropna().astype(str)))
    days = sorted(set(programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")))

    depth_counter: Counter[int] = Counter()
    per_group_max = []
    strict_depth_counter: Counter[int] = Counter()  # overlap by [start, end] only

    for channel in channels:
        for day in days:
            segs = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact_model,
                channel=channel, day=day,
            )
            if not segs:
                continue
            ordered = sorted(segs, key=lambda s: s.start_seconds)
            gmax = 0
            for j, cur in enumerate(ordered):
                depth = 0
                sdepth = 0
                for i in range(j):
                    prev = ordered[i]
                    end = prev.start_seconds + prev.duration_seconds
                    window_end = end + prev.break_length_seconds / 2.0 + SPACING
                    if window_end > cur.start_seconds:
                        depth += 1
                    if end > cur.start_seconds + 1e-9:
                        sdepth += 1
                depth_counter[depth] += 1
                strict_depth_counter[sdepth] += 1
                gmax = max(gmax, depth)
            per_group_max.append((gmax, channel, day, len(ordered)))

    per_group_max.sort(reverse=True)
    total = sum(depth_counter.values())
    print("interaction-window open-depth distribution (over all segments):")
    for d in sorted(depth_counter):
        print(f"  depth {d}: {depth_counter[d]} ({100.0 * depth_counter[d] / total:.1f}%)")
    print("strict [start,end] overlap depth distribution:")
    for d in sorted(strict_depth_counter):
        c = strict_depth_counter[d]
        print(f"  depth {d}: {c} ({100.0 * c / total:.1f}%)")
    print("worst 10 channel-days by max open-depth:")
    for gmax, channel, day, n in per_group_max[:10]:
        print(f"  max_depth={gmax} n={n} {channel} {day}")


if __name__ == "__main__":
    main()
