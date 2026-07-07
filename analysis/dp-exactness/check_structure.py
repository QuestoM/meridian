"""Empirical structure checks for the DP-exactness verdict.

Verifies, on the real programme data via the same builder the service uses:
  1. Segments within each channel-day are time-sorted and non-overlapping
     (required for the chain order of the DP: hour closing and spacing
     adjacency both assume break start times are monotone in segment order).
  2. break_length_seconds is uniform per run (collapses the daily ad-load
     budget to a break-count budget).
  3. Size stats: segments per channel-day, max_breaks, duration ranges.
  4. Break spillover: whether a segment's last break can END after the next
     segment starts (allowed; spacing handles it; DP carries last-break-end).

Read-only. Prints a report to stdout.
"""
from __future__ import annotations

import sys
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


def main() -> None:
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(None, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()
    channels = sorted(set(programmes["Channel"].dropna().astype(str)))
    days = sorted(set(programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")))

    n_groups = 0
    sizes = []
    overlap_groups = 0
    overlap_pairs = 0
    equal_start_pairs = 0
    unsorted_groups = 0
    break_lengths = set()
    max_breaks_vals = set()
    spill_possible = 0   # segments where last break end can pass next segment start
    total_segments = 0
    worst_overlap_examples = []

    for channel in channels:
        for day in days:
            segs = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact_model,
                channel=channel, day=day,
            )
            if not segs:
                continue
            n_groups += 1
            sizes.append(len(segs))
            total_segments += len(segs)
            for s in segs:
                break_lengths.add(s.break_length_seconds)
                max_breaks_vals.add(s.max_breaks)
            ordered = sorted(segs, key=lambda s: s.start_seconds)
            if [s.segment_id for s in ordered] != [s.segment_id for s in segs]:
                unsorted_groups += 1
            had_overlap = False
            for prev, cur in zip(ordered, ordered[1:]):
                if cur.start_seconds == prev.start_seconds:
                    equal_start_pairs += 1
                if cur.start_seconds < prev.start_seconds + prev.duration_seconds - 1e-9:
                    overlap_pairs += 1
                    had_overlap = True
                    if len(worst_overlap_examples) < 8:
                        worst_overlap_examples.append(
                            (channel, day, prev.segment_id, prev.start_seconds,
                             prev.duration_seconds, cur.segment_id, cur.start_seconds)
                        )
                # spillover: last break end at k = max_breaks
                k = prev.max_breaks
                if k >= 1:
                    spacing = prev.duration_seconds / (k + 1)
                    last_start = max(
                        prev.start_seconds,
                        prev.start_seconds + spacing * k - prev.break_length_seconds / 2.0,
                    )
                    if last_start + prev.break_length_seconds > cur.start_seconds:
                        spill_possible += 1

    sizes.sort()
    med = sizes[len(sizes) // 2] if sizes else 0
    print(f"channel-days with segments: {n_groups}")
    print(f"total segments: {total_segments}")
    print(f"segments per channel-day: min={sizes[0] if sizes else 0} "
          f"median={med} max={sizes[-1] if sizes else 0}")
    print(f"groups NOT already sorted by start_seconds: {unsorted_groups}")
    print(f"overlapping segment pairs (next.start < prev.end): {overlap_pairs} "
          f"in {overlap_groups} groups")
    print(f"equal-start segment pairs: {equal_start_pairs}")
    print(f"distinct break_length_seconds values: {sorted(break_lengths)}")
    print(f"distinct max_breaks values: {sorted(max_breaks_vals)}")
    print(f"segment-boundary pairs where a max-load last break can spill past "
          f"next segment start: {spill_possible}")
    for ex in worst_overlap_examples:
        print("  overlap example:", ex)


if __name__ == "__main__":
    main()
