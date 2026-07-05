"""Identity proof for the bisect-based programme-title lookup.

kairos.model.measure._title_for_break used to walk every start-sorted
programme span of the channel until one contained the break: an
O(breaks x programmes) scan per channel that the scale audit projected at
roughly 7.8e9 operations on a 24-month window. It is now a sorted-starts
binary search over the same spans (the pattern of
kairos.model.competitor_features._in_break), which must return the EXACT same
title for every break, including the old tie-breaking (first containing span
in start order wins).

This suite keeps the retired linear scan as a private reference and proves
identity three ways: hand-built edge cases (overlaps, ties, boundaries,
non-covering spans), a seeded randomized sweep with heavy overlap, and the
full real month (every detected break on every channel, marked realdata).
The realdata test also prints an old-vs-new micro-benchmark line.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from kairos.model.measure import _programme_title_lookup, _title_for_break


# --- the retired implementation, verbatim, as the reference ------------------

def _reference_title_lookup(programmes: pd.DataFrame) -> dict[str, list[tuple]]:
    frame = programmes[programmes["start_dt"].notna()].copy()
    lookup: dict[str, list[tuple]] = {}
    for channel, group in frame.groupby("Channel", sort=False):
        spans = []
        for row in group.itertuples(index=False):
            start = getattr(row, "start_dt")
            end = getattr(row, "end_dt")
            title = getattr(row, "Title")
            if pd.notna(start) and pd.notna(end):
                spans.append((start, end, "" if title is None or pd.isna(title) else str(title)))
        spans.sort(key=lambda s: s[0])
        lookup[str(channel)] = spans
    return lookup


def _reference_title_for_break(
    lookup: dict[str, list[tuple]], channel: str, start: pd.Timestamp, end: pd.Timestamp
) -> str:
    for s_start, s_end, title in lookup.get(channel, []):
        if s_start <= start and s_end >= end:
            return title
    return ""


def _programmes(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _both(programmes: pd.DataFrame, channel: str, start: str, end: str) -> tuple[str, str]:
    """(new_title, reference_title) for one query."""
    new = _title_for_break(
        _programme_title_lookup(programmes), channel,
        pd.Timestamp(start), pd.Timestamp(end),
    )
    old = _reference_title_for_break(
        _reference_title_lookup(programmes), channel,
        pd.Timestamp(start), pd.Timestamp(end),
    )
    return new, old


# --- hand-built edge cases ----------------------------------------------------

_OVERLAPPING = _programmes([
    ("Long", "A", "2024-11-04 19:00", "2024-11-04 21:00"),
    ("Short", "A", "2024-11-04 19:30", "2024-11-04 20:30"),
    ("Late", "A", "2024-11-04 21:00", "2024-11-04 22:00"),
])


def test_overlapping_spans_first_by_start_wins() -> None:
    # Both Long and Short contain the break; the old scan returned the first
    # containing span in start order (Long). The bisect must agree.
    new, old = _both(_OVERLAPPING, "A", "2024-11-04 19:45", "2024-11-04 19:50")
    assert new == old == "Long"


def test_earlier_non_covering_span_is_skipped() -> None:
    frame = _programmes([
        ("Opener", "A", "2024-11-04 19:00", "2024-11-04 19:30"),
        ("Feature", "A", "2024-11-04 19:10", "2024-11-04 21:00"),
    ])
    # Opener starts first but ends before the break; Feature covers it.
    new, old = _both(frame, "A", "2024-11-04 19:50", "2024-11-04 19:55")
    assert new == old == "Feature"
    # A break inside both takes the first by start.
    new, old = _both(frame, "A", "2024-11-04 19:12", "2024-11-04 19:20")
    assert new == old == "Opener"


def test_exact_boundary_containment_matches() -> None:
    frame = _programmes([("Show", "A", "2024-11-04 20:00", "2024-11-04 21:00")])
    new, old = _both(frame, "A", "2024-11-04 20:00", "2024-11-04 21:00")
    assert new == old == "Show"


def test_break_past_span_end_does_not_match() -> None:
    frame = _programmes([("Show", "A", "2024-11-04 20:00", "2024-11-04 21:00")])
    new, old = _both(frame, "A", "2024-11-04 20:50", "2024-11-04 21:10")
    assert new == old == ""


def test_unknown_channel_and_uncovered_break_are_empty() -> None:
    new, old = _both(_OVERLAPPING, "B", "2024-11-04 19:45", "2024-11-04 19:50")
    assert new == old == ""
    new, old = _both(_OVERLAPPING, "A", "2024-11-04 18:00", "2024-11-04 18:10")
    assert new == old == ""


def test_equal_starts_keep_stable_source_order() -> None:
    frame = _programmes([
        ("First", "A", "2024-11-04 20:00", "2024-11-04 20:30"),
        ("Second", "A", "2024-11-04 20:00", "2024-11-04 21:00"),
    ])
    # Both start together and both contain the early break: stable sort keeps
    # source order, so the old scan returned First. The bisect must agree.
    new, old = _both(frame, "A", "2024-11-04 20:05", "2024-11-04 20:10")
    assert new == old == "First"
    # Later break outgrows First; only Second covers it.
    new, old = _both(frame, "A", "2024-11-04 20:40", "2024-11-04 20:50")
    assert new == old == "Second"


def test_nan_title_becomes_empty_string_in_both() -> None:
    frame = _programmes([("X", "A", "2024-11-04 20:00", "2024-11-04 21:00")])
    frame.loc[0, "Title"] = None
    new, old = _both(frame, "A", "2024-11-04 20:10", "2024-11-04 20:20")
    assert new == old == ""


def test_randomized_overlapping_spans_identical_to_reference() -> None:
    # Heavy random overlap, seeded: 200 spans across 3 channels, 600 queries.
    rng = np.random.default_rng(20260705)
    base = pd.Timestamp("2024-11-01 02:00")
    rows = []
    for i in range(200):
        channel = ("A", "B", "C")[int(rng.integers(0, 3))]
        start = base + pd.Timedelta(minutes=int(rng.integers(0, 7 * 1440)))
        length = int(rng.integers(5, 240))
        rows.append((f"title-{i}", channel,
                     str(start), str(start + pd.Timedelta(minutes=length))))
    frame = _programmes(rows)
    new_lookup = _programme_title_lookup(frame)
    old_lookup = _reference_title_lookup(frame)
    for _ in range(600):
        channel = ("A", "B", "C", "D")[int(rng.integers(0, 4))]
        q_start = base + pd.Timedelta(minutes=int(rng.integers(0, 7 * 1440)))
        q_end = q_start + pd.Timedelta(minutes=int(rng.integers(0, 30)))
        assert _title_for_break(new_lookup, channel, q_start, q_end) == \
            _reference_title_for_break(old_lookup, channel, q_start, q_end)


# --- the full real month: identity plus micro-benchmark ------------------------

@pytest.mark.realdata
def test_full_real_month_titles_identical_and_bench() -> None:
    from kairos.data import ProgramClassifier
    from kairos.data.loaders import load_programmes, load_spots
    from kairos.model.prepare import keyed_breaks

    spots = load_spots()
    programmes = load_programmes()
    breaks = keyed_breaks(spots, programmes, ProgramClassifier.from_yaml())
    queries = [
        (
            str(getattr(row, "channel")),
            pd.Timestamp(getattr(row, "break_start")).floor("min"),
            pd.Timestamp(getattr(row, "break_end")).floor("min"),
        )
        for row in breaks.itertuples(index=False)
    ]
    assert len(queries) > 2000  # the real month carries thousands of breaks

    new_lookup = _programme_title_lookup(programmes)
    old_lookup = _reference_title_lookup(programmes)

    t0 = perf_counter()
    old_titles = [
        _reference_title_for_break(old_lookup, c, s, e) for c, s, e in queries
    ]
    t_old = perf_counter() - t0
    t0 = perf_counter()
    new_titles = [_title_for_break(new_lookup, c, s, e) for c, s, e in queries]
    t_new = perf_counter() - t0

    assert new_titles == old_titles  # every break on every channel, exact
    matched = sum(1 for t in new_titles if t)
    assert matched > 0  # the lookup genuinely recovers titles on real data
    speedup = (t_old / t_new) if t_new > 0 else float("inf")
    print(
        f"[title-bisect] breaks={len(queries)} matched={matched} identical; "
        f"linear-scan {t_old:.4f}s vs bisect {t_new:.4f}s (x{speedup:.1f})"
    )
