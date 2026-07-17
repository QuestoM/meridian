"""Proof that freezing ``today`` in the weekly-schedule golden is a no-op today.

``tests/golden_weekly_schedule.py`` feeds a fixed ``FROZEN_PACING_DATE`` (never
``date.today()``) into the byte-hash golden so the hash cannot silently start
depending on the wall clock. That freeze is only safe (provably a no-op) while the
delivery-pacing signal is inert, which holds exactly while campaign flights are
empty: ``load_campaigns()`` returns ``[]`` -> ``build_pacing_weights`` returns
all-1.0 weights -> the schedule is byte-identical for any ``today``.

This companion asserts that emptiness directly, and proves ``today``-independence
of the pacing weights while campaigns are empty. The day real campaign flights
land, ``test_pacing_inputs_are_currently_empty`` fails loudly, which is the signal
to re-capture the golden under a chosen fixed date rather than let it drift unseen.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.optimize._types import ProgramSegment  # noqa: E402
from kairos.optimize.pacing import build_pacing_weights, load_campaigns  # noqa: E402

from golden_weekly_schedule import FROZEN_PACING_DATE  # noqa: E402


def _two_segments() -> list[ProgramSegment]:
    """A tiny pair of real segments to feed the pacing-weight builder."""
    return [
        ProgramSegment(
            segment_id="seg-a", channel="קשת 12", day="2024-11-01",
            start_seconds=72000.0, duration_seconds=1800.0, program_type="News",
            baseline_tvr=10.0, cpp=1000.0,
        ),
        ProgramSegment(
            segment_id="seg-b", channel="קשת 12", day="2024-11-01",
            start_seconds=75600.0, duration_seconds=1800.0, program_type="Drama",
            baseline_tvr=12.0, cpp=1200.0,
        ),
    ]


def test_frozen_pacing_date_is_a_fixed_date_not_a_clock_read() -> None:
    """The golden pins an explicit calendar date, so the hash is clock-independent."""
    assert isinstance(FROZEN_PACING_DATE, date)
    assert FROZEN_PACING_DATE == date(2026, 6, 15)


def test_pacing_inputs_are_currently_empty() -> None:
    """The precondition that makes the ``today`` freeze a proven no-op today.

    Campaign flights are header-only, so the delivery-pacing signal (the sole
    consumer of ``today`` in the weekly build) is inert. If this ever fails, real
    flights have landed and the golden must be re-captured under a chosen date.
    """
    assert load_campaigns() == []


def test_pacing_weights_are_today_independent_while_campaigns_empty() -> None:
    """With no campaigns, the pacing weights are all-1.0 for ANY reference date.

    This is the direct proof that the frozen date cannot move the schedule: the
    weights the optimizer would fold are byte-identical whether ``today`` is the
    frozen date or the real clock date, and every weight is the 1.0 identity.
    """
    segments = _two_segments()
    frozen = build_pacing_weights(segments, load_campaigns(), FROZEN_PACING_DATE)
    live = build_pacing_weights(segments, load_campaigns(), date.today())
    assert frozen == live
    assert set(frozen.values()) == {1.0}
    assert set(frozen) == {"seg-a", "seg-b"}
