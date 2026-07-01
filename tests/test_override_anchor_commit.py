"""The anchor guard in the build_weekly_schedule COMMIT path.

The /api/overrides/effect preview and the decision endpoints already resolve
stored overrides against the current segments' semantic anchors (date, start
clock, program). These tests prove the SAME guard now protects the committed
weekly CSV: a stale-anchored override is skipped (the committed day is
byte-identical to a no-override build, and the skip is reported on
``frame.attrs["skipped_overrides"]``), a matching-anchor override binds, and a
blank-anchor legacy override still binds by target_id alone.

The programme frame is synthetic (one channel-day, two programmes) but the run
is the real engine end to end: classifier, pricing, transform, optimizer and
the export row construction. Runs under pytest or directly with the venv
python (``python tests/test_override_anchor_commit.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from kairos.export.schedule import build_weekly_schedule
from kairos.optimize.overrides import Override, OverrideSet

CHANNEL = "קשת 12"


def _programmes() -> pd.DataFrame:
    rows = [
        {"Title": "Evening Drama", "Channel": CHANNEL,
         "start_dt": pd.Timestamp("2026-01-05 20:00:00"), "Duration": 3600.0, "TVR": 10.0},
        {"Title": "Late Panel", "Channel": CHANNEL,
         "start_dt": pd.Timestamp("2026-01-05 21:00:00"), "Duration": 3600.0, "TVR": 8.0},
    ]
    return pd.DataFrame(rows)


def _build(overrides: OverrideSet) -> pd.DataFrame:
    return build_weekly_schedule(
        programmes=_programmes(), overrides=overrides, revenue_weight=1.0,
    )


def _baseline() -> pd.DataFrame:
    return _build(OverrideSet())


def _forbid_override(target_id: str, **anchor: str) -> OverrideSet:
    return OverrideSet(overrides=[Override(
        override_id="anchor-test-1", scope="segment", target_id=target_id,
        kind="forbid", **anchor,
    )])


def test_mismatched_anchor_override_is_skipped_at_commit() -> None:
    """A stale anchor leaves the committed day byte-identical to no-override."""
    base = _baseline()
    row = base.iloc[0]
    assert int(row["num_breaks"]) > 0, "test needs a segment the optimizer gave breaks"
    overrides = _forbid_override(
        str(row["segment_id"]),
        anchor_date=str(row["date"]),
        anchor_start=str(row["start_time"]),
        anchor_title="A Different Programme Entirely",
    )
    committed = _build(overrides)
    assert committed.to_csv(index=False) == base.to_csv(index=False), (
        "a mismatched-anchor override must not bend the committed plan"
    )
    skipped = committed.attrs["skipped_overrides"]
    assert len(skipped) == 1
    assert skipped[0]["override_id"] == "anchor-test-1"
    assert "anchor mismatch" in skipped[0]["reason"]
    assert skipped[0]["found"] is not None


def test_matching_anchor_override_binds_at_commit() -> None:
    """An override whose stored anchor still matches its segment is applied.

    The anchor trio is read from the baseline CSV row (date, start_time,
    program_type), exactly what the decision endpoints record via _row_anchor,
    so this also proves a CSV-derived anchor binds at commit time.
    """
    base = _baseline()
    row = base.iloc[0]
    assert int(row["num_breaks"]) > 0
    overrides = _forbid_override(
        str(row["segment_id"]),
        anchor_date=str(row["date"]),
        anchor_start=str(row["start_time"]),
        anchor_title=str(row["program_type"]),
    )
    committed = _build(overrides)
    target = committed[committed["segment_id"] == row["segment_id"]]
    assert len(target) == 1
    assert int(target.iloc[0]["num_breaks"]) == 0, "a bound forbid must zero the segment"
    assert committed.attrs["skipped_overrides"] == []


def test_blank_anchor_legacy_override_still_binds() -> None:
    """An override with no stored anchor keeps binding by target_id alone."""
    base = _baseline()
    row = base.iloc[0]
    assert int(row["num_breaks"]) > 0
    overrides = _forbid_override(str(row["segment_id"]))
    committed = _build(overrides)
    target = committed[committed["segment_id"] == row["segment_id"]]
    assert int(target.iloc[0]["num_breaks"]) == 0
    assert committed.attrs["skipped_overrides"] == []


def test_anchored_override_with_unknown_target_is_reported_not_applied() -> None:
    """An anchored override whose target_id exists nowhere is skip-reported."""
    base = _baseline()
    overrides = _forbid_override(
        "1999-01-01|nowhere|000",
        anchor_date="1999-01-01", anchor_start="20:00", anchor_title="Ghost Show",
    )
    committed = _build(overrides)
    assert committed.to_csv(index=False) == base.to_csv(index=False)
    skipped = committed.attrs["skipped_overrides"]
    assert len(skipped) == 1
    assert skipped[0]["found"] is None
    assert "not in the current schedule" in skipped[0]["reason"]


def main() -> int:
    tests = [
        test_mismatched_anchor_override_is_skipped_at_commit,
        test_matching_anchor_override_binds_at_commit,
        test_blank_anchor_legacy_override_still_binds,
        test_anchored_override_with_unknown_target_is_reported_not_applied,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
