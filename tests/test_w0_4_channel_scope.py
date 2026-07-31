"""The competitor boundary helper, measured against the real saved plan.

The breach these tests describe is real and is still live: the routes that
serve it belong to other pieces, so this file proves the helper closes it
rather than that the route is closed. The numbers below are the ones measured
on the running instance before the helper existed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kairos_api import channel_scope

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"

OWNED = "רשת 13"
RIVALS = ("קשת 12", "כאן 11", "עכשיו 14")


def _plan_head() -> pd.DataFrame:
    """The exact slice GET /api/schedule serves in break_schedule: head(200)."""
    if not PLAN_PATH.is_file():
        pytest.skip("no saved weekly plan on disk to measure the boundary against")
    return pd.read_csv(PLAN_PATH, encoding="utf-8-sig").head(200)


# ---------------------------------------------------------------------------
# The operator's one channel
# ---------------------------------------------------------------------------

def test_operator_channel_reads_settings_and_degrades_honestly():
    class _Settings:
        operator_channel = "  רשת 13  "

    assert channel_scope.operator_channel(_Settings()) == OWNED
    assert channel_scope.operator_channel({"operator_channel": OWNED}) == OWNED
    assert channel_scope.operator_channel({"operator_channel": ""}) == ""
    assert channel_scope.operator_channel({}) == ""


# ---------------------------------------------------------------------------
# The measured breach, and the helper that closes it
# ---------------------------------------------------------------------------

def test_scope_frame_closes_the_measured_schedule_breach():
    """Measured on the live instance: the 200 rows GET /api/schedule serves are
    96 קשת 12, 73 כאן 11, 28 עכשיו 14 and 3 of the operator's own."""
    frame = _plan_head()
    counts = frame["channel"].astype(str).str.strip().value_counts().to_dict()
    assert counts.get(OWNED) == 3
    assert sum(counts.get(rival, 0) for rival in RIVALS) == 197

    scoped, note = channel_scope.scope_frame(frame, channel=OWNED)
    assert len(scoped) == 3
    assert set(scoped["channel"].astype(str).str.strip()) == {OWNED}
    assert note["scope_channel"] == OWNED
    assert note["scoped"] is True
    assert note["rows_in"] == 200
    assert note["rows_out"] == 3
    assert note["channels_in"] == 4
    assert note["competitor_rows_excluded"] == 197
    assert note["competitor_channels_excluded"] == 3
    assert note["reason"] is None


def test_scope_records_closes_the_measured_break_operations_breach():
    """Measured on the live instance: GET /api/break-operations returns 12
    programmes for each of the four channels."""
    records = [
        {"channel": channel, "program_id": f"{channel}-{index}"}
        for channel in (OWNED, *RIVALS)
        for index in range(12)
    ]
    assert len(records) == 48

    kept, note = channel_scope.scope_records(records, channel=OWNED)
    assert len(kept) == 12
    assert {row["channel"] for row in kept} == {OWNED}
    assert note["competitor_rows_excluded"] == 36
    assert note["competitor_channels_excluded"] == 3
    # The kept rows are copies: scoping never mutates the caller's records.
    kept[0]["program_id"] = "changed"
    assert records[0]["program_id"] != "changed"


def test_an_unconfigured_channel_passes_through_and_says_so():
    records = [{"channel": OWNED}, {"channel": RIVALS[0]}]
    kept, note = channel_scope.scope_records(records, channel="")
    assert len(kept) == 2, "no scope is a pass-through, never a silent empty"
    assert note["scoped"] is False
    assert note["scope_channel"] is None
    assert note["reason"] == channel_scope.NO_OPERATOR_CHANNEL_REASON
    assert note["competitor_rows_excluded"] == 0
    assert note["competitor_channels_excluded"] == 0


def test_a_frame_without_a_channel_column_names_what_stopped_the_scope():
    frame = pd.DataFrame({"date": ["2024-11-01"], "predicted_revenue": [1.0]})
    scoped, note = channel_scope.scope_frame(frame, channel=OWNED)
    assert len(scoped) == 1
    assert note["scoped"] is False
    assert "channel column" in note["reason"]

    empty, empty_note = channel_scope.scope_frame(pd.DataFrame(), channel=OWNED)
    assert empty.empty
    assert empty_note["rows_in"] == 0
    assert empty_note["scoped"] is True


# ---------------------------------------------------------------------------
# The unnamed aggregate
# ---------------------------------------------------------------------------

def test_competitor_aggregate_keeps_the_fact_and_destroys_the_identity():
    records = [
        {"channel": OWNED, "predicted_revenue": 100.0, "num_breaks": 1},
        {"channel": RIVALS[0], "predicted_revenue": 10.0, "num_breaks": 2},
        {"channel": RIVALS[0], "predicted_revenue": 5.0, "num_breaks": 1},
        {"channel": RIVALS[1], "predicted_revenue": 1.0, "num_breaks": 3},
    ]
    aggregate = channel_scope.competitor_aggregate(
        records, sum_fields=("predicted_revenue", "num_breaks"), channel=OWNED
    )
    assert aggregate == {
        "channels": 2,
        "rows": 3,
        "totals": {"predicted_revenue": 16.0, "num_breaks": 6.0},
    }
    rendered = repr(aggregate)
    for rival in RIVALS:
        assert rival not in rendered, "a competitor name may never reach a payload"


def test_competitor_aggregate_on_the_real_plan_names_nobody():
    frame = _plan_head()
    aggregate = channel_scope.competitor_aggregate(
        frame.to_dict("records"), sum_fields=("predicted_revenue",), channel=OWNED
    )
    assert aggregate["channels"] == 3
    assert aggregate["rows"] == 197
    assert aggregate["totals"]["predicted_revenue"] > 0
    for rival in RIVALS:
        assert rival not in repr(aggregate)
