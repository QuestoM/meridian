"""Keeping a fact that expires, and refusing the three ways of losing it.

The feed is the only source in this repository that has ever carried Live and
Rerun, and it publishes only the coming fortnight. So the archive is not a
convenience: a morning it fails to keep is a morning whose labels are gone, and
none of the failures below announce themselves at the time.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kairos.model import feed_archive


def _rows(n=3, *, channel="קשת 12", day="21/08/2026", live="True", rerun="False"):
    return [{
        "Channel": channel, "Title": f"תוכנית {i}", "Date": day,
        "Start time": f"{20 + i:02d}:00:00", "End time": f"{21 + i:02d}:00:00",
        "Duration": "3600", "Live": live, "Rerun": rerun, "SeriesKey": f"key{i}",
    } for i in range(n)]


def test_a_pull_is_kept_with_its_liveness_intact(tmp_path):
    result = feed_archive.keep(_rows(), channel="קשת 12", root=tmp_path)
    assert result["kept"] is True
    entry = result["entry"]
    assert entry["rows"] == 3 and entry["liveness"] is True and entry["live"] == 3
    frame = feed_archive.read_snapshot(entry["file"], root=tmp_path)
    assert list(frame["Live"]) == ["True"] * 3


def test_an_unchanged_pull_is_recorded_without_storing_the_bytes_twice(tmp_path):
    """A rival that stops moving is the common case. "We pulled and it was the
    same" and "we did not pull" must stay different facts, or a silently dead
    job reads as a quiet week."""
    at = datetime(2026, 8, 21, 2, 30, tzinfo=timezone.utc)
    first = feed_archive.keep(_rows(), channel="קשת 12", at=at, root=tmp_path)
    second = feed_archive.keep(_rows(), channel="קשת 12", at=at + timedelta(days=1),
                               root=tmp_path)
    assert second["kept"] is True
    assert second["unchanged_since"] == first["entry"]["at"]
    assert len(feed_archive.read_index(tmp_path)) == 2
    assert len(list(tmp_path.glob("*.csv.gz"))) == 1


def test_an_empty_pull_is_not_archived_as_an_empty_schedule(tmp_path):
    """The refresh already refuses to write an empty pull over the contract,
    because no publication ever claimed the rival airs nothing. Archiving one
    would put that claim into the permanent record by another door."""
    result = feed_archive.keep([], channel="קשת 12", root=tmp_path)
    assert result["kept"] is False
    assert feed_archive.read_index(tmp_path) == []


def test_two_pulls_in_one_second_never_overwrite_each_other(tmp_path):
    at = datetime(2026, 8, 21, 2, 30, 0, tzinfo=timezone.utc)
    feed_archive.keep(_rows(2), channel="קשת 12", at=at, root=tmp_path)
    feed_archive.keep(_rows(4), channel="קשת 12", at=at, root=tmp_path)
    assert len(list(tmp_path.glob("*.csv.gz"))) == 2


def test_the_union_of_pulls_is_what_a_reader_gets(tmp_path):
    """A schedule is a forecast until it airs, so the same broadcast appears in
    many pulls and the window slides. The archive's value is the UNION, with the
    last pull that named each broadcast."""
    at = datetime(2026, 8, 21, 2, 30, tzinfo=timezone.utc)
    feed_archive.keep(_rows(3, day="21/08/2026"), channel="קשת 12", at=at, root=tmp_path)
    feed_archive.keep(_rows(3, day="22/08/2026"), channel="קשת 12",
                      at=at + timedelta(days=1), root=tmp_path)
    frame = feed_archive.broadcasts(tmp_path)
    assert len(frame) == 6
    assert set(frame["Date"]) == {"21/08/2026", "22/08/2026"}


def test_a_broadcast_republished_keeps_the_latest_statement_and_counts_the_pulls(tmp_path):
    at = datetime(2026, 8, 21, 2, 30, tzinfo=timezone.utc)
    feed_archive.keep(_rows(1, live="True", rerun="False"), channel="קשת 12",
                      at=at, root=tmp_path)
    feed_archive.keep(_rows(1, live="False", rerun="True"), channel="קשת 12",
                      at=at + timedelta(days=1), root=tmp_path)
    frame = feed_archive.broadcasts(tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["Live"] == "False" and row["Rerun"] == "True"  # the last word
    assert int(row["pulls"]) == 2
    assert row["first_seen"] != row["last_seen"]


def test_an_empty_archive_answers_none_and_never_an_empty_frame(tmp_path):
    """"Nothing was archived" and "nothing was published" are opposite facts and
    must not arrive at a reader looking the same."""
    assert feed_archive.broadcasts(tmp_path) is None
    assert feed_archive.read_index(tmp_path) == []
    assert feed_archive.coverage(tmp_path)["pulls"] == 0


def test_coverage_names_the_mornings_that_were_missed(tmp_path):
    """The missing days are the product. An archive that reports only what it
    holds reads as complete, and this one is complete only from its first day."""
    at = datetime(2026, 8, 21, 2, 30, tzinfo=timezone.utc)
    feed_archive.keep(_rows(2), channel="קשת 12", at=at, root=tmp_path)
    feed_archive.keep(_rows(3), channel="קשת 12", at=at + timedelta(days=3), root=tmp_path)
    report = feed_archive.coverage(tmp_path)
    assert report["days_with_no_pull"] == ["2026-08-22", "2026-08-23"]


def test_pull_days_are_counted_in_broadcast_time_not_utc(tmp_path):
    """A pull at 23:00 Israel is the NEXT UTC day. Counting in UTC invents a
    missing morning and hides a real one, on a product whose every other date is
    Israeli."""
    evening = datetime(2026, 8, 21, 20, 30, tzinfo=timezone.utc)  # 23:30 in Israel
    feed_archive.keep(_rows(2), channel="קשת 12", at=evening, root=tmp_path)
    feed_archive.keep(_rows(3), channel="קשת 12",
                      at=evening + timedelta(hours=1), root=tmp_path)  # 00:30, next day
    report = feed_archive.coverage(tmp_path)
    assert report["pull_days"] == 2
    assert report["days_with_no_pull"] == []


def test_an_unreadable_index_never_deletes_the_evidence(tmp_path):
    feed_archive.keep(_rows(2), channel="קשת 12", root=tmp_path)
    feed_archive.index_path(tmp_path).write_text("{ this is not json", encoding="utf-8")
    assert feed_archive.read_index(tmp_path) == []
    assert len(list(tmp_path.glob("*.csv.gz"))) == 1  # still on disk, untouched


def test_a_channel_name_that_no_filesystem_agrees_with_still_gets_a_file(tmp_path):
    result = feed_archive.keep(_rows(2, channel="כאן 11"), channel="כאן 11", root=tmp_path)
    assert result["kept"] is True
    assert feed_archive.read_snapshot(result["entry"]["file"], root=tmp_path) is not None
    # The index carries the real name; the filename only has to exist.
    assert feed_archive.read_index(tmp_path)[0]["channel"] == "כאן 11"


def test_a_snapshot_whose_file_is_gone_is_skipped_rather_than_raising(tmp_path):
    at = datetime(2026, 8, 21, 2, 30, tzinfo=timezone.utc)
    first = feed_archive.keep(_rows(2), channel="קשת 12", at=at, root=tmp_path)
    feed_archive.keep(_rows(3, day="22/08/2026"), channel="קשת 12",
                      at=at + timedelta(days=1), root=tmp_path)
    (tmp_path / first["entry"]["file"]).unlink()
    frame = feed_archive.broadcasts(tmp_path)
    assert frame is not None and len(frame) == 3


def test_the_refresh_archives_without_being_asked(tmp_path, monkeypatch):
    """THE WHOLE POINT. refresh() has taken a history_dir since it was written,
    keshet_feed threads it through as a flag, and the daily job passes --all and
    nothing else - so the capability was complete, wired, and never once reached.
    Archiving is now what happens; not archiving is what needs an argument."""
    from kairos.model import keshet_refresh

    archive = tmp_path / "archive"
    monkeypatch.setattr(feed_archive, "DEFAULT_ARCHIVE", archive)

    def convert(payload, *, channel):
        return list(payload), {"window_start": "21/08/2026", "window_end": "21/08/2026",
                               "days": ["21/08/2026"]}

    result = keshet_refresh.refresh(
        fetch=lambda: _rows(3), channel="קשת 12",
        target=tmp_path / "CompetitorProgrammes.csv", convert=convert,
    )
    assert result["refreshed"] is True
    assert result["archived"]["kept"] is True
    assert len(feed_archive.read_index(archive)) == 1


def test_a_failing_archive_never_turns_a_good_pull_into_a_failure(tmp_path, monkeypatch):
    """The schedule is already written by the time the archive runs. A full disk
    must cost the labels, not the plan."""
    from kairos.model import keshet_refresh

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(feed_archive, "keep", boom)

    def convert(payload, *, channel):
        return list(payload), {"window_start": "21/08/2026", "window_end": "21/08/2026",
                               "days": ["21/08/2026"]}

    result = keshet_refresh.refresh(
        fetch=lambda: _rows(3), channel="קשת 12",
        target=tmp_path / "CompetitorProgrammes.csv", convert=convert,
    )
    assert result["refreshed"] is True
    assert result["archived"]["kept"] is False
    assert "no space left" in result["archived"]["reason"]


def test_the_daily_log_says_whether_the_labels_were_kept():
    """A log that says "refreshed" and nothing else is exactly what let every
    pull before this one overwrite the last."""
    from kairos.model.keshet_feed import _archive_line

    assert "3 of 3" in _archive_line([{"archived": {"kept": True}}] * 3)
    assert "NOTHING KEPT" in _archive_line([{"archived": {"kept": False, "reason": "disk"}}])
    assert "identical" in _archive_line([
        {"archived": {"kept": True, "unchanged_since": "2026-08-20T02:30:00+00:00"}},
    ])


def test_the_live_archive_holds_what_the_contract_cannot(tmp_path):
    """Against the real archive: it already holds broadcasts the CURRENT contract
    file no longer carries, which is the whole reason it exists. Skipped where
    the archive has not run."""
    if not feed_archive.index_path().exists():
        pytest.skip("nothing archived on this machine yet")
    frame = feed_archive.broadcasts()
    if frame is None:
        pytest.skip("the archive index holds no readable snapshot")
    import pandas as pd
    from pathlib import Path

    contract = Path("data/reference/CompetitorProgrammes.csv")
    if not contract.exists():
        pytest.skip("no contract file on this machine")
    live = pd.read_csv(contract, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert len(frame) >= len(live)
    assert set(feed_archive.LIVENESS_COLUMNS) <= set(frame.columns)
