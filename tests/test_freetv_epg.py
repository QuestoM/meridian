"""The rival lineup from one publication, and the three ways it could lie.

Every fixture here is a record shape captured from the real feed and then
reduced by hand — the field names, the field TYPES and the values that made a
difference are exactly as they arrived. Nothing in this file reaches the
network, so a publication that is down does not fail a build, and a publication
that changes shape fails one measurement rather than every test at once.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from kairos.model import freetv_epg as ft

# A real record, reduced. `live` is an OBJECT and not a flag — that is the
# capture's own shape and the reason this fixture keeps it.
RECORD = {
    "id": 7897191,
    "title": "חם לי קר לי - פרק אחרון (ש.ח.)",
    "lead": "מסע דוקומנטרי אישי",
    "description": "בפרק האחרון בסדרה",
    "since": "2026-08-19T20:34:00Z",
    "till": "2026-08-19T21:30:00Z",
    "live": {"type_": "LIVE", "id": 3370462},
    "liveBroadcast": False,
    "repeat": True,
}


def record(**changes):
    return {**RECORD, **changes}


# ------------------------------------------------------------ the conversion

def test_a_broadcast_arrives_in_the_contract_the_engine_already_reads():
    rows, status = ft.to_contract_rows([RECORD], channel="כאן 11")
    assert len(rows) == 1
    row = rows[0]
    assert row["Channel"] == "כאן 11"
    assert row["Title"] == RECORD["title"]
    assert row["Duration"] == 56 * 60
    assert status["dropped"] == []
    assert set(row) >= {"Channel", "Title", "Date", "Start time", "End time", "Duration"}


def test_the_clock_is_the_broadcaster_s_and_not_the_publication_s():
    """The feed is UTC and Israel is not. A fixed offset would be wrong for
    half of every year, so the conversion is a real timezone: this instant is
    20:34 UTC in August, which is 23:34 in Israel."""
    rows, _ = ft.to_contract_rows([RECORD], channel="כאן 11")
    assert rows[0]["Start time"] == "23:34:00"
    assert rows[0]["Date"] == "19/08/2026"


def test_the_same_clock_in_winter_lands_an_hour_earlier():
    """The half of the year a fixed +3 would have silently broken."""
    winter = record(since="2026-01-15T20:34:00Z", till="2026-01-15T21:30:00Z")
    rows, _ = ft.to_contract_rows([winter], channel="כאן 11")
    assert rows[0]["Start time"] == "22:34:00"


def test_a_broadcast_running_past_midnight_keeps_its_real_length():
    """Both ends are dated, so the length is a subtraction and the end clock is
    simply the clock it is. The midnight case that breaks a bare end time
    cannot arise here, and this pins that it stays that way."""
    late = record(since="2026-08-19T20:57:00Z", till="2026-08-19T21:23:00Z")
    rows, _ = ft.to_contract_rows([late], channel="כאן 11")
    assert rows[0]["Start time"] == "23:57:00"
    assert rows[0]["End time"] == "00:23:00"
    assert rows[0]["Duration"] == 26 * 60
    assert rows[0]["Date"] == "19/08/2026", "the date is the day it started"


def test_the_channel_reference_is_not_mistaken_for_a_live_flag():
    """THE BUG, pinned. The field called `live` is an object naming the channel,
    and a non-empty dict is truthy — so a generous reading of it marked every
    broadcast in the file as live, including reruns."""
    rows, _ = ft.to_contract_rows([RECORD], channel="כאן 11")
    assert rows[0]["Live"] is False
    assert rows[0]["Rerun"] is True


def test_a_live_broadcast_reads_as_live():
    rows, _ = ft.to_contract_rows(
        [record(liveBroadcast=True, repeat=False)], channel="כאן 11")
    assert rows[0]["Live"] is True
    assert rows[0]["Rerun"] is False


def test_a_record_with_no_start_is_counted_rather_than_hidden():
    rows, status = ft.to_contract_rows([record(since="")], channel="כאן 11")
    assert rows == []
    assert len(status["dropped"]) == 1
    assert "start" in status["dropped"][0]["reason"]


def test_a_record_that_ends_before_it_starts_is_refused():
    """A negative length is not a short programme. Downstream would take it."""
    rows, status = ft.to_contract_rows(
        [record(till="2026-08-19T20:00:00Z")], channel="כאן 11")
    assert rows == []
    assert len(status["dropped"]) == 1


def test_records_come_out_in_broadcast_order():
    later = record(id=2, since="2026-08-20T05:00:00Z", till="2026-08-20T06:00:00Z")
    earlier = record(id=3, since="2026-08-19T04:00:00Z", till="2026-08-19T05:00:00Z")
    rows, _ = ft.to_contract_rows([later, earlier, RECORD], channel="כאן 11")
    assert [r["Date"] for r in rows] == ["19/08/2026", "19/08/2026", "20/08/2026"]


# ------------------------------------------------- the channel it is filed as

def test_a_channel_whose_number_was_reused_is_refused_not_filed():
    """The one failure nothing downstream can catch.

    The rows would be well formed and the file would load; a rival's whole
    evening would sit under another rival's name until somebody noticed by eye.
    """
    with pytest.raises(ft.FreeTvError) as caught:
        ft.verify_channel("כאן 11", {3370462: "ערוץ הקניות"})
    assert "ערוץ הקניות" in str(caught.value)
    assert "ערוץ 11" in str(caught.value)


def test_a_channel_the_publication_stopped_listing_is_refused():
    with pytest.raises(ft.FreeTvError):
        ft.verify_channel("כאן 11", {3340020: "ערוץ 12"})


def test_the_expected_title_still_matching_returns_the_id():
    assert ft.verify_channel("כאן 11", {3370462: "ערוץ 11"}) == 3370462


def test_a_channel_with_no_mapping_says_which_ones_have_one():
    with pytest.raises(ft.FreeTvError) as caught:
        ft.verify_channel("ערוץ הספורט", {})
    assert "כאן 11" in str(caught.value)


def test_every_mapped_channel_is_one_this_engine_has_history_for():
    """A source for a channel the engine cannot join to is a file that loads
    cleanly and contributes zero."""
    from kairos.data.loaders import CHANNELS

    assert set(ft.CHANNELS) <= set(CHANNELS)


# ------------------------------------------------------------- the window

def test_one_day_is_asked_for_at_a_time(monkeypatch):
    """Two days answer 400 LIVE_PROGRAMME_INVALID_TIMESPAN, measured. The loop
    is a limit of the publication and not an oversight, so it is pinned."""
    asked = []
    monkeypatch.setattr(ft, "verify_channel", lambda channel, listed=None: 3370462)
    monkeypatch.setattr(ft, "fetch_day", lambda live_id, day: asked.append(day) or [])
    ft.fetch("כאן 11", days=5, start=date(2026, 8, 19))
    assert asked == [date(2026, 8, 19 + n) for n in range(5)]


def test_a_broadcast_returned_by_two_days_is_kept_once(monkeypatch):
    """A programme straddling midnight comes back on both days it touches."""
    monkeypatch.setattr(ft, "verify_channel", lambda channel, listed=None: 3370462)
    monkeypatch.setattr(ft, "fetch_day", lambda live_id, day: [RECORD])
    assert len(ft.fetch("כאן 11", days=3, start=date(2026, 8, 19))) == 1


def test_a_day_that_fails_fails_the_whole_pull(monkeypatch):
    """A hole in a schedule reads as "the rival broadcasts nothing that day",
    which is a claim no publication made."""
    def flaky(live_id, day):
        if day.day == 20:
            raise ft.FreeTvError("the publication answered 500")
        return [RECORD]

    monkeypatch.setattr(ft, "verify_channel", lambda channel, listed=None: 3370462)
    monkeypatch.setattr(ft, "fetch_day", flaky)
    with pytest.raises(ft.FreeTvError):
        ft.fetch("כאן 11", days=3, start=date(2026, 8, 19))


def test_the_request_carries_the_broadcast_day_in_local_time():
    """Asking for a day in the wrong offset asks for the wrong day."""
    stamped = ft._stamp(datetime(2026, 8, 19, 0, 0, tzinfo=ft.BROADCAST_TZ))
    assert stamped.startswith("2026-08-19T00:00")
    assert stamped.endswith("0300"), "August in Israel is UTC+3"
    winter = ft._stamp(datetime(2026, 1, 15, 0, 0, tzinfo=ft.BROADCAST_TZ))
    assert winter.endswith("0200"), "January in Israel is UTC+2"
