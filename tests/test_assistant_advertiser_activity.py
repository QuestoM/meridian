"""The assistant can read one advertiser across all raw traffic days at once.

This is deliberately not a money-ledger test: pricing and frequency rules may
drop rows, while advertiser activity must preserve every row in the authoritative
raw traffic snapshot.  Re-uploads of one broadcast day replace rather than add,
and every limit applies only after complete aggregates have been computed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from kairos_api import assistant_read_tools_advertiser as activity


HEBREW_COLUMNS = {
    "date": "תאריך",
    "spot_time": "שעה",
    "break_start": "שעת התחלת ברייק",
    "agency": "משרד / MB",
    "spot_type": "סוג תשדיר",
    "advertiser": "מפרסם",
    "campaign": "קמפיין",
    "creative": "שם גרסה",
    "house_number": "House Number",
    "duration_sec": "אורך תשדיר",
    "program": "תוכנית מוזמנת",
    "break_type": "סוג ברייק",
    "pricing_type": "סוג תמחור",
    "position_in_break": "מיקום בברייק",
    "status": "סטטוס",
}


def _row(day: str, *, advertiser: str = "לקוח בדיקה", campaign: str = "קמפיין א",
         creative: str = "גרסה א", clock: str = "20:01:00") -> dict[str, object]:
    return {
        HEBREW_COLUMNS["date"]: day,
        HEBREW_COLUMNS["spot_time"]: clock,
        HEBREW_COLUMNS["break_start"]: "20:00:00",
        HEBREW_COLUMNS["agency"]: "סוכנות",
        HEBREW_COLUMNS["spot_type"]: "פרסומת",
        HEBREW_COLUMNS["advertiser"]: advertiser,
        HEBREW_COLUMNS["campaign"]: campaign,
        HEBREW_COLUMNS["creative"]: creative,
        HEBREW_COLUMNS["house_number"]: "H-1",
        HEBREW_COLUMNS["duration_sec"]: 15,
        HEBREW_COLUMNS["program"]: "תוכנית",
        HEBREW_COLUMNS["break_type"]: "Regular",
        HEBREW_COLUMNS["pricing_type"]: "CPP",
        HEBREW_COLUMNS["position_in_break"]: 2,
        HEBREW_COLUMNS["status"]: "",
    }


def _write(path: Path, rows: list[dict[str, object]], mtime_ns: int) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_schema_and_register_name_the_all_raw_files_source() -> None:
    schema = activity.ADVERTISER_ACTIVITY_READ_TOOL_SCHEMAS[0]
    assert schema["name"] == "get_advertiser_airings"
    assert schema["input_schema"]["required"] == ["name"]
    assert set(schema["input_schema"]["properties"]) >= {
        "name", "date_from", "date_to", "limit", "offset",
    }
    executors: dict = {}
    sources: dict = {}
    activity.register(executors, sources)
    assert executors["get_advertiser_airings"] is activity._read_get_advertiser_airings
    assert "data/daily_input/Wally_*.csv" in sources["get_advertiser_airings"]


def test_shipped_mei_eden_answer_is_one_call_over_full_raw_coverage() -> None:
    payload = activity._read_get_advertiser_airings({"name": "מי   עדן"})
    assert payload["status"] == "ok"
    assert payload["identity"]["canonical_name"] == "מי עדן"
    assert payload["summary"] == {
        "airings": 2,
        "seconds": 30.0,
        "broadcast_days": 1,
        "campaigns": 1,
        "creatives": 1,
        "breaks": 2,
        "agencies": ["יוניברסל"],
        "first_airing_at": "2025-04-27T22:06:50",
        "last_airing_at": "2025-04-27T23:05:28",
    }
    assert payload["coverage"]["rows_read"] == 175
    assert payload["coverage"]["available_days"] == ["2025-04-27"]
    assert payload["coverage"]["complete_for_available_files"] is True
    assert payload["coverage"]["complete_through_today"] is False
    assert [row["break_start"] for row in payload["airings"]] == ["22:03:06", "22:59:40"]
    assert [row["position_in_break"] for row in payload["airings"]] == [5, 4]
    assert {row["house_number"] for row in payload["airings"]} == {"CMK022702"}
    assert {row["creative"] for row in payload["airings"]} == {
        "מי עדן סודה קיצור 15 מחליפה 2"
    }


def test_latest_file_per_actual_broadcast_day_prevents_reupload_double_count(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(activity, "DAILY_INPUT_DIR", tmp_path)
    old = tmp_path / "Wally_old.csv"
    new = tmp_path / "Wally_new.csv"
    next_day = tmp_path / "Wally_next.csv"
    _write(old, [_row("4/27/2025", campaign="גרסה ישנה")], 1_000_000_000)
    _write(new, [_row("4/27/2025", campaign="גרסה חדשה")], 2_000_000_000)
    _write(next_day, [_row("4/28/2025", campaign="יום שני")], 3_000_000_000)

    payload = activity._read_get_advertiser_airings({"name": "לקוח בדיקה"})

    assert payload["summary"]["airings"] == 2
    assert {row["campaign"] for row in payload["airings"]} == {"גרסה חדשה", "יום שני"}
    coverage = payload["coverage"]
    assert coverage["rows_read"] == 3
    assert coverage["authoritative_rows"] == 2
    assert coverage["available_days"] == ["2025-04-27", "2025-04-28"]
    assert coverage["shadowed_day_versions"] == [{
        "day": "2025-04-27",
        "authoritative_file": "Wally_new.csv",
        "shadowed_files": ["Wally_old.csv"],
    }]


def test_filters_and_pagination_never_cap_the_aggregates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(activity, "DAILY_INPUT_DIR", tmp_path)
    _write(tmp_path / "Wally_days.csv", [
        _row("4/27/2025", campaign="א", creative="א", clock="20:01:00"),
        _row("4/28/2025", campaign="ב", creative="ב", clock="20:02:00"),
        _row("4/29/2025", campaign="ג", creative="ג", clock="20:03:00"),
    ], 1_000_000_000)

    paged = activity._read_get_advertiser_airings({
        "name": "לקוח בדיקה", "limit": 1, "offset": 1,
    })
    assert paged["summary"]["airings"] == 3
    assert len(paged["campaigns"]) == 3
    assert len(paged["airings"]) == 1
    assert paged["airings"][0]["day"] == "2025-04-28"
    assert paged["pagination"] == {
        "offset": 1, "limit": 1, "returned": 1, "total": 3,
        "has_more": True, "next_offset": 2,
    }

    filtered = activity._read_get_advertiser_airings({
        "name": "לקוח בדיקה", "date_from": "2025-04-28", "date_to": "2025-04-28",
    })
    assert filtered["summary"]["airings"] == 1
    assert filtered["coverage"]["selected_days"] == ["2025-04-28"]
    assert filtered["coverage"]["available_days"] == [
        "2025-04-27", "2025-04-28", "2025-04-29",
    ]


def test_group_caps_keep_the_true_aggregate_totals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(activity, "DAILY_INPUT_DIR", tmp_path)
    rows = [
        _row("4/27/2025", campaign=f"קמפיין {index}", creative=f"גרסה {index}")
        for index in range(activity.MAX_GROUPS + 5)
    ]
    _write(tmp_path / "Wally_many.csv", rows, 1_000_000_000)

    payload = activity._read_get_advertiser_airings({"name": "לקוח בדיקה", "limit": 1})

    assert payload["summary"]["airings"] == activity.MAX_GROUPS + 5
    assert payload["summary"]["campaigns"] == activity.MAX_GROUPS + 5
    assert payload["campaigns_total"] == activity.MAX_GROUPS + 5
    assert payload["campaigns_omitted"] == 5
    assert len(payload["campaigns"]) == activity.MAX_GROUPS
    assert payload["pagination"]["total"] == activity.MAX_GROUPS + 5
    assert len(payload["airings"]) == 1


def test_invalid_dates_and_incomplete_files_fail_honestly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(activity, "DAILY_INPUT_DIR", tmp_path)
    _write(tmp_path / "Wally_valid.csv", [_row("not-a-date")], 1_000_000_000)
    (tmp_path / "Wally_broken.csv").write_bytes(b"\xff\xfe\x00")

    invalid = activity._read_get_advertiser_airings({
        "name": "לקוח בדיקה", "date_from": "28/04/2025",
    })
    assert "YYYY-MM-DD" in invalid["error"]
    reversed_range = activity._read_get_advertiser_airings({
        "name": "לקוח בדיקה", "date_from": "2025-04-29", "date_to": "2025-04-28",
    })
    assert "on or before" in reversed_range["error"]

    payload = activity._read_get_advertiser_airings({"name": "לקוח בדיקה"})
    assert payload["summary"]["airings"] == 0
    assert payload["coverage"]["rows_without_broadcast_day"] == 1
    assert payload["coverage"]["files_failed"]
    assert payload["coverage"]["complete_for_available_files"] is False
