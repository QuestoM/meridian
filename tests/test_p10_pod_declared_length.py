"""The declared break length against the sum of the pod's spots, driven to a figure.

On the shipped data this comparison is always unavailable, because the plan covers
2024-11-01 to 2024-11-30 and the one traffic file covers 2025-04-27. An honest
state is the right answer there and ``test_p10_pod_arithmetic.py`` asserts it, but
a state is not evidence that the arithmetic behind it works. A surface whose
central figure has never once been computed is a surface nobody has tested.

So this builds a traffic file on a day the plan does cover, over a break the plan
really places, and drives the shipped code path to a number. The plan's declared
length is read from the plan rather than written into the fixture, and the spots
are chosen to land on both sides of it, so a gap and an overflow are both measured
against a length this test never chose.

The fixture writes into a temporary folder and the module's own input directory is
pointed at it, so no file under ``data/daily_input`` is created, changed or read.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kairos_api import break_api_pod as pod

COLUMNS = [
    "תאריך", "שעה", "שעת התחלת ברייק", "משרד / MB", "סוג תשדיר", "מפרסם", "קמפיין",
    "שם גרסה", "House Number", "אורך תשדיר", "תוכנית מוזמנת", "שעת התחלת תוכנית",
    "סוג ברייק", "סוג תמחור", "מחיר", "רייטינג ברייקים מתוכנן", "מיקום בברייק", "סטטוס",
]


def _clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _planned_break():
    """One break the plan really places, on a day the plan really covers."""
    from kairos_api import break_store

    if not break_store.operator_channel():
        pytest.skip("no operator channel is configured, so no plan day can be built")
    days = break_store.plan_days()
    if not days:
        pytest.skip("no saved weekly plan covers the operator's channel")
    day = days[0]
    records = break_store.break_records(break_store.day_plan(day))
    inside = [row for row in records if float(row["start_seconds"]) + float(row["duration_seconds"]) < 86400]
    if not inside:
        pytest.skip(f"the plan for {day} places no break wholly inside the clock day")
    return day, inside[0]


def _write_traffic(folder: Path, day: str, break_start: float, lengths: list[int]) -> None:
    stamp = f"{int(day[5:7])}/{int(day[8:10])}/{day[:4]}"
    start = break_start + 1
    rows = []
    for index, length in enumerate(lengths):
        rows.append({
            "תאריך": stamp,
            "שעה": _clock(start),
            "שעת התחלת ברייק": _clock(break_start + 1),
            "משרד / MB": "בדיקה",
            "סוג תשדיר": "פרסומת",
            "מפרסם": f"מפרסם {index + 1}",
            "קמפיין": f"קמפיין {index + 1}",
            "שם גרסה": f"גרסה {index + 1}",
            "House Number": f"C{index:06d}",
            "אורך תשדיר": length,
            "תוכנית מוזמנת": "תוכנית בדיקה",
            "שעת התחלת תוכנית": _clock(break_start),
            "סוג ברייק": "Regular",
            "סוג תמחור": "CPP",
            "מחיר": "",
            "רייטינג ברייקים מתוכנן": 6.0,
            "מיקום בברייק": 99 if index == len(lengths) - 1 else index + 1,
            "סטטוס": "",
        })
        start += length
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"traffic_{day}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _pod_on(tmp_path, monkeypatch, day, break_start, lengths):
    folder = tmp_path / "daily_input"
    _write_traffic(folder, day, break_start, lengths)
    monkeypatch.setattr(pod, "DAILY_INPUT_DIR", folder)
    pods = pod.pods_for_day(day)
    assert len(pods) == 1, f"the fixture declared one pod and the module read {len(pods)}"
    return pods[0]


def test_the_declared_length_is_read_from_the_plan_and_the_gap_is_stated_in_seconds(tmp_path, monkeypatch):
    day, planned = _planned_break()
    declared_seconds = float(planned["duration_seconds"])
    lengths = [30, 30, 25]
    record = _pod_on(tmp_path, monkeypatch, day, float(planned["start_seconds"]), lengths)

    declared = record["declared_break_length"]
    assert declared["state"] == "real", f"the plan declares no length over this pod: {declared}"
    assert declared["seconds"] == round(declared_seconds, 1)
    assert declared["break_id"] == planned["break_id"]
    assert declared["basis"] and declared["basis_he"]

    against = record["against_declared"]
    assert against["state"] == "real"
    assert against["load_seconds"] == float(sum(lengths))
    assert against["declared_seconds"] == round(declared_seconds, 1)
    assert against["signed_seconds"] == round(declared_seconds - sum(lengths), 1)
    assert against["verdict"] == ("gap" if declared_seconds > sum(lengths) else "overflow")
    assert against["seconds"] == abs(round(declared_seconds - sum(lengths), 1))


def test_a_pod_that_oversells_the_declared_length_reads_as_an_overflow(tmp_path, monkeypatch):
    day, planned = _planned_break()
    declared_seconds = float(planned["duration_seconds"])
    over = int(declared_seconds) + 45
    record = _pod_on(tmp_path, monkeypatch, day, float(planned["start_seconds"]), [over])
    against = record["against_declared"]
    assert against["state"] == "real"
    assert against["verdict"] == "overflow"
    assert against["seconds"] == 45.0
    assert against["signed_seconds"] == -45.0


def test_a_pod_that_fills_the_declared_length_exactly_says_exactly(tmp_path, monkeypatch):
    day, planned = _planned_break()
    declared_seconds = int(float(planned["duration_seconds"]))
    half = declared_seconds // 2
    record = _pod_on(tmp_path, monkeypatch, day, float(planned["start_seconds"]), [half, declared_seconds - half])
    against = record["against_declared"]
    assert against["state"] == "real"
    assert against["verdict"] == "exact"
    assert against["seconds"] == 0.0


def test_the_planned_break_now_serves_its_own_contents_instead_of_declaring_them_unavailable(tmp_path, monkeypatch):
    """The state P3 left open, filled. This is the behaviour that did not exist.

    Before this piece ``contents`` was a fixed unavailable state on every break in
    the product, with no input that could ever change it. Here the same break
    serves the pod that covers its window, with its spots and its arithmetic.
    """
    day, planned = _planned_break()
    folder = tmp_path / "daily_input"
    _write_traffic(folder, day, float(planned["start_seconds"]), [30, 30, 25])
    monkeypatch.setattr(pod, "DAILY_INPUT_DIR", folder)

    contents = pod.contents_state(day, float(planned["start_seconds"]), float(planned["duration_seconds"]))
    assert contents["state"] == "real", f"contents came back {contents.get('state')}: {contents.get('reason')}"
    assert contents["pod"]["arithmetic"]["spot_count"] == 3
    assert contents["pod"]["arithmetic"]["declared_load"]["seconds"] == 85.0
    assert contents["covered_days"] == [day]


def test_a_break_on_a_day_no_traffic_file_covers_stays_a_state_and_names_the_covered_days(tmp_path, monkeypatch):
    day, planned = _planned_break()
    folder = tmp_path / "daily_input"
    _write_traffic(folder, day, float(planned["start_seconds"]), [30])
    monkeypatch.setattr(pod, "DAILY_INPUT_DIR", folder)

    contents = pod.contents_state("1999-01-01", 3600.0, 120.0)
    assert contents["state"] == "unavailable"
    assert contents["spots"] == []
    assert contents["covered_days"] == [day]
    assert contents["reason"] and contents["reason_he"]
    assert contents["path_forward"] and contents["path_forward_he"]
    assert "0" not in str(contents.get("seconds", ""))


def test_a_traffic_file_that_covers_the_day_but_not_this_break_says_which_of_the_two_it_is(tmp_path, monkeypatch):
    """The two absences are different absences, so they carry different reasons."""
    day, planned = _planned_break()
    folder = tmp_path / "daily_input"
    _write_traffic(folder, day, float(planned["start_seconds"]), [30])
    monkeypatch.setattr(pod, "DAILY_INPUT_DIR", folder)

    elsewhere = pod.contents_state(day, float(planned["start_seconds"]) + 7200.0, 120.0)
    assert elsewhere["state"] == "unavailable"
    assert elsewhere["covered_days"] == [day]
    assert "declares no break starting inside" in elsewhere["reason"]
    missing = pod.contents_state("1999-01-01", 3600.0, 120.0)
    assert missing["reason"] != elsewhere["reason"]
