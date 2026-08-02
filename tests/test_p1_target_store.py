"""The plan target store: a number a person sets, and the verdict it enables.

The whole point of this store is that it starts empty and stays empty until
somebody supplies a figure. So the first thing these tests pin is the absence:
no target, no verdict, no number anywhere. Then the round trip, then the
validation that keeps an unusable value out, then the three states and the
boundary between them.

The store path is redirected through ``KAIROS_PLAN_TARGETS_PATH`` so nothing
here writes the deployed file.
"""

from __future__ import annotations

import csv

import pytest
from fastapi import HTTPException

from kairos_api import target_store

CHANNEL = "רשת 13"
START = "2024-11-01"
END = "2024-11-07"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    path = tmp_path / "plan_targets.csv"
    monkeypatch.setenv(target_store.PATH_ENV, str(path))
    return path


def test_the_shipped_store_is_header_only_so_no_target_is_asserted_anywhere():
    """The file checked into the repository holds columns and zero targets."""
    with target_store.TARGETS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []
    assert target_store.TARGETS_PATH.read_text(encoding="utf-8").strip() == ",".join(target_store.COLUMNS)


def test_an_empty_store_reads_as_no_target_and_never_as_zero(store):
    assert target_store.read_all() == []
    assert target_store.target_for(CHANNEL, START, END) is None
    body = target_store.payload(CHANNEL, START, END, None)
    assert body["state"] == "unset"
    assert body["amount_ils"] is None
    assert body["at_risk_band_percent"] is None


def test_a_read_does_not_create_the_file(store):
    target_store.read_all()
    target_store.payload(CHANNEL, START, END, None)
    assert not store.exists()


def test_save_then_read_returns_exactly_what_was_supplied(store):
    saved = target_store.save_target(CHANNEL, START, END, 9_500_000, 5, note="quarterly")
    assert saved["amount_ils"] == 9_500_000.0
    assert saved["at_risk_band_percent"] == 5.0
    read = target_store.target_for(CHANNEL, START, END)
    assert read["amount_ils"] == 9_500_000.0
    assert read["at_risk_band_percent"] == 5.0
    assert read["note"] == "quarterly"
    assert read["set_at"]


def test_saving_the_same_window_twice_replaces_rather_than_duplicates(store):
    target_store.save_target(CHANNEL, START, END, 9_500_000, 5)
    target_store.save_target(CHANNEL, START, END, 9_900_000, 4)
    rows = target_store.read_all()
    assert len(rows) == 1
    assert rows[0]["amount_ils"] == 9_900_000.0


def test_a_target_for_another_window_is_not_returned_for_this_one(store):
    target_store.save_target(CHANNEL, "2024-11-08", "2024-11-14", 9_500_000, 5)
    assert target_store.target_for(CHANNEL, START, END) is None
    assert len(target_store.targets_for_channel(CHANNEL)) == 1


def test_delete_removes_the_row_and_reports_whether_it_did(store):
    target_store.save_target(CHANNEL, START, END, 9_500_000, 5)
    assert target_store.delete_target(CHANNEL, START, END) is True
    assert target_store.delete_target(CHANNEL, START, END) is False
    assert target_store.target_for(CHANNEL, START, END) is None


@pytest.mark.parametrize(
    "amount, band",
    [(0, 5), (-1, 5), ("", 5), (9_500_000, -1), (9_500_000, 101), (9_500_000, "")],
)
def test_an_unusable_number_is_refused_and_nothing_is_written(store, amount, band):
    with pytest.raises(HTTPException) as raised:
        target_store.save_target(CHANNEL, START, END, amount, band)
    assert raised.value.status_code == 400
    assert target_store.read_all() == []


def test_a_window_that_ends_before_it_starts_is_refused(store):
    with pytest.raises(HTTPException):
        target_store.save_target(CHANNEL, END, START, 9_500_000, 5)


def test_an_unsupported_metric_is_refused_rather_than_stored_unmeasurable(store):
    with pytest.raises(HTTPException):
        target_store.save_target(CHANNEL, START, END, 9_500_000, 5, metric="grp")


def test_no_target_means_the_verdict_is_unavailable_with_its_reason():
    verdict = target_store.verdict(10_123_070.8, None)
    assert verdict["state"] == "unavailable"
    assert verdict["reason"] == "no_target"
    assert verdict["variance_ils"] is None
    assert verdict["variance_percent"] is None


def test_no_projection_means_unavailable_even_when_a_target_exists():
    target = {"amount_ils": 9_500_000.0, "at_risk_band_percent": 5.0}
    verdict = target_store.verdict(None, target)
    assert verdict["state"] == "unavailable"
    assert verdict["reason"] == "no_projection"


@pytest.mark.parametrize(
    "projected, expected",
    [
        (9_500_000, "on_plan"),
        (10_000_000, "on_plan"),
        (9_499_999, "at_risk"),
        (9_025_000, "at_risk"),
        (9_024_999, "behind"),
        (1_000_000, "behind"),
    ],
)
def test_the_three_states_and_the_boundary_between_them(projected, expected):
    """A five percent band puts the at-risk floor at 9,025,000 exactly."""
    target = {"amount_ils": 9_500_000.0, "at_risk_band_percent": 5.0}
    assert target_store.verdict(projected, target)["state"] == expected


def test_the_verdict_publishes_the_threshold_it_was_decided_by():
    target = {"amount_ils": 9_500_000.0, "at_risk_band_percent": 5.0}
    verdict = target_store.verdict(9_400_000, target)
    assert "5 percent" in verdict["threshold_en"]
    assert "5 אחוז" in verdict["threshold_he"]
    assert verdict["variance_ils"] == -100_000.0
    assert verdict["variance_percent"] == pytest.approx(-1.05, abs=0.01)


def test_the_stored_file_keeps_its_columns_and_stays_parseable(store):
    target_store.save_target(CHANNEL, START, END, 9_500_000, 5)
    with store.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames) == list(target_store.COLUMNS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["channel"] == CHANNEL
    assert rows[0]["metric"] == "projected_revenue"


def test_a_redirected_store_backs_up_into_its_own_tree_and_never_the_repository(store, tmp_path):
    """A test writing the store must not leave a file in the repository."""
    repo_backups = target_store.BACKUP_DIR
    before = sorted(repo_backups.glob("plan_targets_*.csv")) if repo_backups.exists() else []
    target_store.save_target(CHANNEL, START, END, 9_500_000, 5)
    target_store.save_target(CHANNEL, START, END, 9_600_000, 5)
    after = sorted(repo_backups.glob("plan_targets_*.csv")) if repo_backups.exists() else []
    assert after == before
    assert sorted((tmp_path / "_backups").glob("plan_targets_*.csv"))
