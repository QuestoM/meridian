"""P13: the technical verdict on a commercial's own file.

THE SHIPPED STORE IS HEADER-ONLY AND THAT IS WHY THIS FILE IS SHAPED THIS WAY.
Nothing in this repository observes a media file, so every fact is honestly
unavailable today. An empty store makes almost any assertion pass for free, so
every test that matters here supplies a real row and shows the verdict MOVES.
The empty case is asserted too, but on its own it would prove nothing.

The rule under test, and the one that would do damage if it were ever inverted:
unavailable is not a pass and is not a failure. A file nobody inspected has not
been cleared to air, so it can never read `verified`; it has not been found wrong
either, so it must not read `failed` and must not block a lock. If absence
blocked, nothing could be locked at all today, and a product that refuses
everything has not done the job JS-8 describes.
"""

from __future__ import annotations

import pytest

from kairos_api.media_store import (
    ASSETS_PATH,
    COLUMNS,
    FAILED,
    UNAVAILABLE,
    VERIFIED,
    read_assets,
    write_assets,
)
from kairos_api.media_verdict import verdict_for, verdicts_for


def _asset(**over):
    row = {
        "creative_id": "CID100",
        "duration_seconds": 30.0,
        "container_format": "mxf",
        "aspect_ratio": "16:9",
        "has_audio": "true",
        "measured_at": "2026-08-10T00:00:00Z",
        "source": "test",
    }
    row.update(over)
    return row


# 1. The shipped store, and the honest empty ---------------------------------
def test_the_shipped_store_is_a_header_and_not_a_fabricated_row() -> None:
    """A seeded media row would clear a file nobody inspected. There are none."""
    assert ASSETS_PATH.exists()
    assert read_assets() == []
    header = ASSETS_PATH.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.split(",") == list(COLUMNS)


def test_an_uninspected_commercial_is_unavailable_and_does_not_block() -> None:
    result = verdict_for("CID_NOBODY_MEASURED", booked_seconds=30.0, assets={})
    assert result["state"] == UNAVAILABLE
    assert result["blocks_lock"] is False, (
        "absence blocked the lock, so nothing could ever be locked while the "
        "media store is header-only, which is every pod shipping today"
    )
    assert {fact["state"] for fact in result["facts"].values()} == {UNAVAILABLE}
    assert result["reason_he"] and result["reason"]


# 2. THE INSTRUMENT MOVES ----------------------------------------------------
def test_a_clean_file_verifies() -> None:
    table = {"CID100": _asset() | {"duration_seconds": 30.0, "has_audio": True}}
    result = verdict_for("CID100", booked_seconds=30.0, assets=table)
    assert result["state"] == VERIFIED
    assert result["blocks_lock"] is False
    assert {fact["state"] for fact in result["facts"].values()} == {VERIFIED}


@pytest.mark.parametrize(
    "over, broken",
    [
        ({"duration_seconds": 25.0}, "duration"),
        ({"container_format": "avi"}, "format"),
        ({"aspect_ratio": "4:3"}, "aspect_ratio"),
        ({"has_audio": False}, "audio"),
    ],
)
def test_each_fact_can_fail_on_its_own_and_blocks(over, broken) -> None:
    """Four separate probes. One combined case would let three of the four be
    dead and still pass, which is the failure mode this campaign keeps paying
    for: a check nobody has seen fire."""
    row = _asset() | {"has_audio": True}
    row.update(over)
    result = verdict_for("CID100", booked_seconds=30.0, assets={"CID100": row})
    assert result["state"] == FAILED
    assert result["blocks_lock"] is True
    assert result["facts"][broken]["state"] == FAILED
    assert result["facts"][broken]["reason_he"], "a failure names its cause in Hebrew"
    others = [name for name in result["facts"] if name != broken]
    assert all(result["facts"][name]["state"] == VERIFIED for name in others), (
        "one broken fact took its neighbours down with it, so the verdict cannot "
        "say WHICH thing is wrong with the file"
    )


def test_a_measured_file_with_nothing_to_compare_it_to_does_not_fail() -> None:
    """Measured, but the booking carries no duration. The file is not wrong."""
    result = verdict_for("CID100", booked_seconds=None, assets={"CID100": _asset()})
    assert result["facts"]["duration"]["state"] == UNAVAILABLE
    assert result["facts"]["duration"]["measured_seconds"] == 30.0
    assert result["blocks_lock"] is False


def test_an_unparseable_duration_is_unknown_and_never_zero(tmp_path) -> None:
    path = tmp_path / "media_assets.csv"
    write_assets([_asset(duration_seconds="not a number")], path)
    from kairos_api import media_store

    asset = media_store.assets_by_creative(path)["CID100"]
    assert asset["duration_seconds"] is None, "a malformed figure read as a number"
    result = verdict_for("CID100", booked_seconds=30.0, assets={"CID100": asset})
    assert result["facts"]["duration"]["state"] == UNAVAILABLE
    assert result["blocks_lock"] is False


# 3. The pod, and what its lock does -----------------------------------------
def test_a_pod_blocks_only_on_a_measured_failure() -> None:
    spots = [
        {"creative": {"value": "CID100"}, "duration": {"seconds": 30.0}},
        {"creative": {"value": "CID_NOBODY_MEASURED"}, "duration": {"seconds": 20.0}},
    ]
    empty = verdicts_for(spots)
    assert empty["blocks_lock"] is False
    assert empty["counts"][UNAVAILABLE] == 2
    assert empty["assets_on_file"] == 0


def test_the_round_trip_keeps_a_column_the_writer_never_heard_of(tmp_path) -> None:
    """The store-column rule: a writer is not the authority on which columns
    exist, because migrations and seeds add them and never edit writers."""
    path = tmp_path / "media_assets.csv"
    write_assets([_asset(loudness_lufs="-23")], path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith(",".join(COLUMNS))
    assert "loudness_lufs" in header
