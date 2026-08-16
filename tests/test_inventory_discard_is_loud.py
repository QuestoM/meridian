"""Inventory input is usable, scoped, and loud when malformed.

The shipped replay is rebuilt from the canonical historical spot log for the
saved operator channel. Synthetic malformed fixtures preserve the fail-closed
contract: a present file that loses every temporal coordinate must never pass
for a neutral or absent signal.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd
import pytest

from kairos.optimize.inventory import InventoryInputError, load_inventory
from kairos.data.loaders import load_spots

HEADER = "Channel,Date_dt,hour_of_day\n"


def _write(tmp_path, body: str):
    target = tmp_path / "inventory.csv"
    target.write_text(HEADER + body, encoding="utf-8")
    return target


def test_shipped_inventory_exactly_replays_the_saved_operator_source() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "data" / "Spots - inventory.csv"
    settings = json.loads((root / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
    operator = settings["operator_channel"]

    pool = load_inventory(target, require_usable=True)
    actual = {key: slot.booked for key, slot in pool.items()}

    spots = load_spots()
    owned = spots.loc[spots["Channel"].astype(str).str.strip().eq(operator)].copy()
    assert owned["air_dt"].notna().all()
    expected = (
        owned.assign(
            day=owned["air_dt"].dt.strftime("%Y-%m-%d"),
            hour=owned["air_dt"].dt.hour,
        )
        .groupby(["Channel", "day", "hour"], sort=True)
        .size()
        .to_dict()
    )

    assert actual == expected
    assert {slot.channel for slot in pool.values()} == {operator}
    assert len(pool) == 682
    assert sum(slot.booked for slot in pool.values()) == 18_669
    assert len({slot.day for slot in pool.values()}) == 30


def test_total_discard_warns_with_count_and_reason(tmp_path, caplog) -> None:
    """A real date with no hour is rejected row by row and explained."""
    target = _write(tmp_path, "ch1,2024-11-04,\nch1,2024-11-04,\nch1,2024-11-05,\n")
    with caplog.at_level(logging.WARNING):
        pool = load_inventory(target)

    assert pool == {}
    message = caplog.text
    assert "discarded ALL 3 rows" in message, "the count must be in the message"
    assert "3 on hour" in message, "the failing FIELD must be named, not just the count"
    assert "inert" in message, "the consequence must be stated, not left to inference"


def test_money_moving_run_mode_refuses_a_present_file_that_yields_nothing(tmp_path) -> None:
    target = _write(tmp_path, "ch1,2024-11-04,\nch1,2024-11-05,\n")

    with pytest.raises(InventoryInputError, match="produced no usable"):
        load_inventory(target, require_usable=True)


def test_saved_plan_run_refuses_before_it_builds_or_writes(tmp_path, monkeypatch) -> None:
    from kairos.optimize import inventory as inventory_module
    from kairos_api import recompute_api

    target = _write(tmp_path, "ch1,2024-11-04,\nch1,2024-11-05,\n")
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", target)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("the invalid input must stop the run before plan work")

    monkeypatch.setattr(recompute_api, "build_weekly_schedule", should_not_run)
    monkeypatch.setattr(recompute_api, "write_weekly_schedule", should_not_run)

    with pytest.raises(InventoryInputError, match="saved plan was not changed"):
        recompute_api._run_recompute()


def test_async_run_refuses_before_it_creates_a_job(tmp_path, monkeypatch) -> None:
    from fastapi import HTTPException
    from kairos.optimize import inventory as inventory_module
    from kairos_api import jobs, recompute_api

    target = _write(tmp_path, "ch1,2024-11-04,\n")
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", target)
    monkeypatch.setattr(recompute_api, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(jobs, "running_job", lambda _kind: None)

    def should_not_submit(*_args, **_kwargs):
        raise AssertionError("an invalid input must not create a background job")

    monkeypatch.setattr(jobs, "submit", should_not_submit)

    with pytest.raises(HTTPException) as raised:
        recompute_api.start_recompute_job()
    assert raised.value.status_code == 422
    assert "produced no usable" in str(raised.value.detail)


def test_common_shipped_writer_rechecks_inventory_at_commit(tmp_path, monkeypatch) -> None:
    from kairos.export import schedule
    from kairos.optimize import inventory as inventory_module

    source = _write(tmp_path, "ch1,2024-11-04,\n")
    output = tmp_path / "weekly_break_schedule.csv"
    frame = pd.DataFrame(
        [{column: "" for column in schedule.COLUMNS}], columns=schedule.COLUMNS
    )
    monkeypatch.delenv("KAIROS_PLAN_READONLY", raising=False)
    monkeypatch.setattr(inventory_module, "DEFAULT_INVENTORY_PATH", source)
    monkeypatch.setattr(schedule, "DEFAULT_OUTPUT_PATH", output)

    with pytest.raises(InventoryInputError, match="saved plan was not changed"):
        schedule.write_weekly_schedule(frame=frame, replace_shipped_plan=True)
    assert not output.exists()
    assert not Path(str(output) + ".writes.meta.json").exists(), (
        "a refused plan is not a write and must leave no provenance mutation"
    )


def test_partial_discard_warns_and_keeps_the_good_rows(tmp_path, caplog) -> None:
    target = _write(tmp_path, "ch1,2024-11-04,9\nch1,2024-11-04,\n")
    with caplog.at_level(logging.WARNING):
        pool = load_inventory(target)

    assert len(pool) == 1
    assert "discarded 1 of 2 rows" in caplog.text
    assert "kept 1 slots" in caplog.text


def test_a_clean_file_says_nothing(tmp_path, caplog) -> None:
    """No warning when nothing was thrown away, or the signal becomes noise."""
    target = _write(tmp_path, "ch1,2024-11-04,9\nch1,2024-11-04,10\n")
    with caplog.at_level(logging.WARNING):
        pool = load_inventory(target)

    assert len(pool) == 2
    assert "discarded" not in caplog.text


def test_the_warning_never_names_a_channel(tmp_path, caplog) -> None:
    """Competitor boundary: this file carries other broadcasters' rows.

    Counts may travel; rival names may not. The loader reports how many rows
    failed, never whose they were.
    """
    target = _write(tmp_path, "RIVALCHANNEL,2024-11-04,\nRIVALCHANNEL,2024-11-05,\n")
    with caplog.at_level(logging.WARNING):
        load_inventory(target)

    assert "RIVALCHANNEL" not in caplog.text
