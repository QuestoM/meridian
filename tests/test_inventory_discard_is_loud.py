"""A loader that throws every row away must say so, with the count and reason.

Measured 2026-08-09: ``load_inventory`` read all 994 rows of the shipped
``data/Spots - inventory.csv`` and discarded 100% of them, because every row
carries an empty ``hour_of_day``. The pool came back empty, so the inventory
placement steer sat at 1.0 everywhere -- indistinguishable from having no file
at all, while the file sat on disk looking like data.

The parse is deliberately NOT fixed here: repairing it would activate a lever and
move real money, so it is owner-gated. What is fixed is the silence.
"""

from __future__ import annotations

import logging

from kairos.optimize.inventory import load_inventory

HEADER = "Channel,Date_dt,hour_of_day\n"


def _write(tmp_path, body: str):
    target = tmp_path / "inventory.csv"
    target.write_text(HEADER + body, encoding="utf-8")
    return target


def test_total_discard_warns_with_count_and_reason(tmp_path, caplog) -> None:
    """The shipped file's exact failure: a real date, no hour, every row dropped."""
    target = _write(tmp_path, "ch1,2024-11-04,\nch1,2024-11-04,\nch1,2024-11-05,\n")
    with caplog.at_level(logging.WARNING):
        pool = load_inventory(target)

    assert pool == {}
    message = caplog.text
    assert "discarded ALL 3 rows" in message, "the count must be in the message"
    assert "3 on hour" in message, "the failing FIELD must be named, not just the count"
    assert "inert" in message, "the consequence must be stated, not left to inference"


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
