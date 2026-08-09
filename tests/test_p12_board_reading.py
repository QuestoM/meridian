"""How the board READS: the model version, and the money.

Two defects of one kind, measured in a real browser on the real panel. Both are
about a value that is correct in the payload and wrong on the screen, which is
the only kind of defect a page like this can still have once its figures are
right.

**The model version.** It is stored as `2026-07-29`, a calendar day, and three
sites in this piece printed it back exactly as stored. dd/mm/yyyy is the format
an Israeli operator reads and shell/dates.js is the one file that decides it, so
the day is read out through that file, in both locales, and only when the stored
name is a day at all.

**The money.** One renderer printed a current figure at two decimals and a stale
one at none, in the same block, so a reader comparing two rows saw a difference
in shape that was about the reading's state rather than about the money, and the
exact stale figure was on no screen. One form now, with the state still said in
words beside it.
"""

from __future__ import annotations

import copy
import re

from tests.test_p12_board_harness import (
    BOARD_DIR,
    drive_board,
    read_board,
    served_payload,
)

# dd/mm/yyyy anywhere in a run of text. Used to prove a date is NOT invented for
# a version name that is not a day, so it is deliberately loose.
A_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

# The stored name on this tree, and what an Israeli operator reads it as.
STORED_VERSION = "2026-07-29"
READ_VERSION = "29/07/2026"


def _money(board, row_id, money):
    """The published board with one row's money block replaced."""
    doctored = copy.deepcopy(board)
    for row in doctored["candidates"]:
        if row["id"] == row_id:
            row["money"] = money
    return doctored


def test_the_model_version_reads_as_an_israeli_date_and_never_as_iso(tmp_path):
    """The last raw ISO string on this board, in both locales.

    The version name is a calendar day and it was printed as stored. The date
    guard could not catch it: its raw-ISO rule enumerates calendar-day payload
    FIELDS and a version NAME is not one of them, which is why this is measured
    on the rendered page rather than asserted against the source.
    """
    board = read_board()
    assert board["shipped"]["version_name"] == STORED_VERSION, "the fixture moved"
    for locale in ("he", "en"):
        result = drive_board(tmp_path / locale,
                             {"/api/model/candidates": served_payload(board)},
                             locale=locale)
        assert READ_VERSION in result["reference_note"], result["reference_note"]
        # Not merely on the reference row: nowhere on the page, because the log
        # line under the live verdict prints the same stored value.
        assert STORED_VERSION not in result["body"], [
            line for line in result["body"].splitlines() if STORED_VERSION in line]
        assert READ_VERSION in result["live_block"], result["live_block"]


def test_a_version_name_that_is_not_a_day_is_printed_exactly_as_it_was_stored(tmp_path):
    """Honest math on a string: a name is read as a day only when it is one.

    Nothing constrains a model version to be named after a calendar day, and a
    formatter that invents a date for `retention-v3` would be a worse defect
    than the one it replaced. This is measured against a payload rather than the
    published file because the published file carries exactly one version name.
    """
    board = read_board()
    doctored = copy.deepcopy(board)
    doctored["shipped"]["version_name"] = "retention-v3"
    doctored["decision_log"]["version_name"] = "retention-v3"
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(doctored)},
                         locale="en", board=doctored)
    assert "retention-v3" in result["reference_note"], result["reference_note"]
    assert not A_DATE.search(result["reference_note"]), result["reference_note"]


def test_a_model_version_with_no_name_says_so_rather_than_showing_a_blank(tmp_path):
    """Absent is not empty. It was rendered as an empty code span at two sites."""
    board = read_board()
    doctored = copy.deepcopy(board)
    doctored["shipped"].pop("version_name", None)
    doctored["decision_log"].pop("version_name", None)
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(doctored)},
                         locale="en", board=doctored)
    assert "No version name recorded" in result["reference_note"], result["reference_note"]
    assert "No version name recorded" in result["live_block"], result["live_block"]
    assert not A_DATE.search(result["reference_note"]), result["reference_note"]


def test_the_exact_stale_figure_is_on_the_screen_and_not_only_at_the_terminal(tmp_path):
    """+902,998.61, which the screen rounded to +902,999 and no reader could recover."""
    board = read_board()
    stale = [row for row in board["candidates"]
             if (row.get("money") or {}).get("state") == "stale"]
    assert stale, "no stale money on this tree, so this proves nothing"
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(board)},
                         locale="en")
    for row in stale:
        exact = f'{row["money"]["last_known_revenue_delta"]:,.2f}'
        assert exact in result["body"], (row["id"], exact)
    # And the state is still stated, in the one channel that can carry it
    # without destroying the figure.
    assert "Stale" in result["body"]


def test_current_and_stale_money_are_printed_to_one_precision(tmp_path):
    """The defect was two forms in one block, so it is measured with both present.

    On the published tree every row is stale, which is exactly the arrangement
    that hid this: a page with only one state cannot show the disagreement. One
    row is made current so the table holds both at once.
    """
    board = read_board()
    stale_row = next(row for row in board["candidates"]
                     if (row.get("money") or {}).get("state") == "stale")
    other = next(row for row in board["candidates"] if row["id"] != stale_row["id"])
    doctored = _money(board, other["id"], {
        "state": "measured",
        "revenue_delta": 902998.61,
        "whole_plan_delta": 1234567.89,
        "rows": 4321,
        "basis": "one basis",
        "measured_at": board["measured_at"],
    })
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(doctored)},
                         locale="en", board=doctored)
    cells = [cell for cell in result["money_cells"] if "₪" in cell]
    assert len(cells) >= 2, result["money_cells"]
    # Every shekel figure on the table carries exactly two decimals, whatever
    # the freshness of the reading it came from.
    for cell in cells:
        for figure in re.findall(r"[-+]?[\d,]+(?:\.\d+)?(?= ₪)", cell):
            assert re.search(r"\.\d{2}$", figure), (cell, figure)


def test_no_call_site_can_choose_a_second_money_form(tmp_path):
    """The knob is gone, not defaulted.

    A rule that lives in a default argument is a rule any later call site can
    opt out of in one word, which is how this defect arrived. Shekels takes no
    digits argument at all now, and this reads the source because that is where
    the property lives.
    """
    detail = (BOARD_DIR / "board-detail.jsx").read_text(encoding="utf-8")
    signature = re.search(r"export function Shekels\(\{([^}]*)\}\)", detail)
    assert signature, detail[:200]
    assert "digits" not in signature.group(1), signature.group(1)
    for name in ("board-detail.jsx", "CandidateBoard.jsx"):
        source = (BOARD_DIR / name).read_text(encoding="utf-8")
        for call in re.findall(r"<Shekels[^/]*/>", source):
            assert "digits" not in call, (name, call)
