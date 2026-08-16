"""What this surface discloses, pinned where it was measured to be silent.

Round 1 of the wave-two critic sweep found five things that were true of the
artifact and not on the screen or the terminal reading it, and one sentence that
was on the terminal and not true of the artifact. Each is a test here.

- The verdict act printed one sentence about the money for all three states, and
  on the stale state it was false: the record it writes carries the console's own
  ``revenue_delta`` beside ``money_state`` stale, measured at 948456.21 on a
  ``competitor`` verdict written against a copy of the store.
- ``--decision`` accepted the store's two keys only, on the one command where the
  steward chooses, while every screen this piece renders shows the pair as ship
  and no ship.
- The ``*`` beside every recorded verdict carried its meaning in a ``title``
  attribute, which a keyboard reaches never. Five of five rows carry the mark.
- Nothing on the board said whether anything had ever been adopted, while the
  column beside it said "Shipped" of a decision.
- The board printed ISO instants sliced to nineteen characters in four places,
  against a design rule that says dd/mm/yyyy in both locales and names one file
  that decides it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts import adopt_candidate_decide as verdict
from scripts import adopt_candidate_words as words

from tests.test_p12_board_harness import drive_board, read_board, served_payload

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates"

# An ISO instant as a payload carries it, which is what four places on this
# screen were printing at a reader.
ISO_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

# The shape shell/dates.js prints, which is the only shape a date may take.
ISRAELI_STAMP = re.compile(r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2}")


def test_every_money_state_a_record_can_hold_has_a_sentence_in_both_halves():
    for state in ("measured", "stale", "not_measured"):
        entry = words.RECORD_MONEY[state]
        assert entry["en"].strip() and entry["he"].strip(), state


def test_a_stale_verdict_does_not_claim_the_record_carries_no_figure():
    """The claim that was measured false, refused as a claim.

    The evidence block opens as the console's own, and that function writes
    ``revenue_delta`` from the stored measurement whenever one exists, letting
    ``money_state`` carry the staleness. So the record does carry a figure and
    only its state says not to use it.
    """
    lines = verdict.render({
        "candidate_id": "competitor", "decision": "not_shipped", "outcome": "ready",
        "checks": [], "money": {"state": "stale", "last_known_revenue_delta": 948456.21},
        "rescore_verdict": "not_distinguishable", "reason_is_hebrew": True,
    })
    money = [line for line in lines if line.startswith("Money ")]
    assert len(money) == 1, lines
    assert "rather than carrying a figure" not in money[0]
    assert words.RECORD_MONEY["stale"]["en"] in money[0]


def test_a_record_with_no_measurement_at_all_still_says_so():
    lines = verdict.render({
        "candidate_id": "competitor", "decision": "not_shipped", "outcome": "ready",
        "checks": [], "money": {"state": "not_measured"},
        "rescore_verdict": "not_distinguishable", "reason_is_hebrew": True,
    })
    money = [line for line in lines if line.startswith("Money ")][0]
    assert "carries no figure at all" in money


@pytest.mark.parametrize("typed,key", [
    ("ship", "shipped"), ("shipped", "shipped"),
    ("no-ship", "not_shipped"), ("no_ship", "not_shipped"),
    ("not-shipped", "not_shipped"), ("not_shipped", "not_shipped"),
    ("SHIP", "shipped"),
])
def test_the_verdict_flag_takes_the_word_the_screen_showed_and_records_the_store_key(typed, key):
    assert verdict.normalise_decision(typed) == key


def test_every_spelling_the_flag_offers_resolves_to_a_verdict_the_store_holds():
    for spelling in verdict.DECISION_CHOICES:
        assert verdict.normalise_decision(spelling) in verdict.DECISIONS


def test_the_two_words_the_screens_use_are_both_accepted_at_the_keyboard():
    """The divergence this closes, taken from the render tables themselves."""
    from scripts.adopt_candidate_render import DECISION_TAGS

    for key, shown in DECISION_TAGS.items():
        assert verdict.normalise_decision(shown.replace(" ", "-")) == key


def test_the_board_explains_its_own_mark_in_ink_rather_than_in_a_tooltip(tmp_path):
    board = read_board()
    marked = [row for row in board["candidates"]
              if (row.get("decision") or {}).get("state") and not (row.get("decision") or {}).get("on_rescore")]
    assert marked, "no marked verdict on this tree, so this proves nothing"
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(board)})
    body = result["body"]
    assert "*" in body
    # The sentence itself, on the page, not in an attribute.
    assert "different splits" in body or "פיצולים שונים" in body


def test_the_board_states_whether_anything_has_ever_been_adopted(tmp_path):
    board = read_board()
    assert all(row.get("adopted") is False for row in board["candidates"]), \
        "something is adopted on this tree, so the sentence under test is the wrong one"
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(board)}, locale="en")
    assert "No candidate adoption is recorded" in result["body"]
    assert "a recorded verdict does not replace it" in result["body"]


def test_the_board_says_how_many_artifacts_it_is_comparing(tmp_path):
    board = read_board()
    result = drive_board(tmp_path, {"/api/model/candidates": served_payload(board)}, locale="en")
    assert f"{len(board['candidates'])}" in result["body"]
    assert "candidate artifacts compared with the released artifact" in result["body"]


@pytest.mark.parametrize("locale", ["he", "en"])
def test_no_machine_timestamp_reaches_this_screen_in_either_locale(tmp_path, locale):
    """dd/mm/yyyy in both locales, from the one file that decides it.

    The window string is excluded deliberately: it is a pre-joined range on the
    payload rather than two ends a formatter can read, so it is the one raw ISO
    left on this screen and it is reported as a payload-shape change rather than
    silently accepted here.
    """
    board = read_board()
    result = drive_board(tmp_path / locale, {"/api/model/candidates": served_payload(board)}, locale=locale)
    assert not ISO_INSTANT.search(result["body"]), result["body"][:400]
    assert ISRAELI_STAMP.search(result["body"]), result["body"][:400]


def test_the_name_that_opens_an_artifact_looks_like_it_opens_something():
    css = (BOARD_DIR / "candidate-board.css").read_text(encoding="utf-8")
    resting = re.search(r"\.cb-pick \.cb-name \{[^}]*\}", css)
    assert resting, "the row name has no resting style, so its affordance is hover only"
    assert "text-decoration" in resting.group(0)


def test_the_unit_beside_a_file_size_is_a_word_with_both_halves():
    from_words = (BOARD_DIR / "board-words.js").read_text(encoding="utf-8")
    assert "'detail.bytes'" in from_words
    detail = (BOARD_DIR / "board-detail.jsx").read_text(encoding="utf-8")
    assert "bytes`" not in detail, "the unit is still a hardcoded English word in a template"
