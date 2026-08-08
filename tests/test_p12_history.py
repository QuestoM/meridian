"""Every verdict on record, not the newest one, and the two the join dropped.

JS-19's done condition has two halves and the second one is "a later reader can
see what was tried". Before this module every read of the decision log on this
piece kept ``taken[0]`` and filtered ``subject == "candidate"``, so on the tree
this was written against two of the seven records reached no surface at all: the
earlier of two verdicts on one candidate, and the one verdict whose subject is
the live model itself.

The tests are written against records rather than against the store, through the
``records`` seam, so a reading can be measured on a log this repository does not
happen to hold. Four of them are taken on the real tree, because the finding is
about this tree and an assertion about a fixture would not have found it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adopt_candidate_history as history
from scripts import adopt_candidate_registry as registry

ROOT = Path(__file__).resolve().parents[1]
BOARD_JSON = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates" / "candidate-board.json"
DECISIONS = ROOT / "models" / "releases" / "decisions.jsonl"

VERSION = "mv-2026-07-29-35cc2e21"


def _record(**overrides):
    base = {
        "decision_id": "md-1",
        "recorded_at": "2026-08-01T10:00:00+00:00",
        "actor": "steward",
        "model_version_id": VERSION,
        "model_version_name": "2026-07-29",
        "subject": "candidate",
        "candidate_id": "twin",
        "decision": "not_shipped",
        "reason": "not far enough",
        "release_note_he": "",
        "evidence": {},
    }
    base.update(overrides)
    return base


def _log(records, known=("twin",)):
    """Newest first, which is the order the store's own reader returns."""
    return history.decision_log(known, version_id=VERSION, version_name="2026-07-29",
                                records=list(records))


# ---------------------------------------------------------------------------
# The reading a count cannot give
# ---------------------------------------------------------------------------


def test_a_candidate_with_no_verdict_carries_the_empty_state_and_not_a_missing_key():
    log = _log([])
    block = history.history_for(log, "twin")
    assert block["state"] == "none"
    assert block["rows"] == []
    assert block["count"] == 0
    assert block["reading_en"] and block["reading_he"]


def test_one_verdict_reads_as_one_and_not_as_a_number():
    block = history.history_for(_log([_record()]), "twin")
    assert block["state"] == "one"
    assert block["not_shown_by_the_latest"] == 0


def test_the_same_word_twice_for_the_same_reason_is_a_repeat():
    block = history.history_for(_log([
        _record(decision_id="md-2", recorded_at="2026-08-02T10:00:00+00:00"),
        _record(decision_id="md-1"),
    ]), "twin")
    assert block["state"] == "repeated"


def test_the_same_word_twice_for_two_reasons_is_a_restatement_and_not_a_repeat():
    """The defect this module exists for, in one assertion.

    A column reading "no ship (2)" says an artifact was refused twice. On the
    real tree the first refusal was for want of a current measurement and the
    second was on the measurement, which are two different kinds of no, and only
    the second is a verdict about the model.
    """
    block = history.history_for(_log([
        _record(decision_id="md-2", recorded_at="2026-08-02T10:00:00+00:00",
                reason="the figures move too little"),
        _record(decision_id="md-1", reason="no current measurement to decide on"),
    ]), "twin")
    assert block["state"] == "restated"
    assert block["not_shown_by_the_latest"] == 1
    assert "2" in block["reading_en"] and "2" in block["reading_he"]


def test_two_different_words_are_a_reversal():
    block = history.history_for(_log([
        _record(decision_id="md-2", recorded_at="2026-08-02T10:00:00+00:00", decision="shipped"),
        _record(decision_id="md-1"),
    ]), "twin")
    assert block["state"] == "reversed"


def test_the_newest_is_in_force_and_every_earlier_one_names_what_replaced_it():
    block = history.history_for(_log([
        _record(decision_id="md-3", recorded_at="2026-08-03T10:00:00+00:00"),
        _record(decision_id="md-2", recorded_at="2026-08-02T10:00:00+00:00"),
        _record(decision_id="md-1"),
    ]), "twin")
    assert [row["in_force"] for row in block["rows"]] == [True, False, False]
    assert [row["superseded_by"] for row in block["rows"]] == [None, "md-3", "md-2"]


# ---------------------------------------------------------------------------
# The subject every read filtered out
# ---------------------------------------------------------------------------


def test_a_verdict_on_the_live_model_is_not_a_candidate_row_and_is_not_dropped():
    log = _log([_record(subject="current", candidate_id=None, decision_id="md-live")])
    assert log["live_model"]["count"] == 1
    assert log["live_model"]["state"] == "standing"
    assert history.history_for(log, "twin")["count"] == 0


def test_a_live_model_verdict_against_an_older_version_is_not_standing():
    log = _log([_record(subject="current", candidate_id=None, decision_id="md-live",
                        model_version_id="mv-older", model_version_name="2026-01-01")])
    assert log["live_model"]["state"] == "superseded"
    assert log["live_model"]["rows"][0]["against_version_in_force"] is False
    assert log["against_another_version"] == ["md-live"]
    assert log["all_against_version_in_force"] is False


def test_no_live_model_verdict_is_an_empty_state_and_never_a_silence():
    log = _log([_record()])
    assert log["live_model"]["state"] == "none"
    assert log["live_model"]["reading_en"] and log["live_model"]["reading_he"]


# ---------------------------------------------------------------------------
# Accounting for the whole of an append-only file
# ---------------------------------------------------------------------------


def test_every_record_in_the_log_is_accounted_for_by_the_tally():
    records = [_record(decision_id="md-1"),
               _record(decision_id="md-2", candidate_id="gone"),
               _record(decision_id="md-3", subject="current", candidate_id=None)]
    log = _log(records)
    tally = log["tally"]
    assert tally["on_the_shelf"] + tally["on_the_live_model"] + tally["off_the_shelf"] == log["records"]
    assert log["off_the_shelf"] == ["gone"]


def test_a_verdict_about_an_artifact_that_is_not_on_the_shelf_is_named_rather_than_lost():
    log = _log([_record(candidate_id="gone")], known=("twin",))
    assert log["off_the_shelf"] == ["gone"]
    assert history.history_for(log, "twin")["state"] == "none"


# ---------------------------------------------------------------------------
# What may not reach a browser bundle
# ---------------------------------------------------------------------------


def test_the_board_copy_carries_no_steward_sentence_on_any_row():
    block = history.history_for(_log([_record(reason="a sentence somebody typed")]), "twin")
    assert block["rows"][0]["reason"] == "a sentence somebody typed"
    published = history.for_the_board(block)
    assert all("reason" not in row for row in published["rows"])
    assert "a sentence somebody typed" not in json.dumps(published, ensure_ascii=False)
    assert published["reason_en"] and published["reason_he"]


def test_the_live_board_copy_carries_the_console_s_own_name_for_its_subject():
    published = history.for_the_board_live(
        _log([_record(subject="current", candidate_id=None)])["live_model"])
    assert published["subject_en"] == "the shipped model"
    assert published["subject_he"] == "המודל המשודר"


@pytest.mark.skipif(not DECISIONS.is_file(), reason="no decision log on this tree")
def test_no_reason_from_the_real_decision_log_appears_in_the_published_board():
    """The rule the published board has always held, now that it carries history.

    The board is imported by a browser bundle and the wall on this act is a
    route wall, not a bundler one. Every reason in the store is checked against
    the file rather than the shape being trusted.
    """
    published = BOARD_JSON.read_text(encoding="utf-8")
    reasons = [str(json.loads(line).get("reason") or "").strip()
               for line in DECISIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    reasons = [reason for reason in reasons if len(reason) > 20]
    assert reasons, "no reason in the log is long enough to search for, so this proves nothing"
    assert [reason for reason in reasons if reason in published] == []


# ---------------------------------------------------------------------------
# The tree this was found on
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DECISIONS.is_file(), reason="no decision log on this tree")
def test_every_record_on_this_tree_reaches_a_surface_of_this_piece():
    """The measurement that started this round, as a standing guard.

    Before it, five of the seven records on this tree reached the published
    board and two reached nothing: the earlier verdict on one candidate, and the
    verdict whose subject is the live model. It is asserted over whatever the log
    holds rather than over seven, because the log is append-only.
    """
    payload = registry.registry()
    log = payload["decision_log"]
    seen = {row["decision_id"] for block in log["candidates"].values() for row in block["rows"]}
    seen |= {row["decision_id"] for row in log["live_model"]["rows"]}
    stored = {json.loads(line)["decision_id"]
              for line in DECISIONS.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert stored, "the decision log is empty, so this proves nothing"
    assert stored - seen == set()


@pytest.mark.skipif(not DECISIONS.is_file(), reason="no decision log on this tree")
def test_the_terminal_prints_every_verdict_and_not_only_the_newest():
    from scripts import adopt_candidate_render as render

    payload = registry.registry()
    text = "\n".join(render.render(payload))
    for block in payload["decision_log"]["candidates"].values():
        for row in block["rows"]:
            assert str(row["recorded_at"])[:19] in text, row["decision_id"]
    for row in payload["decision_log"]["live_model"]["rows"]:
        assert str(row["recorded_at"])[:19] in text


def test_the_registry_row_and_its_history_cannot_disagree_about_the_count():
    """One reader for one log.

    The join used to walk the decision log itself for the newest verdict while
    the history reading walked it again, and two readers of one append-only file
    is how two surfaces of one piece come to hold different verdicts.
    """
    payload = registry.registry()
    for row in payload["candidates"]:
        assert row["decisions"] == row["history"]["count"]
        if row["history"]["rows"]:
            assert row["latest_decision"]["decision_id"] == row["history"]["rows"][0]["decision_id"]
            assert row["decision_on_rescore"] == row["history"]["rows"][0]["on_rescore"]
        else:
            assert row["latest_decision"] is None


@pytest.mark.skipif(not BOARD_JSON.is_file(), reason="the board has not been published")
def test_the_published_board_carries_the_whole_log_and_no_row_carries_a_sentence():
    """The shape the contract freezes, checked on the file a browser imports."""
    board = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    live = board["live_model"]
    assert live["subject_en"] and live["subject_he"]
    assert live["reason_en"] and live["reason_he"]
    log = board["decision_log"]
    assert set(log) == {"records", "tally", "off_the_shelf",
                        "all_against_version_in_force", "version_name"}
    for row in board["candidates"]:
        block = row["history"]
        assert block["count"] == len(block["rows"])
        assert block["reading_en"] and block["reading_he"]
        for record in block["rows"]:
            assert "reason" not in record
            assert set(record) >= {"decision_id", "recorded_at", "actor", "decision",
                                   "in_force", "on_rescore", "against_version_in_force",
                                   "release_note", "superseded_by"}
