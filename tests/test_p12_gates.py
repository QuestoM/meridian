"""The gate reading: the count that lies, and the sentence that replaces it.

JS-19's sequence reads the gates before it reads the money, and its target asks
for every gate delta with its held-out figure. Both were reachable only from the
adoption checks, which is the last command a steward runs, and neither was on the
board this piece publishes at all.

Building it surfaced the defect underneath it. The model console's own comparison
returns every gate key on which two artifacts do not hold the same value, and a
key the candidate does not carry comes back as one of those. Measured on this
tree, three of the five candidates return ten such rows and every one is an
absence: counting the rows reads ten gates decided the other way when the truth
is none. So the rows are split here, and the sentence is chosen from the split.

Every figure asserted below was measured on the delivered tree before the code
that reports it was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adopt_candidate_gates as gates
from scripts import adopt_candidate_registry as registry
from scripts import adopt_candidate_rescore as rescore
from scripts import adopt_candidate_state as state
from scripts import adopt_candidate_words as words

ROOT = Path(__file__).resolve().parents[1]
BOARD_JSON = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates" / "candidate-board.json"


def _evidence(identifier: str) -> dict:
    paths = rescore.Paths()
    shipped = rescore.read_artifact(paths.shipped)
    candidate = rescore.read_artifact(
        paths.candidates_dir / f"tv_break_coefficients_{identifier}.json")
    return state.gate_evidence(shipped, candidate)


def _board() -> dict:
    if not BOARD_JSON.is_file():
        pytest.skip("no board has been published on this tree")
    return json.loads(BOARD_JSON.read_text(encoding="utf-8"))


def test_a_row_of_ten_differences_with_no_disagreement_in_it_says_so():
    """The defect, pinned on the row it was measured on.

    ``competitor`` differs from the shipped artifact on ten gate keys and holds a
    different value on none of them, because it carries none of them. A count of
    ten is the wrong reading of that and the sentence is the right one.
    """
    summary = gates.gate_summary(_evidence("competitor"))
    assert summary["not_identical"] == 10
    assert summary["differing"] == 0
    assert summary["absent"] == 10
    assert summary["state"] == "absent_only_candidate"
    assert "does not carry" in summary["reading_en"]
    assert "decided differently" not in summary["reading_en"]


def test_exactly_one_candidate_on_this_tree_decides_a_gate_differently():
    """Measured across the whole shelf, because that is the finding.

    Four of the five carry no gate metadata the shipped artifact carries. One
    does, and it turns one gate off. Anything that reported five candidates with
    ten differences each would be reporting the opposite.
    """
    deciding = {}
    for path in rescore.candidate_files(rescore.Paths()):
        identifier = rescore.candidate_id(path)
        summary = gates.gate_summary(_evidence(identifier))
        if summary["differing"]:
            deciding[identifier] = summary["differing_keys"]
    assert deciding == {"calibrated": ["placebo_correction_active"]}


def test_the_one_real_disagreement_carries_both_values_and_neither_is_judged():
    summary = gates.gate_summary(_evidence("calibrated"))
    row = next(row for row in summary["rows"] if row["key"] == "placebo_correction_active")
    assert row["shipped"] is True
    assert row["candidate"] is False
    assert row["shipped_absent"] is False and row["candidate_absent"] is False


def test_one_difference_is_never_printed_as_a_plural():
    """A sentence that reads "1 gate keys are carried" is a sentence nobody wrote."""
    summary = gates.gate_summary(_evidence("calibrated"))
    assert summary["differing"] == 1
    for half in ("reading_en", "reading_he"):
        assert "1 gate keys" not in summary[half]
        assert "1 מפתחות" not in summary[half]
    assert summary["reading_en"].startswith("One gate key")


def test_a_reading_states_the_absences_beside_a_difference_and_never_a_bare_zero():
    """The clause is added when there is something to say and omitted otherwise."""
    with_none = gates.gate_summary({"verdicts": [
        {"key": "a", "shipped": True, "candidate": False,
         "shipped_absent": False, "candidate_absent": False}]})
    assert "A further" not in with_none["reading_en"]
    with_absence = gates.gate_summary({"verdicts": [
        {"key": "a", "shipped": True, "candidate": False,
         "shipped_absent": False, "candidate_absent": False},
        {"key": "b", "shipped": 1, "candidate": None,
         "shipped_absent": False, "candidate_absent": True}]})
    assert "A further 1 keys" in with_absence["reading_en"]


def test_two_artifacts_that_agree_everywhere_read_as_agreeing_and_not_as_empty():
    summary = gates.gate_summary({"verdicts": [], "held_out": []})
    assert summary["state"] == "same"
    assert summary["reading_en"].strip() and summary["reading_he"].strip()
    assert summary["held_out_state"] == "none"


def test_the_held_out_sentence_is_a_measurement_and_not_a_constant():
    """It used to assert that the amounts disagree, of every pair.

    Measured: on ``calibrated`` all three held-out blocks are reported by both
    artifacts at the same amount, so the confound the sentence asserted is one
    that pair does not carry. On ``competitor`` it is real: the same
    counter-programming gate was decided on 2,532 breaks and on 506.
    """
    even = gates.gate_summary(_evidence("calibrated"))
    assert even["held_out_state"] == "even"
    assert even["held_out_comparable"] == even["held_out_blocks"] == 3
    assert "do not agree" not in even["held_out_basis_en"]

    uneven = gates.gate_summary(_evidence("competitor"))
    assert uneven["held_out_state"] == "uneven"
    assert "counterprogramming_holdout" in uneven["held_out_uneven"]
    assert "do not agree" in uneven["held_out_basis_en"]
    row = next(row for row in uneven["held_out"] if row["block"] == "counterprogramming_holdout")
    assert (row["shipped_size"], row["candidate_size"]) == (2532, 506)
    assert row["comparable"] is False


def test_the_rule_and_whether_it_bit_are_two_different_sentences():
    """One is unconditionally true and the other is about this pair only."""
    assert gates.HELD_OUT_RULE["en"].strip() and gates.HELD_OUT_RULE["he"].strip()
    for entry in gates.HELD_OUT_STATE.values():
        assert entry["en"].strip() and entry["he"].strip()
    summary = gates.gate_summary(_evidence("calibrated"))
    assert summary["held_out_rule_en"] == gates.HELD_OUT_RULE["en"]
    assert summary["held_out_basis_en"] != summary["held_out_rule_en"]


def test_every_held_out_amount_carries_its_noun_as_a_word_in_both_halves():
    """34,560 minutes and 2,532 breaks are two different things.

    The unit is a key inside the artifact, ``n_test_minutes``, and no screen may
    hold its own map from that key to a noun. The noun travels with the figure.
    """
    summary = gates.gate_summary(_evidence("calibrated"))
    minutes = next(row for row in summary["held_out"]
                   if row["block"] == "detrend_seasonality_holdout")
    assert minutes["shipped_size"] == 34560
    assert minutes["shipped_unit_en"] == "minutes"
    assert minutes["shipped_unit_he"] == "דקות"
    breaks = next(row for row in summary["held_out"] if row["block"] == "series_gate_holdout")
    assert breaks["shipped_unit_en"] == "breaks"
    assert breaks["shipped_unit_he"] == "ברייקים"


def test_no_reading_claims_a_share_of_all_gate_keys():
    """The list of keys the console compares is not read here, so it is not stated.

    A denominator over every gate key would need a constant inside a frozen
    module, and a key both artifacts leave out is indistinguishable from a key
    both agree on from the comparison alone. Every count is a count of what was
    measured.
    """
    for entry in gates.GATE_READING.values():
        for half in ("en", "he"):
            assert "of the gate keys" not in entry[half]
            assert "%" not in entry[half]


def test_the_registry_carries_the_gate_reading_on_every_candidate_row():
    payload = registry.registry(rescore.Paths())
    assert payload["candidates"], "no candidate on this tree"
    for row in payload["candidates"]:
        summary = row["gates"]
        assert summary["state"]
        assert summary["reading_en"].strip() and summary["reading_he"].strip()


def test_the_board_carries_what_the_registry_measured_to_the_last_field():
    board, payload = _board(), registry.registry(rescore.Paths())
    measured = {row["id"]: row["gates"] for row in payload["candidates"]}
    for row in board["candidates"]:
        published, source = row["gates"], measured[row["id"]]
        assert published["state"] == source["state"]
        assert published["differing"] == source["differing"]
        assert published["absent"] == source["absent"]
        assert published["reading_he"] == source["reading_he"]
        assert published["held_out_state"] == source["held_out_state"]
        assert len(published["held_out"]) == len(source["held_out"])


def test_one_stored_value_reads_the_same_at_the_terminal_and_on_the_screen():
    """A float of 1.0 is 1.0 here and 1 in a browser, and JavaScript cannot tell.

    So the rendering happens once, on the side that still knows the type, and
    the screen prints what it is given. Measured on the shipped artifact's
    ``first_break_multiplier``, which is stored as 1.0.
    """
    summary = gates.gate_summary(_evidence("competitor"))
    row = next(row for row in summary["rows"] if row["key"] == "first_break_multiplier")
    assert row["shipped"] == 1.0
    assert row["shipped_text"] == "1.0"
    assert row["candidate_absent"] is True
    assert row["candidate_text"] is None
    flag = next(row for row in summary["rows"] if row["key"] == "series_layer_active")
    assert flag["shipped_text"] == "false"


def test_the_published_gate_values_are_stored_values_and_never_free_text():
    """The board is a browser bundle, so what rides in it is checked.

    A gate value is a flag, a number or one of the engine's own words. Nothing
    here is a sentence somebody typed, so nothing here can carry a name.
    """
    for row in _board()["candidates"]:
        for entry in row["gates"]["rows"]:
            for side in ("shipped", "candidate"):
                value = entry[side]
                assert value is None or isinstance(value, (bool, int, float, str))
                if isinstance(value, str):
                    assert len(value) < 40 and " " not in value


def test_a_verdict_record_carries_what_the_gates_of_that_artifact_decided():
    """The record is the thing a later reader finds, so it has to carry it.

    The console's own ``gate_counts`` in the same block is the LIVE ledger, three
    active and five tested and lost, and it is about the shipped model rather
    than about the candidate being decided on. So a reader holding only the
    record could see the state of the model and not the state of the artifact.
    """
    from scripts import adopt_candidate_decide as decide

    evidence = decide.evidence_for("calibrated")
    assert evidence["gates"]["state"] == "decides_differently"
    assert evidence["gates"]["differing_keys"] == ["placebo_correction_active"]
    assert evidence["gates"]["reading_he"].strip()
    # And the console's own keys are still there, under their own names.
    assert "gate_counts" in evidence and "gate_total" in evidence


# The window, which was the one raw ISO left on this screen after round 7.
# It shipped as one pre-joined string with an English preposition inside it.

def test_the_evaluation_carries_two_calendar_days_and_no_joined_string():
    stored = rescore.load_rescore(rescore.Paths()) or {}
    evaluation = stored.get("evaluation") or {}
    assert evaluation.get("window_from") == "2024-11-01"
    assert evaluation.get("window_to") == "2024-11-30"
    assert "window" not in evaluation, "the pre-joined field is still being emitted"


def test_the_window_reading_is_composed_where_it_is_read_and_in_both_halves():
    evaluation = {"window_from": "2024-11-01", "window_to": "2024-11-30"}
    assert words.window_line(evaluation) == "2024-11-01 to 2024-11-30"
    assert words.window_line(evaluation, "he") == "2024-11-01 עד 2024-11-30"
    # And an absent window is a state, never a half range.
    assert "not recorded" in words.window_line({})
    assert words.window_line({"window_from": "2024-11-01"}).endswith("not recorded")


def test_the_board_publishes_the_two_ends_for_the_screen_to_format():
    evaluation = _board()["evaluation"]
    assert evaluation["window_from"] == "2024-11-01"
    assert evaluation["window_to"] == "2024-11-30"
    assert "window" not in evaluation
