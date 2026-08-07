"""The fit basis and the producer's own recommendation.

Two facts about the artifacts on this tree were measured by nothing and stated
by nothing, and one of them falsified a sentence that was on screen.

The evaluation's limit asserted that every artifact was fitted on all of the
breaks it is scored on, so the in-sample optimism was common to every row and
the difference between two rows survived it. ``spotclip`` records a fit over
2,336 of the 2,532, having dropped 196. It is the row the table ranks first.

And ``spotclip`` carries its own producer's out-of-sample test, which advised
against adopting it. That is not a figure this surface ranks, and the tests
below hold that line as carefully as they hold the finding: a self-test is the
artifact's own split under its own fit, and the sentence saying so has to travel
with it.

The tree-facing tests here are pinned against the artifacts in the repository on
purpose. They are the reason the rule exists, so a tree where they stopped being
true should say so rather than quietly passing on a scratch fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adopt_candidate_basis as basis
from scripts import adopt_candidate_words as words

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "models" / "tv_break_coefficients.json"
CANDIDATES = ROOT / "models" / "candidates"


def _metadata(path: Path) -> dict:
    return (json.loads(path.read_text(encoding="utf-8")) or {}).get("metadata") or {}


def test_the_clip_variant_on_this_tree_was_fitted_on_fewer_breaks_than_it_is_scored_on():
    """The measurement the whole finding rests on, read from the artifact itself."""
    metadata = _metadata(CANDIDATES / "tv_break_coefficients_spotclip.json")
    assert metadata["total_breaks_measured"] == 2336
    assert metadata["base_breaks"] == 2532
    assert metadata["dropped_by_spot_clip"] == 196
    row = basis.basis_row("spotclip", metadata, 2532)
    assert row["state"] == "fewer"
    assert row["not_fitted_on"] == 196
    assert row["share_not_fitted_on"] == pytest.approx(196 / 2532)


def test_every_other_artifact_on_this_tree_was_fitted_on_all_of_them():
    """So the finding is about one row and the surface may name it as one row."""
    rows = [basis.basis_row("shipped", _metadata(SHIPPED), 2532)]
    for path in sorted(CANDIDATES.glob("tv_break_coefficients_*.json")):
        identifier = path.stem.replace("tv_break_coefficients_", "")
        rows.append(basis.basis_row(identifier, _metadata(path), 2532))
    summary = basis.fit_basis(rows, 2532)
    assert summary["state"] == "uneven"
    assert summary["uneven"] == ["spotclip"]
    assert summary["unknown"] == []
    assert summary["largest_shortfall"] == 196


def test_the_limit_sentence_is_chosen_by_the_measurement_and_never_asserted():
    """Three states, three sentences, and the wrong one may not survive the wrong tree."""
    # The two sentences share most of their words, so they are told apart by the
    # claim that differs: one says the optimism is shared and the other says it
    # is not. Matching on the shared clause is what let the constant look right.
    common = basis.limit_for({"state": "common"})
    assert common["state"] == "in_sample"
    assert "carry the same optimism" in common["en"]
    assert "not the same in every row" not in common["en"]

    uneven = basis.limit_for({"state": "uneven", "uneven": ["spotclip"],
                              "largest_shortfall": 196, "largest_shortfall_at": "spotclip"})
    assert uneven["state"] == "in_sample_uneven"
    assert "not the same in every row" in uneven["en"]
    assert "carry the same optimism" not in uneven["en"]
    assert uneven["uneven"] == ["spotclip"]

    unknown = basis.limit_for({"state": "unknown", "unknown": ["mystery"]})
    assert unknown["state"] == "in_sample_unknown"
    assert unknown["unknown"] == ["mystery"]


def test_the_lifting_condition_for_an_uneven_tree_says_the_size_is_not_computable():
    """Honest math on the finding itself.

    The count of breaks is recorded on every artifact and the identity of them
    is recorded on none, so the confound is measured to exist and its effect on
    the metric is not computable from anything on disk. Saying only the first
    half invites a reader to assume somebody sized it.
    """
    uneven = basis.limit_for({"state": "uneven", "uneven": ["spotclip"]})
    assert "not computable from anything on disk" in uneven["unblocked_by_en"]
    assert uneven["unblocked_by_he"].strip()


def test_an_artifact_that_records_nothing_is_unknown_rather_than_a_match():
    assert basis.basis_row("silent", {}, 2532)["state"] == "unknown"
    assert basis.basis_row("silent", {"total_breaks_measured": None}, 2532)["state"] == "unknown"
    # A boolean is not a count. json.loads gives back True for `true`, and
    # `int(True)` is 1, which would report a fit over one break.
    assert basis.basis_row("odd", {"total_breaks_measured": True}, 2532)["state"] == "unknown"


def test_a_scored_on_of_zero_cannot_divide_and_reports_unknown():
    row = basis.basis_row("any", {"total_breaks_measured": 10}, 0)
    assert row["state"] == "unknown"
    assert row["share_not_fitted_on"] is None


def test_the_clip_variant_carries_its_own_producer_advising_against_adopting_it():
    """The second fact about the row this table ranks first."""
    reported = basis.self_reported(
        "spotclip", _metadata(CANDIDATES / "tv_break_coefficients_spotclip.json"))
    assert reported["state"] == "advised_against"
    assert reported["adopt_recommended"] is False
    assert reported["n_test"] == 461
    assert "keep OFF" in reported["reason"]
    assert reported["reading_en"] and reported["reading_he"]


def test_a_self_test_with_no_recommendation_is_its_own_state_and_not_a_yes():
    """afterwindow records a self-test and reaches no verdict from it.

    Folding that into either recommendation would put a verdict in the artifact's
    mouth that it never gave, which is the same class of error as the constant
    limit: a state nobody measured, rendered as one somebody did.
    """
    reported = basis.self_reported(
        "afterwindow", _metadata(CANDIDATES / "tv_break_coefficients_afterwindow.json"))
    assert reported["state"] == "recorded_without_a_verdict"
    assert reported["adopt_recommended"] is None


def test_an_artifact_with_no_self_test_says_so_rather_than_going_silent():
    reported = basis.self_reported("competitor", _metadata(
        CANDIDATES / "tv_break_coefficients_competitor.json"))
    assert reported["state"] == "absent"
    assert "records no out-of-sample test" in reported["reading_en"]
    assert reported["reading_he"].strip()


def test_the_shipped_artifact_is_a_row_in_this_like_any_other():
    assert basis.basis_row("shipped", _metadata(SHIPPED), 2532)["state"] == "all"
    assert basis.self_reported("shipped", _metadata(SHIPPED))["state"] == "absent"


def test_a_self_test_is_never_rendered_without_the_sentence_that_stops_it_being_a_rank():
    """The line this piece exists to hold, on the one block that could break it.

    Ranking two self-tests against each other is exactly the mistake the
    common-basis re-score was built to replace, so the block that prints one
    prints the reason it is not comparable, every time.
    """
    payload = {"candidates": [{"self_reported": basis.self_reported(
        "spotclip", _metadata(CANDIDATES / "tv_break_coefficients_spotclip.json"))}]}
    lines = basis.render_self_tests(payload)
    assert lines
    assert any(words.SELF_TEST_BASIS["en"] in line for line in lines)


def test_the_self_test_block_is_absent_when_nothing_recorded_one():
    assert basis.render_self_tests({"candidates": [
        {"self_reported": {"id": "a", "state": "absent"}}]}) == []
    assert basis.render_self_tests({}) == []


def test_the_fit_basis_block_names_only_the_rows_the_limit_is_about():
    """A list of six rows saying five are fine buries the one that is not."""
    payload = {"fit_basis": {"state": "uneven", "rows": [
        {"id": "fine", "state": "all", "fitted_on": 2532, "scored_on": 2532},
        {"id": "clipped", "state": "fewer", "fitted_on": 2336, "scored_on": 2532,
         "not_fitted_on": 196},
    ]}}
    lines = basis.render_fit_basis(payload)
    assert len(lines) == 1
    assert "clipped" in lines[0]
    assert "2336" in lines[0] and "2532" in lines[0] and "196" in lines[0]
    assert "fine" not in lines[0]


def test_the_fit_basis_block_is_absent_when_every_row_covers_the_evaluation():
    assert basis.render_fit_basis({"fit_basis": {"state": "common", "rows": []}}) == []


@pytest.mark.parametrize("table", ["SELF_TEST", "FIT_BASIS"])
def test_the_new_string_tables_carry_both_halves(table):
    for key, entry in getattr(words, table).items():
        assert entry["en"].strip(), f"{table}.{key} has no English"
        assert entry["he"].strip(), f"{table}.{key} has no Hebrew"


def test_no_string_this_module_emits_carries_a_banned_mark():
    for table in (words.SELF_TEST, words.FIT_BASIS):
        for entry in table.values():
            for half in entry.values():
                assert "—" not in half and "!" not in half
    for limit in (words.LIMIT_UNEVEN, words.LIMIT_UNKNOWN, words.SELF_TEST_BASIS):
        for half in limit.values():
            assert "—" not in half and "!" not in half
