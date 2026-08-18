"""One term stated twice becomes one card — and two terms never become one.

Half of this suite is about what must NOT merge. A collapse that is too eager is
worse than none at all: a reviewer shown a single card where the agreement said
two different things approves a term that is not in the agreement, and the
evidence that it was ever two is gone. So every refusal here is a measured case
from the corpus, not an invented one.
"""

from __future__ import annotations

import pytest

from kairos.trade import collapse
from kairos.trade.documents import Citation, ClauseDisposition, TermInstance

DOC = "d1"


def cite(clause: str, quote: str = "ציטוט", page: int = 1) -> Citation:
    return Citation(document_id=DOC, page=page, clause_id=clause, quote=quote)


def inst(iid, term="target-cpp", params=None, clauses=("2.2",),
         scope=None, window=None, confidence="high", notes="", missing=()):
    return TermInstance(
        instance_id=iid, term_id=term, params=dict(params or {}),
        citations=[cite(c, f"ציטוט {c}") for c in clauses],
        confidence=confidence, scope=dict(scope or {}), window=dict(window or {}),
        missing=list(missing), notes=notes,
    )


CPP = {"audience": "גברים בני 18-44", "cpp": 2400}


# ------------------------------------------------------------ what merges

def test_a_clause_read_twice_through_a_reference_becomes_one_card():
    """The measured case: 7.1 says "as defined in 2.2", so 2.2 is read again.

    Both instances carry the same CPP for the same audience and both cite 2.2.
    Before this, the reviewer got two identical cards.
    """
    kept, _, report = collapse.collapse([
        inst("x-008", params=CPP, clauses=("2.2",)),
        inst("x-031", params=CPP, clauses=("2.2", "7.1")),
    ])
    assert len(kept) == 1
    assert kept[0].instance_id == "x-008", "the earlier reading should survive"
    assert report["merged"][0]["clauses_gained"] == ["7.1"]


def test_the_surviving_card_names_every_clause_the_term_was_stated_in():
    """The reviewer gains evidence by merging. That is the point."""
    kept, _, _ = collapse.collapse([
        inst("x-008", params=CPP, clauses=("2.2",)),
        inst("x-031", params=CPP, clauses=("2.2", "7.1")),
    ])
    assert {c.clause_id for c in kept[0].citations} == {"2.2", "7.1"}


def test_the_fuller_reading_of_one_clause_wins_over_the_thinner_one():
    """Measured: one reading of 2.2 caught the programmes it names, one did not.

    Reached through a reference, the second reading kept the daypart and the
    lengths but lost the three programme names. It is the same envelope read
    less completely, so the complete one is what survives.
    """
    rich = {"dayparts": ["כלל רצועות השידור"], "lengths_seconds": [30],
            "programmes": ["גמר קרב השפים", "מוצ\"ש ספורט", "ערב הגאלה"]}
    thin = {"dayparts": ["כלל רצועות השידור"], "lengths_seconds": [30]}
    kept, _, _ = collapse.collapse([
        inst("x-008", params=CPP, clauses=("2.2",), scope=rich),
        inst("x-031", params=CPP, clauses=("2.2", "7.1"), scope=thin),
    ])
    assert len(kept) == 1
    assert kept[0].scope["programmes"] == rich["programmes"]


def test_a_parameter_one_reading_missed_is_taken_from_the_other():
    kept, _, _ = collapse.collapse([
        inst("a", params={"audience": "", "cpp": 2400}),
        inst("b", params={"audience": "נשים 25-54", "cpp": 2400}),
    ])
    assert len(kept) == 1
    assert kept[0].params == {"audience": "נשים 25-54", "cpp": 2400}


def test_a_field_no_longer_absent_stops_being_reported_as_missing():
    kept, _, _ = collapse.collapse([
        inst("a", params={"cpp": 2400}, missing=["audience"]),
        inst("b", params={"audience": "נשים 25-54", "cpp": 2400}),
    ])
    assert "audience" not in kept[0].missing


def test_the_same_daypart_listed_in_another_order_is_the_same_daypart():
    """A list the document happened to write in an order is not a disagreement."""
    kept, _, _ = collapse.collapse([
        inst("a", params=CPP, scope={"dayparts": ["פריים", "יום"]}),
        inst("b", params=CPP, scope={"dayparts": ["יום", "פריים"]}),
    ])
    assert len(kept) == 1


def test_a_fuller_phrasing_of_the_same_audience_merges_within_one_clause():
    kept, _, _ = collapse.collapse([
        inst("a", params={"audience": "גברים בני 18-44", "cpp": 2400}),
        inst("b", params={"audience": "גברים 18-44", "cpp": 2400}),
    ])
    assert len(kept) == 1
    assert kept[0].params["audience"] == "גברים בני 18-44", "the fuller reading should win"


def test_the_higher_confidence_survives():
    kept, _, _ = collapse.collapse([
        inst("a", params=CPP, confidence="medium"),
        inst("b", params=CPP, confidence="high"),
    ])
    assert kept[0].confidence == "high"


def test_both_sets_of_notes_survive_without_repeating_themselves():
    kept, _, _ = collapse.collapse([
        inst("a", params=CPP, notes="הערה אחת"),
        inst("b", params=CPP, notes="הערה אחת | הערה שנייה"),
    ])
    assert kept[0].notes == "הערה אחת | הערה שנייה"


# ---------------------------------------------------------- what must not

def test_two_different_prices_are_a_contradiction_and_never_a_duplicate():
    """The difference between two prices is the whole product.

    A tolerance here would merge 2,400 and 2,412 into one card and delete the
    disagreement that precedence exists to judge.
    """
    kept, _, _ = collapse.collapse([
        inst("a", params={"audience": "גברים בני 18-44", "cpp": 2400}),
        inst("b", params={"audience": "גברים בני 18-44", "cpp": 2412}),
    ])
    assert len(kept) == 2


def test_a_contradictory_document_keeps_its_contradictions():
    """Measured on heb-contradictory-2026: two readings of one clause naming
    different audiences and different tolerances. That document exists to carry
    contradictions, and a collapse that dissolved them would pass every other
    test while destroying the corpus."""
    kept, _, _ = collapse.collapse([
        inst("a", term="trp-delivery-guarantee", clauses=("4.1",),
             params={"audience": "קהל היעד שהוגדר בה", "tolerance_percent": 5}),
        inst("b", term="trp-delivery-guarantee", clauses=("4.1",),
             params={"audience": "בתי אב יהודיים", "tolerance_percent": 10}),
    ])
    assert len(kept) == 2


def test_the_same_term_for_two_different_scopes_stays_two_terms():
    """Merging these would widen a term to slots it was never sold for."""
    kept, _, _ = collapse.collapse([
        inst("a", params=CPP, clauses=("6.1",), scope={"positions": ["הודעת חסות"]}),
        inst("b", params=CPP, clauses=("7.1",), scope={}),
    ])
    assert len(kept) == 2


def test_two_different_terms_never_merge_however_alike_their_parameters():
    kept, _, _ = collapse.collapse([
        inst("a", term="target-cpp", params=CPP),
        inst("b", term="effective-cpp-cap", params=CPP),
    ])
    assert len(kept) == 2


def test_two_empty_readings_in_different_clauses_are_two_silences():
    """An interpretive instance says almost nothing. Two of them saying nothing
    in different clauses have not agreed about anything."""
    kept, _, _ = collapse.collapse([
        inst("a", params={"audience": "", "cpp": None}, clauses=("2.2",)),
        inst("b", params={"audience": "", "cpp": None}, clauses=("9.9",)),
    ])
    assert len(kept) == 2


def test_a_different_window_keeps_the_terms_apart():
    kept, _, _ = collapse.collapse([
        inst("a", params=CPP, clauses=("3.1",), window={"start": "2026-01-01"}),
        inst("b", params=CPP, clauses=("3.2",), window={"start": "2026-07-01"}),
    ])
    assert len(kept) == 2


# -------------------------------------------------------- the bookkeeping

def test_a_clause_whose_instance_was_folded_still_points_at_the_survivor():
    """Otherwise the clause maps to an id that no longer exists, and
    DocumentExtraction.validate refuses the whole extraction."""
    kept, dispositions, _ = collapse.collapse(
        [inst("x-008", params=CPP, clauses=("2.2",)),
         inst("x-031", params=CPP, clauses=("2.2", "7.1"))],
        [ClauseDisposition(clause_id="2.2", disposition="mapped",
                           instance_ids=("x-008",)),
         ClauseDisposition(clause_id="7.1", disposition="mapped",
                           instance_ids=("x-031",))],
    )
    surviving = {i.instance_id for i in kept}
    for disposition in dispositions:
        assert set(disposition.instance_ids) <= surviving, disposition
    assert dispositions[1].instance_ids == ("x-008",)


def test_a_clause_mapping_to_both_sides_of_a_merge_names_the_survivor_once():
    _, dispositions, _ = collapse.collapse(
        [inst("a", params=CPP), inst("b", params=CPP)],
        [ClauseDisposition(clause_id="2.2", disposition="mapped",
                           instance_ids=("a", "b"))],
    )
    assert dispositions[0].instance_ids == ("a",)


def test_dispositions_that_map_to_nothing_are_left_alone():
    _, dispositions, _ = collapse.collapse([], [
        ClauseDisposition(clause_id="9.9", disposition="irrelevant",
                          irrelevant_class="signature-block", reason="נוסח סטנדרטי"),
    ])
    assert dispositions[0].disposition == "irrelevant"
    assert dispositions[0].reason == "נוסח סטנדרטי"


def test_the_report_accounts_for_every_instance_that_went_in():
    kept, _, report = collapse.collapse([
        inst("a", params=CPP), inst("b", params=CPP), inst("c", term="brand-scope"),
    ])
    assert report["instances_before"] == 3
    assert report["instances_after"] == len(kept) == 2
    assert len(report["merged"]) == 1


def test_order_is_preserved_so_the_proposal_still_reads_like_the_document():
    kept, _, _ = collapse.collapse([
        inst("a", term="definitions"), inst("b", params=CPP),
        inst("c", term="brand-scope"), inst("d", params=CPP),
    ])
    assert [i.instance_id for i in kept] == ["a", "b", "c"]


def test_nothing_to_collapse_changes_nothing():
    given = [inst("a", params=CPP), inst("b", term="brand-scope")]
    kept, _, report = collapse.collapse(given)
    assert kept == given
    assert report["merged"] == []
