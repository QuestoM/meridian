"""A reading with no values in it is a lead, not a proposal a person must decide.

The defect this closes is not a crash, it is weight. The reading of a fifty
clause agreement produced 228 proposals, and a reviewer had to work through all
of them line by line — including a discount ladder whose only rung was 0% at a
threshold of 0, and a measurement source whose every field came back unknown.
Those carry the SHAPE of a term and nothing in it. They cannot be checked
against the document, because there is nothing in them to check.

The split is measured, not asserted, and the measurement is pinned here against
the real corpus readings so a change to the rule has to face it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kairos.trade import standing

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "trade_corpus" / "agreements"


# ------------------------------------------------------------ the rule itself

def test_a_term_carrying_values_is_a_proposal():
    inst = {"params": {"percent": 15, "base": "net_of_discount",
                       "form": "invoice_deduction"}, "confidence": "medium"}
    assert standing.standing(inst) == standing.CONFIDENT
    assert standing.reason(inst, "he") == ""


def test_a_term_with_no_values_at_all_is_an_interpretation():
    inst = {"params": {}, "confidence": "high"}
    assert standing.standing(inst) == standing.INTERPRETIVE
    assert "ולא חילץ ממנו" in standing.reason(inst, "he")
    assert "extracted no value" in standing.reason(inst, "en")


def test_the_shape_of_the_failure_this_was_built_for():
    """A ladder with one rung of nothing, measured on the real corpus."""
    inst = {"params": {"tiers": [{"threshold": 0, "discount_percent": 0}],
                       "basis": "unstated", "mechanics": "unstated",
                       "period": "campaign"},
            "confidence": "low"}
    assert standing.standing(inst) == standing.INTERPRETIVE


def test_high_confidence_does_not_rescue_an_empty_term():
    """The rule reads the ANSWER, never the model's opinion of itself.

    Measured on the corpus with the shipped routing: withholding every
    low-confidence proposal would have withheld 68 and buried 42 CORRECT terms,
    because a smaller model rates itself low on things it got right. A rule
    whose safety depends on which model answered is not a rule.
    """
    assert standing.standing({"params": {}, "confidence": "high"}) == standing.INTERPRETIVE
    assert standing.standing(
        {"params": {"percent": 15}, "confidence": "low"}) == standing.CONFIDENT


def test_a_partly_filled_term_stays_a_proposal():
    """Half an answer is still an answer; the threshold is stated once."""
    inst = {"params": {"points": 500, "audience": "<UNKNOWN>", "window": "campaign"},
            "confidence": "medium"}
    assert standing.emptiness(inst["params"]) == pytest.approx(1 / 3)
    assert standing.standing(inst) == standing.CONFIDENT


def test_it_reads_an_object_and_a_term_instance_the_same_way():
    from kairos.trade.documents import Citation, TermInstance

    inst = TermInstance(
        instance_id="x-1", term_id="agency-commission", params={},
        citations=[Citation(document_id="d", page=1, clause_id="1", quote="x")],
        confidence="low",
    )
    assert standing.standing(inst) == standing.INTERPRETIVE
    assert standing.standing({"params": {}, "confidence": "low"}) == standing.INTERPRETIVE


# -------------------------------------------------- the measurement it rests on

def _readings():
    """Every corpus document's most recent measured reading, if one is on disk."""
    for directory in sorted(CORPUS.iterdir()):
        measured = directory / "measured-extraction.json"
        if measured.exists():
            yield directory.name, json.loads(measured.read_text(encoding="utf-8"))


def test_the_split_is_worth_making_on_the_real_corpus():
    """It has to move a real number, or it is ceremony.

    Measured when this landed, on the readings in the corpus: 16 of 228
    proposals carry no answer. The assertion is a floor rather than the exact
    figure, because the readings on disk are re-measured whenever the accuracy
    harness runs and this test must not become a reason not to re-run it.
    """
    readings = dict(_readings())
    if not readings:
        pytest.skip("no measured reading on disk to split")
    total = sum(len(r.get("instances", [])) for r in readings.values())
    interpretive = sum(
        1 for r in readings.values()
        for inst in r.get("instances", [])
        if standing.is_interpretive(inst)
    )
    assert total > 100, "the corpus reading is too small to conclude anything from"
    assert interpretive >= 5, (
        "the split moves almost nothing, so either the readings changed or the "
        f"rule stopped matching them: {interpretive} of {total}"
    )
    # and it must not swallow the review — a rule that withholds a third of the
    # terms is not separating leads, it is hiding the agreement.
    assert interpretive / total < 0.25, (
        f"{interpretive} of {total} proposals withheld; that is not a lead list"
    )


def test_no_interpretation_carries_a_value_anywhere_in_it():
    """The claim the reason text makes to the reader has to be true."""
    for name, reading in _readings():
        for inst in reading.get("instances", []):
            if not standing.is_interpretive(inst):
                continue
            share = standing.emptiness(inst.get("params"))
            assert share >= standing.EMPTY_SHARE, (
                f"{name}/{inst['instance_id']} is called an interpretation at "
                f"{share:.0%} empty"
            )
