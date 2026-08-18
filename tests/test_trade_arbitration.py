"""The two-reader arbitration chain, exercised without touching a provider.

WHY THIS FILE EXISTS, stated plainly because it was paid for: the first live
run of the bench spent a full clause-by-clause pass and a whole-document call -
217 seconds of real model time - and then crashed in the scoring adapter on a
field name that does not exist on DocumentExtraction. A typo two frames above
the measurement cost the measurement. Every seam in that chain is now reachable
in a second with a fake ``call``, so the next crash of that class costs nothing.

The fake reader returns canned records, which is the point: what is under test
is the CODE's discipline - dropping a reading that cannot be anchored, aligning
two readings, applying a ruling, refusing to let an unverifiable quote pass as
evidence - not the model's judgement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kairos.trade import arbitrate, extract_wholedoc  # noqa: E402
from kairos.trade.documents import (Citation, Clause, ClauseDisposition,  # noqa: E402
                                    DocumentExtraction, TermInstance)

DOC = "doc-test"

CLAUSES = [
    Clause(clause_id="1", text="1. תוקף ההסכם: מיום 01.03.2026 ועד ליום 31.08.2026.", pages=(1,)),
    Clause(clause_id="2", text="2. עמלת הסוכנות תעמוד על 15% מהמחזור נטו לאחר הנחות.", pages=(1,)),
    Clause(clause_id="3", text="3. לא ישודר יותר מתשדיר אחד של המפרסם באותו מקבץ.", pages=(1,)),
]


def _fake_call(payload):
    """A ``call`` that ignores the prompt and returns what the test decided."""
    def call(**_kwargs):
        return payload
    return call


def _pipeline_instance(instance_id, term_id, clause_id, params, quote):
    return TermInstance(
        instance_id=instance_id, term_id=term_id, params=params,
        citations=[Citation(document_id=DOC, page=1, clause_id=clause_id, quote=quote)],
        confidence="medium",
    )


# ------------------------------------------------------- the second reader

def test_the_whole_reader_drops_every_reading_it_cannot_anchor():
    """Three ways a reading is unanchored, and none of them may become evidence."""
    payload = {"instances": [
        # good
        {"clause_id": "2", "term_id": "agency-commission", "quote": "15% מהמחזור נטו",
         "params": {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
         "confidence": "high"},
        # a clause the segmenter never produced
        {"clause_id": "99", "term_id": "agency-commission", "quote": "15%",
         "params": {}, "confidence": "high"},
        # a term outside the taxonomy
        {"clause_id": "2", "term_id": "invented-term", "quote": "15% מהמחזור נטו",
         "params": {}, "confidence": "high"},
        # a quote that is not verbatim in its clause
        {"clause_id": "3", "term_id": "frequency-caps", "quote": "תשדיר אחד לכל מקבץ",
         "params": {"unit": "break", "cap": 1}, "confidence": "high"},
        # not even an object
        "רק מחרוזת",
    ]}
    result = extract_wholedoc.read_whole_document(CLAUSES, _fake_call(payload))
    assert len(result["instances"]) == 1
    assert result["instances"][0]["term_id"] == "agency-commission"
    reasons = sorted(d["reason"] for d in result["dropped"])
    assert reasons == [
        "clause id not in the document",
        "not an object",
        "quote is not verbatim in its clause",
        "term id not in the taxonomy",
    ]


def test_records_become_instances_that_carry_their_citation():
    payload = {"instances": [
        {"clause_id": "2", "term_id": "agency-commission", "quote": "15% מהמחזור נטו",
         "params": {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
         "confidence": "high"},
    ]}
    result = extract_wholedoc.read_whole_document(CLAUSES, _fake_call(payload))
    instances = extract_wholedoc.instances_from_records(
        result["instances"], CLAUSES, document_id=DOC)
    assert len(instances) == 1
    assert instances[0].citations[0].clause_id == "2"
    assert instances[0].citations[0].quote == "15% מהמחזור נטו"


# ------------------------------------------------------------- the alignment

def test_alignment_names_all_four_outcomes():
    pipeline = [
        _pipeline_instance("p1", "agency-commission", "2",
                           {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
                           "15% מהמחזור נטו"),
        _pipeline_instance("p2", "frequency-caps", "3", {"unit": "break", "cap": 1},
                           "לא ישודר יותר מתשדיר אחד"),
        _pipeline_instance("p3", "effective-window", "1",
                           {"starts_on": "2026-03-01", "ends_on": "2026-08-31"},
                           "מיום 01.03.2026 ועד ליום 31.08.2026"),
    ]
    whole = [
        # identical params -> agreed
        {"clause_id": "3", "term_id": "frequency-caps", "params": {"unit": "break", "cap": 1},
         "scope": {}, "window": {}, "quote": "תשדיר אחד", "confidence": "high",
         "missing": [], "notes": ""},
        # same pair, different params -> contested
        {"clause_id": "2", "term_id": "agency-commission",
         "params": {"percent": 15, "base": "gross", "form": "invoice_deduction"},
         "scope": {}, "window": {}, "quote": "15%", "confidence": "high",
         "missing": [], "notes": ""},
        # only the whole reader saw this one
        {"clause_id": "1", "term_id": "payment-terms", "params": {"terms": "שוטף"},
         "scope": {}, "window": {}, "quote": "תוקף", "confidence": "low",
         "missing": [], "notes": ""},
    ]
    alignment = extract_wholedoc.align(pipeline, whole)
    assert alignment["agreed"] == [("3", "frequency-caps")]
    assert alignment["params_differ"] == [("2", "agency-commission")]
    assert alignment["pipeline_only"] == [("1", "effective-window")]
    assert alignment["whole_only"] == [("1", "payment-terms")]


# ---------------------------------------------------------------- the judge

def _contested():
    pipeline = [
        _pipeline_instance("p1", "agency-commission", "2",
                           {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
                           "15% מהמחזור נטו"),
        _pipeline_instance("p2", "frequency-caps", "3", {"unit": "break", "cap": 1},
                           "לא ישודר יותר מתשדיר אחד"),
    ]
    whole = [
        {"clause_id": "2", "term_id": "agency-commission",
         "params": {"percent": 15, "base": "gross", "form": "invoice_deduction"},
         "scope": {}, "window": {}, "quote": "15% מהמחזור נטו", "confidence": "high",
         "missing": [], "notes": ""},
        {"clause_id": "3", "term_id": "frequency-caps", "params": {"unit": "break", "cap": 1},
         "scope": {}, "window": {}, "quote": "תשדיר אחד של המפרסם", "confidence": "high",
         "missing": [], "notes": ""},
    ]
    return extract_wholedoc.align(pipeline, whole)


def test_the_judge_is_not_called_when_the_readers_agree():
    """No disagreement, no third call: an agreement re-decided is money wasted."""
    pipeline = [_pipeline_instance("p2", "frequency-caps", "3", {"unit": "break", "cap": 1},
                                   "לא ישודר יותר מתשדיר אחד")]
    whole = [{"clause_id": "3", "term_id": "frequency-caps",
              "params": {"unit": "break", "cap": 1}, "scope": {}, "window": {},
              "quote": "תשדיר אחד", "confidence": "high", "missing": [], "notes": ""}]
    alignment = extract_wholedoc.align(pipeline, whole)

    def exploding_call(**_kwargs):
        raise AssertionError("the judge was called with nothing to decide")

    ruled = arbitrate.arbitrate(CLAUSES, alignment, exploding_call, document_id=DOC)
    assert ruled["called"] is False
    assert ruled["agreed_count"] == 1
    assert len(ruled["instances"]) == 1


def test_each_verdict_produces_the_reading_it_names():
    alignment = _contested()
    payload = {"rulings": [
        {"clause_id": "2", "term_id": "agency-commission", "verdict": "b",
         "reason_he": "הסעיף אומר נטו לאחר הנחות, והבסיס שנרשם ברוטו אינו מה שכתוב"},
    ]}
    ruled = arbitrate.arbitrate(CLAUSES, alignment, _fake_call(payload), document_id=DOC)
    assert ruled["called"] is True
    commission = [i for i in ruled["instances"] if i.term_id == "agency-commission"]
    assert len(commission) == 1
    # the whole-document reading governed, so its params are the ones that survive
    assert commission[0].params["base"] == "gross"
    assert "הוכרע בבוררות" in commission[0].notes
    assert ruled["rulings"][0]["outcome"] == "whole-document reading kept"


def test_a_revised_verdict_lets_the_judge_write_the_term_itself():
    alignment = _contested()
    payload = {"rulings": [
        {"clause_id": "2", "term_id": "agency-commission", "verdict": "revised",
         "params": {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
         "quote": "15% מהמחזור נטו", "confidence": "high",
         "reason_he": "שתי הקריאות פספסו את צורת הגבייה; הסעיף קובע ניכוי מהחשבונית"},
    ]}
    ruled = arbitrate.arbitrate(CLAUSES, alignment, _fake_call(payload), document_id=DOC)
    commission = [i for i in ruled["instances"] if i.term_id == "agency-commission"][0]
    assert commission.params["base"] == "net_of_discount"
    assert commission.confidence == "high"
    assert ruled["rulings"][0]["outcome"] == "arbiter's own reading"


def test_a_ruling_with_an_unverifiable_quote_is_kept_but_demoted():
    """The judge's evidence is checked like anyone else's."""
    alignment = _contested()
    payload = {"rulings": [
        {"clause_id": "2", "term_id": "agency-commission", "verdict": "revised",
         "params": {"percent": 12, "base": "gross", "form": "invoice_deduction"},
         "quote": "משפט שאיננו במסמך", "confidence": "high",
         "reason_he": "הכרעה שנשענת על מובאה שאינה בסעיף"},
    ]}
    ruled = arbitrate.arbitrate(CLAUSES, alignment, _fake_call(payload), document_id=DOC)
    commission = [i for i in ruled["instances"] if i.term_id == "agency-commission"][0]
    assert commission.confidence == "low"
    assert "לא נשענה על מובאה מילולית" in commission.notes
    assert commission.citations[0].quote in CLAUSES[1].text


def test_neither_drops_the_term_and_says_so():
    alignment = _contested()
    payload = {"rulings": [
        {"clause_id": "2", "term_id": "agency-commission", "verdict": "neither",
         "reason_he": "הסעיף מתאר נוהג ולא מחייב עמלה"},
    ]}
    ruled = arbitrate.arbitrate(CLAUSES, alignment, _fake_call(payload), document_id=DOC)
    assert not [i for i in ruled["instances"] if i.term_id == "agency-commission"]
    assert ruled["rulings"][0]["outcome"] == "dropped"


def test_the_judge_cannot_add_a_clause_that_does_not_exist():
    """The coverage denominator is the segmenter's, and no ruling may move it."""
    alignment = _contested()
    payload = {"rulings": [
        {"clause_id": "77", "term_id": "agency-commission", "verdict": "revised",
         "params": {"percent": 9, "base": "gross", "form": "invoice_deduction"},
         "quote": "אין סעיף כזה", "confidence": "high", "reason_he": "סעיף מומצא"},
    ]}
    ruled = arbitrate.arbitrate(CLAUSES, alignment, _fake_call(payload), document_id=DOC)
    assert all(i.citations[0].clause_id in {"1", "2", "3"} for i in ruled["instances"])
    assert ruled["rulings"] == []


# ------------------------------------------------- the bench's scoring adapter

def test_the_scoring_adapter_builds_a_valid_extraction():
    """The exact frame that crashed the first live run, now one second to check.

    ``_as_extraction`` swaps a reading onto the segmenter's own clause ledger. It
    invented an ``agreement_id`` field DocumentExtraction does not have and
    passed a list where a tuple was required, and both only surfaced after two
    paid model stages had already run.
    """
    import trade_arbitration_bench as bench

    base = DocumentExtraction(
        document_id=DOC, clauses=CLAUSES,
        instances=[_pipeline_instance("p1", "frequency-caps", "3",
                                      {"unit": "break", "cap": 1},
                                      "לא ישודר יותר מתשדיר אחד")],
        dispositions=[
            ClauseDisposition(clause_id="1", disposition="irrelevant",
                              irrelevant_class="preamble-recitals", reason="מבוא"),
            ClauseDisposition(clause_id="2", disposition="unmapped", reason="לא מופה"),
            ClauseDisposition(clause_id="3", disposition="mapped", instance_ids=("p1",)),
        ],
    )
    swapped = bench._as_extraction(base, [
        _pipeline_instance("w1", "agency-commission", "2",
                           {"percent": 15, "base": "gross", "form": "invoice_deduction"},
                           "15% מהמחזור נטו"),
    ])
    coverage = swapped.coverage()  # raises if any clause lost its disposition
    assert coverage.total_clauses == 3
    assert swapped.document_id == DOC
    assert [d.clause_id for d in swapped.dispositions] == ["1", "2", "3"]
    by_clause = {d.clause_id: d for d in swapped.dispositions}
    # clause 2 carries the new reading
    assert by_clause["2"].disposition == "mapped"
    # clause 3 was mapped by the PIPELINE and is unmapped for THIS reading: a
    # reading may not borrow another reading's disposition score
    assert by_clause["3"].disposition == "unmapped"
    # clause 1 is irrelevant no matter who reads it, so that verdict is inherited
    assert by_clause["1"].disposition == "irrelevant"


# --------------------------------------------------- the output-budget seam

def test_a_whole_document_stage_gets_its_own_output_ceiling():
    """The silent, expensive failure this seam exists to prevent.

    A clause-level stage answers with one small object; a whole-document stage
    answers with dozens. When both shared one 4000-token ceiling the second
    reader was truncated mid-response, the instances array never arrived, and
    the run reported zero terms with zero errors - the worst shape a failure
    can take. The resolution is pinned here rather than trusted.
    """
    from kairos.trade import extract_provider

    caller = extract_provider.StageCaller(
        client=None, stats=extract_provider.RunStats(),
        max_tokens_by_stage={"wholedoc": 16000, "arbitrate": 16000},
    )
    assert caller.max_tokens_by_stage.get("wholedoc") == 16000
    assert caller.max_tokens_by_stage.get("parameterise", caller.max_tokens) == 4000
    # and the default caller is unchanged for every shipped clause-level stage
    plain = extract_provider.StageCaller(client=None, stats=extract_provider.RunStats())
    assert plain.max_tokens_by_stage == {}
    assert plain.max_tokens == 4000


def test_the_notes_field_cannot_eat_the_output_budget():
    """The other half of the same failure: an uncapped prose field."""
    schema = extract_wholedoc._wholedoc_schema()
    notes = schema["properties"]["document_notes"]
    assert notes.get("maxLength"), "an uncapped notes field can starve the instances array"
    assert schema["required"] == ["instances"]


def test_a_truncated_answer_is_a_failure_and_not_a_result():
    """The same class again, and this time it published a number.

    The provider used to accept any response carrying a ``tool_use`` block. A
    response that stopped at the output ceiling carries one — a half-written
    one — so it passed every check and travelled on as an answer. Measured on
    the corpus: the arbiter's reply on the document with fifty-four
    disagreements was exactly 16,000 output tokens against a 16,000 ceiling, the
    run recorded zero rulings with zero failures, and the accuracy report
    published 14.6% recall for arbitration on that document as though the judge
    had weighed the disagreements and dismissed them.

    ``stop_reason`` is the provider's own word for it and is now believed.
    """
    from kairos.trade import extract_provider

    class _Cut:
        stop_reason = "max_tokens"
        usage = type("U", (), {"input_tokens": 61454, "output_tokens": 16000})()
        content = [type("B", (), {"type": "tool_use", "input": {"rulings": []}})()]

    class _Client:
        class messages:
            @staticmethod
            def create(**_kwargs):
                return _Cut()

    stats = extract_provider.RunStats()
    caller = extract_provider.StageCaller(
        client=_Client(), stats=stats, pace_seconds=0.0,
        max_tokens_by_stage={"arbitrate": 16000},
    )
    with pytest.raises(extract_provider.TruncatedAnswer):
        caller.call(stage="arbitrate", tier="reason", system="s", content="c",
                    tool_name="record_rulings", tool_schema={"type": "object"})
    # and the run's own record says what happened rather than counting it a pass
    assert [call.ok for call in stats.calls] == [False]
    assert stats.calls[0].error == "truncated_max_tokens"


def test_the_judge_divides_its_contests_so_no_answer_can_be_cut_in_half():
    """The document goes in every call; the disagreements are what get split."""
    sent = []

    def call(**kwargs):
        sent.append(kwargs["content"])
        return {"rulings": []}

    many = []
    for index in range(40):
        many.append(_pipeline_instance(f"p{index}", "agency-commission", "2",
                                       {"percent": index}, "15% מהמחזור נטו"))
    # forty contests against one clause, all of them params_differ
    alignment = {
        "agreed": [],
        "params_differ": [("2", "agency-commission")] * 40,
        "pipeline_only": [],
        "whole_only": [],
        "pipeline_by_key": {("2", "agency-commission"): many[0]},
        "whole_by_key": {},
    }
    arbitrate.arbitrate(CLAUSES, alignment, call, document_id=DOC)
    assert len(sent) == 3, f"40 contests at {arbitrate.CONTESTS_PER_CALL} per call should be 3 calls"
    for content in sent:
        # the judge's whole value is holding the document while it decides
        assert "עמלת הסוכנות" in content, "a batch went out without the document"


def test_a_truncated_classify_batch_falls_to_the_retry_instead_of_killing_the_run():
    """The guard must not turn a recoverable batch into a dead run.

    classify_clauses climbs a ladder on purpose: a batch the model answers badly
    is retried one clause at a time, and a clause still unanswered lands as an
    honest ``unmapped`` disposition. Its own comment says why — never a crash
    that loses the other forty clauses already paid for.

    A truncated answer is exactly the case that ladder is for: the provider's
    remedy for it is "send less work", which is what the next rung does. So the
    refusal is caught here rather than propagating.
    """
    from kairos.trade import extract_provider, extract_stages

    seen = []

    def call(**kwargs):
        seen.append(kwargs["content"])
        # The whole batch truncates; a single clause answers.
        if kwargs["content"].count("<clause") > 1:
            raise extract_provider.TruncatedAnswer("ceiling")
        clause_id = kwargs["content"].split('id="')[1].split('"')[0]
        return {"classifications": [
            {"clause_id": clause_id, "labels": ["agency-commission"], "note": ""}]}

    labels = extract_stages.classify_clauses(CLAUSES, call)
    assert set(labels) == {"1", "2", "3"}, "a truncated batch lost clauses instead of retrying them"
    assert all(entry["labels"] == ["agency-commission"] for entry in labels.values())
    assert len(seen) == 4, "expected one batch call, then one call per clause"
