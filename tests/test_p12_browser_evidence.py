"""P12: browser verdicts freeze the same evidence as terminal verdicts."""

from __future__ import annotations

from kairos_api import model_console_api_payloads as payloads
def test_candidate_decision_carries_the_full_published_comparison() -> None:
    evidence = payloads.complete_decision_evidence("candidate", "spotclip")
    assert set(("rescore", "gates", "evaluation", "limit", "fit_basis",
                "baselines", "cell_structure")) <= set(evidence)
    assert evidence["rescore"]["rmse"]
    assert evidence["rescore"]["cells"]["compared"] == 36
    assert evidence["gates"]["not_identical"] == 10
    assert evidence["evaluation"]["breaks"] == 2532


def test_an_unpublished_candidate_is_honestly_unavailable() -> None:
    evidence = payloads.complete_decision_evidence("candidate", "not-on-board")
    assert evidence["comparison_state"] == "unavailable"
    assert "absent" in evidence["comparison_reason"]


def test_current_model_decisions_keep_the_console_evidence_shape(monkeypatch) -> None:
    expected = {"gate_counts": {"active": 3}, "gate_total": 8}
    monkeypatch.setattr(payloads, "decision_evidence", lambda subject, candidate_id: expected)
    assert payloads.complete_decision_evidence("current", None) is expected
