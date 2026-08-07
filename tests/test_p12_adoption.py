"""The adoption act: what it refuses, what it writes, and what it puts back.

Everything runs against a temporary tree. The real
``models/tv_break_coefficients.json`` is never written by this suite, which is
deliberate twice over: it is the engine's own frozen artifact, and it is not on
this piece's ownership row.

The three cases that matter are all here and all measured rather than described:
an adoption that would move a shipped figure is escalated and writes nothing, an
approval naming a different figure does not release it, and an adoption that
lands is byte-exactly undone by a revert.
"""

from __future__ import annotations

import json

import pytest

from scripts import adopt_candidate_adoption as adoption
from scripts import adopt_candidate_rescore as rescore

VERSION = {"id": "mv-test-1", "name": "2026-08-07", "short": "abc12345"}

SHIPPED = {
    "method": "measured_detrended_pooled",
    "metadata": {"computed_at": "2026-08-01T00:00:00+00:00", "source_fingerprints": {"a": "1"},
                 "first_break_multiplier": 1.0, "total_breaks_measured": 100},
    "coefficients": {"News_first_long": -0.05, "Other_last_short": -0.02},
    "detail": {"News_first_long": {"coefficient": -0.05, "ci_low": -0.08, "ci_high": -0.02, "n": 27},
               "Other_last_short": {"coefficient": -0.02, "ci_low": -0.04, "ci_high": -0.01, "n": 40}},
}


def _candidate(coefficients=None, metadata=None):
    payload = json.loads(json.dumps(SHIPPED))
    if coefficients:
        payload["coefficients"] = coefficients
        for cell, value in coefficients.items():
            payload["detail"].setdefault(cell, {})["coefficient"] = value
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(
        json.dumps(SHIPPED, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_twin.json").write_text(
        json.dumps(_candidate(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_mover.json").write_text(
        json.dumps(_candidate({"News_first_long": -0.09, "Other_last_short": -0.02}),
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paths = rescore.Paths(root=tmp_path)

    stored = rescore.rescore(paths, _frame())
    rescore.save_rescore(stored, paths)

    monkeypatch.setattr(adoption, "live_version", lambda: dict(VERSION))
    monkeypatch.setattr(adoption, "ship_decision",
                        lambda identifier, version_id: {"decision_id": "md-test"})
    monkeypatch.setattr(adoption, "money_state", lambda identifier: _money(identifier))
    return paths


def _frame():
    import pandas as pd

    cells = ["News_first_long", "Other_last_short"] * 30
    values = [-0.05, -0.02] * 30
    return pd.DataFrame({"channel_name": cells, "log_effect": values,
                         "break_start": pd.date_range("2024-11-01", periods=60, freq="h")})


def _money(identifier):
    if identifier == "twin":
        return {"state": "measured", "revenue_delta": 0.0, "moved_fields": [],
                "scope": {"rows": 2540}, "measured_at": "2026-08-07T00:00:00+00:00"}
    return {"state": "measured", "revenue_delta": 963477.37,
            "moved_fields": ["revenue_delta"], "scope": {"rows": 2540},
            "measured_at": "2026-08-07T00:00:00+00:00"}


def _digest(paths):
    return rescore.sha256_file(paths.shipped)


def test_a_plan_writes_nothing_at_all(tree):
    before = _digest(tree)
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing", paths=tree)
    assert plan["outcome"] == "ready"
    assert plan["performed"] is False
    assert _digest(tree) == before
    assert not adoption.adoptions_log(tree).exists()


def test_a_movement_of_zero_needs_no_owner_approval_and_may_land(tree):
    state = adoption.preconditions("twin", tree, approved_by="steward", reason="testing")
    assert state["money_moves"] is False
    assert state["escalated"] is False
    assert state["passed"] is True
    names = {check["id"] for check in state["checks"]}
    assert "no_shipped_figure_moves" in names
    assert "owner_approval_matches_movement" not in names


def test_a_movement_that_is_not_zero_is_escalated_and_writes_nothing(tree):
    before = _digest(tree)
    plan = adoption.adopt("mover", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    assert plan["outcome"] == "escalated"
    assert plan["performed"] is False
    assert _digest(tree) == before
    blocked = plan["blocked_on"]
    assert "owner_approval_matches_movement" in blocked


def test_an_approval_that_names_a_different_figure_does_not_release_it(tree):
    directory = adoption.approvals_dir(tree)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mover.json").write_text(
        json.dumps({"approved_revenue_delta": 900000.0, "approved_by": "owner"}), encoding="utf-8")
    state = adoption.preconditions("mover", tree, approved_by="steward", reason="testing")
    assert state["escalated"] is True
    assert state["passed"] is False


def test_an_approval_that_names_the_exact_figure_releases_it(tree):
    directory = adoption.approvals_dir(tree)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mover.json").write_text(
        json.dumps({"approved_revenue_delta": 963477.37, "approved_by": "owner"}), encoding="utf-8")
    state = adoption.preconditions("mover", tree, approved_by="steward", reason="testing")
    assert state["escalated"] is False
    assert state["passed"] is True


def test_an_adoption_that_lands_stamps_its_verdict_into_the_artifact(tree):
    before = _digest(tree)
    plan = adoption.adopt("twin", adopted_by="steward", reason="identical predictions",
                          release_note_he="עודכן קובץ המודל", paths=tree, perform=True)
    assert plan["outcome"] == "adopted"
    stamp = json.loads(tree.shipped.read_text(encoding="utf-8"))["metadata"]["adoption"]
    assert stamp["from_candidate"] == "twin"
    assert stamp["adopted_by"] == "steward"
    assert stamp["superseded_sha256"] == before
    assert stamp["superseded_version_id"] == VERSION["id"]
    assert stamp["ship_decision_id"] == "md-test"
    assert stamp["rescore_verdict"] == "identical"
    assert stamp["measured_revenue_delta"] == 0.0
    assert stamp["revert_with"].endswith(plan["adoption_id"])


def test_the_replaced_artifact_is_kept_whole_beside_the_adoption(tree):
    before = tree.shipped.read_bytes()
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    directory = adoption.adoptions_dir(tree) / plan["adoption_id"]
    assert (directory / adoption.PREVIOUS_NAME).read_bytes() == before
    assert (directory / adoption.MANIFEST_NAME).is_file()
    assert (directory / adoption.ADOPTED_NAME).read_bytes() == tree.shipped.read_bytes()


def test_a_revert_puts_back_the_exact_bytes(tree):
    before = tree.shipped.read_bytes()
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    assert tree.shipped.read_bytes() != before
    dry = adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo", paths=tree)
    assert dry["outcome"] == "ready"
    assert tree.shipped.read_bytes() != before
    done = adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                           paths=tree, perform=True)
    assert done["outcome"] == "reverted"
    assert done["record"]["restored_exactly"] is True
    assert tree.shipped.read_bytes() == before


def test_a_revert_refuses_when_the_artifact_is_not_the_one_it_left(tree):
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    tree.shipped.write_text(json.dumps({"coefficients": {}, "metadata": {}, "detail": {}}),
                            encoding="utf-8")
    later = tree.shipped.read_bytes()
    result = adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                             paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert tree.shipped.read_bytes() == later


def test_the_same_adoption_cannot_be_reverted_twice(tree):
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                    paths=tree, perform=True)
    again = adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                            paths=tree, perform=True)
    assert again["outcome"] == "refused"


def test_reverting_an_adoption_that_never_happened_is_refused_by_name(tree):
    result = adoption.revert("ad-nope", reverted_by="steward", reason="undo",
                             paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert "ad-nope" in result["reason_en"]


@pytest.mark.parametrize("missing", ["adopted_by", "reason"])
def test_an_unnamed_steward_or_a_missing_reason_refuses_even_with_perform(tree, missing):
    before = _digest(tree)
    arguments = {"adopted_by": "steward", "reason": "testing"}
    arguments[missing] = ""
    plan = adoption.adopt("twin", paths=tree, perform=True, **arguments)
    assert plan["outcome"] == "refused"
    assert _digest(tree) == before


def test_a_stale_rescore_stops_an_adoption(tree):
    target = tree.candidates_dir / "tv_break_coefficients_twin.json"
    target.write_text(json.dumps(_candidate({"News_first_long": -0.051,
                                             "Other_last_short": -0.02})), encoding="utf-8")
    before = _digest(tree)
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    assert "rescore_current" in plan["blocked_on"]
    assert _digest(tree) == before


def test_a_candidate_the_rescore_calls_worse_may_not_be_adopted(tree, monkeypatch):
    monkeypatch.setattr(adoption, "_stored_verdict", lambda identifier, paths: "worse")
    state = adoption.preconditions("twin", tree, approved_by="steward", reason="testing")
    assert state["passed"] is False
    assert "not_measured_worse" in state["blocked_on"]


def test_a_candidate_that_does_not_exist_is_refused_and_the_known_ones_are_named(tree):
    state = adoption.preconditions("nope", tree, approved_by="steward", reason="testing")
    assert state["passed"] is False
    reasons = {check["id"]: check["reason_en"] for check in state["checks"]}
    assert "mover" in reasons["candidate_exists"] and "twin" in reasons["candidate_exists"]


def test_every_check_answers_in_both_languages(tree):
    state = adoption.preconditions("mover", tree, approved_by="steward", reason="testing")
    for check in state["checks"]:
        assert check["reason_en"].strip()
        assert check["reason_he"].strip()


def test_the_adoption_log_records_both_the_landing_and_the_undo(tree):
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                    paths=tree, perform=True)
    actions = [record["action"] for record in adoption.adoptions(tree)]
    assert actions == ["reverted", "adopted"]
