"""Recording the verdict: what it refuses, what it writes, and where it lands.

JS-19's done condition is a stored verdict a later reader can find. These tests
hold the three things that makes true: the verdict must rest on the common-basis
re-score rather than on two artifacts' own held-out figures, a ship verdict must
carry the money in shekels and the sentence the operator side reads, and the
record must land in the model console's own store rather than in a second one.

Everything runs against a temporary tree and a temporary decision store, through
the environment variable the store itself publishes for that purpose, so no test
here appends to the product's decision log.
"""

from __future__ import annotations

import json

import pytest

from scripts import adopt_candidate_decide as decide
from scripts import adopt_candidate_rescore as rescore
from scripts import adopt_candidate_state as state
from scripts import adopt_candidate_words as words

VERSION = {"id": "mv-test-1", "name": "2026-08-07", "short": "abc12345"}

ARTIFACT = {
    "method": "measured_detrended_pooled",
    "metadata": {"computed_at": "2026-08-01T00:00:00+00:00", "source_fingerprints": {"a": "1"},
                 "first_break_multiplier": 1.0, "total_breaks_measured": 60},
    "coefficients": {"News_first_long": -0.05, "Other_last_short": -0.02},
    "detail": {"News_first_long": {"coefficient": -0.05, "ci_low": -0.08, "ci_high": -0.02, "n": 27},
               "Other_last_short": {"coefficient": -0.02, "ci_low": -0.04, "ci_high": -0.01, "n": 40}},
}

MEASURED = {"state": "measured", "revenue_delta": 923843.08, "revenue_delta_pct": 0.31,
            "moved_fields": ["revenue_delta on the operator's own channel"],
            "scope": {"rows": 2540, "basis": "the weekly plan the run path computes"},
            "measured_at": "2026-08-07T02:42:07+00:00"}


def _frame():
    import pandas as pd

    return pd.DataFrame({"channel_name": ["News_first_long", "Other_last_short"] * 30,
                         "log_effect": [-0.05, -0.02] * 30,
                         "break_start": pd.date_range("2024-11-01", periods=60, freq="h")})


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(
        json.dumps(ARTIFACT, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_twin.json").write_text(
        json.dumps(ARTIFACT, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paths = rescore.Paths(root=tmp_path)
    rescore.save_rescore(rescore.rescore(paths, _frame()), paths)

    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "models" / "releases"))
    monkeypatch.setattr(decide, "live_version", lambda: dict(VERSION))
    monkeypatch.setattr(decide, "money_state", lambda identifier: dict(MEASURED))

    from kairos_api import model_console_api_payloads as payloads

    monkeypatch.setattr(payloads, "decision_evidence",
                        lambda subject, candidate_id: {"gate_counts": {"active": 3}, "gate_total": 13,
                                                       "money_state": "measured",
                                                       "revenue_delta": MEASURED["revenue_delta"],
                                                       "revenue_delta_pct": MEASURED["revenue_delta_pct"],
                                                       "scope": MEASURED["scope"],
                                                       "measured_at": MEASURED["measured_at"]})
    return paths


def _log(paths):
    path = paths.releases_dir / "decisions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_plan_appends_nothing_to_the_decision_log(tree):
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree)
    assert result["outcome"] == "ready"
    assert result["recorded"] is False
    assert _log(tree) == []


def test_a_recorded_verdict_lands_in_the_store_the_console_reads(tree):
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree, perform=True)
    assert result["outcome"] == "recorded"
    records = _log(tree)
    assert len(records) == 1
    assert records[0]["subject"] == "candidate"
    assert records[0]["candidate_id"] == "twin"
    assert records[0]["decision"] == "not_shipped"
    assert records[0]["decision_id"] == result["record"]["decision_id"]


def test_the_record_carries_the_common_basis_rescore_and_not_only_the_money(tree):
    """The defect this whole piece exists to close, asserted on the record.

    A verdict taken by reading two artifacts' own held-out figures compares two
    experiments on two different test sets. This one carries the comparison it
    was taken on, so a later reader can tell which kind it is.
    """
    decide.decide("twin", decision="not_shipped", actor="steward", reason="לא מבחין",
                  paths=tree, perform=True)
    evidence = _log(tree)[0]["evidence"]
    assert evidence["rescore"]["state"] == "identical"
    assert evidence["rescore"]["rmse"] == pytest.approx(evidence["rescore"]["shipped_rmse"])
    assert evidence["evaluation"]["breaks"] == 60
    assert evidence["basis_en"] and evidence["basis_he"]
    assert state.decision_rests_on_rescore(_log(tree)[0]) is True


def test_a_verdict_taken_before_this_comparison_existed_is_readable_as_such(tree):
    """The other half of the same fact, on the records already on the shelf."""
    assert state.decision_rests_on_rescore({"evidence": {"money_state": "measured"}}) is False
    assert state.decision_rests_on_rescore({}) is False
    assert state.decision_rests_on_rescore(None) is False


def test_the_evidence_keeps_the_keys_the_console_card_already_renders(tree):
    """A superset, not a rival shape, so P7's own card renders this record."""
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree)
    evidence = result["evidence"]
    for key in ("money_state", "revenue_delta", "revenue_delta_pct", "scope", "measured_at",
                "gate_counts", "gate_total"):
        assert key in evidence, key


def test_a_stale_rescore_refuses_the_verdict(tree):
    target = tree.candidates_dir / "tv_break_coefficients_twin.json"
    payload = json.loads(json.dumps(ARTIFACT))
    payload["coefficients"]["News_first_long"] = -0.051
    target.write_text(json.dumps(payload), encoding="utf-8")
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert "rescore_current" in result["blocked_on"]
    assert _log(tree) == []


def test_a_ship_verdict_without_a_release_note_is_refused(tree):
    result = decide.decide("twin", decision="shipped", actor="steward", reason="עדיף",
                           paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert "release_note_written" in result["blocked_on"]
    assert _log(tree) == []


def test_a_ship_verdict_without_a_measured_money_figure_is_refused(tree, monkeypatch):
    monkeypatch.setattr(decide, "money_state", lambda identifier: {
        "state": "not_measured", "reason_en": "not measured", "reason_he": "לא נמדד"})
    result = decide.decide("twin", decision="shipped", actor="steward", reason="עדיף",
                           release_note_he="עודכן קובץ המודל", paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert "money_measured" in result["blocked_on"]


def test_a_no_ship_verdict_needs_neither_a_release_note_nor_a_measured_figure(tree, monkeypatch):
    monkeypatch.setattr(decide, "money_state", lambda identifier: {
        "state": "not_measured", "reason_en": "not measured", "reason_he": "לא נמדד"})
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree, perform=True)
    assert result["outcome"] == "recorded"
    assert result["money_direction"] == "unknown"


def test_an_unnamed_steward_or_a_missing_reason_refuses_even_with_perform(tree):
    for missing in ("actor", "reason"):
        arguments = {"actor": "steward", "reason": "לא מבחין"}
        arguments[missing] = ""
        result = decide.decide("twin", decision="not_shipped", paths=tree, perform=True,
                               **arguments)
        assert result["outcome"] == "refused"
    assert _log(tree) == []


def test_a_release_note_carrying_a_gate_verdict_is_refused_by_the_store_verbatim(tree):
    """The store's own guard, shown rather than duplicated here."""
    result = decide.decide("twin", decision="shipped", actor="steward", reason="עדיף",
                           release_note_he="המקדם עודכן", paths=tree, perform=True)
    assert result["outcome"] == "refused"
    assert result["refusal"]
    assert "מקדם" in result["refusal"]
    assert _log(tree) == []


def test_the_money_direction_is_computed_from_the_measurement_and_never_typed(tree):
    assert decide.money_direction(MEASURED) == "up"
    assert decide.money_direction({**MEASURED, "revenue_delta": -1.0}) == "down"
    assert decide.money_direction({**MEASURED, "revenue_delta": 0.0}) == "none"
    assert decide.money_direction({"state": "stale", "revenue_delta": 5.0}) == "unknown"
    result = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree, perform=True)
    assert _log(tree)[0]["money_direction"] == "up"
    assert result["money_direction"] == "up"


def test_a_candidate_that_does_not_exist_is_refused_and_the_known_ones_are_named(tree):
    result = decide.decide("nope", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree, perform=True)
    assert result["outcome"] == "refused"
    reasons = {check["id"]: check["reason_en"] for check in result["checks"]}
    assert "twin" in reasons["candidate_exists"]


def test_every_check_answers_in_both_languages(tree):
    result = decide.decide("twin", decision="shipped", actor="", reason="", paths=tree)
    for check in result["checks"]:
        assert check["reason_en"].strip(), check["id"]
        assert check["reason_he"].strip(), check["id"]


def test_an_english_reason_is_flagged_because_the_console_renders_it_right_to_left(tree):
    hebrew = decide.decide("twin", decision="not_shipped", actor="steward",
                           reason="לא מבחין", paths=tree)
    english = decide.decide("twin", decision="not_shipped", actor="steward",
                            reason="not distinguishable", paths=tree)
    assert hebrew["reason_is_hebrew"] is True
    assert english["reason_is_hebrew"] is False
    assert any("right-to-left" in line for line in decide.render(english))
    assert not any("right-to-left" in line for line in decide.render(hebrew))


def test_the_english_reason_rides_in_the_evidence_so_neither_language_is_lost(tree):
    decide.decide("twin", decision="not_shipped", actor="steward", reason="לא מבחין",
                  reason_en="Not distinguishable from the shipped model.", paths=tree,
                  perform=True)
    assert _log(tree)[0]["evidence"]["reason_en"].startswith("Not distinguishable")


def _modules():
    """Every module of this piece, from disk, so a new one cannot escape the law.

    The discovery below used to walk one module. That was the same shape of hole
    one level up: an authored table put beside the arithmetic it belongs to,
    rather than in the words file, was governed by nothing at all. Round 8 put
    the gate strings beside the gate arithmetic in ``adopt_candidate_gates.py``,
    and under the old test both halves of every one of them were unchecked.
    """
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    found = [importlib.import_module(f"scripts.{path.stem}")
             for path in sorted((root / "scripts").glob("adopt_candidate*.py"))]
    assert len(found) >= 8, "the module discovery found almost nothing"
    return found


def _string_tables():
    """Every table of two-language entries in this piece, found rather than listed.

    This was a hardcoded list of five names. That is the same defect round 3
    found in the module-size test and fixed there with a glob: a list of names
    stops covering new code the moment new code is added, and it does it
    silently, so the law it holds quietly stops applying exactly when there is
    something new to hold it against. Round 6 added two tables and both would
    have escaped. Round 8 widened the search from one module to every module of
    the piece, for the same reason. Discovered now, with a floor so an accident
    that makes the discovery return nothing fails instead of passing vacuously.
    """
    found = {}
    for module in _modules():
        for name in dir(module):
            if not name.isupper():
                continue
            value = getattr(module, name)
            if isinstance(value, dict) and value and all(
                    isinstance(entry, dict) and "en" in entry for entry in value.values()):
                found[f"{module.__name__.split('_')[-1]}.{name}"] = value
    assert len(found) >= 9, f"the discovery found only {sorted(found)}"
    for known in ("words.VERDICTS", "words.BASIS", "words.RULE", "words.HOW",
                  "words.NEXT_ACT", "words.SELF_TEST", "words.FIT_BASIS",
                  "words.CELL_READING", "gates.GATE_READING", "gates.HELD_OUT_STATE"):
        assert known in found, f"{known} was not discovered"
    return found


@pytest.mark.parametrize("table", sorted(_string_tables()))
def test_every_authored_string_this_piece_emits_exists_in_both_languages(table):
    """The law is that an authored string is two strings, so the table is walked."""
    for key, entry in _string_tables()[table].items():
        assert entry.get("en", "").strip(), f"{table}.{key} has no English"
        assert entry.get("he", "").strip(), f"{table}.{key} has no Hebrew"


def test_the_evaluation_and_the_limit_carry_both_halves_too():
    from scripts import adopt_candidate_gates as gates

    for pair in (words.METRIC, words.TARGET_SD, words.DECISION_BASIS, words.WINDOW_JOIN,
                 words.SELF_TEST_BASIS, gates.GATE_ABSENT, gates.HELD_OUT_RULE,
                 gates.GATE_ALSO_ABSENT):
        assert pair["en"].strip() and pair["he"].strip()
    for limit in (words.IN_SAMPLE_LIMIT, words.LIMIT_UNEVEN, words.LIMIT_UNKNOWN):
        assert limit["en"].strip() and limit["he"].strip()
        assert limit["unblocked_by_en"].strip() and limit["unblocked_by_he"].strip()
