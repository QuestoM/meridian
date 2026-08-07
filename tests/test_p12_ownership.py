"""The write surface: every write this act makes lands on this piece's row.

The adoption act ends by copying a candidate over
``models/tv_break_coefficients.json``, and that path is absent from the P12 row
of section 8.2, so it is frozen by absence. Nothing had landed, because every
candidate on the tree fails an earlier check, so the violation was latent. These
tests hold the guard that closes it: the write is refused by default, it is
released only by a ruling recorded on disk that names that exact path, and the
refusal is a first-class check a steward reads rather than an exception.
"""

from __future__ import annotations

import json

import pytest

from scripts import adopt_candidate_adoption as adoption
from scripts import adopt_candidate_ownership as ownership
from scripts import adopt_candidate_rescore as rescore

ARTIFACT = {
    "method": "measured_detrended_pooled",
    "metadata": {"computed_at": "2026-08-01T00:00:00+00:00", "source_fingerprints": {"a": "1"},
                 "first_break_multiplier": 1.0, "total_breaks_measured": 100},
    "coefficients": {"News_first_long": -0.05},
    "detail": {"News_first_long": {"coefficient": -0.05, "ci_low": -0.08, "ci_high": -0.02, "n": 9}},
}


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A tree with one candidate, no ruling on record, and every other check met."""
    import pandas as pd

    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    text = json.dumps(ARTIFACT, ensure_ascii=False, indent=1) + "\n"
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(text, encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_twin.json").write_text(
        text, encoding="utf-8")
    paths = rescore.Paths(root=tmp_path)
    frame = pd.DataFrame({"channel_name": ["News_first_long"] * 20, "log_effect": [-0.05] * 20,
                          "break_start": pd.date_range("2024-11-01", periods=20, freq="h")})
    rescore.save_rescore(rescore.rescore(paths, frame), paths)
    monkeypatch.setattr(adoption, "live_version", lambda: {"id": "mv-1", "name": "n"})
    monkeypatch.setattr(adoption, "ship_decision", lambda i, v: {"decision_id": "md-1"})
    monkeypatch.setattr(adoption, "money_state", lambda i: {
        "state": "measured", "revenue_delta": 0.0, "moved_fields": [], "scope": {"rows": 1}})
    return paths


def _rule(paths, path=ownership.PENDING_PATH):
    target = ownership.ruling_path(paths.releases_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"path": path, "granted_by": "the lead"}), encoding="utf-8")


def test_the_row_this_guard_is_held_against_is_the_one_in_the_specification():
    """A guard on the guard: the constants must still name the real paths."""
    assert ownership.WRITE_SURFACE == ("models/candidates", "models/releases",
                                       "tv-break-dashboard/src/model/candidates")
    assert ownership.PENDING_PATH == "models/tv_break_coefficients.json"
    assert "8.2" in ownership.SPEC_ROW and "P12" in ownership.SPEC_ROW


def test_a_write_inside_the_row_is_allowed_and_one_outside_it_is_refused(tmp_path):
    releases = tmp_path / "models" / "releases"
    assert ownership.may_write(tmp_path, releases / "holdout_rescores.json", releases)
    assert ownership.may_write(tmp_path, tmp_path / "models" / "candidates" / "x.json", releases)
    assert not ownership.may_write(tmp_path, tmp_path / "data" / "breaks.csv", releases)
    assert not ownership.may_write(tmp_path, tmp_path / "models" / "audience_model.json", releases)
    with pytest.raises(ownership.WriteOutsideTheRow) as refusal:
        ownership.guard(tmp_path, tmp_path / "kairos_api" / "server.py", releases)
    assert "kairos_api/server.py" in str(refusal.value)
    assert "8.2" in str(refusal.value)


def test_a_path_that_only_looks_like_the_row_is_not_on_it(tmp_path):
    """``models/candidates_old`` is not ``models/candidates``."""
    releases = tmp_path / "models" / "releases"
    assert not ownership.may_write(tmp_path, tmp_path / "models" / "candidates_old" / "x", releases)


def test_the_shipped_artifact_is_refused_until_a_ruling_names_it(tree):
    assert not ownership.may_write(tree.root, tree.shipped, tree.releases_dir)
    _rule(tree)
    assert ownership.may_write(tree.root, tree.shipped, tree.releases_dir)


def test_a_ruling_about_some_other_path_does_not_release_this_one(tree):
    _rule(tree, path="models/audience_model.json")
    assert ownership.ruling(tree.releases_dir) is None
    assert not ownership.may_write(tree.root, tree.shipped, tree.releases_dir)


def test_an_unreadable_ruling_is_no_ruling_rather_than_an_error(tree):
    target = ownership.ruling_path(tree.releases_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{ not json", encoding="utf-8")
    assert ownership.ruling(tree.releases_dir) is None


def test_an_adoption_with_no_ruling_stops_at_the_check_and_writes_nothing(tree):
    """The state of this tree today, asserted rather than described."""
    before = rescore.sha256_file(tree.shipped)
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    assert plan["outcome"] == "refused"
    assert plan["blocked_on"] == ["write_target_is_owned"]
    assert rescore.sha256_file(tree.shipped) == before
    assert not adoption.adoptions_log(tree).exists()


def test_the_check_names_the_path_the_row_and_what_would_release_it(tree):
    state = adoption.preconditions("twin", tree, approved_by="steward", reason="testing")
    check = next(row for row in state["checks"] if row["id"] == "write_target_is_owned")
    assert check["passed"] is False
    assert ownership.PENDING_PATH in check["reason_en"]
    assert ownership.SPEC_ROW in check["reason_en"]
    assert check["reason_he"].strip()
    assert ownership.RULING_FILE in check["how_en"]
    assert check["how_he"].strip()
    assert state["ownership"]["ruled"] is False


def test_the_same_adoption_lands_once_the_ruling_is_on_record(tree):
    _rule(tree)
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=tree, perform=True)
    assert plan["outcome"] == "adopted"
    assert plan["ownership"]["ruled"] is True
    assert json.loads(tree.shipped.read_text(encoding="utf-8"))["metadata"]["adoption"]


def test_the_guard_refuses_even_when_the_checks_are_bypassed(tree):
    """The check is a screen. The guard is the thing that cannot be walked past.

    A caller reaching the write directly, in a later refactor or a test that
    patches the checks away, must still be stopped, which is why the guard sits
    at the line that writes and not only in the precondition list.
    """
    with pytest.raises(ownership.WriteOutsideTheRow):
        adoption._write_atomic(tree.shipped, "{}", tree)


def test_the_state_block_says_which_file_would_record_the_ruling(tree):
    state = ownership.state(tree.root, tree.releases_dir)
    assert state["ruled"] is False and state["ruling"] is None
    assert state["ruling_file"].endswith(ownership.RULING_FILE)
    assert state["write_surface"] == list(ownership.WRITE_SURFACE)
