"""What each artifact was built for, and what it was built from.

Two absences are measured here rather than asserted. The first is the purpose:
every candidate's own metadata may carry one line saying what it was for, and
until this round no surface this piece owns carried it, so a steward read five
opaque identifiers and inferred each one's intent from its coefficient table.
The second is the provenance: every artifact records the data it read, by
digest, and nothing checked those digests against the files on disk or against
each other.

The rules under test are the ones that make both honest. A purpose is a stored
value and never an inference, an absent one is an absence naming the field that
would fill it, the source check is the engine's own freshness guard rather than
a second implementation, and a file that is gone is a third state and not a
failed comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adopt_candidate_origin as origin
from scripts import adopt_candidate_rescore as rescore

ROOT = Path(__file__).resolve().parents[1]
BOARD_JSON = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates" / "candidate-board.json"


def _artifact(name: str) -> dict:
    path = ROOT / "models" / "candidates" / f"tv_break_coefficients_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _shipped() -> dict:
    return json.loads((ROOT / "models" / "tv_break_coefficients.json").read_text(encoding="utf-8"))


def _tree(tmp_path: Path, *, fingerprints: dict, purpose=None, files: dict) -> dict:
    """A scratch tree whose source files can be made to move, or to vanish."""
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    metadata = {"source_fingerprints": fingerprints}
    if purpose is not None:
        metadata["purpose"] = purpose
    return metadata


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_purpose_is_the_artifacts_own_sentence_verbatim():
    """A stored value, carried, never rewritten and never summarised."""
    metadata = _artifact("afterwindow")["metadata"]
    row = origin.origin_row("afterwindow", metadata, root=ROOT)
    assert row["purpose_state"] == "recorded"
    assert row["purpose"] == metadata["purpose"]
    assert row["purpose"] == "after-window de-bias verification recompute (post-clip code, reference data)"


def test_an_artifact_that_records_no_purpose_gets_an_absence_and_never_a_guess():
    """The one place this piece could most easily invent a fact, and does not.

    Two of the five candidates on this tree record nothing about what they were
    for. Everything needed to write a plausible sentence about them is on the
    row already, the gates and the coefficient delta and the verdict, and none
    of it may become a purpose.
    """
    for name in ("calibrated", "placebo_corrected"):
        row = origin.origin_row(name, _artifact(name)["metadata"], root=ROOT)
        assert row["purpose_state"] == "absent"
        assert row["purpose"] is None
        assert row["purpose_reading_en"] == origin.PURPOSE["absent"]["en"]
        # And the sentence names the field that would supply it.
        assert origin.PURPOSE_KEY in row["purpose_reading_en"]


def test_the_shipped_artifact_records_no_purpose_either_and_says_so():
    row = origin.origin_row("shipped", _shipped()["metadata"], root=ROOT)
    assert row["purpose_state"] == "absent"
    assert row["purpose"] is None


def test_every_artifact_on_this_tree_was_fitted_from_the_files_on_disk_now():
    """The measurement, and it is the good kind: it strengthens the comparison.

    All six artifacts record the same three source files at the same digests,
    and all three are on disk with those bytes, which are the bytes the
    evaluation rebuilds its breaks from. So the source data is not one of the
    differences between any two rows on this board.
    """
    shipped = _shipped()["metadata"]
    rows = [origin.origin_row("shipped", shipped, root=ROOT)]
    for name in ("afterwindow", "calibrated", "competitor", "placebo_corrected", "spotclip"):
        rows.append(origin.origin_row(name, _artifact(name)["metadata"], root=ROOT,
                                      shipped_metadata=shipped))
    for row in rows:
        assert row["sources_state"] == "verified", row["id"]
        assert row["sources_recorded"] == 3
        assert all(item["on_disk"] and item["matches"] for item in row["sources"])
    for row in rows[1:]:
        assert row["same_sources_as_shipped"] == "same"
        assert row["differs_on"] == []


def test_a_source_file_that_moved_reads_moved_and_names_the_file(tmp_path):
    metadata = _tree(tmp_path, fingerprints={"data/a.csv": _digest("one"),
                                             "data/b.csv": _digest("two")},
                     files={"data/a.csv": "one", "data/b.csv": "moved"})
    row = origin.origin_row("x", metadata, root=tmp_path)
    assert row["sources_state"] == "moved"
    assert row["sources_changed"] == ["data/b.csv"]
    assert "data/b.csv" in row["sources_reading_en"]
    by_file = {item["file"]: item for item in row["sources"]}
    assert by_file["data/a.csv"]["matches"] is True
    assert by_file["data/b.csv"]["matches"] is False


def test_a_source_file_that_is_gone_is_a_third_state_and_not_a_failed_match(tmp_path):
    """A file nobody could hash did not fail the comparison. Tri-state."""
    metadata = _tree(tmp_path, fingerprints={"data/a.csv": _digest("one"),
                                             "data/gone.csv": _digest("two")},
                     files={"data/a.csv": "one"})
    row = origin.origin_row("x", metadata, root=tmp_path)
    assert row["sources_state"] == "unverifiable"
    assert row["sources_changed"] == []
    by_file = {item["file"]: item for item in row["sources"]}
    assert by_file["data/gone.csv"]["on_disk"] is False
    assert by_file["data/gone.csv"]["matches"] is None


def test_an_artifact_recording_no_sources_is_absent_rather_than_unverifiable(tmp_path):
    """Two different facts. One is a producer that wrote nothing down, the other
    is a tree that has lost an input, and the acts that fix them are different."""
    row = origin.origin_row("x", {"purpose": "something"}, root=tmp_path)
    assert row["sources_state"] == "absent"
    assert row["sources"] == []
    assert row["sources_recorded"] == 0


def test_the_source_verdict_agrees_with_the_engines_own_guard_in_every_state(tmp_path):
    """The product already answers this question, and this module answers it again.

    Deliberately, and for a measured reason: importing anything under
    ``kairos.model`` costs 17 seconds on this machine because that package's own
    init probes TensorFlow, and the registry that carries this figure is the
    first command of the steward's job and reads in under a second. So the four
    lines are here, over the guard's own hasher, and this test is what stops the
    two drifting: every state the guard can reach is produced and both must
    agree. The 17 seconds are paid once, here, where nobody is waiting.
    """
    from kairos.model.freshness import coefficient_freshness

    cases = [
        ({"data/a.csv": _digest("one")}, {"data/a.csv": "one"}, "fresh", "verified"),
        ({"data/a.csv": _digest("one")}, {"data/a.csv": "moved"}, "stale", "moved"),
        ({"data/a.csv": _digest("one")}, {}, "unknown", "unverifiable"),
    ]
    for index, (fingerprints, files, status, state) in enumerate(cases):
        work = tmp_path / str(index)
        work.mkdir()
        metadata = _tree(work, fingerprints=fingerprints, files=files)
        assert coefficient_freshness(metadata, root=work)["status"] == status
        assert origin.origin_row("x", metadata, root=work)["sources_state"] == state


def test_two_artifacts_that_read_different_data_are_named_as_differing(tmp_path):
    mine = _tree(tmp_path, fingerprints={"data/a.csv": _digest("one")},
                 files={"data/a.csv": "one"})
    theirs = {"source_fingerprints": {"data/a.csv": _digest("other")}}
    row = origin.origin_row("x", mine, root=tmp_path, shipped_metadata=theirs)
    assert row["same_sources_as_shipped"] == "differs"
    assert row["differs_on"] == ["data/a.csv"]
    assert "data/a.csv" in row["agreement_reading_en"]


def test_one_side_recording_nothing_is_unknown_and_never_a_match(tmp_path):
    mine = _tree(tmp_path, fingerprints={"data/a.csv": _digest("one")},
                 files={"data/a.csv": "one"})
    row = origin.origin_row("x", mine, root=tmp_path, shipped_metadata={})
    assert row["same_sources_as_shipped"] == "unknown"
    assert row["differs_on"] == []


def test_a_row_with_nothing_to_compare_against_carries_no_agreement_at_all(tmp_path):
    """The shipped artifact's own row. A row agreeing with itself is a true
    sentence about nothing, so the keys are absent instead."""
    row = origin.origin_row("shipped", _shipped()["metadata"], root=ROOT)
    assert "same_sources_as_shipped" not in row
    assert "agreement_reading_en" not in row


def test_nothing_here_claims_the_command_that_produced_an_artifact_is_recorded():
    """The half of the provenance this tree cannot answer, measured as absent.

    Not one metadata key on any of the six artifacts names a script, a flag or a
    command line, so the surface says the artifact can be identified and not
    rebuilt, and names what would close it.
    """
    metadata = _shipped()["metadata"]
    for name in ("afterwindow", "calibrated", "competitor", "placebo_corrected", "spotclip"):
        metadata = {**metadata, **_artifact(name)["metadata"]}
    assert [key for key in metadata
            if any(word in key.lower() for word in ("command", "argv", "script", "recipe"))] == []
    row = origin.origin_row("afterwindow", _artifact("afterwindow")["metadata"], root=ROOT)
    assert row["recipe_state"] == "not_recorded"
    assert row["recipe_en"] and row["recipe_he"]
    assert row["recipe_unblocked_by_en"] and row["recipe_unblocked_by_he"]


def test_the_recipe_gap_carries_both_halves():
    for key in ("en", "he", "unblocked_by_en", "unblocked_by_he"):
        assert origin.RECIPE[key].strip()


def _registry():
    from scripts import adopt_candidate_registry as registry

    return registry.registry(rescore.Paths())


def test_the_terminal_prints_one_purpose_line_per_artifact_including_the_live_one():
    payload = _registry()
    lines = origin.render_purposes(payload)
    assert lines, "the purpose block rendered nothing"
    body = "\n".join(lines)
    assert "shipped (live)" in body
    for row in payload["candidates"]:
        recorded = (row.get("origin") or {}).get("purpose")
        assert any(row["id"] in line and (recorded or "no purpose recorded") in line
                   for line in lines), row["id"]


def test_the_terminal_provenance_states_the_shared_answer_once_when_it_is_shared():
    """The discipline the fit-basis block already keeps: six rows saying the same
    thing buries the one that would not, so the shared answer is stated once and
    only a row that reads differently is named."""
    lines = origin.render_provenance(_registry())
    body = "\n".join(lines)
    assert body.count(origin.SOURCES["verified"]["en"]) == 1
    assert "Spots.xlsx" in body and "a540fe3bee3f" in body
    assert origin.RECIPE["en"] in body


def test_the_provenance_names_every_row_that_reads_differently():
    """Proven on a payload built to disagree, because a block that has only ever
    seen agreement has not been shown to report a disagreement."""
    shipped = {"id": "shipped", "sources": [{"file": "a", "sha256": "1", "short": "1",
                                             "on_disk": True, "matches": True}],
               "sources_state": "verified", "sources_reading_en": "fine"}
    odd = {**shipped, "id": "odd", "sources_state": "moved",
           "sources_reading_en": "the bytes moved",
           "same_sources_as_shipped": "differs", "differs_on": ["a"],
           "agreement_reading_en": "it read other data"}
    payload = {"shipped": {"origin": shipped},
               "candidates": [{"origin": odd}, {"origin": dict(shipped, id="same")}]}
    body = "\n".join(origin.render_provenance(payload))
    assert "the bytes moved" in body
    assert "it read other data" in body


def test_the_registry_and_the_board_carry_the_same_purpose_as_the_artifact_on_disk():
    """Three surfaces, one stored value, and no chance for two of them to drift."""
    published = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    joined = {row["id"]: row for row in _registry()["candidates"]}
    for row in published["candidates"]:
        on_disk = _artifact(row["id"])["metadata"].get("purpose")
        assert row["origin"]["purpose"] == on_disk
        assert joined[row["id"]]["origin"]["purpose"] == on_disk


def test_the_stored_rescore_row_carries_the_origin_too():
    stored = rescore.load_rescore(rescore.Paths()) or {}
    assert (stored.get("shipped") or {}).get("origin")
    for row in stored.get("candidates") or []:
        assert row["origin"]["id"] == row["id"]
        assert row["origin"]["sources_state"] in ("verified", "moved", "unverifiable", "absent")


def test_the_source_digest_helper_kept_its_absent_marker(tmp_path):
    """It moved modules this round and its output feeds the stored fingerprint,
    so a missing file has to keep changing that string the way it always did."""
    paths = rescore.Paths(root=tmp_path)
    (tmp_path / "data" / "reference").mkdir(parents=True)
    (tmp_path / "data" / "reference" / "Spots.xlsx").write_bytes(b"x")
    digests = origin.data_fingerprint(paths)
    assert set(digests) == set(origin.SOURCE_FILES)
    assert digests["Programmes.xlsx"] == "absent"
    assert digests["Spots.xlsx"] != "absent"
    assert rescore.data_fingerprint is origin.data_fingerprint


def test_reading_the_registry_never_imports_the_model_package(tmp_path):
    """The stopwatch guard, because this round nearly cost the story 17 seconds.

    ``kairos.model``'s own init probes TensorFlow and costs 17 s to import.
    Round 10 first reached for the freshness guard that lives under it, which
    would have made ``show``, the first command of JS-19, thirty times slower
    while answering a question about three digests. Measured rather than
    remembered: the module that joins the registry is imported in a clean
    interpreter and the model package must not come with it.
    """
    import subprocess
    import sys

    probe = ("import sys;"
             "import scripts.adopt_candidate_registry;"
             "print([name for name in sys.modules if name.startswith('kairos.model')])")
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("[]"), result.stdout


@pytest.mark.parametrize("table", ["PURPOSE", "SOURCES", "AGREEMENT"])
def test_every_reading_this_module_emits_exists_in_both_languages(table):
    for key, entry in getattr(origin, table).items():
        assert entry["en"].strip(), key
        assert entry["he"].strip(), key
