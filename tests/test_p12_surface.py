"""What a candidate would change beyond its coefficients, and whether the
citations that claim the engine reads a field are still true.

The interesting case is the one this repository actually holds: an artifact that
predicts exactly what the shipped model predicts and is still a different engine
input. Two tests pin it against the real files, because a synthetic version of
that case would prove only that the code can detect a case somebody invented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import adopt_candidate_surface as surface

ROOT = Path(__file__).resolve().parents[1]


def _artifact(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_a_dropped_metadata_key_the_engine_reads_is_named_with_who_reads_it():
    shipped = {"metadata": {"first_break_multiplier": 1.9, "purpose": "x"},
               "coefficients": {"a": 1.0}, "detail": {}}
    candidate = {"metadata": {"purpose": "x"}, "coefficients": {"a": 1.0}, "detail": {}}
    report = surface.artifact_surface(shipped, candidate)
    assert report["metadata_dropped"] == ["first_break_multiplier"]
    dropped = report["engine_inputs_dropped"]
    assert [item["field"] for item in dropped] == ["first_break_multiplier"]
    assert "kairos/service.py" in dropped[0]["read_by"]


def test_a_dropped_metadata_key_nothing_reads_is_reported_but_not_as_an_engine_input():
    shipped = {"metadata": {"purpose": "x", "note": "y"}, "coefficients": {}, "detail": {}}
    candidate = {"metadata": {"purpose": "x"}, "coefficients": {}, "detail": {}}
    report = surface.artifact_surface(shipped, candidate)
    assert report["metadata_dropped"] == ["note"]
    assert report["engine_inputs_dropped"] == []


def test_a_dropped_cell_is_named_rather_than_counted():
    shipped = {"metadata": {}, "coefficients": {"a": 1.0, "b": 2.0}, "detail": {}}
    candidate = {"metadata": {}, "coefficients": {"a": 1.0}, "detail": {}}
    assert surface.artifact_surface(shipped, candidate)["cells_dropped"] == ["b"]


def test_a_moved_credible_bound_is_reported_even_when_the_point_does_not_move():
    shipped = {"metadata": {}, "coefficients": {"a": -0.05},
               "detail": {"a": {"coefficient": -0.05, "ci_low": -0.08, "ci_high": -0.02}}}
    candidate = {"metadata": {}, "coefficients": {"a": -0.05},
                 "detail": {"a": {"coefficient": -0.05, "ci_low": -0.07, "ci_high": -0.03}}}
    intervals = surface.artifact_surface(shipped, candidate)["intervals"]
    assert intervals["bounds_moved"] == 2
    assert intervals["max_abs_move"] == pytest.approx(0.01)
    assert intervals["max_abs_move_at"] in ("a.ci_low", "a.ci_high")


def test_the_placebo_candidate_on_this_tree_moves_every_bound_while_moving_no_point():
    """The measured case the whole module exists for, pinned against real files."""
    shipped = _artifact("models/tv_break_coefficients.json")
    candidate = _artifact("models/candidates/tv_break_coefficients_placebo_corrected.json")
    assert shipped["coefficients"] == candidate["coefficients"]
    report = surface.artifact_surface(shipped, candidate)
    assert report["intervals"]["bounds_moved"] == 72
    assert report["detail_fields_dropped"] == ["predictive_high", "predictive_low"]
    assert len(report["metadata_dropped"]) == 9


def test_the_candidate_with_a_ship_verdict_on_this_tree_drops_an_engine_input():
    """The after-window candidate is the one somebody already voted to ship.

    It does not carry ``first_break_multiplier``, which the service folds into
    the optimizer assumptions on every run, so adopting it would quietly remove
    an engine input. Nothing before this check could see that.
    """
    shipped = _artifact("models/tv_break_coefficients.json")
    candidate = _artifact("models/candidates/tv_break_coefficients_afterwindow.json")
    dropped = surface.artifact_surface(shipped, candidate)["engine_inputs_dropped"]
    assert "first_break_multiplier" in [item["field"] for item in dropped]


@pytest.mark.parametrize("field,citation", sorted(surface.ENGINE_READ_METADATA.items())
                         + sorted(surface.ENGINE_READ_DETAIL.items()))
def test_every_engine_citation_still_points_at_a_line_that_names_the_field(field, citation):
    """A citation that has rotted is worse than no citation, so it is checked.

    The claim "the engine reads this field" is the whole basis for refusing an
    adoption, and it is stated as a file and a line. This reads that line.
    """
    match = re.match(r"^(?P<path>[\w./_]+):(?P<line>\d+)", citation)
    assert match, citation
    lines = (ROOT / match.group("path")).read_text(encoding="utf-8").splitlines()
    number = int(match.group("line"))
    assert 1 <= number <= len(lines), citation
    assert field in lines[number - 1], f"{citation} no longer names {field}"
