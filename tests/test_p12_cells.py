"""The coefficient delta: what moved, what it bought, and what nothing means.

JS-19's done condition names three things recorded against a new model version:
the gate deltas, the coefficient deltas and the measured money movement. The
third was absent, and the tests here hold the two properties that make it worth
having rather than merely present.

The first is attribution. A metric delta says a candidate is a thousandth
closer; it cannot say whether one cell was fixed or thirty-six moved and
cancelled, and those are two different artifacts. Every figure here carries the
breaks it was measured on and the squared error it moved.

The second is tri-state. A cell one artifact carries and the other does not has
no delta. It is ``added`` or ``dropped`` with ``None``, and never zero.

Two tests pin the measured cases this repository actually holds, because a
synthetic version of them would prove only that the code detects a case somebody
invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import adopt_candidate_cells as cells
from scripts import adopt_candidate_rescore as rescore
from scripts import adopt_candidate_words as words

ROOT = Path(__file__).resolve().parents[1]


def _artifact(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _deltas(shipped, candidate, cell_keys, shipped_errors, candidate_errors):
    return cells.cell_deltas(shipped, candidate, np.array(cell_keys),
                             np.array(shipped_errors, dtype=float),
                             np.array(candidate_errors, dtype=float))


def test_a_moved_coefficient_is_named_with_the_breaks_it_was_measured_on():
    report = _deltas({"a": -0.05, "b": -0.04}, {"a": -0.03, "b": -0.04},
                     ["a", "a", "a", "b"], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0])
    rows = {row["cell"]: row for row in report["rows"]}
    assert rows["a"]["state"] == "moved"
    assert rows["a"]["delta"] == pytest.approx(0.02)
    assert rows["a"]["breaks"] == 3
    assert rows["a"]["squared_error_delta"] == pytest.approx(-1.5)
    assert rows["b"]["state"] == "unchanged"
    assert rows["b"]["squared_error_delta"] == pytest.approx(0.0)


def test_a_cell_only_one_artifact_carries_has_no_delta_and_never_a_zero():
    """The tri-state. A missing coefficient is not a coefficient of zero."""
    report = _deltas({"a": -0.05, "gone": -0.09}, {"a": -0.05, "new": -0.01},
                     ["a"], [1.0], [1.0])
    rows = {row["cell"]: row for row in report["rows"]}
    assert rows["gone"]["state"] == "dropped" and rows["gone"]["delta"] is None
    assert rows["new"]["state"] == "added" and rows["new"]["delta"] is None
    assert rows["gone"]["candidate"] is None and rows["new"]["shipped"] is None
    summary = report["summary"]
    assert summary["cells_dropped"] == ["gone"] and summary["cells_added"] == ["new"]
    # The dropped and added cells are not counted as compared, because there is
    # nothing on the other side to compare them with.
    assert summary["cells_compared"] == 1


def test_the_same_move_in_a_heavier_cell_buys_more_than_in_a_thin_one():
    """Attribution is the whole point: a coefficient list cannot say this."""
    report = _deltas({"heavy": 0.0, "thin": 0.0}, {"heavy": 0.1, "thin": 0.1},
                     ["heavy"] * 10 + ["thin"], [1.0] * 11, [0.5] * 10 + [0.5])
    rows = {row["cell"]: row for row in report["rows"]}
    assert rows["heavy"]["delta"] == rows["thin"]["delta"]
    assert abs(rows["heavy"]["squared_error_delta"]) == pytest.approx(
        10 * abs(rows["thin"]["squared_error_delta"]))
    assert rows["heavy"]["share_of_absolute"] > rows["thin"]["share_of_absolute"]


def test_cells_that_move_against_each_other_are_reported_as_cancelling():
    """The finding a metric delta hides: movement that nets to nothing."""
    report = _deltas({"up": 0.0, "down": 0.0}, {"up": 0.1, "down": -0.1},
                     ["up", "down"], [1.0, 1.0], [2.0, 0.0])
    summary = report["summary"]
    assert summary["net_squared_error_delta"] == pytest.approx(0.0)
    assert summary["total_abs_squared_error_delta"] == pytest.approx(2.0)
    assert summary["cancelled_share"] == pytest.approx(1.0)
    assert summary["cells_improved"] == 1 and summary["cells_worsened"] == 1


def test_nothing_moving_reads_as_nothing_moving_and_not_as_a_zero_measurement():
    report = _deltas({"a": -0.05}, {"a": -0.05}, ["a"], [1.0], [1.0])
    summary = report["summary"]
    assert summary["cells_moved"] == 0
    assert summary["max_abs_delta"] is None and summary["max_abs_delta_at"] is None
    assert summary["carries_the_move"] == []
    # The cancellation is a share of a movement, so with no movement it is not
    # a zero. It was a zero in the payload while the terminal said "no move",
    # which is the same figure stated two ways on two surfaces.
    assert summary["cancelled_share"] is None
    assert "No coefficient moves" in summary["reading_en"]
    assert summary["reading_he"].strip()


def test_the_published_payload_carries_no_cancellation_for_the_candidate_that_moves_none():
    """The tri-state where a route would read it, not only where a person does.

    ``models/releases/holdout_rescores.json`` is what this piece publishes for
    the console route that section 6 of the contract is blocked on, and a reader
    of that file must not find 0.0 in a field that means "none of its movement
    cancelled" on a candidate that made no movement at all.
    """
    stored = json.loads((ROOT / "models/releases/holdout_rescores.json").read_text(encoding="utf-8"))
    for row in stored["candidates"]:
        summary = row["cell_deltas"]["summary"]
        if summary["cells_moved"]:
            assert isinstance(summary["cancelled_share"], float), row["id"]
        else:
            assert summary["cancelled_share"] is None, row["id"]


def test_the_named_cells_genuinely_reach_the_share_the_sentence_claims():
    generator = np.random.default_rng(11)
    keys = [f"c{index}" for index in range(20)]
    shipped = {key: 0.0 for key in keys}
    candidate = {key: float(value) for key, value in zip(keys, generator.normal(0, 0.05, 20))}
    errors_before = np.ones(20)
    errors_after = np.abs(generator.normal(1.0, 0.6, 20))
    report = _deltas(shipped, candidate, keys, errors_before, errors_after)
    summary = report["summary"]
    named = set(summary["carries_the_move"])
    total = summary["total_abs_squared_error_delta"]
    carried = sum(abs(row["squared_error_delta"]) for row in report["rows"]
                  if row["cell"] in named)
    assert carried / total >= cells.CARRIES_SHARE
    # And the smallest such set: dropping the last name falls short of the bar.
    assert (carried - abs(min(
        (row["squared_error_delta"] for row in report["rows"] if row["cell"] in named),
        key=abs))) / total < cells.CARRIES_SHARE


def test_a_movement_spread_over_many_cells_is_reported_as_spread_not_as_carried():
    """Naming sixteen of thirty-six cells is the opposite of a finding."""
    keys = [f"c{index}" for index in range(20)]
    report = _deltas({key: 0.0 for key in keys}, {key: 0.1 for key in keys},
                     keys, np.ones(20), np.full(20, 2.0))
    summary = report["summary"]
    assert summary["concentrated"] is False
    assert "no small set of cells carries this" in summary["reading_en"]
    assert not any("carrying most of it" in line
                   for line in cells.render_summary(report))


def test_one_cell_carrying_the_move_is_named_because_that_is_a_finding():
    report = _deltas({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0,
                      "f": 0.0, "g": 0.0, "h": 0.0},
                     {"a": 0.9, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0,
                      "f": 0.0, "g": 0.0, "h": 0.0},
                     list("abcdefgh"), np.ones(8), [9.0] + [1.0] * 7)
    summary = report["summary"]
    assert summary["concentrated"] is True and summary["carries_the_move"] == ["a"]
    assert any("carrying most of it" in line for line in cells.render_summary(report))


def test_the_render_never_prints_a_missing_coefficient_as_a_number():
    report = _deltas({"gone": -0.09}, {"new": -0.01}, ["gone"], [1.0], [1.0])
    text = "\n".join(cells.render_table("x", report, limit=0))
    assert "not carried" in text
    assert "cells the candidate adds: new" in text
    assert "cells the candidate drops: gone" in text


def test_the_table_refuses_to_rank_rows_that_all_moved_nothing():
    """Thirty-six identical rows sorted by zero is noise dressed as a table."""
    report = _deltas({"a": -0.05, "b": -0.04}, {"a": -0.05, "b": -0.04},
                     ["a", "b"], [1.0, 1.0], [1.0, 1.0])
    short = cells.render_table("twin", report, limit=12)
    assert not any(line.startswith("  a ") for line in short)
    assert any("--all" in line for line in short)
    # And --all still prints every value, so no capability is lost.
    assert any(row["cell"] == "a" for row in report["rows"])
    assert any("-0.050000" in line for line in cells.render_table("twin", report, limit=0))


def test_the_placebo_candidate_on_this_tree_moves_no_point_while_moving_every_bound():
    """The measured pair this block exists to separate, pinned to the real files.

    Its credible bounds move on all 72 bounds, which the adoption checks have
    always reported. Its points move on none, which nothing could see before,
    and reporting only the first invites the reading that the second is why.
    """
    shipped = _artifact("models/tv_break_coefficients.json")
    candidate = _artifact("models/candidates/tv_break_coefficients_placebo_corrected.json")
    report = _deltas(shipped["coefficients"], candidate["coefficients"],
                     list(shipped["coefficients"]),
                     np.ones(len(shipped["coefficients"])),
                     np.ones(len(shipped["coefficients"])))
    assert report["summary"]["cells_compared"] == 36
    assert report["summary"]["cells_moved"] == 0


def test_the_afterwindow_candidate_on_this_tree_moves_every_one_of_its_36_cells():
    """The measured absence this module closed: 36 of 36, reported nowhere."""
    shipped = _artifact("models/tv_break_coefficients.json")
    candidate = _artifact("models/candidates/tv_break_coefficients_afterwindow.json")
    report = _deltas(shipped["coefficients"], candidate["coefficients"],
                     list(shipped["coefficients"]),
                     np.ones(36), np.ones(36))
    summary = report["summary"]
    assert summary["cells_compared"] == 36 and summary["cells_moved"] == 36
    assert summary["max_abs_delta_at"] == "PrimeShow2_first_short"
    assert summary["max_abs_delta"] == pytest.approx(0.014066885, abs=1e-9)


def test_no_cell_key_on_this_tree_names_a_channel():
    """The competitor boundary, measured on the keys rather than asserted.

    The keys look like PrimeShow2_first_short and a reader who has not been told
    will read the first part as a channel. Every key is a programme class, a
    break position and a break length, and the four classes are enumerated here
    so a key shaped like anything else fails rather than shipping.
    """
    shipped = _artifact("models/tv_break_coefficients.json")
    classes = {"News", "Other", "PrimeShow1", "PrimeShow2"}
    positions = {"first", "middle", "last"}
    lengths = {"short", "standard", "long"}
    for key in shipped["coefficients"]:
        programme, position, length = key.rsplit("_", 2)
        assert programme in classes, key
        assert position in positions and length in lengths, key


def test_the_rescore_carries_the_coefficient_delta_onto_every_candidate_row(tmp_path):
    """Computed where the errors are, so a moved cell can be attributed."""
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(
        json.dumps({"coefficients": {"a": 0.1, "b": -0.2}, "metadata": {}, "detail": {}}),
        encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_moved.json").write_text(
        json.dumps({"coefficients": {"a": 0.3, "b": -0.2}, "metadata": {}, "detail": {}}),
        encoding="utf-8")
    frame = pd.DataFrame({"channel_name": ["a", "b"] * 8,
                          "log_effect": [0.1, -0.3] * 8,
                          "break_start": pd.date_range("2024-11-01", periods=16, freq="h")})
    row = rescore.rescore(rescore.Paths(root=tmp_path), frame)["candidates"][0]
    summary = row["cell_deltas"]["summary"]
    assert summary["cells_moved"] == 1 and summary["max_abs_delta_at"] == "a"
    assert {entry["cell"] for entry in row["cell_deltas"]["rows"]} == {"a", "b"}


def test_candidate_row_tells_an_unknown_name_apart_from_an_unscored_candidate(tmp_path):
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_known.json").write_text(
        json.dumps({"coefficients": {"a": 0.1}}), encoding="utf-8")
    paths = rescore.Paths(root=tmp_path)
    assert rescore.candidate_row("nobody", paths) is None
    assert rescore.candidate_row("known", paths) == {"id": "known"}


@pytest.mark.parametrize("table", ["CELL_READING"])
def test_every_authored_string_the_coefficient_delta_emits_has_both_halves(table):
    for key, entry in getattr(words, table).items():
        assert entry.get("en", "").strip(), f"{table}.{key} has no English"
        assert entry.get("he", "").strip(), f"{table}.{key} has no Hebrew"


def test_the_two_standalone_sentences_carry_both_halves_too():
    for pair in (words.CELL_KEY_SHAPE, words.CELL_READ_BY):
        assert pair["en"].strip() and pair["he"].strip()
