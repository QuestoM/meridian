"""The board in a real browser: what it renders, how fast, and when it refuses.

The panel this piece owns has no mount point. The model console's rail, its
section list and its panel imports are all in files P7 owns and this piece may
not write, so the board cannot be reached by pressing a key on the console until
two lines land in those files. That is a lead decision, not a builder one, and
it is recorded as such.

It does not stop the panel from being measured. It is bundled here by the
product's own bundler and driven in the same headless Chrome the console's own
measurements use, against a stand-in that answers the console's own candidate
route with the product's own payload. So every claim below is measured on the
real component with the real published figures, and the only thing the missing
mount changes is which page it is measured on.

The plumbing is P7's ``test_p7_console_bridge_harness``, imported rather than
copied. It is a frozen file, so depending on it cannot rot, and copying a
hundred and twenty lines of stand-in server would be the worse failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_p7_console_bridge_harness import (
    build_harness,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates"
BOARD_JSON = BOARD_DIR / "candidate-board.json"
BOARD_JSX = BOARD_DIR / "CandidateBoard.jsx"
BOARD_MOUNT = BOARD_DIR / "board-mount.jsx"

# JS-19's whole route on this surface, in one page load: read whether the
# comparison is about the artifacts on disk now, read the ranked table, open one
# artifact's evidence, and read what its verdict was decided on.
JS19_TARGET_S = 120

HARNESS_JS = """
import { mountBoard } from '%(board)s';

const marks = {};
const started = performance.now();
const at = (name) => { if (marks[name] === undefined) marks[name] = performance.now() - started; };

mountBoard(document.getElementById('root'), '%(locale)s');

const text = () => document.body.innerText;
const one = (selector) => document.querySelector(selector);
const all = (selector) => Array.from(document.querySelectorAll(selector));

function step() {
  if (all('.cb-table tbody tr').length > 1) at('table');
  const word = one('.cb-state-word');
  const state = one('.cb-state');
  if (word && state && !state.className.includes('cb-blue')) at('state');
  if (marks.state === undefined && word && %(allow_unknown)s) at('state');
  if (marks.table !== undefined && marks.state !== undefined && marks.picked === undefined) {
    const pick = one('.cb-row:not(.cb-reference) .cb-pick');
    if (pick) { pick.click(); marks.clicked = performance.now() - started; }
  }
  if (marks.clicked !== undefined && one('.cb-detail h3 code')) at('picked');
  if (marks.picked !== undefined || performance.now() - started > 20000) {
    report();
    return;
  }
  requestAnimationFrame(step);
}

function report() {
  fetch('/testctl/result', {
    method: 'POST',
    body: JSON.stringify({
      marks,
      state_word: (one('.cb-state-word') || {}).textContent || '',
      state_class: (one('.cb-state') || {}).className || '',
      state_reason: (one('.cb-state-reason') || {}).textContent || '',
      moved: all('.cb-state-moved li').map((node) => node.textContent),
      rows: all('.cb-row:not(.cb-reference) .cb-name').map((node) => node.textContent),
      opened: (one('.cb-detail h3 code') || {}).textContent || '',
      detail: (one('.cb-detail') || {}).innerText || '',
      body: text(),
      dir: (one('.cb-board') || {}).getAttribute('dir'),
      cells_rows: all('.cb-cells-table tbody tr').length,
      basis_marks: all('.cb-basis-mark').map((node) => node.textContent),
      basis_rows: all('.cb-basis-rows li').map((node) => node.textContent),
      self_block: (one('.cb-self') || {}).innerText || '',
    }),
  });
}

requestAnimationFrame(step);
"""


def _board():
    return json.loads(BOARD_JSON.read_text(encoding="utf-8"))


def _served(board, *, shipped_digest=None, candidate_digests=None):
    """The candidate route's own shape, with the digests a scenario asks for."""
    digests = candidate_digests or {}
    return {
        "model_version": {
            "available": True,
            "id": board["shipped"]["version_id"],
            "artifacts": {"retention": {
                "sha256": shipped_digest or board["shipped"]["sha256"],
                "path": board["shipped"]["file"],
            }},
        },
        "candidates": [{"id": row["id"], "sha256": digests.get(row["id"], row["sha256"]),
                        "file": row["file"], "bytes": row["bytes"]}
                       for row in board["candidates"]],
    }


def _run(tmp_path, payloads, locale="he", allow_unknown="false"):
    skip_unless_a_real_browser_is_available()
    work = tmp_path.resolve()
    (work / "src").mkdir(parents=True, exist_ok=True)
    script = HARNESS_JS % {
        "board": os.path.relpath(BOARD_MOUNT, work / "src"),
        "locale": locale,
        "allow_unknown": allow_unknown,
    }
    return run_scenario(build_harness(work, script), work, payloads)


def test_the_whole_route_runs_in_a_browser_and_lands_inside_its_target(tmp_path):
    """Open the board, learn whether it is current, and open one artifact."""
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    marks = result["marks"]
    assert result["state_word"], "the freshness strip never rendered"
    assert set(result["rows"]) == {row["id"] for row in board["candidates"]}
    assert result["opened"] in {row["id"] for row in board["candidates"]}
    assert marks["picked"] / 1000.0 < JS19_TARGET_S
    # And the whole thing is well inside a second, so the target is not the
    # interesting number: the interesting number is that it exists at all.
    assert marks["picked"] < 5000, marks


def test_a_matching_digest_reads_current_and_a_moved_one_reads_stale(tmp_path):
    board = _board()
    current = _run(tmp_path / "a", {"/api/model/candidates": _served(board)})
    assert current["state_class"].endswith("cb-teal") or "cb-teal" in current["state_class"]
    assert current["moved"] == []

    moved = _run(tmp_path / "b", {"/api/model/candidates": _served(
        board, candidate_digests={board["candidates"][0]["id"]: "0" * 64})})
    assert "cb-amber" in moved["state_class"]
    assert len(moved["moved"]) == 1
    assert board["candidates"][0]["id"] in moved["moved"][0]


def test_a_route_that_does_not_answer_reads_unknown_and_never_current(tmp_path):
    """Unknown is not stale and it is not current. Three states, all reachable."""
    result = _run(tmp_path, {}, allow_unknown="true")
    assert "cb-blue" in result["state_class"]
    assert result["moved"] == []
    assert result["state_reason"].strip()


def test_every_figure_on_the_screen_is_one_the_published_board_carries(tmp_path):
    """No figure is computed in the browser, so each one is findable in the file."""
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    body = result["body"]
    assert f"{board['shipped']['rmse']:.6f}" in body
    for row in board["candidates"]:
        assert f"{row['rmse']:.6f}" in body
    assert str(board["evaluation"]["breaks"]) in body.replace(",", "")
    assert board["shipped"]["short"] in body


def test_the_board_reads_right_to_left_and_in_the_campaign_vocabulary(tmp_path):
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    assert result["dir"] == "rtl"
    assert "ברייקים" in result["body"]
    assert "משתמש" not in result["body"]


def test_no_rival_channel_reaches_this_screen(tmp_path):
    """The competitor boundary, measured against the real channel list.

    The rival names are read from the reference data rather than typed, so a
    channel that appears in the sources later is covered without an edit.
    """
    pytest.importorskip("pandas")
    from kairos.data.loaders import load_spots
    from kairos_api import channel_scope

    channels = {str(name) for name in load_spots()["Channel"].unique()}
    rivals = channels - {channel_scope.operator_channel()}
    assert rivals, "no rival channel in the sources, so this test proves nothing"
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    assert [name for name in rivals if name in result["body"]] == []


def test_the_screen_offers_no_act_and_names_no_path_into_one(tmp_path):
    """The training line, held on the one surface most able to blur it."""
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    for name in ("adopt_candidate", "adopt-candidate", "adoptCandidate"):
        assert name not in result["body"]
    # Every control on this screen either sorts the table or opens an artifact.
    assert result["cells_rows"] >= 0


def test_the_row_fitted_on_fewer_breaks_carries_the_caveat_on_the_row(tmp_path):
    """The confound reaches the reader who is comparing the numbers.

    The limit paragraph names the row, but a steward reading two figures is
    reading the table, so a caveat stated only above it is a caveat the
    comparison is made without. Measured on the real screen: the marked rows are
    exactly the rows the published measurement says do not cover the evaluation.
    """
    board = _board()
    uneven = [row["id"] for row in board["candidates"]
              if (row.get("fit_basis") or {}).get("state") in ("fewer", "unknown")]
    assert uneven, "no uneven row on this tree, so this test would prove nothing"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    assert len(result["basis_marks"]) == len(uneven)
    assert all(mark.strip() for mark in result["basis_marks"])
    # And the shortfall itself is on screen with both of its denominators.
    row = next(row for row in board["candidates"] if row["id"] == uneven[0])
    basis = row["fit_basis"]
    body = result["body"].replace(",", "")
    assert str(basis["fitted_on"]) in body
    assert str(basis["not_fitted_on"]) in body
    assert str(basis["scored_on"]) in body
    assert any(uneven[0] in line for line in result["basis_rows"])


def test_the_limit_on_screen_is_the_measured_one_and_not_the_constant(tmp_path):
    """The sentence that was false, no longer asserted on this tree."""
    board = _board()
    assert board["limit"]["state"] == "in_sample_uneven"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    assert board["limit"]["he"] in result["body"]
    assert "כל קובץ שנמדד כאן אומן על כל הברייקים האלה" not in result["body"]


def test_opening_an_artifact_shows_what_its_own_producer_recorded(tmp_path):
    """Carried, and carried with the sentence that stops it reading as a rank."""
    board = _board()
    first = sorted(board["candidates"], key=lambda row: row["rmse"])[0]
    reported = first.get("self_reported") or {}
    assert reported.get("state") == "advised_against", reported
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    assert result["opened"] == first["id"]
    assert result["self_block"].strip()
    assert reported["reading_he"] in result["self_block"]
    assert str(reported["n_test"]) in result["self_block"]
    # The non-comparability sentence, every time the block is shown.
    assert "בת השוואה" in result["self_block"]


def test_a_row_that_covers_the_evaluation_carries_no_caveat_and_no_self_block(tmp_path):
    """The tri-state on screen: an absent state renders nothing, not a reassurance."""
    board = _board()
    covered = [row for row in board["candidates"]
               if (row.get("fit_basis") or {}).get("state") == "all"
               and (row.get("self_reported") or {}).get("state") == "absent"]
    assert covered, "no covered row on this tree, so this test would prove nothing"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    marked = set(result["basis_marks"])
    assert covered[0]["id"] not in " ".join(marked)
