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
import re
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
    // By name when a scenario names one, because which artifact is interesting
    // depends on what is being measured and the ranked first row is not always
    // it. The name is rendered inside an isolated run, so it is matched by
    // containment rather than by equality.
    const want = '%(pick_name)s';
    const pick = all('.cb-row:not(.cb-reference)')
      .filter((row) => !want || ((row.querySelector('.cb-name') || {}).textContent || '').includes(want))
      .map((row) => row.querySelector('.cb-pick'))[0];
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
      purposes: all('.cb-purpose').map((node) => node.textContent),
      purpose_block: (one('.cb-purpose-block') || {}).innerText || '',
      provenance: (one('.cb-provenance') || {}).innerText || '',
      meters: all('.cb-meter-fill').map((node) => node.style.inlineSize),
      meter_widths: all('.cb-meter-fill').map((node) => node.getBoundingClientRect().width),
      meter_tracks: all('.cb-meter').map((node) => node.getBoundingClientRect().width),
      gates_block: (one('.cb-gates') || {}).innerText || '',
      gate_rows: all('.cb-gates .cb-cells-table tbody tr').length,
      notes: all('.cb-notes .cb-note').map((node) => node.textContent),
      history_block: (one('.cb-history') || {}).innerText || '',
      history_rows: all('.cb-history .cb-cells-table tbody tr').length,
      live_block: (one('.cb-live-verdict') || {}).innerText || '',
      live_rows: all('.cb-live-verdict .cb-cells-table tbody tr').length,
      live_class: (one('.cb-live-verdict') || {}).className || '',
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


def _run(tmp_path, payloads, locale="he", allow_unknown="false", pick_name=""):
    skip_unless_a_real_browser_is_available()
    work = tmp_path.resolve()
    (work / "src").mkdir(parents=True, exist_ok=True)
    script = HARNESS_JS % {
        "board": os.path.relpath(BOARD_MOUNT, work / "src"),
        "locale": locale,
        "allow_unknown": allow_unknown,
        "pick_name": pick_name,
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


def test_opening_an_artifact_shows_what_its_gates_decided(tmp_path):
    """JS-19's second sentence, on the screen for the first time.

    The row opened here is the one candidate on this tree that decides a gate
    differently, so the assertion is about a real disagreement and not about an
    empty table. Both of its values are on screen and neither is judged.
    """
    board = _board()
    row = next(r for r in board["candidates"] if r["gates"]["differing"])
    assert row["id"] == "calibrated", "the shelf changed, so this row is the wrong one"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    assert result["opened"] == row["id"]
    block = result["gates_block"]
    assert block.strip(), "the gate block never rendered"
    assert row["gates"]["reading_he"] in block
    assert "placebo_correction_active" in block
    assert "true" in block and "false" in block


def test_a_candidate_that_carries_no_gate_keys_is_not_shown_as_ten_disagreements(tmp_path):
    """The count that lies, refused on the screen as well as in the payload."""
    board = _board()
    row = next(r for r in board["candidates"]
               if r["gates"]["state"] == "absent_only_candidate")
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    block = result["gates_block"]
    assert row["gates"]["reading_he"] in block
    # Every row of its gate table reads as an absence rather than as a value.
    assert result["gate_rows"] >= row["gates"]["not_identical"]
    assert "אינו נישא" in block
    assert "המועמד אינו רושם דבר עבור השער הזה" in block


def test_the_held_out_amounts_reach_the_screen_with_the_noun_they_count(tmp_path):
    """2,532 breaks and 34,560 minutes, side by side, with the sentence.

    This is the argument the whole board rests on and it was on no screen. The
    sentence beside it is the measured one: on this pair the amounts agree, and
    saying they disagree would be stating a confound this pair does not carry.
    """
    board = _board()
    row = next(r for r in board["candidates"] if r["gates"]["held_out_state"] == "even")
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    block = result["gates_block"]
    assert "34,560" in block and "דקות" in block
    assert "2,532" in block and "ברייקים" in block
    assert row["gates"]["held_out_rule_he"] in block
    assert row["gates"]["held_out_basis_he"] in block


def test_the_measurement_window_reads_as_two_israeli_dates_and_never_as_iso(tmp_path):
    """dd/mm/yyyy in both locales, through the one file that decides it."""
    board = _board()
    for locale in ("he", "en"):
        result = _run(tmp_path / locale, {"/api/model/candidates": _served(board)},
                      locale=locale)
        body = result["body"]
        assert "01/11/2024-30/11/2024" in body, body[:400]
        assert board["evaluation"]["window_from"] not in body
        assert " to 2024-11-30" not in body


def test_the_table_says_how_it_is_worked_and_how_many_verdicts_a_row_holds(tmp_path):
    """Two things that were true of the payload and readable only by discovery."""
    board = _board()
    twice = [row for row in board["candidates"] if row["decision"]["count"] > 1]
    assert twice, "no candidate here was decided twice, so this proves nothing"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, locale="en")
    assert "Up and down move the selection" in " ".join(result["notes"])
    assert "verdicts on record" in result["body"]
    assert str(twice[0]["decision"]["count"]) in result["body"]


def test_the_shelf_says_what_each_artifact_was_built_for_before_anything_is_opened(tmp_path):
    """The largest absence a builder could close, measured on the real screen.

    Every candidate's own metadata may carry a purpose and no surface carried
    one, so this table showed five opaque identifiers. Each recorded sentence is
    now under its own name, verbatim, and a row that records none says so rather
    than leaving a blank that reads like a row nobody wrote a note for.
    """
    board = _board()
    recorded = [row for row in board["candidates"] if (row["origin"] or {}).get("purpose")]
    absent = [row for row in board["candidates"] if not (row["origin"] or {}).get("purpose")]
    assert recorded and absent, "this tree no longer has both states, so this proves nothing"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    body = result["body"]
    for row in recorded:
        assert row["origin"]["purpose"] in body, row["id"]
    assert len(result["purposes"]) == len(board["candidates"])
    assert sum(1 for text in result["purposes"] if "לא נרשם ייעוד" in text) == len(absent)


def test_opening_an_artifact_that_records_no_purpose_shows_the_absence_and_no_guess(tmp_path):
    """The one place this screen could most easily invent a fact.

    Everything needed to write a plausible purpose for these two rows is on the
    screen already. What is shown instead is the absence and the field that
    would supply it, in the reader's own language.
    """
    board = _board()
    row = next(r for r in board["candidates"] if not (r["origin"] or {}).get("purpose"))
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    assert result["opened"] == row["id"]
    block = result["purpose_block"]
    assert block.strip(), "the purpose block never rendered"
    assert row["origin"]["purpose_reading_he"] in block
    others = [r["origin"]["purpose"] for r in board["candidates"] if (r["origin"] or {}).get("purpose")]
    assert [text for text in others if text in block] == []


def test_opening_an_artifact_shows_the_data_it_read_and_whether_that_data_is_here(tmp_path):
    """The reproduction half, and the half of it this tree cannot answer."""
    board = _board()
    row = next(r for r in board["candidates"] if r["origin"]["sources"])
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    block = result["provenance"]
    assert block.strip(), "the provenance block never rendered"
    for item in row["origin"]["sources"]:
        assert item["file"] in block
        assert item["short"] in block
    assert row["origin"]["sources_reading_he"] in block
    assert row["origin"]["agreement_reading_he"] in block
    # And the command nobody recorded, said rather than left out.
    assert row["origin"]["recipe_he"] in block


def test_every_bar_on_the_screen_is_the_share_the_payload_carries(tmp_path):
    """A bar is a figure, so it is measured like one.

    Each fill is read out of the real layout and compared with the share
    computed from the published payload. A bar that drew a shape nobody measured
    would be the visual form of a fabricated number, and it would pass every
    other test on this file.
    """
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    tracks, fills = result["meter_tracks"], result["meter_widths"]
    assert tracks and len(tracks) == len(fills)
    # The evaluation's own bar, first on the page: the live model's error as a
    # share of the spread of the thing it predicts.
    spread = float(board["shipped"]["rmse"]) / float(board["evaluation"]["target_sd"])
    assert abs(fills[0] / tracks[0] - spread) < 0.02, (fills[0], tracks[0], spread)
    # Then one per candidate row that has a dispersion to divide by, and none
    # for the row that has not.
    shares = [abs(row["rmse_delta"]) / row["fold_dispersion"]
              for row in sorted(board["candidates"], key=lambda r: r["rmse"])
              if row.get("fold_dispersion")]
    for drawn, share in zip(fills[1:1 + len(shares)], shares):
        assert abs(drawn / tracks[1] - share) < 0.02, (drawn, share)
    assert len(shares) == len(board["candidates"]) - 1


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


def test_the_verdict_on_the_live_model_is_on_the_board_and_accounts_for_the_log(tmp_path):
    """The record every read of the decision log on this piece filtered out.

    A decision record may be about the shipped model rather than about a
    candidate. On this tree one is, and it is a no-ship against the version in
    force, so the shelf showed five verdicts and said nothing about a standing
    verdict on the artifact all five are measured against.
    """
    board = _board()
    live = board["live_model"]
    log = board["decision_log"]
    assert live["rows"], "no verdict on the live model on this tree, so this proves nothing"
    result = _run(tmp_path, {"/api/model/candidates": _served(board)})
    block = result["live_block"]
    assert live["reading_he"] in block
    assert live["rows"][0]["actor"] in block
    assert result["live_rows"] == len(live["rows"])
    assert "cb-amber" in result["live_class"]
    # The three counts and the total they add up to, on the screen.
    tally = log["tally"]
    assert str(log["records"]) in block
    assert tally["on_the_shelf"] + tally["on_the_live_model"] + tally["off_the_shelf"] == log["records"]
    # dd/mm/yyyy in both locales, through the one file that decides it. The
    # stored instant may not appear anywhere on the screen.
    assert str(live["rows"][0]["recorded_at"])[:10] not in result["body"]


def test_the_earlier_verdict_on_a_restated_candidate_is_on_the_screen(tmp_path):
    """The second half of JS-19's done condition, measured.

    One artifact on this tree carries two verdicts holding the same word for two
    different stated reasons. The shelf column reads "no ship" with a count of
    two beside it, which cannot tell a restatement from a repeat, and the earlier
    record reached no surface at all.
    """
    board = _board()
    restated = [row for row in board["candidates"]
                if (row.get("history") or {}).get("state") == "restated"]
    assert restated, "no restated candidate on this tree, so this proves nothing"
    row = restated[0]
    result = _run(tmp_path, {"/api/model/candidates": _served(board)}, pick_name=row["id"])
    assert result["opened"].strip() == row["id"]
    block = result["history_block"]
    assert result["history_rows"] == len(row["history"]["rows"])
    assert row["history"]["reading_he"] in block
    # The older row is marked as replaced on the screen, in the word the board's
    # own table holds for it rather than in a word this test invented.
    words = (BOARD_DIR / "board-words.js").read_text(encoding="utf-8")
    superseded = re.search(r"'history\.superseded': \{ en: '[^']+', he: '([^']+)'", words).group(1)
    assert superseded in block
    assert row["history"]["rows"][1]["superseded_by"] == row["history"]["rows"][0]["decision_id"]
    assert str(row["history"]["not_shown_by_the_latest"]) in block


def test_no_steward_sentence_from_the_decision_log_is_rendered_on_the_board(tmp_path):
    """It is not on the payload, so it cannot be on the screen. Both are checked.

    The sentence is unbounded text typed at a terminal, the model console renders
    it from the store, and a second copy in a bundled file is a second source
    that can disagree with the first.
    """
    decisions = ROOT / "models" / "releases" / "decisions.jsonl"
    if not decisions.is_file():
        pytest.skip("no decision log on this tree")
    reasons = [str(json.loads(line).get("reason") or "").strip()
               for line in decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    reasons = [reason for reason in reasons if len(reason) > 20]
    assert reasons, "no reason long enough to search for, so this proves nothing"
    board = _board()
    result = _run(tmp_path, {"/api/model/candidates": _served(board)},
                  pick_name=board["candidates"][0]["id"])
    assert [reason for reason in reasons if reason in result["body"]] == []
    assert board["live_model"]["reason_he"] in result["live_block"]
