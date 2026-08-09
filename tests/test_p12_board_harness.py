"""The plumbing the board's browser measurements share.

Three files drive this piece's panel in a real browser and one copy of the
stand-in, the harness script and the payload shapes serves all of them. It was
inside ``test_p12_board_page.py`` until that file passed 450 lines, which is the
size at which a file is split rather than compressed.

It holds no test of its own. It is named ``test_p12_*`` because that is the
naming this piece's row owns, exactly as P7's own ``test_p7_console_bridge_harness``
is, and pytest collects nothing from it.

The lower plumbing is P7's ``test_p7_console_bridge_harness``, imported rather
than copied. It is a frozen file, so depending on it cannot rot, and copying a
hundred and twenty lines of stand-in server would be the worse failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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

mountBoard(document.getElementById('root'), '%(locale)s', %(payload)s);

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
      reference_note: (one('.cb-reference-note') || {}).innerText || '',
      money_cells: all('.cb-row:not(.cb-reference) td:nth-child(7)').map((node) => node.innerText),
    }),
  });
}

requestAnimationFrame(step);
"""


def read_board():
    return json.loads(BOARD_JSON.read_text(encoding="utf-8"))


def served_payload(board, *, shipped_digest=None, candidate_digests=None):
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


def drive_board(tmp_path, payloads, locale="he", allow_unknown="false", pick_name="",
                board=None):
    """Drive the panel in a browser.

    ``board`` renders a payload other than the published one, and is the only
    way to measure a rule against a shape the shipped file does not contain: the
    file carries one model version name and it is a calendar day, so the rule
    for a version named anything else has nothing on this tree to be measured
    on. Left None the panel renders the published board, which is what every
    measurement that is about THIS tree's figures uses.
    """
    skip_unless_a_real_browser_is_available()
    work = tmp_path.resolve()
    (work / "src").mkdir(parents=True, exist_ok=True)
    script = HARNESS_JS % {
        "board": os.path.relpath(BOARD_MOUNT, work / "src"),
        "locale": locale,
        "allow_unknown": allow_unknown,
        "pick_name": pick_name,
        "payload": "undefined" if board is None else json.dumps(board),
    }
    return run_scenario(build_harness(work, script), work, payloads)

