"""The published board: one source, no act, both languages, inside the row.

The board is a snapshot of a measurement, imported by a panel at build time
because the console's routes are frozen and this piece cannot publish one. A
snapshot is only honest if three things hold, and each is a test here: every
figure in it comes from the stored measurement rather than being recomputed on
the way out, nothing in it names the act that writes under models/, and the file
itself carries no channel but the operator's own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import adopt_candidate_board as board
from scripts import adopt_candidate_ownership as ownership
from scripts import adopt_candidate_rescore as rescore

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates"
BOARD_JSON = BOARD_DIR / "candidate-board.json"
WORDS = BOARD_DIR / "board-words.js"

# What the panel may import. Everything else is either its own or a path it does
# not own, and a panel reaching into a frozen module by any other door is the
# thing the contract exists to prevent.
#
# `shell/bidi` and `shell/styles.css` are the direction primitive and the
# stylesheet that defines its three classes. They arrived when the lead's
# isolation sweep rewired every surface in the product onto one home for
# direction, this board included, and that is the correct dependency: consuming
# a frozen shell primitive read-only is what `shell/format` beside them already
# does. What was not correct was that the sweep landed and this guard did not
# know, so the piece shipped a red test.
#
# `shell/dates` joined them for the same reason. The design rules say dd/mm/yyyy
# in both locales and name one file that decides it, and this board was printing
# ISO instants sliced to nineteen characters in four places. The product's own
# date guard could not catch it: its raw-ISO rule names eight calendar-DAY fields
# and these are instants, so the rule that exists for exactly this class of
# defect passed the board while the board was committing it.
#
# `shell/card.css` arrived the same way and was caught the same way: it landed
# on 2026-08-08 while this round was building, when the card became one home
# with one inset, and the sweep wired every mount point in the product to it
# including this row's. The dependency is correct and it is the mount point's
# rather than the panel's, which is the division `board-mount.jsx` already
# states: a mount point loads the sheets a page needs and a panel reads them.
ALLOWED_IMPORTS = {
    "react", "react-dom/client",
    "../../shell/format", "../../shell/bidi", "../../shell/dates",
    "../../studio/dom-controls",
    "../console/console-api",
    "../../tokens.css", "../../shell/styles.css", "../../shell/card.css",
}

# Both doors, which is the point. The guard read `from '<target>'` only, so every
# side-effect import in the tree was invisible to it: `import '../../tokens.css'`
# was on the allowed list and had never once been matched by the pattern that
# checks the list, and `import '../../shell/styles.css'` entered the row without
# the guard seeing it at all. A rule that cannot see half of what it governs is
# not a weaker rule, it is a rule about a different thing.
IMPORT_PATTERN = re.compile(r"(?:from|import)\s+'([^']+)'")

# Written as an escape rather than as the character, for the reason
# shell/bidi.jsx gives for the isolate characters it holds and test_p12_basis.py
# now gives for this one: a file that bans a mark should not be the one file in
# the tree that contains it, or every sweep for that mark finds its own guard.
# The reasoning was applied to one of this piece's three guard files and the
# other two kept the literal.
EM_DASH = "\u2014"


def imports_outside(directory: Path, allowed: set[str]) -> dict[str, list[str]]:
    """Every import in a directory that the allowed set does not name.

    Relative imports inside the row are the piece's own and are skipped. Exposed
    rather than inlined so a test can point it at a scratch copy and prove the
    guard bites, which is the only way to know a guard works.
    """
    outside: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.js*")):
        for target in IMPORT_PATTERN.findall(path.read_text(encoding="utf-8")):
            if target.startswith("./"):
                continue
            if target not in allowed:
                outside.setdefault(path.name, []).append(target)
    return outside


def _published():
    if not BOARD_JSON.is_file():
        pytest.skip("no board has been published on this tree")
    return json.loads(BOARD_JSON.read_text(encoding="utf-8"))


def _stored():
    return rescore.load_rescore(rescore.Paths()) or {}


def test_the_board_file_is_on_this_piece_ownership_row():
    paths = rescore.Paths()
    assert ownership.may_write(paths.root, board.board_path(paths), paths.releases_dir)
    assert board.board_path(paths) == BOARD_JSON


def test_every_score_on_the_board_is_the_stored_measurement_to_the_last_digit():
    published, stored = _published(), _stored()
    scores = {row["id"]: row for row in stored.get("candidates") or []}
    assert published["shipped"]["rmse"] == (stored.get("shipped") or {}).get("rmse")
    assert published["shipped"]["sha256"] == (stored.get("shipped") or {}).get("sha256")
    assert published["measured_at"] == stored.get("measured_at")
    assert published["fingerprint"] == stored.get("fingerprint")
    assert len(published["candidates"]) == len(scores)
    for row in published["candidates"]:
        score = scores[row["id"]]
        assert row["rmse"] == score["rmse"]
        assert row["sha256"] == score["sha256"]
        assert row["rmse_delta"] == (score["paired"] or {})["rmse_delta"]
        assert row["paired_statistic"] == (score["paired"] or {})["paired_statistic"]
        assert row["fold_dispersion"] == (score["paired"] or {})["fold_dispersion"]
        assert row["verdict"] == (score["verdict"] or {})["state"]
        summary = (score["cell_deltas"] or {})["summary"]
        assert row["cells"]["moved"] == summary["cells_moved"]
        assert row["cells"]["compared"] == summary["cells_compared"]
        assert row["cells"]["cancelled_share"] == summary["cancelled_share"]


def test_the_cell_rows_it_carries_are_the_ranked_head_and_it_says_of_how_many():
    published, stored = _published(), _stored()
    scores = {row["id"]: row for row in stored.get("candidates") or []}
    for row in published["candidates"]:
        rows = (scores[row["id"]]["cell_deltas"] or {})["rows"]
        cells = row["cells"]
        assert cells["top_of"] == len(rows)
        assert len(cells["top"]) <= board.TOP_CELLS
        moved = [abs(float(item["squared_error_delta"])) for item in cells["top"]]
        assert moved == sorted(moved, reverse=True)
        largest = max((abs(float(item["squared_error_delta"])) for item in rows), default=0.0)
        if rows:
            assert abs(moved[0] - largest) < 1e-12


def test_a_candidate_that_moves_nothing_carries_no_cancellation_share():
    """Tri-state on the board as well as in the payload it was published from."""
    published = _published()
    still = [row for row in published["candidates"] if row["cells"]["moved"] == 0]
    assert still, "no candidate on this tree moves nothing, so this proves nothing"
    for row in still:
        assert row["cells"]["cancelled_share"] is None


def test_a_stale_money_row_carries_the_magnitude_and_never_a_current_figure():
    published = _published()
    stale = [row for row in published["candidates"] if row["money"]["state"] == "stale"]
    assert stale, "no stale money on this tree, so this proves nothing"
    for row in stale:
        assert row["money"]["revenue_delta"] is None
        assert isinstance(row["money"]["last_known_revenue_delta"], (int, float))
        assert row["money"]["reason_he"].strip()


def test_the_board_names_no_path_into_the_act_that_writes_under_models():
    text = BOARD_JSON.read_text(encoding="utf-8")
    assert board.offending_names(text) == []
    assert "--perform" not in text
    for path in sorted(BOARD_DIR.glob("*.js*")):
        assert board.offending_names(path.read_text(encoding="utf-8")) == []


def test_the_write_is_refused_when_the_payload_names_the_act(tmp_path):
    paths = rescore.Paths(root=tmp_path)
    with pytest.raises(board.ActNamedInAPublishedFile):
        board.save_board({"note": "run scripts/adopt_candidate.py show"}, paths)
    assert not board.board_path(paths).exists()


def test_publishing_writes_exactly_one_file_and_it_is_the_board(tmp_path):
    paths = rescore.Paths(root=tmp_path)
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    board.save_board({"candidates": []}, paths)
    written = {path for path in tmp_path.rglob("*") if path.is_file()} - before
    assert {path.relative_to(tmp_path).as_posix() for path in written} == {
        f"{board.BOARD_DIR}/{board.BOARD_FILE}"}


def test_publishing_writes_nothing_under_models(tmp_path):
    """So the act that publishes the screen is not training by section 4.1's test."""
    paths = rescore.Paths(root=tmp_path)
    board.save_board({"candidates": []}, paths)
    assert not (tmp_path / "models").exists()


def test_no_rival_channel_is_in_the_published_file():
    pytest.importorskip("pandas")
    from kairos.data.loaders import load_spots
    from kairos_api import channel_scope

    channels = {str(name) for name in load_spots()["Channel"].unique()}
    rivals = channels - {channel_scope.operator_channel()}
    assert rivals, "no rival channel in the sources, so this test proves nothing"
    text = BOARD_JSON.read_text(encoding="utf-8")
    assert [name for name in rivals if name in text] == []


def test_every_word_on_the_board_carries_both_halves():
    text = WORDS.read_text(encoding="utf-8")
    entries = re.findall(r"'([\w.]+)': \{ en: (.+?), he: (.+?) \},", text)
    assert len(entries) > 40, "the words table did not parse, so this proves nothing"
    for key, english, hebrew in entries:
        assert english.strip(" '\""), key
        assert hebrew.strip(" '\""), key
        assert re.search(r"[֐-׿]", hebrew), key


def test_the_panel_imports_nothing_it_does_not_own_beyond_the_published_surface():
    assert imports_outside(BOARD_DIR, ALLOWED_IMPORTS) == {}


def test_the_import_guard_catches_a_side_effect_import_and_not_only_a_from(tmp_path):
    """Prove the guard bites, on the door it used to be blind to.

    A test that has never failed has never been shown to work, and this one did
    not work: the old pattern matched ``from '<target>'`` and the two imports
    that actually reached outside this row are written ``import '<target>';``
    with no ``from`` in them. Both shapes are injected here and both must be
    caught, so a future edit that narrows the pattern back fails here rather
    than silently reopening the door.
    """
    (tmp_path / "panel.jsx").write_text(
        "import React from 'react';\n"
        "import { thing } from '../../shell/frozen-module';\n"
        "import '../../shell/frozen-sheet.css';\n"
        "import local from './board-words';\n",
        encoding="utf-8")
    caught = imports_outside(tmp_path, {"react"})
    assert caught == {"panel.jsx": ["../../shell/frozen-module", "../../shell/frozen-sheet.css"]}


def test_every_allowed_import_is_one_the_row_actually_uses():
    """No headroom in the allowed set.

    An allowance nobody exercises is indistinguishable from an allowance for a
    dependency that has since been removed, and it is where the next unnoticed
    reach lands. Every name on the list has to be matched by something in the
    row, which also means the list cannot be padded in advance of a need.
    """
    used = set()
    for path in sorted(BOARD_DIR.glob("*.js*")):
        used.update(IMPORT_PATTERN.findall(path.read_text(encoding="utf-8")))
    unused = {name for name in ALLOWED_IMPORTS if name not in used}
    assert unused == set()


def _prose(text: str) -> str:
    """Everything a person reads: the comments and the quoted strings.

    The exclamation law is about words, and in JavaScript ``!`` is the negation
    operator, so testing the whole file would ban ``if (!row)``. What it may not
    appear in is a sentence, which is a comment or a string literal, and that is
    what this returns.
    """
    comments = re.findall(r"//[^\n]*|/\*.*?\*/", text, re.S)
    strings = re.findall(r"'([^'\n]*)'|\"([^\"\n]*)\"", text)
    return "\n".join(comments + [part for pair in strings for part in pair])


@pytest.mark.parametrize("path", sorted(path.name for path in BOARD_DIR.glob("*.*")))
def test_the_frontend_files_of_this_piece_keep_the_laws(path):
    target = BOARD_DIR / path
    text = target.read_text(encoding="utf-8")
    assert EM_DASH not in text
    assert not re.search(r"[\U0001F300-\U0001FAFF]", text)
    if target.suffix == ".json":
        # A published measurement is data, not source, so the file-size law does
        # not apply to it. Every other law does, and it is checked whole.
        assert "!" not in text
        return
    assert len(text.splitlines()) < 450, path
    assert "!" not in _prose(text)


def test_the_next_act_block_says_one_thing_once_when_it_is_one_thing():
    """Ten lines to say one thing, under a heading that promises per candidate.

    The act is genuinely computed per row. On this tree every row collapses onto
    the same branch, because none of the candidates has measured money yet, so
    the block printed five identical pairs of lines. That is honest and useless:
    it hides the one fact a steward needs, which is that a single command clears
    the whole shelf.

    When the acts differ the list is the right shape, so both directions are
    asserted here rather than only the one that prompted the change.
    """
    from scripts import adopt_candidate_render as render

    def _row(identifier, en, command):
        return {"id": identifier, "file": f"{identifier}.json", "bytes": 0,
                "produced_on": None, "next_act": {"en": en, "command": command}}

    same = {"candidates": [_row(f"c{n}", "measure the money", "adopt measure") for n in range(5)]}
    text = "\n".join(render._render_notes(same)) if hasattr(render, "_render_notes") else None
    if text is None:
        import pytest

        pytest.skip("the notes renderer is not exposed under that name")
    assert "the same for all 5 candidates" in text
    assert text.count("adopt measure") == 1, "the one command is printed once"

    differing = {"candidates": [_row("c0", "measure the money", "adopt measure"),
                                _row("c1", "decide", "adopt decide")]}
    listed = "\n".join(render._render_notes(differing))
    assert "Next act, per candidate" in listed
    assert "adopt measure" in listed and "adopt decide" in listed
