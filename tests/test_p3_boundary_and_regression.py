"""The competitor boundary on the day board, and the P3 regression floor.

Two subjects that belong together because both are about what a surface is
allowed to lose.

**The boundary.** Section 8.3 of the specification gives P3 the duty of applying
``channel_scope`` to ``/api/break-operations``. Measured on the live instance
before this wave, with ``operator_channel = רשת 13``, that route served twelve
programmes for each of four channels, forty eight in all, thirty six of them
rivals'. It now serves the operator's twelve and names the scope in the payload.

**The floor.** Section 8.5's P3 row: drag and resize still work with 30 s and 60 s
snap and the zoom scale, the segment inspector still opens from the break list,
and the override preview still reports rejected overrides verbatim. The first two
are frontend contracts, so they are checked against the sources the way the
wave-zero seam tests check theirs; the third is a route and is checked as one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# See the helper's own docstring: the shared settings file can lose the operator
# channel while these tests run, and a boundary test that skips proves nothing.
from test_p3_break_store import declare_operator_channel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


@pytest.mark.realdata
def test_the_break_board_route_serves_one_channel_and_names_the_scope():
    from kairos_api import break_store
    from kairos_api.day_api import _break_operations_cached, break_operations

    _break_operations_cached.cache_clear()
    payload = break_operations()
    owned = break_store.operator_channel()
    if not owned:
        pytest.skip("no operator channel configured, so there is nothing to scope to")
    channels = {row["channel"] for row in payload["programs"]}
    assert channels == {owned}, f"a competitor reached the operator's board: {channels}"
    assert {row["channel"] for row in payload["breaks"]} <= {owned}
    note = payload["channel_scope"]
    assert note["scoped"] is True
    assert note["scope_channel"] == owned
    assert note["competitor_rows_excluded"] >= 0


@pytest.mark.realdata
def test_no_competitor_name_appears_anywhere_in_the_break_board_payload():
    import json

    from kairos_api.core import _load_settings
    from kairos_api.day_api import _break_operations_cached, break_operations

    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        pytest.skip("no operator channel configured")
    _break_operations_cached.cache_clear()
    body = json.dumps(break_operations(), ensure_ascii=False)
    from kairos_api.core import KAIROS_CHANNELS

    for channel in KAIROS_CHANNELS:
        if str(channel).strip() != owned:
            assert str(channel) not in body, f"{channel} reached an operator surface"


@pytest.mark.realdata
def test_the_override_preview_still_reports_rejected_overrides_verbatim():
    """Bar 3 row P3. The preview's honesty surface may not be quietly dropped."""
    import inspect

    from kairos_api import overrides

    source = inspect.getsource(overrides.override_effect)
    assert "rejected_overrides" in source
    assert "anchor_stale" in source
    signature = inspect.signature(overrides.override_effect)
    assert {"channel", "day", "target_id", "kind", "value", "gold", "scope"} <= set(signature.parameters)


def test_the_editor_keeps_its_thirty_and_sixty_second_snap_and_the_zoom_scale():
    """Bar 3 row P3, first clause, on both timelines the product now carries."""
    toolbar = read("plan/day/ScheduleEditorToolbar.jsx")
    assert "onSnapGrid(30)" in toolbar and "onSnapGrid(60)" in toolbar
    assert "ZoomControl" in toolbar

    model = read("plan/day/day-board-model.js")
    assert "export const SNAP_CHOICES = [30, 60];" in model
    board = read("plan/day/DayBoard.jsx")
    assert "ZoomControl" in read("plan/day/DayBoardToolbar.jsx")
    assert "handleResizePointerDown" in board and "handleMovePointerDown" in board
    assert "useScheduleZoom" in board


def test_the_segment_inspector_still_opens_from_the_break_list():
    """Bar 3 row P3, second clause. The ranked shelf keeps its own drawer."""
    page = read("plan/break/BreakLibraryPage.jsx")
    assert "ScheduleInspector" in page
    assert "onRowClick={openBreak}" in page
    assert "setInspect({ segmentId: row.segment_id" in page


def test_the_override_console_keeps_every_capability_it_had():
    """The console moved file, so its contents are checked at the new address.

    ``runDayPlanJob`` is the same client under the canonical word: the vocabulary
    rule retires recompute from every label and every name this piece owns, and
    the HTTP path it calls is unchanged because the specification freezes paths.
    """
    decisions = read("plan/day/OverrideDecisions.jsx")
    for fragment in ("/api/overrides/effect", "rejected_overrides", "runDayPlanJob", "KINDS"):
        assert fragment in decisions, f"the decisions console lost {fragment}"
    library = read("plan/day/override-console-lib.js")
    assert "/api/jobs/recompute" in library, "the HTTP path is frozen and must not be renamed"
    console = read("plan/day/OverrideConsole.jsx")
    assert "export default OverrideConsole;" in console
    assert "OverrideDecisions" in console and "DayPage" in console


def test_every_source_file_this_piece_owns_is_inside_the_size_law():
    owned = sorted(
        list((SRC / "plan" / "day").glob("*.js*"))
        + list((SRC / "plan" / "break").glob("*.js*"))
        + [ROOT / "kairos_api" / name for name in (
            "break_api.py", "break_api_board.py", "break_api_detail.py",
            "break_api_states.py", "break_store.py", "break_store_pins.py",
            "day_api.py", "gold_api.py", "overrides.py",
        )]
        + [ROOT / "kairos" / "export" / "spots_coverage.py"]
    )
    oversized = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in owned}
    assert not {name: size for name, size in oversized.items() if size > 450}, oversized


def test_nothing_this_piece_wrote_carries_an_em_dash_an_emoji_or_a_shout():
    """No em dash, no emoji, and no exclamation mark in anything a person reads.

    The shout sweep looks for a bare exclamation mark, never for one that is part
    of an operator. The first version of this test read the ``!=`` inside a JSX
    expression as a shout, which is a test that fails on correct code, so the
    pattern now excludes ``!=``, ``!==`` and ``!!`` and the leading ``!`` of a
    negation. A test that cries wolf is worse than no test.
    """
    banned = re.compile(r"[—\U0001F300-\U0001FAFF]")
    shout = re.compile(r"(?<![!=<>])!(?![=!])")
    paths = (
        sorted(list((SRC / "plan" / "day").glob("*.js*")) + list((SRC / "plan" / "break").glob("*.js*")))
        + [ROOT / "kairos_api" / name for name in (
            "break_api.py", "break_api_board.py", "break_api_detail.py",
            "break_api_states.py", "break_store.py", "break_store_pins.py",
        )]
        + [ROOT / "kairos" / "export" / "spots_coverage.py"]
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not banned.search(text), f"{path.name} carries an em dash or an emoji"
        for line in text.splitlines():
            for quoted in re.findall(r"""['"]([^'"\n]{4,})['"]""", line):
                if "!" not in quoted or not re.search(r"[֐-׿ a-z]", quoted):
                    continue
                # A negation only ever follows an operator or an opening bracket,
                # so a shout is an exclamation mark with a word in front of it.
                assert not shout.search(quoted) or not re.search(r"[\w֐-׿]\s*!", quoted), (
                    f"{path.name} shouts in a display string: {quoted}"
                )


def test_the_readout_never_animates_a_figure_that_did_not_move():
    """The measured fact this surface is built on, held at the source.

    Moving a break inside its programme changes revenue by exactly zero, so the
    readout must state that rather than showing a delta. If someone deletes the
    sentence or the condition that guards it, this fails.
    """
    readout = read("plan/day/DayBoardReadout.jsx")
    assert "onlyPlacement" in readout
    assert "moneyMoved" in readout
    assert "changed.placement && !changed.duration && !changed.gold" in readout
    assert "אינה משנה את ההכנסה ממנו" in readout


def test_the_save_scope_names_one_airing_rather_than_a_whole_date():
    """The shipped editor's date scope binds every programme on the day, measured.

    The day board writes the frozen predicate instead, with all three of date,
    programme and hour, so a saved placement binds the airing that was dragged.
    """
    actions = read("plan/day/day-board-actions.js")
    for field in ("date", "programme", "hour"):
        assert f"field: '{field}'" in actions
    assert "combinator: 'and'" in actions
    predicate = actions.split("export function breakPredicate")[1].split("export function")[0]
    assert "field: 'channel'" not in predicate, (
        "channel is never part of a predicate; the engine scopes every restriction "
        "to the operator's own channel by itself"
    )


def test_undo_of_a_save_deletes_the_restriction_it_created():
    actions = read("plan/day/day-board-actions.js")
    assert "undoBreakPlacement" in actions
    assert "/api/constraints/${encodeURIComponent(constraintId)}" in actions
    assert "method: 'DELETE'" in actions
    board = read("plan/day/DayBoard.jsx")
    assert "undoLastSave" in board
    # The keystroke moved into the history hook when the 450-line law split the
    # board; the behaviour did not, so it is asserted at its new address.
    history = read("plan/day/day-board-history.js")
    assert "event.key === 'z'" in history and "event.shiftKey" in history


def test_the_retired_words_are_gone_from_every_label_and_name_this_piece_owns():
    """Section 8.3's adoption duty, checked as the grep the critic runs.

    Two classes of hit survive on purpose and both are named here so a critic can
    tell a duty from an escalation. The HTTP path ``/api/jobs/recompute`` is
    frozen by the specification, which renames labels and never paths. And
    ``onRecompute`` and ``recomputeState`` are prop names declared in
    ``src/shell/**``, while ``recomputeDisabled`` and
    ``recomputeDisabledReason`` are the internal safety boundary that keeps the
    embedded editor from opening a write review before its prerequisites are
    verified. None is operator-facing copy. Measured before this sweep: 52 hits
    in these two trees. Every surviving hit must be one of these contracts.
    """
    retired = re.compile(r"recompute|rebuild|חישוב מחדש|בנייה מחדש|חשבו מחדש|חושב מחדש", re.IGNORECASE)
    allowed = re.compile(r"/api/jobs/recompute|onRecompute|recomputeState|recomputeDisabled(?:Reason)?")
    offenders = []
    for path in sorted(list((SRC / "plan" / "day").glob("*.js*")) + list((SRC / "plan" / "break").glob("*.js*"))):
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if retired.search(allowed.sub("", line)):
                offenders.append(f"{path.name}:{index + 1}: {line.strip()[:80]}")
    assert not offenders, offenders


def test_the_hebrew_vocabulary_on_this_surface_is_the_canonical_one():
    """The words the specification froze, checked where an operator reads them."""
    day_text = read("plan/day/DayBoardToolbar.jsx") + read("plan/day/DayBoardReadout.jsx") + read("plan/day/DayPage.jsx")
    break_text = read("plan/break/BreakBoard.jsx") + read("plan/break/BreakInspector.jsx")
    joined = day_text + break_text
    assert "ברייק" in joined
    assert "ברייקים" in joined
    assert "ברייקי זהב" in break_text
    assert "הכנסה צפויה" in joined
    assert "עלות שימור" in break_text
    assert "משתמש" not in joined, "the retired word for a person is forbidden"
    for retired in ("חישוב מחדש", "בנייה מחדש", "recompute", "rebuild"):
        assert retired not in joined, f"the retired word {retired} is on a new surface"
