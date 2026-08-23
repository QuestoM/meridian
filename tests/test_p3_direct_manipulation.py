"""The mechanics the day board is graded on, held at the source.

The reference for this destination is a drawing tool for selection, drag and
undo, and a professional editing timeline for exact durations. Both references
have concrete mechanics, and a mechanic that is not asserted anywhere is a
mechanic the next edit can quietly remove. So each one is pinned here against
the source that implements it, the way the wave-zero seam tests pin theirs.

Nothing here is a screenshot check. Each assertion names a behaviour a critic
can also reproduce in a browser: type a clock and the break moves there, press
an arrow and it steps by a published increment, press Cmd Z and it goes back.

This file carries the gesture and the scope a saved gesture writes. Three
neighbours carry the rest, and the four are the 450-line law applied to what was
one 775-line file: ``test_p3_framing.py`` for the scale a chip is drawn at and the
two framings, ``test_p3_save_scope.py`` for the save path this file's last test
measures, and ``test_p3_surface_readout.py`` for what a person reads once the
gesture is done. Nothing moved in behaviour.

The last two tests are the ones a critic had to write for this piece. The editor
saved a placement scoped to the whole broadcast date, no test asserted what a
saved body binds, and the resolver matches a date scope against every segment on
the day. One holds the rule on every row the editor draws, the other holds the
exact figures on the one chip the critic dragged.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest

# The scaffolding is the save-scope file's, so both files measure one thing one
# way: the same reader of the shipped source, the same runner that executes the
# shipped module in node rather than restating it, and the same fixtures, which
# put every store write below in a temporary directory and never in the
# operator's own placements.
from test_p3_save_scope import (
    bound_segments,
    client,  # noqa: F401
    editor_rows,
    isolated,  # noqa: F401
    node_board_model,
    node_pin_bodies,
    read,
    same_airing_key,
)


def test_both_the_start_and_the_length_are_typed_targets_not_readouts():
    """The editing timeline's device: the position is a field, not a display."""
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "<output" not in toolbar, "a number a person may want to state must not be a readout"
    assert "commitStart" in toolbar and "commitLength" in toolbar
    assert "parseClock" in toolbar
    assert toolbar.count("onKeyDown") >= 2, "Enter commits both fields"
    board = read("plan/day/DayBoard.jsx")
    assert "onStart={" in board and "onLength={" in board


def test_a_typed_clock_parses_the_two_forms_a_person_writes_and_refuses_the_rest():
    """The parser is small enough to check exactly, so it is checked exactly."""
    model = read("plan/day/day-board-model.js")
    body = model.split("export function parseClock")[1].split("export function")[0]
    assert "parts.length < 2 || parts.length > 3" in body, "HH:MM and HH:MM:SS, nothing else"
    assert "return null" in body, "a half-typed time must not move a break"
    assert "minute > 59 || second > 59" in body


def test_the_nudge_increments_are_published_where_the_person_reads_them():
    """A keyboard taught in place, on the surface that performs it."""
    model = read("plan/day/day-board-model.js")
    assert "if (event.altKey) return 1;" in model
    assert "if (event.shiftKey) return grid * 5;" in model
    help_surface = read("plan/day/DayPage.jsx") + read("plan/day/DayBreakNavigator.jsx")
    for fragment in ("Shift", "Alt", "G ", "Enter"):
        assert fragment in help_surface, f"the help text stopped naming {fragment}"
    assert "מקשי החיצים" in help_surface, "the keyboard is taught in Hebrew too"


def test_undo_and_redo_are_one_stack_read_in_two_directions():
    model = read("plan/day/day-board-model.js")
    for name in ("emptyHistory", "pushAction", "undoAction", "redoAction"):
        assert f"export function {name}" in model
    assert "future: []" in model, "a new act after an undo drops the redone tail"
    history = read("plan/day/day-board-history.js")
    assert "if (event.shiftKey) redo(); else undo();" in history
    assert "event.metaKey || event.ctrlKey" in history
    board = read("plan/day/DayBoard.jsx")
    assert "useBoardHistory" in board, "the board reaches undo through the one hook that owns it"


def test_a_drag_that_ends_where_it_began_records_nothing():
    drag = read("plan/day/day-board-drag.js")
    assert "if (!moved) return;" in drag
    assert "setSnapMark" in drag, "the snap line is the in-flight signifier"


def test_a_pending_edit_never_shadows_the_break_s_gold_state():
    """Gold is the plan's answer, and an unsaved move must not overwrite it.

    Reproduced in a browser on רשת 13 / 2024-11-01, break 001~1, before this was
    closed: select the break, press ArrowRight once so one unsaved edit exists,
    then press G. The override landed, the day came back gold and two sibling
    chips rendered gold, while this one stayed grey with the toolbar Gold button
    reading off, because the placement edit had snapshotted is_gold false and
    false ?? true is false. The mark was then unclearable from the surface, and
    each further press wrote another identical override row instead of taking it
    off: measured c199a87597c8 and 25906b93d511, both carrying the same note.

    The same sequence is driven here through the shipped module.
    """
    measured = node_board_model("""
      const item = { break_id: 'b', offset_seconds: 100, duration_seconds: 120, is_gold: false };
      const start = m.liveBreak(item, {});
      const edits = m.applyEdit({}, item, { ...start, offsetSeconds: start.offsetSeconds + 60 });
      const served = { ...item, is_gold: true };
      const live = m.liveBreak(served, edits);
      const roundTrip = m.applyEdit(edits, served, { ...live, offsetSeconds: served.offset_seconds });
      const goldAct = m.applyEdit({}, served, { ...live, isGold: false, goldEdit: true });
      process.stdout.write(JSON.stringify({
        edit: edits[item.break_id],
        editedGold: live.isGold,
        stillEdited: live.edited,
        moves: m.movesFrom(edits),
        roundTrip,
        goldActHeld: goldAct[item.break_id],
        goldActLive: m.liveBreak(served, goldAct).isGold,
      }));
    """)
    assert "is_gold" not in measured["edit"], "a placement edit must not carry a copy of the gold state"
    assert measured["editedGold"] is True, "the chip has to render the gold the plan came back with"
    assert measured["stillEdited"] is True, "and it is still an edited break, so the move is not lost"
    assert measured["moves"][0]["is_gold"] is None, "an untouched flag travels as untouched, not as false"
    assert measured["roundTrip"] == {}, "a move back to the plan's own offset still leaves no trace"
    # A gold act on top of the pending move keeps the move and adds its own decision.
    assert measured["goldActHeld"] == {"offset_seconds": 160, "duration_seconds": 120, "is_gold": False}
    assert measured["goldActLive"] is False, "an act that IS a gold act still decides the gold state"


def test_the_gold_button_reads_the_same_live_state_the_chip_does():
    """One source for the mark, so the board and its toolbar cannot disagree.

    Measured in a browser after the fix, same break and same two keystrokes: the
    edited chip renders is-gold with its two siblings, the toolbar button carries
    is-on, and pressing G again clears all three and leaves the override store
    back at its header row.
    """
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "className={live.isGold ? 'day-chip-button is-on' : 'day-chip-button'}" in toolbar
    chip = read("plan/day/DayBoardChip.jsx")
    assert "live.isGold ? 'is-gold' : ''" in chip
    board = read("plan/day/DayBoard.jsx")
    assert "const live = liveOf(item);" in board, "the toolbar and the chip read one live state"
    assert "if (live.isGold) {" in read("plan/day/day-board-writes.js"), "and the act decides from it"
    # Marking is not idempotent at the store, so two presses inside one round
    # trip are coalesced into the one decision they are.
    actions = read("plan/day/day-board-actions.js")
    assert "onceInFlight(`mark:${breakId}`" in actions
    assert "onceInFlight(`clear:${breakId}`" in actions


def test_a_break_carrying_a_saved_placement_offers_its_inverse_with_an_empty_history():
    """The act that moves the money has to be reversible after a reload.

    Measured on רשת 13 / 2024-11-01 before this closed: select 003~1, press
    ArrowRight once, click Save. The settlement panel reported a realised change
    of -25,400 ILS against a prediction of 0 and offered to put it back. Press
    reload: the panel is gone, ``lastSave`` is null, ``history.past`` is empty,
    and the only trace left on the surface was an 11 px lock glyph carrying
    ``aria-hidden``. One arrow key and one click cost 25,400 ILS with no route
    back from this destination.

    So the inverse is read off the break the server served rather than off the
    session, and both halves of that are asserted: the shipped model answers from
    the payload alone with the undo stack empty, and the toolbar that renders the
    control reads no history at all.
    """
    measured = node_board_model("""
      const free = { break_id: 'x~1', ordinal: 1 };
      const pinned = { break_id: 'x~1', ordinal: 1, saved_placement: { break_id: 'x~1', constraint_id: '0679cf29dc86', saved_at: '2026-08-01T09:00:00Z', note: '' } };
      process.stdout.write(JSON.stringify({
        free: m.inversePlacement(free),
        pinned: m.inversePlacement(pinned),
        orphan: m.inversePlacement({ break_id: 'x~1', saved_placement: { break_id: 'x~1', constraint_id: '' } }),
        history: m.emptyHistory(),
      }));
    """)
    assert measured["history"] == {"past": [], "future": []}, "the session stack is empty, as it is after a reload"
    assert measured["free"] is None, "a break the plan placed itself has nothing to reverse"
    assert measured["pinned"] == {
        "breakId": "x~1",
        "constraintId": "0679cf29dc86",
        "savedAt": "2026-08-01T09:00:00Z",
        "note": "",
    }, "the inverse names the restriction it has to delete"
    assert measured["orphan"] is not None, "a record whose restriction is already gone is still removable"

    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "const savedPin = inversePlacement(selectedItem);" in toolbar
    assert "{savedPin && (" in toolbar, "the control is rendered from the served break, not from a flag"
    assert "Remove the saved placement" in toolbar and "הסרת הנעיצה השמורה" in toolbar
    for forbidden in ("lastSave", "history", "settlement"):
        assert forbidden not in toolbar, f"the control must not depend on {forbidden}, which a reload empties"

    board = read("plan/day/DayBoard.jsx")
    assert "function removeSavedPlacement(item)" in board
    assert "predicted: inverseOf(settlement)," in board, "it settles like the save it reverses"
    assert "onRemoveSaved={() => selectedItem && removeSavedPlacement(selectedItem)}" in board

    acts = read("plan/day/day-board-writes.js")
    assert "export async function removeSavedPlacement({" in acts
    assert "const inverse = inversePlacement(item);" in acts
    assert "await settleAfter('undo', predicted, async () => {" in acts, "the same settlement the save printed"
    assert "await undoBreakPlacement(inverse);" in acts, "the same inverse the session undo performs"
    assert "forgetRecord(inverse.breakId);" in acts


def test_the_inverse_is_read_off_the_record_itself_when_no_break_carries_it():
    """The chip is not the only place a saved placement can be reversed from.

    Measured on רשת 13 / 2024-11-01: pinning 001~2 one snap unit right re-plans
    that programme from four breaks to one, so the day falls 1,067,845.55 to
    1,020,401.35 and the id the record names stops existing. The control asserted
    above hangs off the break chip, so at that moment it hangs off nothing.

    The record survives that, because it carries both ids the inverse needs. This
    drives the shipped model on a record alone, with no break and no session.
    """
    measured = node_board_model("""
      const record = { break_id: 's~2', constraint_id: 'a498ae76c186', saved_at: '2026-08-01T20:16:31Z', note: '' };
      process.stdout.write(JSON.stringify({
        fromRecord: m.inverseOfRecord(record),
        fromBreak: m.inversePlacement({ break_id: 's~2', saved_placement: record }),
        nothing: m.inverseOfRecord(null),
        idless: m.inverseOfRecord({ constraint_id: 'a498ae76c186' }),
      }));
    """)
    assert measured["fromRecord"] == {
        "breakId": "s~2",
        "constraintId": "a498ae76c186",
        "savedAt": "2026-08-01T20:16:31Z",
        "note": "",
    }
    assert measured["fromBreak"] == measured["fromRecord"], "one inverse, whichever side it is read from"
    assert measured["nothing"] is None
    assert measured["idless"] is None, "a record with no break id addresses nothing and must not offer to"

    rows = read("plan/day/DayBoardReadout.jsx").split("export function StrandedPlacements")[1].split("export function")[0]
    for forbidden in ("lastSave", "history", "settlement", "forecast"):
        assert forbidden not in rows, f"the row must not depend on {forbidden}, which a reload empties"


def test_the_chip_says_gold_and_saved_in_its_name_and_not_only_in_a_glyph():
    """Both glyphs are decorative, so both states have to be in the label."""
    chip = read("plan/day/DayBoardChip.jsx")
    assert chip.count('aria-hidden="true"') >= 2, "the star and the lock are decoration"
    assert "if (live.isGold) states.push(" in chip
    assert "if (saved) states.push(" in chip
    assert "aria-label={[identity, ...states].join(', ')}" in chip
    assert "ברייק זהב" in chip and "נעיצה שמורה של המפעיל" in chip



@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    from test_p3_break_store import declare_operator_channel

    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.mark.realdata
def test_the_body_the_editor_saves_binds_the_airing_it_was_dragged_on():
    """The defect a critic measured, closed at the body the surface produces.

    The editor saved a placement scoped to the whole broadcast date, and the
    restriction resolver matches a date scope against every segment on that date.
    Measured through the engine's own resolver on the 82 real segments of
    2024-11-01: the date scope binds 82 of 82, the predicate the day board sends
    binds the dragged airing. Driven through the running product with one break
    dragged one snap unit right, the same drag either way: at the date scope the
    day fell from 1,062,669.88 to 273,093.70 and from 80 breaks to 23, which is
    789,576.18 ILS and 74.3 per cent of the day for one click. At the predicate it
    fell to 1,032,180.49 and 77 breaks, 30,489.39 ILS, and the inverse put it back
    to 1,062,669.88 with a gap of 0.0.

    Every row the editor actually draws is checked, not one chosen row, and the
    body is executed out of the shipped module rather than restated here.

    **The rule asserted is the exact one, and it is not always one airing.** The
    frozen predicate contract can name a date, a programme and an hour and nothing
    finer, so two airings of one title inside one hour are one airing to it.
    Measured on this day: the predicate names exactly one segment for 37 of the 82
    and for 30 of the 48 that carry breaks, and for the other 18 it names the 2 to
    4 same-hour repeats of that title. Of the eight rows the editor draws, seven
    bind one segment and the 04:08 music-clip row binds two, and all eight together
    bind 6 of 82. So the assertion is the rule rather than the number 1: a saved
    body binds precisely the airings that share the dragged one's date, title and
    hour. Making that exactly one needs a field the frozen contract does not have,
    which is raised rather than faked here.
    """
    from kairos_api import break_store
    from kairos_api.break_api_board import break_predicate

    rows = editor_rows()
    if not rows:
        pytest.skip("the editor has no breaks to draw, so there is no body to measure")
    unresolved = [row["item"]["id"] for row in rows if not row["segment_id"]]
    assert not unresolved, f"these rows address no segment, so a save cannot record a placement: {unresolved}"

    day = rows[0]["item"]["date"]
    plan = break_store.day_plan(day)
    known = {record["break_id"] for record in break_store.break_records(plan)}
    produced = node_pin_bodies(rows)
    assert len(produced) == len(rows)

    union: set[str] = set()
    for made in produced:
        body, target = made["body"], made["target"]
        assert body["scope_type"] == "always", "a scope the resolver reads as a whole date is what this closes"
        assert "scope_value" not in body, "the airing is named by the predicate, never by a scope value"
        assert target["item"]["break_id"] in known, "a saved placement has to name a break the plan carries"
        segment = plan.segment(target["item"]["break_id"].rsplit("~", 1)[0])
        assert body["where"] == break_predicate(segment), (
            "the surface and the engine's own preview must describe the same airing"
        )
        expected = sorted(
            other.segment_id for other in plan.segments
            if same_airing_key(other) == same_airing_key(segment)
        )
        bound = bound_segments(plan, body)
        assert bound == expected, "a saved body binds its own airing and the ones the contract cannot tell from it"
        assert segment.segment_id in bound
        union.update(bound)

    assert len(union) * 4 < len(plan.segments), (
        f"every row the editor draws, saved at once, binds {len(union)} of {len(plan.segments)} segments"
    )

    # And the body this surface used to send, on the same drag, for the contrast.
    shipped_before = {
        "scope_type": "date",
        "scope_value": day,
        "channel": rows[0]["item"]["channel"],
        "effect": "FIX_OFFSET",
        "offset_seconds": produced[0]["body"]["offset_seconds"],
        "duration_seconds": produced[0]["body"]["duration_seconds"],
        "order_index": produced[0]["body"]["order_index"],
    }
    assert len(bound_segments(plan, shipped_before)) == len(plan.segments) == 82, (
        "the date scope binds every segment on the day, which is the measurement that forced this"
    )

    alone = [made for made in produced if len(bound_segments(plan, made["body"])) == 1]
    # MOST rows bind exactly one segment; the rest bind airings the frozen
    # contract genuinely cannot tell apart, which the per-row assertion above
    # already proves (bound == expected, computed from same_airing_key). So the
    # rule here is "the predicate is nearly always exact", not a fixed count.
    #
    # It used to allow exactly one ambiguous row and that number tracked the
    # plan rather than the contract: when the segment capacity guard emptied the
    # short promo blocks, more of the draggable rows landed on
    # קובץ פרומו/פרסומות airings that share a (day, title, hour) key, and the
    # count moved to three of thirteen with no change to the save's behaviour.
    assert len(alone) * 2 > len(produced), (
        f"only {len(alone)} of {len(produced)} editor rows bind exactly one segment; "
        f"the predicate has stopped being exact for most drags"
    )


# The chip the critic dragged, and what one click of it costs either way.
#
# Measured on this instance, same drag, the only difference being the body: at
# the whole-date scope the day falls to 275,614.11 and 24 breaks, which is the
# figure and the count the critic reported; at the predicate the shipped module
# now builds it falls to 1,037,270.00 and keeps all 80. The absolute figures are
# asserted because they are the claim, and the day's own baseline is checked
# first so a plan moved by another input skips with its reason rather than
# failing on somebody else's change or passing while measuring nothing.
CRITIC_DAY = "2024-11-01"
CRITIC_SEGMENT = "2024-11-01|רשת 13|005"
DAY_AS_MEASURED = 1_062_669.88
AT_THE_PREDICATE = 1_037_270.00
AT_THE_WHOLE_DATE = 275_614.11


@pytest.mark.realdata
def test_the_dragged_chip_keeps_the_day_where_the_scope_it_used_to_send_took_74_per_cent(client):
    from kairos_api import break_store

    rows = [row for row in editor_rows() if row["segment_id"] == CRITIC_SEGMENT]
    if not rows:
        pytest.skip(f"the editor draws no row on {CRITIC_SEGMENT}, so this drag is not reachable")
    before = client.get("/api/plan/day", params={"day": CRITIC_DAY}).json()["totals"]
    if before["revenue"] != pytest.approx(DAY_AS_MEASURED, abs=0.005):
        pytest.skip(
            f"this channel-day now plans at {before['revenue']} rather than the {DAY_AS_MEASURED} "
            "these figures were measured against, so the exact ones are not this test's to assert"
        )

    made = node_pin_bodies(rows[:1])[0]
    break_id = made["target"]["item"]["break_id"]
    assert bound_segments(break_store.day_plan(CRITIC_DAY), made["body"]) == [CRITIC_SEGMENT], (
        "the body this surface produces resolves to exactly this one segment"
    )

    created = client.post("/api/constraints", json=made["body"]).json()["constraint_id"]
    recorded = client.post(f"/api/breaks/{quote(break_id, safe='')}/placement", json={
        "constraint_id": str(created),
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
    })
    assert recorded.status_code == 201, recorded.text
    after = client.get("/api/plan/day", params={"day": CRITIC_DAY}).json()
    assert after["totals"]["revenue"] == pytest.approx(AT_THE_PREDICATE, abs=0.005)
    assert after["totals"]["breaks"] == 80, "the day keeps every break it had"
    assert [row["break_id"] for row in after["breaks"] if row["saved_placement"]] == [break_id], (
        "and the chip comes back saved, which is what the Remove control hangs off"
    )
    client.delete(f"/api/constraints/{quote(str(created), safe='')}")
    client.delete(f"/api/breaks/{quote(break_id, safe='')}/placement")
    back = client.get("/api/plan/day", params={"day": CRITIC_DAY}).json()["totals"]
    assert back["revenue"] == pytest.approx(DAY_AS_MEASURED, abs=0.005)

    old = client.post("/api/constraints", json={
        "scope_type": "date",
        "scope_value": CRITIC_DAY,
        "channel": rows[0]["item"]["channel"],
        "effect": "FIX_OFFSET",
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
        "order_index": made["body"]["order_index"],
    }).json()["constraint_id"]
    break_store.invalidate()
    wrecked = client.get("/api/plan/day", params={"day": CRITIC_DAY}).json()
    assert wrecked["totals"]["revenue"] == pytest.approx(AT_THE_WHOLE_DATE, abs=0.005)
    assert wrecked["totals"]["breaks"] == 24
    assert [row["break_id"] for row in wrecked["breaks"] if row["saved_placement"]] == [], (
        "and it wrote no record, so no surface offered a way back from it"
    )
    client.delete(f"/api/constraints/{quote(str(old), safe='')}")
    break_store.invalidate()

