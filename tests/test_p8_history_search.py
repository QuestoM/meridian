"""P8 History: what the search covers, and why an unwindowed list is empty.

Split out of ``tests/test_p8_history_empty.py`` under the 450-line law, which is
the pattern that file was itself split out of. It earns its own file for a second
reason: the sentence under an empty list is the one this destination has now got
wrong four times, each time by a different route, and this is the route the fix
had not taken. The three closed before it were all under a day window, and this
one is on the landing state every reader arrives at.

Measured live on 8038 at 02:55 on 2026-08-02. The record held 5,770 entries, the
default page served 200 of them, and typing an operator's name into the search
box emptied the list under "nothing here matches those filters". That operator
had 23 entries on the record, ``GET /api/history?limit=200&actor=planner``
answered all 23, and their name was in the payload's own ``actors`` list.

The rule enforced here is this piece's own: name the control that emptied the
list, print in the payload's own figures what dropping it would reveal, and carry
every control that reaches the rest. The module is executed rather than grepped,
because a source assertion is what pinned the defective branch in place twice.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "tv-break-dashboard" / "src" / "history"


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


# The payload measured live on 8038 at 02:55 on 2026-08-02, on the landing state
# every reader arrives at: no day window, no filter, the default page. Every
# figure is the endpoint's own.
SEARCH_PROBE = """
import { RECORD_EMPTY, actorForNeedle, byActorLine, emptyPage }
  from './tv-break-dashboard/src/history/history-search.js';
const live = {total: 5770, matched: 5770, served: 200, newer: 0, older: 5570,
  next_before: '2026-08-02T02:33:04.095+00:00|account:5197',
  actors: ['admin', 'engine', 'planner', 'yield', 'yieldo'],
  counts: {change: 2107, preview: 1149, run: 142, restore_point: 200, restore: 4, sign_in: 2168}};
const at = (over, view) => emptyPage({...live, ...over}, {limit: 200, wide: 500, ...view});
const whole = {matched: 200, served: 200, newer: 0, older: 0, next_before: null};
const none = {matched: 0, served: 0, newer: 0, older: 0, next_before: null};
const figures = (line) => line.map((text) => (text.replace(/,/g, '').match(/\\d+(?:\\.\\d+)?/g) || []));
console.log(JSON.stringify({
  onANeedle: at({}, {needle: 'planner'}),
  onANeedleFigures: figures(at({}, {needle: 'planner'}).covers),
  atTheWideLimit: at({served: 500, older: 5270}, {needle: 'planner', limit: 500}),
  underAnotherFilter: at({matched: 2107, served: 200, older: 1907}, {kind: 'change', needle: 'planner'}),
  onTheWholeSet: at(whole, {kind: 'restore_point', needle: 'planner'}),
  pastTheLast: at({served: 0, newer: 5770, older: 0}, {needle: 'planner'}),
  onAKindWithNone: at({...none, counts: {...live.counts, restore: 0}}, {kind: 'restore'}),
  onAnActorWithNone: at(none, {actor: 'yieldo'}),
  onAnActorAndAKind: at(none, {kind: 'restore', actor: 'yieldo'}),
  onNoFilterAtAll: at({}, {}),
  onAnEmptyRecord: at({total: 0, ...none}, {}),
  nothingLoaded: emptyPage(undefined, undefined),
  exact: actorForNeedle(live.actors, 'yieldo'),
  padded: actorForNeedle(live.actors, '  YieldO '),
  ambiguous: actorForNeedle(live.actors, 'yiel'),
  byLabel: actorForNeedle(live.actors, 'המנוע'),
  absent: actorForNeedle(live.actors, 'zzqq'),
  noActors: actorForNeedle(undefined, 'planner'),
  offer: byActorLine('planner'),
  emptySentence: RECORD_EMPTY,
}));
"""


def test_a_search_over_one_page_never_reports_the_record_as_holding_nothing() -> None:
    """The named gap of round three, and the last route the fix had not taken.

    Measured live on 8038 as the reader arrives: the record held 5,770 entries,
    the default page served 200 of them, and typing an operator's name into the
    search box emptied the list under "nothing here matches those filters". That
    operator had 23 entries on the record at that moment, ``GET
    /api/history?limit=200&actor=planner`` answered all 23, and their name was in
    the payload's own ``actors`` list, so the dropdown two centimetres from the
    box found every one of them server side while the box beside it found none.

    The rule is this destination's own and older than the defect: name the
    control that emptied the list, print what dropping it would reveal in the
    payload's own figures, and carry every control that reaches the rest. The
    module is executed rather than grepped, because a source assertion is what
    pinned the defective branch in place the last two times."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", SEARCH_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    empty = measured["emptySentence"]

    # The state the defect was reported in. The sentence names the search, the
    # line beside it says how little of the record the search read, and the four
    # controls that reach the rest are all on the row.
    found = measured["onANeedle"]
    assert found["line"][0] == "Nothing on this page matches that search."
    assert found["covers"][0] == (
        "The search covers this page, which holds 200 of the 5,770 entries on the record, "
        "3.5 percent of them.")
    assert found["covers"][1].startswith("החיפוש פועל על העמוד הזה, שמחזיק 200 מתוך 5,770")
    assert "3.5 אחוז" in found["covers"][1]
    assert found["actor"] == "planner", "the control that answers is offered by name"
    assert [found["wide"], found["older"], found["clear"], found["newest"]] == [True, True, True, False]
    # Every figure in both languages is the payload's own: the page, the matched
    # set, and the share of one in the other. Nothing else is printed.
    assert measured["onANeedleFigures"] == [["200", "5770", "3.5"], ["200", "5770", "3.5"]]

    # A wider page is a larger share and no longer offers itself.
    wider = measured["atTheWideLimit"]
    assert "500 of the 5,770 entries on the record, 8.7 percent" in wider["covers"][0]
    assert wider["wide"] is False, "a control that would change nothing is not offered"

    # With a kind or an actor also narrowing it, the denominator is that set and
    # the sentence says so rather than calling it the record.
    assert "of the 2,107 entries the other filters match, 9.5 percent" in (
        measured["underAnotherFilter"]["covers"][0])
    assert "שאר הסינון" in measured["underAnotherFilter"]["covers"][1]

    # And when the page does hold the whole matched set, the search really did
    # read all of it, so it says so and claims no reach it does not have.
    assert measured["onTheWholeSet"]["line"][0] == (
        "Nothing matches that search. Every one of the 200 entries this page holds was searched.")
    assert measured["onTheWholeSet"]["covers"] is None
    assert measured["onTheWholeSet"]["actor"] == "", (
        "with the whole set searched, a filter by name would answer nothing new")

    # The other four states, each naming the control that emptied the list.
    assert measured["pastTheLast"]["line"][0] == (
        "This page is past the last of the 5,770 matching entries.")
    assert measured["pastTheLast"]["newest"] is True
    assert measured["onAKindWithNone"]["line"][0] == (
        "No entry of kind Restore is on the record. 5,770 entries are, in other kinds.")
    assert measured["onAnActorWithNone"]["line"][0] == (
        "Nothing by yieldo is on the record. 5,770 entries are, by others.")
    assert measured["onAnActorAndAKind"]["line"][0].startswith("Nothing of kind Restore by yieldo"), (
        "an actor narrowed by a kind is not the same claim as an actor absent from the record")
    assert measured["onNoFilterAtAll"]["line"][0] == (
        "Nothing here matches those filters. 5,770 entries are on the record.")

    # The one sentence that says the record itself holds nothing, and the only
    # state that may say it.
    said_empty = [name for name in ("onANeedle", "atTheWideLimit", "underAnotherFilter",
                                    "onTheWholeSet", "pastTheLast", "onAKindWithNone",
                                    "onAnActorWithNone", "onAnActorAndAKind", "onNoFilterAtAll",
                                    "onAnEmptyRecord")
                  if measured[name]["line"][0] == empty[0]]
    assert said_empty == ["onAnEmptyRecord"], f"the empty-record sentence fired on {said_empty}"
    assert measured["onAnEmptyRecord"]["line"][1] == empty[1]
    assert measured["nothingLoaded"]["line"] == empty, "a body that has not arrived claims nothing"

    # The needle read as an actor: exact in either printed form, and nothing at
    # all when two names could be meant, because a guess is worse than the four
    # controls beside it.
    assert measured["exact"] == "yieldo" and measured["padded"] == "yieldo"
    assert measured["ambiguous"] == "", "yiel is both yield and yieldo, so neither is offered"
    assert measured["byLabel"] == "engine", "a reader typing the Hebrew word gets the same filter"
    assert measured["absent"] == "" and measured["noActors"] == ""

    # And the offer says why that control answers where the search did not.
    assert measured["offer"][0] == (
        "planner is on the record, and the operator filter reads all of it rather than this page.")
    assert measured["offer"][1].startswith("ברישום"), (
        "measured in the browser: a Hebrew sentence opening on an ASCII login takes its direction "
        "from that login and renders the whole line left to right inside a right to left page")
    assert "planner" in measured["offer"][1] and "סינון המפעיל" in measured["offer"][1]


def test_the_surface_carries_the_four_controls_that_reach_past_the_page() -> None:
    """The half of it the surface owns. The sentence and the controls have to be
    where the list emptied, not five lines lower in a provenance footer: the
    words were already written there when a compliance owner read "nothing
    matches those filters" and attested."""
    page = _read("HistoryPage.jsx")
    assert "<ReachEmptyPage locale={locale} body={body} kind={kind} actor={actor} needle={needle.trim()} limit={limit} wide={WIDE_LIMIT}" in page
    assert "onActor={(name) => { setBefore(''); setNeedle(''); setActor(name); }}" in page, (
        "the offer moves the question into the control that answers it and drops the one that did not")
    assert "onOlder={() => setBefore((body && body.next_before) || '')}" in page
    assert "onWide={() => setLimit(WIDE_LIMIT)}" in page

    reach = _read("HistoryReach.jsx")
    empty = reach.split("export function ReachEmptyPage")[1]
    assert "emptyPage(body, { kind, actor, needle, limit, wide })" in empty
    for control in ("onActor(state.actor)", "onWide", "onOlder", "onClear", "onNewest"):
        assert control in empty, f"{control} is not on the row of controls"
    assert "state.covers ?" in empty and "byActorLine(state.actor)" in empty
