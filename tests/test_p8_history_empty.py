"""P8 History: why an empty list is empty, executed rather than read.

Split out of ``tests/test_p8_history_reach.py`` under the 450-line law, and it
earns its own file for a second reason: this is the sentence the destination has
now got wrong three times, each time by a different route, and each time because
a surface printed the reassuring branch while the payload held the real one.

The defect closed here, measured live and reproduced two clicks from the landing
state: "Up to" set to 28/07/2026, then the Change tab. The tab row read
Everything 30, Change 0, Run 29, Restore point 0, Restore 1, Account 0, Preview
0, and the sentence directly beneath it read "Nothing was recorded in those
days". Thirty entries were, and the page had every figure in hand.

The rule, which is this piece's own and older than the defect: a count the
product has may not be printed as nothing, and a list that dropped entries must
say what it dropped. The module is executed here rather than grepped, because a
source assertion is what pinned the defective branch in place last round.
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


EMPTY_PROBE = """
import { EMPTY_WINDOW, changesSourceLine, emptyWindow }
  from './tv-break-dashboard/src/history/history-reach.js';
// The payload measured live at the moment the defect was reproduced: "Up to" set
// to 28/07/2026, over the real record, before any kind was picked.
const window = {window_total: 30, total: 5400, matched: 30, served: 30,
  counts: {change: 0, preview: 0, run: 29, restore_point: 0, restore: 1, sign_in: 0}};
const at = (over, filters) => emptyWindow({...window, ...over}, filters);
const digits = (line) => line.map((text) => (text.replace(/,/g, '').match(/\\d+/g) || []));
console.log(JSON.stringify({
  onTheChangeTab: at({matched: 0, served: 0}, {kind: 'change'}),
  onThePoints: at({matched: 0, served: 0}, {kind: 'restore_point'}),
  onAnActor: at({matched: 0, served: 0}, {actor: 'admin'}),
  onTheEngine: at({matched: 0, served: 0}, {actor: 'engine'}),
  onASearch: at({}, {needle: 'nothing on this page'}),
  onEmptyDays: at({window_total: 0, matched: 0, served: 0, counts: {}}, {kind: 'change'}),
  pastTheLast: at({served: 0}, {}),
  noControlAtAll: at({}, {}),
  nothingLoaded: emptyWindow(undefined, undefined),
  changeDigits: digits(at({matched: 0, served: 0}, {kind: 'change'}).line),
  searchDigits: digits(at({}, {needle: 'x'}).line),
  emptySentence: EMPTY_WINDOW,
  changes: changesSourceLine({records: 5088}),
  changesUnread: changesSourceLine(undefined),
}));
"""


def test_a_day_window_never_says_the_record_is_empty_while_its_own_tabs_count_it() -> None:
    """The named gap of round three, reproduced live and closed here.

    Two clicks from the landing state: "Up to" 28/07/2026, then the Change tab.
    The tab row read Everything 30, Change 0, Run 29, Restore 1, and the sentence
    directly under it said nothing was recorded in those days. Thirty entries
    were, and the page had every figure in hand. It reproduced on every tab and
    for any actor or needle, because one branch rendered the empty-record
    sentence for every windowed empty list and the three explaining sentences
    beside it were all guarded on there being no window at all.

    The previous version of this test asserted the shape of that branch in the
    source, which is what pinned it in place, so the test moves with the fix: the
    module is executed and the sentence it returns is read. The rule it enforces
    is the one this destination already applies to the run count and to the
    dropped page. A count the product has in hand may not be printed as nothing,
    and a page that dropped entries must say what it dropped."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", EMPTY_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    empty = measured["emptySentence"]

    assert measured["onTheChangeTab"]["line"][0] == (
        "No entry of kind Change was recorded in those days. 30 entries were, in other kinds.")
    assert measured["onTheChangeTab"]["line"][1] == (
        "בימים האלה לא נרשמה אף רשומה מסוג שינוי. נרשמו 30 רשומות מסוגים אחרים.")
    assert measured["onTheChangeTab"]["clear"] is True, "and it carries the control that undoes it"
    assert measured["onThePoints"]["line"][0].startswith("No entry of kind Restore point"), (
        "the same defect reproduced on the restore points, and it is the same fix")
    assert "admin" in measured["onAnActor"]["line"][0] and "30 entries were, by others" in measured["onAnActor"]["line"][0]
    assert "The engine" in measured["onTheEngine"]["line"][0], "the actor is named as the row names it"
    assert measured["onASearch"]["scope"] is True, (
        "a search runs over this page only, which is the reason it found nothing")
    assert "30" in measured["onASearch"]["line"][0] and "30" in measured["onASearch"]["line"][1]
    assert measured["pastTheLast"]["line"][0] == (
        "This page is past the last of the 30 entries matching in those days.")
    assert measured["noControlAtAll"]["line"][0].endswith("30 entries were recorded in those days.")

    # The one sentence that says the record itself is empty, and the only state
    # that may say it: the days hold nothing at all.
    said_empty = [name for name in ("onTheChangeTab", "onThePoints", "onAnActor", "onTheEngine",
                                    "onASearch", "pastTheLast", "noControlAtAll", "onEmptyDays")
                  if measured[name]["line"][0] == empty[0]]
    assert said_empty == ["onEmptyDays"], f"the empty-record sentence fired on {said_empty}"
    assert measured["onEmptyDays"]["line"][1] == empty[1]
    assert measured["onEmptyDays"]["clear"] is False, (
        "with nothing recorded in those days, dropping the filters would change nothing")
    assert measured["nothingLoaded"]["line"] == empty, "a body that has not arrived claims nothing"

    # Every figure is the payload's own, in both languages, and none is invented.
    assert measured["changeDigits"] == [["30"], ["30"]]
    assert measured["searchDigits"] == [["30", "30"], ["30", "30"]]

    # The provenance line the lead flagged beside it: the recorder's own line
    # count printed under the same word as the Change tab, which counts one of the
    # three kinds those lines become.
    assert measured["changes"][0] == (
        "The request recorder holds 5,088 lines, which become changes, previews and sign-ins.")
    assert "5,088" in measured["changes"][1] and measured["changes"][1].startswith("רישום הבקשות")
    assert "0" in measured["changesUnread"][0], "a source that says nothing is counted as nothing"


def test_the_surface_reads_that_sentence_and_carries_the_controls_it_names() -> None:
    page = _read("HistoryPage.jsx")
    assert "const emptied = emptyWindow(body, { kind, actor, needle: needle.trim() });" in page
    assert "<ReachEmpty locale={locale} empty={emptied} onClear={clearFilters} onNewest={() => setDays('', '')} />" in page
    assert "const windowed = Boolean(fromDay || untilDay);" in page
    assert "{!runsBlocked && !windowed ? (" in page, (
        "and the branch with no day window decides the same way, in history-search.js")
    assert "const clearFilters = useCallback(() => { setBefore(''); setKind(''); setActor(''); setNeedle(''); }, []);" in page
    assert "onClear={clearFilters}" in page.split("<ReachMissed")[1], (
        "the link that missed and the list that emptied drop the filters the same way")

    reach = _read("HistoryReach.jsx")
    assert "const line = state.line || EMPTY_WINDOW;" in reach, (
        "the empty-record sentence survives only as the fallback for a body that has not arrived")
    assert "state.clear ? (" in reach and "CLEAR_FILTERS[0]" in reach
    assert "state.scope ? (" in reach and "SEARCH_SCOPE[0]" in reach


# --- and the record has to reach those days before it can be empty in them -------

# The payload measured live on the real record at 22:35 on 2026-08-01, with the
# day window the defect was reported on. Every figure is the endpoint's own:
# record_starts is the oldest day anything still survives, and each source
# carries where it starts and what it drops.
RECORD_PROBE = """
import { EMPTY_WINDOW, attestationStartLine, emptyWindow, recordStartLine, windowOutOfReach }
  from './tv-break-dashboard/src/history/history-reach.js';
const sources = {
  changes: {records: 5423, starts: '2026-08-01',
    retention: {pruned: true, keeps: 5000, prune_at: 6000, unit: 'lines'}},
  restore_points: {records: 200, starts: '2026-08-01',
    retention: {pruned: true, keeps: 200, prune_at: 200, unit: 'restore_points'}},
  runs: {records: 124, starts: '2026-07-28', state: 'available',
    retention: {pruned: false, keeps: null, prune_at: null, unit: 'runs'}},
};
const record = {total: 5749, record_starts: '2026-07-26', sources};
const emptyAt = (until) => ({...record, window_total: 0, matched: 0, served: 0, counts: {},
  window: {since: null, until, before: null}});
const days = ['2026-06-14', '2026-07-20', '2026-07-25', '2026-07-26', '2026-07-27', '2026-08-01'];
// The same record with nothing bounded, which is a product that drops nothing:
// then an empty window really is an empty window, on every day.
const unbounded = {...record, sources: {...sources,
  changes: {...sources.changes, retention: {pruned: false, keeps: null}},
  restore_points: {...sources.restore_points, retention: {pruned: false, keeps: null}}}};
console.log(JSON.stringify({
  swept: days.map((until) => {
    const state = emptyWindow(emptyAt(until), {});
    return [until, state.line[0] === EMPTY_WINDOW[0], Boolean(state.reach), state.clear];
  }),
  sweptUnbounded: days.map((until) => {
    const state = emptyWindow({...unbounded, ...emptyAt(until), sources: unbounded.sources}, {});
    return [until, state.line[0] === EMPTY_WINDOW[0], Boolean(state.reach)];
  }),
  beyond: emptyWindow(emptyAt('2026-07-20'), {}).line,
  onTheChangeTab: emptyWindow({...record, window_total: 30, matched: 0, served: 0,
    counts: {change: 0, run: 29, restore: 1},
    window: {since: null, until: '2026-07-28', before: null}}, {kind: 'change'}),
  onTheRunTab: emptyWindow({...record, window_total: 30, matched: 0, served: 0,
    counts: {change: 30, run: 0},
    window: {since: null, until: '2026-07-28', before: null}}, {kind: 'run'}),
  footer: recordStartLine(record),
  footerUnknown: recordStartLine({sources}),
  attested: attestationStartLine({...record, day: '2026-06-14'}),
  attestedInside: attestationStartLine({...record, day: '2026-07-28'}),
  attestedUnknown: attestationStartLine({day: '2026-06-14'}),
  noWindow: windowOutOfReach({...record, window_total: 0, window: {since: '2026-06-14', until: null}}),
}));
"""


def test_the_empty_record_sentence_cannot_fire_for_a_window_older_than_the_record() -> None:
    """The named gap of round three, closed and pinned.

    Measured live on the running instance at 22:13 and reproduced at 22:35: "Up
    to" 20/07/2026, every tab reading 0, and the body reading "nothing was
    recorded in those days" over a record whose oldest surviving line was stamped
    26 July. The request recorder keeps its newest 5,000 lines and the version
    store its newest 200 restore points, so the days the reader asked about had
    been pruned out from under them while they read.

    The rule this pins: the record is empty in those days only when it reaches
    them. A window entirely older than the day the record starts on gets the
    sentence that says so, in both languages, with the day and the retention that
    caused it. And a product that prunes nothing keeps the old sentence, because
    then an empty window is genuinely an empty window."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", RECORD_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    # The sweep across the boundary. Before the record starts the empty-record
    # sentence may not fire; on the day it starts and after it, it may.
    assert measured["swept"] == [
        ["2026-06-14", False, True, False],
        ["2026-07-20", False, True, False],
        ["2026-07-25", False, True, False],
        ["2026-07-26", True, False, False],
        ["2026-07-27", True, False, False],
        ["2026-08-01", True, False, False],
    ], "the empty-record sentence fired for a window the record does not reach"

    # And it is retention that makes the difference, not the calendar: with
    # nothing bounded, every one of those days is an honest empty day.
    assert [row[1] for row in measured["sweptUnbounded"]] == [True] * 6
    assert [row[2] for row in measured["sweptUnbounded"]] == [False] * 6

    assert measured["beyond"][0] == (
        "The record does not reach those days. It starts on 2026-07-26; the request recorder "
        "keeps the newest 5,000 lines and the version store keeps the newest 200 restore points.")
    assert measured["beyond"][1].startswith("הרישום אינו מגיע לימים האלה. הוא מתחיל ב-2026-07-26;")
    assert "5,000" in measured["beyond"][1] and "200" in measured["beyond"][1]

    # One kind at a time, for the same reason: the change record stops on
    # 2026-08-01, so a window ending on 28 July holds no change and never could.
    assert measured["onTheChangeTab"]["line"][0] == (
        "The record of kind Change does not reach those days. It starts on 2026-08-01; "
        "the request recorder keeps the newest 5,000 lines.")
    assert measured["onTheChangeTab"]["clear"] is True, "and dropping the kind still shows the rest"
    # The run log drops nothing, so an absent run is an absent run.
    assert measured["onTheRunTab"]["line"][0].startswith("No entry of kind Run was recorded")
    assert measured["onTheRunTab"]["reach"] is not True

    # The footer says it whether the list is empty or full, and says nothing at
    # all while the start is unknown.
    assert measured["footer"][0] == (
        "The record starts on 2026-07-26; the request recorder keeps the newest 5,000 lines "
        "and the version store keeps the newest 200 restore points.")
    assert measured["footer"][1].startswith("הרישום מתחיל ב-2026-07-26;")
    assert measured["footerUnknown"] is None, "a start nobody can name is not printed"

    # The attestation strip names the record its count is over, and warns when the
    # day asked for is older than that record.
    assert measured["attested"][0] == (
        "The record behind this count starts on 2026-07-26, so the days before it hold no "
        "evidence either way.")
    assert measured["attestedInside"][0] == "The record behind this count starts on 2026-07-26."
    assert measured["attestedUnknown"] is None
    assert measured["noWindow"] is False, (
        "an open-ended window reaches today, so it is never out of the record's reach")


def test_the_two_records_on_the_attestation_strip_each_name_themselves() -> None:
    """The half of it a component owns. Measured before the fix: the strip printed
    "2,562 changes and points recorded" for a window opening on 14 June directly
    beside "and this record starts on 2026-06-14", which is the guardrail store's
    baseline and not the record that count was taken over."""
    since = _read("HistorySince.jsx")
    assert "the regulatory limit record starts on ${guardrails.record_starts}" in since
    assert "ורישום מגבלות הרגולציה מתחיל ב-${guardrails.record_starts}" in since
    assert "const startLine = attestationStartLine(body);" in since
    assert "const covered = !(body && body.record_starts) || String(body.day || '') >= String(body.record_starts);" in since, (
        "the warning and the sentence read the same payload, never the control")
    assert "and this record starts on ${guardrails.record_starts}" not in since, (
        "the sentence that named no record is gone from both languages")
    assert "והרישום הזה מתחיל ב-${guardrails.record_starts}" not in since

    page = _read("HistoryPage.jsx")
    assert "<ReachStart locale={locale} body={body} />" in page, (
        "and the provenance footer prints the record's own start under every list")
    reach = _read("HistoryReach.jsx")
    assert "export function ReachStart({ locale, body })" in reach
    assert "if (!line) return null;" in reach
