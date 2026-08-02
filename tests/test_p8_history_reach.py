"""P8 History, the reach: how far back a page goes, and what it says it dropped.

The defect this file pins, measured by a blind critic on the running instance
and reproduced here before the fix: ``GET /api/history?kind=change`` matched
2,027 entries, served exactly 500 of them spanning one afternoon, and the
surface printed "Showing 500 rows over 500 of 6,049 recorded entries" without
ever mentioning the 1,527 it had dropped. No control on the page could go
further back, because the only date parameter, ``since``, narrows the newest
side. A compliance owner reading "2,011 changes recorded since 15 July" above a
list that begins this afternoon could only conclude that nothing changed before
today.

It is the same error class this piece already closed for the run count: a number
the product may not print must say so rather than print a reassuring one. Here
the product can print it, so it prints all of it and carries a control that
reaches the rest.

Two halves, and both are tested here rather than described:

- the endpoint takes ``until`` and ``before``, and the payload says where the
  window sits in the matched set, so a walk reaches every matching entry,
- and the surface sends both, prints what the page does not hold, and jumps to
  the day a link asked for.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.version_store as vs
from kairos.observability import run_log as run_log_module
from kairos_api import activity_log

from test_p8_history import (  # noqa: F401 - fixtures are used by name
    OWNED,
    RIVALS,
    _run_record,
    auth_env,
    history_env,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "tv-break-dashboard" / "src" / "history"


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


def _spread_runs(days: int, per_day: int) -> list[str]:
    """A run log spread over whole calendar days, so the reach can be measured.

    Every stamp is well before 21:00 UTC, which is where the broadcast day turns
    over in this zone, so the UTC day and the broadcast day agree and the test
    measures the reach rather than the timezone.
    """
    rows = []
    stamps = []
    for day in range(days):
        for index in range(per_day):
            stamp = f"2026-07-{10 + day:02d}T{8 + index:02d}:00:00+00:00"
            stamps.append(stamp)
            rows.append(_run_record(f"{day:02d}{index:02d}".ljust(32, "a"), stamp,
                                    OWNED, "2024-11-11", 1000.0 + index, 80, 95.0))
    run_log_module.DEFAULT_RUN_LOG_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return stamps


# --- what the page says about itself --------------------------------------------

def test_the_page_says_how_much_of_the_matched_set_it_is_serving(history_env) -> None:
    """The disclosure half. Before this round the payload carried ``matched`` and
    the page served ``limit`` of it with no field saying which part."""
    _spread_runs(days=4, per_day=3)
    body = history_env.get("/api/history", params={"kind": "run", "limit": 5}).json()

    assert body["matched"] == 12, "twelve runs match, over four days"
    assert body["served"] == 5 == len(body["entries"])
    assert body["newer"] == 0, "the first page starts at the newest end"
    assert body["older"] == 7, "and it says how many it did not serve"
    assert body["served"] + body["older"] + body["newer"] == body["matched"]
    assert body["next_before"], "the step that reaches them rides the payload"
    assert body["page_max"] == 500
    assert body["window"] == {"since": None, "until": None, "before": None}


def test_every_matching_entry_is_reachable_by_walking_the_cursor(history_env) -> None:
    """The reach half, and the reason the step is a cursor rather than a day: a
    single day can hold more entries than a page, so a day-granular step would
    serve the same rows forever. The cursor is the sort key itself."""
    stamps = _spread_runs(days=4, per_day=3)
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params = {"kind": "run", "limit": 5}
        if cursor:
            params["before"] = cursor
        body = history_env.get("/api/history", params=params).json()
        pages += 1
        seen.extend(entry["id"] for entry in body["entries"])
        assert body["newer"] == len(seen) - body["served"], "the position is the walk's own count"
        if not body["next_before"]:
            break
        cursor = body["next_before"]
        assert pages < 10, "the walk terminates"

    assert pages == 3, "twelve entries over pages of five"
    assert len(seen) == len(set(seen)) == 12, "every matching entry, exactly once"
    assert body["older"] == 0, "the last page says nothing is older"
    oldest = f"run:{'0000'.ljust(32, 'a')}"
    assert seen[-1] == oldest, f"the walk ends on the oldest entry, recorded {stamps[0]}"

    newest_page = history_env.get("/api/history", params={"kind": "run", "limit": 5}).json()
    assert oldest not in [entry["id"] for entry in newest_page["entries"]], (
        "and it is not on the newest page, which is what made it unreachable before")


def test_a_day_the_newest_page_cannot_reach_is_reached_by_the_day_window(history_env) -> None:
    """``since`` narrows the newest side and cannot reach backwards at all, which
    is why the page needed a second day parameter rather than a second use of the
    first one."""
    _spread_runs(days=4, per_day=3)
    newest = history_env.get("/api/history", params={"kind": "run", "limit": 3}).json()
    assert {entry["facts"]["day"] for entry in newest["entries"]} == {"2024-11-11"}
    assert [entry["ts"][:10] for entry in newest["entries"]] == ["2026-07-13"] * 3

    reached = history_env.get("/api/history", params={"kind": "run", "limit": 3, "until": "2026-07-11"}).json()
    assert reached["matched"] == 6, "the two oldest days, and nothing newer"
    assert [entry["ts"][:10] for entry in reached["entries"]] == ["2026-07-11"] * 3
    assert reached["window"]["until"] == "2026-07-11"

    forwards = history_env.get("/api/history", params={"kind": "run", "since": "2026-07-11", "limit": 500}).json()
    assert forwards["matched"] == 9, "since still narrows the other side, unchanged"


def test_the_two_days_pin_one_calendar_day_in_the_broadcast_zone(history_env) -> None:
    """The compliance question is a day: who changed this on 20 July. Both days
    are inclusive and both are read in the zone the list is grouped by."""
    _spread_runs(days=4, per_day=3)
    body = history_env.get("/api/history", params={
        "since": "2026-07-12", "until": "2026-07-12", "limit": 500}).json()
    assert body["matched"] == 3
    assert {entry["ts"][:10] for entry in body["entries"]} == {"2026-07-12"}

    # A stamp after 21:00 UTC is the next broadcast day in this zone, so the day
    # a reader picks and the day heading they land on cannot disagree.
    rows = [_run_record("f" * 32, "2026-07-12T21:30:00+00:00", OWNED, "2024-11-11", 1.0, 1, 90.0)]
    run_log_module.DEFAULT_RUN_LOG_PATH.write_text(
        json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    same_day = history_env.get("/api/history", params={"until": "2026-07-12", "limit": 500}).json()
    next_day = history_env.get("/api/history", params={"since": "2026-07-13", "limit": 500}).json()
    assert same_day["matched"] == 0, "21:30 UTC is already the following day in Tel Aviv"
    assert next_day["matched"] == 1


def test_the_counts_follow_the_day_window_so_a_tab_never_counts_another_set(history_env) -> None:
    """A tab printing 2,027 over a list of three is counting a set the reader is
    not looking at, which is the same defect one layer down."""
    _spread_runs(days=4, per_day=3)
    whole = history_env.get("/api/history", params={"limit": 500}).json()
    assert whole["counts"]["run"] == 12 and whole["window_total"] == whole["total"]

    windowed = history_env.get("/api/history", params={"until": "2026-07-11", "limit": 500}).json()
    assert windowed["counts"]["run"] == 6, "the tab counts what clicking it would reveal"
    assert windowed["window_total"] == 6
    assert windowed["total"] == whole["total"], "the record's own size is unchanged"


def test_a_malformed_day_or_cursor_is_refused_rather_than_guessed(history_env) -> None:
    _spread_runs(days=2, per_day=2)
    assert history_env.get("/api/history", params={"until": "yesterday"}).status_code == 400
    assert history_env.get("/api/history", params={"until": "2026-13-99x"}).status_code == 400
    assert history_env.get("/api/history", params={"before": "../../etc/passwd"}).status_code == 400
    assert history_env.get("/api/history", params={"before": "not-a-cursor"}).status_code == 400
    # A well-formed cursor from a set that no longer holds it is an empty page,
    # never a page of the wrong entries.
    empty = history_env.get("/api/history", params={"before": "1900-01-01T00:00:00+00:00|run:x"}).json()
    assert empty["entries"] == [] and empty["older"] == 0
    assert empty["newer"] == empty["matched"], "everything matched is newer than that point"


def test_a_day_the_calendar_does_not_have_is_refused_rather_than_answered(history_env) -> None:
    """The guard was a shape regex, and a shape is not a calendar. Measured on the
    running instance before this fix, over the real record: ``until=2026-13-99``
    answered 200 over all 5,400 entries, ``until=2026-02-31`` answered 200 over
    none, ``until=2026-07-32`` served 74, and ``/api/history/since`` answered 500
    on every one of them, because the day is compared as a string here and parsed
    three modules later. Only a crafted call reaches it, since the control is an
    input of type date, but a route that answers an impossible question with a
    number is a route that will one day be quoted."""
    _spread_runs(days=2, per_day=2)
    impossible = ("2026-13-99", "2026-02-31", "2026-00-10", "2026-07-32", "0000-01-01")
    for day in impossible:
        assert history_env.get("/api/history", params={"until": day}).status_code == 400, day
        assert history_env.get("/api/history", params={"since": day}).status_code == 400, day
        answered = history_env.get("/api/history/since", params={"day": day})
        assert answered.status_code == 400, f"{day} answered {answered.status_code} rather than a refusal"
        assert "calendar" in answered.json()["detail"]

    # And a day the calendar does have still answers, on all three parameters.
    assert history_env.get("/api/history", params={"until": "2026-02-28"}).status_code == 200
    assert history_env.get("/api/history", params={"since": "2026-07-11"}).json()["matched"] == 2
    assert history_env.get("/api/history/since", params={"day": "2026-07-11"}).json()["counts"]["run"] == 2


def test_the_boundary_still_holds_on_every_new_parameter(history_env) -> None:
    """A new way to ask is a new way to leak. Neither day nor cursor widens the
    channel scope, the training filter or the activity scope."""
    _spread_runs(days=4, per_day=3)
    rivals = [_run_record(f"{letter * 32}", "2026-07-11T09:00:00+00:00", rival, "2024-11-11", 999.0, 9, 90.0)
              for letter, rival in zip("cde", RIVALS)]
    with run_log_module.DEFAULT_RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        for row in rivals:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    first = history_env.get("/api/history", params={"limit": 5, "until": "2026-07-11"})
    assert first.status_code == 200
    for rival in RIVALS:
        assert rival not in first.text, f"{rival} reached the timeline through the day window"
    cursor = first.json()["next_before"]
    stepped = history_env.get("/api/history", params={"limit": 5, "until": "2026-07-11", "before": cursor})
    for rival in RIVALS:
        assert rival not in stepped.text, f"{rival} reached the timeline through the cursor"
    assert stepped.json()["matched"] == first.json()["matched"] == 6, "six own runs, three rivals dropped"


# --- the module, executed rather than read --------------------------------------

REACH_PROBE = """
import { reachState, reachLine, olderLine, missedLine, SEARCH_SCOPE }
  from './tv-break-dashboard/src/history/history-reach.js';
const digits = (text) => (text.replace(/,/g, '').match(/\\d+/g) || []);
const whole = reachState({matched: 74, served: 74, newer: 0, older: 0, next_before: null});
const first = reachState({matched: 2027, served: 500, newer: 0, older: 1527, next_before: 'a|b'});
const second = reachState({matched: 2027, served: 500, newer: 500, older: 1027, next_before: 'c|d'});
const last = reachState({matched: 2027, served: 27, newer: 2000, older: 0, next_before: null});
console.log(JSON.stringify({
  nothing: reachState(undefined),
  whole: [whole.windowed, whole.from, whole.to],
  first: [first.windowed, first.from, first.to, first.older, first.cursor, first.paged],
  second: [second.from, second.to],
  last: [last.from, last.to, last.older, last.cursor, last.paged],
  firstLine: reachLine(first),
  firstOlder: olderLine(first),
  firstDigits: [digits(reachLine(first)[0]), digits(reachLine(first)[1]),
    digits(olderLine(first)[0]), digits(olderLine(first)[1])],
  scope: SEARCH_SCOPE,
  missed: [missedLine('point_gone', 200, 500)[0], missedLine('paged_out', 200, 500)[0],
    missedLine('absent', 200, 500)[1]],
}));
"""


def test_the_reach_module_prints_the_payloads_own_figures_and_nothing_else() -> None:
    """Executed rather than grepped, because this is the sentence a compliance
    owner reads to decide whether the list in front of them is the whole record.
    Every figure in it is the endpoint's own, and nothing is said at all while
    the page holds everything that matched."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", REACH_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    assert measured["nothing"]["windowed"] is False, (
        "a body that has not arrived claims nothing, which is this piece's own rule")
    assert measured["whole"] == [False, 1, 74], (
        "a page that holds the whole matched set has nothing to disclose and says nothing")
    assert measured["first"] == [True, 1, 500, 1527, "a|b", False]
    assert measured["second"] == [501, 1000], "the second page counts from where the first ended"
    assert measured["last"] == [2001, 2027, 0, "", True], (
        "the last page offers no further step and still offers the way back")
    assert measured["firstLine"][0] == "Entries 1 to 500 of 2,027 matching."
    assert measured["firstOlder"][0] == "1,527 matching entries are older than this page."
    assert measured["firstDigits"] == [["1", "500", "2027"], ["1", "500", "2027"],
                                       ["1527"], ["1527"]], (
        "both languages carry the payload's own figures and invent none")
    assert "older than it" in measured["scope"][0], (
        "a search over part of the set is not evidence about the rest")
    assert "200 points on record" in measured["missed"][0]
    assert "older than the 500 entries on this page" in measured["missed"][1]
    assert measured["missed"][2].startswith("הרשומה")


DAY_PROBE = """
import { dayOfAddress } from './tv-break-dashboard/src/history/history-address.js';
console.log(JSON.stringify({
  change: dayOfAddress('change:2026-07-20T12:30:00.123456+00:00:57'),
  afterMidnightInTelAviv: dayOfAddress('change:2026-07-20T21:30:00.123456+00:00:57'),
  restore: dayOfAddress('restore:8cd305da057e:2026-08-01T09:56:33.300213+00:00'),
  account: dayOfAddress('account:2026-07-19T05:00:00+00:00:3'),
  point: dayOfAddress('version:1337540bd866'),
  run: dayOfAddress('run:1e5f2a4c9b8d3e7f'),
  none: dayOfAddress(''),
}));
"""


def test_the_day_a_link_lands_on_is_read_from_the_address_when_it_carries_one() -> None:
    """A link into a part of the record this page does not reach is answered with
    the day the entry is on, and the day is the broadcast day, so the jump lands
    under the heading the entry is actually filed under. An address that carries
    no stamp says nothing rather than guessing a day."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", DAY_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert measured["change"] == "2026-07-20"
    assert measured["afterMidnightInTelAviv"] == "2026-07-21", (
        "21:30 UTC is the next day in the zone this list is grouped by")
    assert measured["restore"] == "2026-08-01"
    assert measured["account"] == "2026-07-19"
    assert measured["point"] == "" and measured["run"] == "" and measured["none"] == "", (
        "an opaque id carries no day, so the note offers the controls instead of a guess")


# --- the surface -----------------------------------------------------------------

def test_the_surface_asks_for_the_window_and_prints_what_the_page_does_not_hold() -> None:
    page = _read("HistoryPage.jsx")
    assert "fetchTimeline({ limit, kind, actor, since: fromDay, until: untilDay, before })" in page
    assert "<ReachDays locale={locale} from={fromDay} until={untilDay} onDays={setDays} />" in page
    assert "<ReachPager locale={locale} body={body}" in page
    assert "onOlder={setBefore}" in page and "onNewest={() => setBefore('')}" in page
    # A cursor is a position inside one result set, so every control that changes
    # what the list matches starts the reach again at the newest end.
    assert page.count("setBefore('')") >= 5
    assert "onClick={() => { setBefore(''); setKind(name); }}" in page
    assert "onChange={(event) => { setBefore(''); setActor(event.target.value); }}" in page
    assert "const windowTotal = body && body.window_total !== undefined ? body.window_total : total;" in page
    assert "{windowTotal}" in page, "the Everything tab counts inside the window too"

    reach = _read("HistoryReach.jsx")
    assert reach.count('type="date"') == 2, "from and up to, both inclusive"
    assert "if (!reach.windowed) return null;" in reach, (
        "nothing is disclosed while the page holds the whole matched set")
    assert "onClick={() => onOlder(reach.cursor)}" in reach
    assert "reach.paged ? (" in reach, "and the way back to the newest end"


def test_only_the_newest_read_may_set_the_body() -> None:
    """Measured live while driving the new day control on the running instance:
    a segmented date field fires one read per segment, typing a four-digit year
    fired four, the year 0202 answered last, and the page settled on an empty
    list under a date the reader had already finished typing. An out-of-order
    answer on this destination is not a flicker, it is a wrong record."""
    page = _read("HistoryPage.jsx")
    assert "const reading = useRef(0);" in page
    assert "const ticket = (reading.current += 1);" in page
    body = page.split("const load = useCallback")[1]
    assert body.index("if (ticket !== reading.current) return;") < body.index("setBody(result.data)"), (
        "the guard stands before anything the reader can see is committed")
    assert "matchesSearch(entry, text, locale)" in page, (
        "and the search over the loaded page is still the one applied")


# Why an empty list is empty is decided in one module and executed by
# tests/test_p8_history_empty.py. This is the half of it the endpoint owns: the
# figures that sentence is built from have to be in the payload.

def test_the_payload_says_how_far_back_each_record_still_reaches(history_env) -> None:
    """The half of round three's defect the endpoint owns.

    Two of the four records are bounded and drop their oldest rows in silence.
    Measured on the running instance on 2026-08-01, five hours into a working
    day: exactly 200 restore points survived and the request recorder held 5,227
    lines stamped no earlier than 14:42, while the oldest surviving entry on the
    merged timeline was a restore of 2026-07-26. Asked for 20 July, the page
    answered that nothing was recorded in those days, because nothing in the
    payload said the record stops.
    """
    body = history_env.get("/api/history", params={"limit": 1}).json()
    sources = body["sources"]

    # The fixture's own two owned runs are the only thing recorded, and the run
    # log is append-only, so its start is the first run rather than a floor.
    assert sources["runs"]["starts"] == "2026-07-20"
    assert sources["runs"]["retention"] == {"pruned": False, "keeps": None,
                                            "prune_at": None, "unit": "runs"}
    # The two bounded stores hold nothing yet, so they start nowhere and still
    # publish what they would keep.
    assert sources["restore_points"]["starts"] is None
    assert sources["changes"]["starts"] is None
    assert sources["restore_points"]["retention"]["keeps"] == vs.MAX_VERSIONS == 200
    assert sources["changes"]["retention"]["keeps"] == activity_log.MAX_KEPT_ENTRIES == 5000
    assert sources["changes"]["retention"]["prune_at"] == activity_log.PRUNE_TRIGGER == 6000
    assert sources["changes"]["retention"]["pruned"] is True
    assert body["record_starts"] == "2026-07-20", "the oldest day any record still holds"

    # One write, which lands in both bounded stores, and both now start today.
    assert history_env.post("/api/versions/snapshot", json={"label": "a point"}).status_code == 200
    after = history_env.get("/api/history", params={"limit": 1}).json()
    assert after["sources"]["restore_points"]["starts"] == after["today"]
    assert after["sources"]["changes"]["starts"] == after["today"]
    assert after["record_starts"] == "2026-07-20", "and the record still reaches its oldest day"

    # The attestation reads the same figure, because a count since a day is only
    # evidence for the days the record covers.
    since = history_env.get("/api/history/since", params={"day": "2026-06-14"}).json()
    assert since["record_starts"] == "2026-07-20"
    assert "record_starts" in since["guardrails"], (
        "the regulatory limit record keeps its own start, under its own key, so the "
        "strip can name which record each day belongs to")


def test_a_window_the_record_cannot_cover_is_empty_for_a_reason_the_payload_carries(history_env) -> None:
    """The read the defect was reported on, in the fixture's own terms: a day
    window entirely older than anything the record still holds."""
    body = history_env.get("/api/history", params={"until": "2026-07-01", "limit": 500}).json()
    assert body["matched"] == 0 and body["window_total"] == 0, "the list is empty"
    assert body["total"] > 0, "and the record is not"
    assert body["record_starts"] == "2026-07-20" > body["window"]["until"], (
        "so the page can tell an empty day from a day the record does not reach")


def test_the_payload_carries_every_figure_that_sentence_needs(history_env) -> None:
    """The surface can only say what it was told. This is the read the defect was
    reproduced on: a day window that holds entries, and a kind inside it that
    holds none."""
    _spread_runs(days=4, per_day=3)
    body = history_env.get("/api/history", params={
        "until": "2026-07-11", "kind": "change", "limit": 500}).json()
    assert body["entries"] == [] and body["matched"] == 0, "the list under the tab is empty"
    assert body["window_total"] == 6, "and the page knows six entries were recorded in those days"
    assert body["counts"]["change"] == 0 and body["counts"]["run"] == 6, (
        "with the tab counts that say which kind holds them")
