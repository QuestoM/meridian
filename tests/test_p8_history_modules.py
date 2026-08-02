"""P8 History, the modules: the rules that are executed rather than read.

Split out of ``tests/test_p8_history_frontend.py`` under the 450-line law.

Every test here runs a real module with node and asserts what it returns. A
source assertion proves a helper is called; only running it proves what it
returns, and these three decide whether an engine key reaches a person's eye,
whether a shared link opens the entry it names, and whether a count the product
may not make is printed as zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


ADDRESS_PROBE = """
import { addressQuery, missedReason, pointAddress, isPointAddress }
  from './tv-break-dashboard/src/history/history-address.js';
const reason = (over) => missedReason({
  wanted: 'version:1337540bd866', kind: '', actor: '', needle: '', pagedOut: false, ...over});
console.log(JSON.stringify({
  point: addressQuery('version:1337540bd866'),
  restore: addressQuery('restore:8cd305da057e:2026-08-01T09:56:33.300213+00:00'),
  none: addressQuery(''),
  built: pointAddress('1337540bd866'),
  empty: pointAddress(''),
  isPoint: [isPointAddress('version:abc'), isPointAddress('restore:abc'), isPointAddress('')],
  fromTheRestoreFilter: reason({kind: 'restore'}),
  fromASearch: reason({needle: 'planner8'}),
  fromAnActor: reason({actor: 'admin'}),
  onThePointsAndGone: reason({kind: 'restore_point'}),
  onThePointsAndPagedOut: reason({kind: 'restore_point', pagedOut: true}),
  unfilteredAndPagedOut: reason({pagedOut: true}),
  unfilteredAndAbsent: reason({}),
  aChangeAddressOnThePoints: reason({wanted: 'change:1', kind: 'restore_point'}),
}));
"""


def test_the_link_that_answers_how_to_put_it_back_moves_the_list_to_where_the_point_is() -> None:
    """The version ids inside an opened restore are the destination's core
    control, and clearing the filters is not enough to make them work.

    Measured on the running instance at HEAD: the record holds 5,323 entries,
    the endpoint caps a page at 500, and the newest 500 span sixteen minutes, so
    both recorded restores and the four points they name sit outside an
    unfiltered page. There are 200 restore points and they all fit in one page,
    so narrowing to the points is the only query that can promise the point is
    in range. Executed rather than read, because a grep over the surface would
    pass on a rule that is wrong."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", ADDRESS_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert measured["point"] == {"kind": "restore_point", "actor": "", "needle": "", "limit": 500}, (
        "a point address opens on the points, with nothing else narrowing the list")
    assert measured["restore"]["kind"] == "" and measured["none"]["kind"] == ""
    assert measured["built"] == "version:1337540bd866"
    assert measured["empty"] == "", "an unrecorded id builds no address, so the jump does nothing"
    assert measured["isPoint"] == [True, False, False]
    # And when it is still not there, the note names the one reason that is true.
    assert measured["onThePointsAndGone"] == "point_gone", (
        "the whole set of points is loaded and it is not in it, so no control here can find it")
    assert measured["fromTheRestoreFilter"] == "filtered"
    assert measured["fromASearch"] == "filtered"
    assert measured["fromAnActor"] == "filtered"
    assert measured["onThePointsAndPagedOut"] == "filtered", (
        "with points paged out the reader still has a control to drop, so it is not the terminal note")
    assert measured["unfilteredAndPagedOut"] == "paged_out"
    assert measured["unfilteredAndAbsent"] == "absent"
    assert measured["aChangeAddressOnThePoints"] == "filtered", (
        "the terminal note is for a point on the point list, never for another kind")


STEM_PROBE = """
import { pathStem, actorLabel, forceLabel, forceUnit, FORCE_LABELS }
  from './tv-break-dashboard/src/history/history-labels.js';
const cases = [
  ['/api/breaks/2024-11-01|X|000~1/placement', '/api/breaks/\\u2026/placement'],
  ['/api/versions/d347532c0ed6/restore', '/api/versions/\\u2026/restore'],
  ['/api/constraints/891e3b05ec4b', '/api/constraints/\\u2026'],
  ['/api/settings', '/api/settings'],
  ['/api/auth/users/chan1/affiliation', '/api/auth/users/chan1/affiliation'],
  ['/api/uploads/spots/check', '/api/uploads/spots/check'],
];
const wrong = cases.filter(([input, want]) => pathStem(input) !== want).map(([input]) => input);
console.log(JSON.stringify({wrong, person: actorLabel('admin', 'he'),
  token: actorLabel('auth-disabled', 'he'),
  forceKeys: Object.keys(FORCE_LABELS).sort(),
  floorUnit: forceUnit('min_retention_floor'),
  floorHe: forceLabel('min_retention_floor', 'he'),
  unknown: forceLabel('something_nobody_labelled', 'he')}));
"""

# Every guardrail and assumption the run log records, present in every record of
# output/run_log.jsonl. Re-measured this round: 545 records, and the union of the
# recorded keys equals their intersection, so all fifteen are on all of them. The
# log is append-only, so the record count grows and the key set does not.
RECORDED_FORCE_KEYS = sorted((
    "default_break_length_seconds", "default_max_breaks", "first_break_multiplier",
    "gold_breaks_max_per_day", "max_ad_seconds_per_hour", "max_breaks_per_hour",
    "max_daily_ad_seconds", "min_break_spacing_seconds", "min_retention_floor",
    "protected_max_ad_seconds_per_hour", "protected_program_types", "retention_baseline",
    "retention_impact_per_break", "revenue_weight", "risk_lambda",
))


def test_the_path_stem_and_the_actor_label_do_what_they_claim() -> None:
    """Read by running the real module rather than by reading it. Source
    assertions prove a helper is called; only running it proves what it returns,
    and this one decides whether an engine key reaches a person's eye."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", STEM_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert measured["wrong"] == []
    assert measured["person"] == "admin", "a real username passes through untouched"
    assert measured["token"] == "ללא כניסה"
    assert measured["forceKeys"] == RECORDED_FORCE_KEYS, (
        "every guardrail and assumption the run log records has a word of its own")
    assert measured["floorUnit"] == "fraction"
    assert measured["floorHe"] == "רצפת שימור"
    assert measured["unknown"] == "something_nobody_labelled", (
        "an unlabelled key keeps its own name rather than disappearing")


RUNS_PROBE = """
import { RUNS_AVAILABLE, RUNS_REMEDY, RUNS_UNREADABLE, RUNS_WITHHELD,
  runsCountLine, runsCounted, runsSourceLine, runsSourceState }
  from './tv-break-dashboard/src/history/history-runs.js';
const figures = (pair) => pair.filter((text) => /[0-9]/.test(text));
console.log(JSON.stringify({
  states: [runsSourceState(undefined), runsSourceState({}), runsSourceState({runs: {}}),
    runsSourceState({runs: {state: RUNS_AVAILABLE}}), runsSourceState({runs: {state: RUNS_WITHHELD}})],
  counted: [runsCounted(RUNS_AVAILABLE), runsCounted(RUNS_UNREADABLE), runsCounted(RUNS_WITHHELD)],
  available: runsSourceLine(RUNS_AVAILABLE, 83, 'the operator channel'),
  unknownFigures: [
    ...figures(runsSourceLine(RUNS_WITHHELD, 0, null)),
    ...figures(runsSourceLine(RUNS_UNREADABLE, 0, null)),
    ...figures(runsCountLine(RUNS_WITHHELD)),
    ...figures(runsCountLine(RUNS_UNREADABLE)),
  ],
  withheldCount: runsCountLine(RUNS_WITHHELD),
  unreadableCount: runsCountLine(RUNS_UNREADABLE),
  remedy: RUNS_REMEDY,
}));
"""


def test_a_run_count_the_product_may_not_make_is_never_printed_as_zero() -> None:
    """Measured by a blind critic on the running instance, in the two places a
    person reads first: with `operator_channel` blank in the shared settings the
    attestation strip said "ו-0 הרצות" and the run tab said "הרצה 0", while only
    the footer told the truth. Re-measured while closing it: the run log holds
    545 records over four channels and the product may not attribute a single
    one of them, so the honest word is unknown, and a zero there is an
    attestation that nothing ran.

    Executed rather than read, because this is the rule a compliance owner
    attests to and a grep over the surface would pass on a rule that is wrong."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", RUNS_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert measured["states"] == [
        "unreadable", "unreadable", "unreadable", "available", "withheld_no_operator_channel"], (
        "a source that cannot say what it is was not read, which is never the same as available")
    assert measured["counted"] == [True, False, False]
    assert "83" in measured["available"][0] and "83" in measured["available"][1], (
        "the one state that may count prints the figure it counted")
    assert "the operator channel" in measured["available"][0], "and the scope it is on"
    assert measured["unknownFigures"] == [], (
        "not one sentence a person reads while the count is unknown carries a figure")
    assert "cannot be counted" in measured["withheldCount"][0]
    assert "no operator channel" in measured["withheldCount"][0].lower()
    assert measured["withheldCount"][1].startswith("לא ניתן לספור את ההרצות")
    assert "run log could not be read" in measured["unreadableCount"][0], (
        "the other unknown names its own cause, which is a different one")
    assert measured["remedy"] == ["Set the operator channel", "הגדרת ערוץ המפעיל"]


# The seven restriction rows of restore point e105c8d1da22, as GET
# /api/versions/e105c8d1da22/diff served them on 2026-08-02, rebuilt from one
# template because the seven differ in three places: the record id, and the day
# and the hour inside the predicate. Every other column, including the note a
# person reads first, is identical across all seven.
ROWS_PROBE = """
import { rowIdentity } from './tv-break-dashboard/src/history/history-rows.js';
const SHOW = 'משחקי השף עונה 7 ש.ח';
const occurrence = (date, hour) => JSON.stringify({combinator: 'and', conditions: [
  {field: 'programme', operator: 'is', value: SHOW},
  {field: 'date', operator: 'is', value: date},
  {field: 'hour', operator: 'eq', value: hour}]});
const constraint = (date, hour) => ({constraint_id: date.replace(/-/g, ''), scope_type: 'always',
  scope_value: '', channel: '', effect: 'forbid', notes: `No breaks in the last 8 minutes of ${SHOW}`,
  where_json: occurrence(date, hour), restriction_id: '0d78c01b219d', rule_kind: 'clean_tail',
  author: 'נועה, מחלקת תוכן', reason: 'גמר העונה', expires_on: '2024-12-31'});
const flat = (id) => `${id.label ? `${id.label}: ` : ''}${id.title} :: ${id.parts.map(
  (part) => `${part.label} ${part.values.join(' ')}`).join(' | ')}`;
const seven = [['2024-11-04', 13], ['2024-11-07', 13], ['2024-11-09', 16], ['2024-11-18', 13],
  ['2024-11-23', 2], ['2024-11-27', 13], ['2024-11-29', 1]].map(
  ([date, hour]) => flat(rowIdentity('constraints', constraint(date, hour), 'he')));
const pin = {override_id: '5830332d2091', scope: 'segment', target_id: '2024-11-01|רשת 13|001',
  kind: 'gold', value: '', gold: 'True', notes: 'gold from the day board, break 1',
  source: 'manual', status: 'active', anchor_date: '2024-11-01', anchor_start: '00:44',
  anchor_title: 'Reality'};
const condition = {advertiser_id: 'בנק הפועלים', rule_id: 'r1', scope_positions: '1,2',
  scope_genres: 'ANY', scope_dayparts: 'prime', effect: 'premium', value: '2.5',
  mode: 'multiplier', notes: ''};
console.log(JSON.stringify({
  seven,
  distinct: new Set(seven).size,
  pinHe: flat(rowIdentity('overrides', pin, 'he')),
  pinEn: flat(rowIdentity('overrides', pin, 'en')),
  conditionHe: flat(rowIdentity('conditions', condition, 'he')),
  conditionEn: flat(rowIdentity('conditions', condition, 'en')),
  eventHe: flat(rowIdentity('events', {event_id: 'hol-2024-03-24', name: 'פורים', type: 'holiday',
    start_date: '2024-03-24', end_date: '2024-03-24', active: 'True', price_multiplier: '1.0'}, 'he')),
  agencyHe: flat(rowIdentity('agencies', {agency_id: 'AGY_02', name: 'יוניברסל',
    display_name: 'יוניברסל מקאן', agency_type: 'מדיה מלא', status: 'active'}, 'he')),
  linkHe: flat(rowIdentity('agency_links', {agency_id: 'AGY_01', advertiser: 'בנק הפועלים',
    source: 'observed', observed_date: '2025-04-27'}, 'he')),
  channelScope: flat(rowIdentity('constraints', {constraint_id: 'c1', scope_type: 'channel',
    scope_value: 'a rival name', effect: 'forbid'}, 'en')),
  unreadable: flat(rowIdentity('constraints', {constraint_id: 'c2', where_json: '{not json'}, 'en')),
  nested: rowIdentity('constraints', {where_json: JSON.stringify({combinator: 'or', conditions: [
    {field: 'programme', operator: 'is', value: SHOW}, {field: 'genre', operator: 'is_not', value: 'News'}]})},
    'en').parts[0],
  empty: flat(rowIdentity('overrides', {}, 'he')),
  plainName: flat(rowIdentity('advertisers', 'בנק הפועלים', 'he')),
}));
"""


def test_a_row_a_restore_would_add_or_remove_is_named_rather_than_dumped() -> None:
    """The defect a blind critic measured on this destination: the preview that
    decides a restore printed JSON.stringify of the whole row, cut at 77
    characters, for every store except settings. Re-measured here: 81 of the 83
    rows the 200 restore points would add or remove carried neither an id key
    nor a name key, so each printed as its record. On restore point e105c8d1da22
    the seven added restrictions read as seven cuts of the same 78-character
    string differing only in a twelve-character record id, with the note, the
    occurrence, the effect and the author all past the cut.

    Executed rather than read, because what failed was what the function
    returned, not whether it was called."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", ROWS_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    assert measured["distinct"] == 7, "the seven rows of one rule read as seven different rows"
    for line in measured["seven"]:
        assert '{"' not in line and "…" not in line, "no row is printed as a cut record"
        assert "משחקי השף עונה 7 ש.ח" in line, "the note the store carries is read in full"
    assert "04.11.2024 13:00" in measured["seven"][0], (
        "and the occurrence that tells one from the next is the thing that differs")
    assert "23.11.2024 02:00" in measured["seven"][4]

    # The words are the surface's, in the reader's language, from one place.
    assert "השפעה איסור" in measured["seven"][0] and "Effect Forbid" not in measured["seven"][0]
    assert measured["pinHe"].startswith("ברייק זהב") and measured["pinEn"].startswith("Gold break")
    assert "ריאליטי" in measured["pinHe"] and "Reality" in measured["pinEn"]
    assert "מקדם" in measured["conditionHe"] and "Coefficient" in measured["conditionEn"]
    assert "פריים טיים" in measured["conditionHe"] and "Prime time" in measured["conditionEn"]
    assert measured["conditionHe"].startswith("מפרסם: בנק הפועלים"), (
        "a title that is a bare key is named, so nobody has to guess what it is")
    assert "חג" in measured["eventHe"] and "יוניברסל מקאן" in measured["agencyHe"]
    assert measured["linkHe"].startswith("מפרסם: בנק הפועלים") and "AGY_01" in measured["linkHe"]

    # The competitor boundary, by construction rather than by review.
    assert "רשת 13" not in measured["pinHe"] and "2024-11-01|" not in measured["pinHe"], (
        "a pin is identified by its programme, its day and its clock, never by target_id")
    assert "01.11.2024" in measured["pinHe"] and "00:44" in measured["pinHe"]
    assert "a rival name" not in measured["channelScope"], (
        "the one legacy scope that can carry a channel name is named without its value")
    assert "does not name" in measured["channelScope"]

    # Tri-state: real, unavailable, unknown.
    assert "cannot read" in measured["unreadable"], "a predicate that will not parse says so"
    assert measured["nested"]["values"] == ["2 more conditions on it"], (
        "a predicate this surface will not put into words is counted, never guessed at")
    assert measured["empty"] == "נעיצה בלי סוג רשום :: ", (
        "a row that names itself in no way says what it is and what is missing")
    assert measured["plainName"] == "בנק הפועלים :: ", (
        "the one store whose diff carries names rather than rows still reads as a name")


# Six changed rows, exactly as GET /api/versions/{id}/diff served them on
# 2026-08-02. The rate-card row is the shape that printed as a record cut at
# seventy-seven characters on 61 of the 200 restore points; the channel row is
# the only settings field that can carry a channel name and it is given a rival
# one here, which the live store has never held.
FIELDS_PROBE = """
import { changeRows } from './tv-break-dashboard/src/history/history-fields.js';
const flat = (file, row, locale) => changeRows(file, row, locale).map(
  (out) => `${out.field} | ${out.cur} | ${out.ver}`);
const card = {field: 'pricing_overrides', from: {pricing_activation: {show: false, events: true}},
  to: {pricing_activation: {show: true, events: true}, base_price_per_second_per_tvr_point: 120.0,
    premiums: {program_type: {News: 1.15}, day_of_week: {7: 1.2}, position_in_break: {1: 1.3}}}};
console.log(JSON.stringify({
  cardEn: flat('settings', card, 'en'),
  cardHe: flat('settings', card, 'he'),
  floorHe: flat('settings', {field: 'min_retention_floor', from: 0.78, to: 0.72}, 'he'),
  switchEn: flat('settings', {field: 'audience_model_activation', from: false, to: true}, 'en'),
  localeHe: flat('settings', {field: 'locale', from: 'en', to: 'he'}, 'he'),
  listEn: flat('settings', {field: 'protected_program_types', from: ['News', 'Children'], to: ['News']}, 'en'),
  ownChannel: flat('settings', {field: 'operator_channel', from: 'רשת 13', to: ''}, 'he'),
  rivalChannel: flat('settings', {field: 'operator_channel', from: 'רשת 13', to: 'a rival name'}, 'he'),
  linkHe: flat('agency_links', {id: 'AGY_01', field: 'advertiser', from: 'קרסו מוטורס', to: 'לקוח מבקר 944199'}, 'he'),
  unknownEn: flat('settings', {field: 'a_key_this_surface_has_no_word_for', from: 1, to: 2}, 'en'),
}));
"""


def test_a_field_whose_value_holds_other_values_is_read_as_those_values() -> None:
    """The second half of the same defect the row chips had. A settings change
    carries the value the store holds, and one of them is a whole nested object:
    measured on this deployment, 61 of the 200 restore points carry a rate-card
    row, and it printed as JSON cut at seventy-seven characters.

    What is asserted is what a person would decide a restore on: which fields
    inside the card differ, what each is now, what each would become, and that
    the one field that can carry a channel name never prints one this product
    cannot vouch for."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", FIELDS_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    for rows in measured.values():
        for line in rows:
            assert '{"' not in line and "…" not in line and "[object" not in line, (
                "no value reaches a person as a dumped record")

    # The rate card, read as the five fields inside it that differ.
    assert measured["cardEn"] == [
        "Rate card · Specific show layer | Off | On",
        "Rate card · Base price per rating point per second | Not set | 120",
        "Rate card · Programme type premium, News | Not set | 1.15",
        "Rate card · Day of week premium, Sunday | Not set | 1.2",
        "Rate card · Position in break premium, First | Not set | 1.3",
    ], "the field that did not move is not printed, and every one that did is"
    assert "Sunday" in measured["cardEn"][3], (
        "the rate card is keyed by the ISO weekday, where 7 is Sunday, and it is read into the Israeli week")
    assert measured["cardHe"][0] == "כרטיס תעריפים · שכבת תוכנית ספציפית | כבוי | פועל"
    assert measured["cardHe"][2] == "כרטיס תעריפים · מקדם סוג תוכנית, חדשות | לא הוגדר | 1.15"

    # The words are the product's own, in the reader's language.
    assert measured["floorHe"] == ["רף שימור | 0.78 | 0.72"], "the shell's own word for the key"
    assert measured["switchEn"] == ["Audience model switch | Off | On"], "a switch is not true and false"
    assert measured["localeHe"] == ["שפה | אנגלית | עברית"], "a token is not a word"
    assert measured["listEn"] == ["Protected programme types | News, Children | News"]
    assert measured["unknownEn"] == ["a_key_this_surface_has_no_word_for | 1 | 2"], (
        "a key with no word falls back to itself, which is honest and visibly unfinished")

    # Tri-state: a side that holds nothing says so rather than reading as blank.
    assert measured["ownChannel"] == ["הערוץ שלכם | רשת 13 | לא הוגדר"]

    # The competitor boundary, by construction rather than by today's data.
    assert "a rival name" not in measured["rivalChannel"][0], (
        "a stored channel that is not the operator's own is named as one without its name")
    assert measured["rivalChannel"][0].startswith("הערוץ שלכם | רשת 13 | ערוץ אחר")

    # Every other store keeps the column name its own identity already reads.
    assert measured["linkHe"] == ["advertiser | קרסו מוטורס | לקוח מבקר 944199"]
