"""P8 History: a refused write is printed as refused and counted as nothing.

The defect this file pins, measured by a blind critic on their own instance on
2026-08-02 and re-measured here against this repository's own recorder.

The request recorder has stored the status the server answered with on every
line since the log existed, and nothing on this destination read it. The act was
derived from the method and the path alone, so a write the wall refused carried
the sentence of one that happened: four consecutive rows read "the regulatory
limit was saved" at 10:35, two of them at 403, the only difference a small red
number at the far end, and the opened entry offered "open the surface this
changed" for a request that changed nothing. 680 of the 2,264 change entries on
the record (30.0 percent) answered 400 or more, 42 of them 405 and 30 of them
404, which are routes that do not exist. And the compliance strip summed all of
them: ``since?day=2026-08-02`` answered ``changed: 2652`` while 743 of the 2,451
change entries in that window had been refused, so 28.0 percent of the figure a
compliance owner attests to had changed nothing.

Measured on this repository's own store at 00:33 on 2026-08-04: 3,263 recorded
requests, 811 of them refused (24.9 percent), 528 of those 403, not one 5xx and
not one line without a status.

Three rules are enforced here and nothing else is:

- an act is printed as what it did, so a refused attempt never wears the words
  of one that happened and never opens the door to the surface it did not change,
- a figure that attests counts only what landed, and what was attempted is
  reported beside it rather than folded into it,
- and an act whose result nobody recorded is neither, because unknown and
  unchanged are different answers and only one of them can be attested to.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.version_store as vs
from kairos_api import activity_log, history_api_actions, history_api_timeline

from test_p8_history import (  # noqa: F401 - fixtures are used by name
    _as,
    auth_env,
    history_env,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "tv-break-dashboard" / "src" / "history"


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


def _changes(client, **params) -> list[dict]:
    """Every change entry the timeline serves, with the filters applied."""
    query = {"kind": "change", "limit": 500}
    query.update(params)
    return client.get("/api/history", params=query).json()["entries"]


# --- what the status code means, decided once -----------------------------------

def test_the_outcome_is_derived_from_the_status_the_recorder_stored() -> None:
    """The nine codes this store actually holds, plus the ones that cannot be
    read as either answer. 405 and 404 mean the route does not exist, and a route
    that does not exist changed nothing.

    A 5xx is on neither side. The server failed rather than declined and a
    failure can land after a write has begun, so a refusal would be a certainty
    the record cannot support. Measured on this repository's own recorder at
    00:33 on 2026-08-04: 3,261 recorded requests and not one 5xx among them."""
    applied = (200, 201, 204, 302)
    refused = (400, 401, 403, 404, 405, 409, 422)
    for code in applied:
        assert history_api_actions.outcome_for(code) == "applied", code
    for code in refused:
        assert history_api_actions.outcome_for(code) == "refused", code
    for value in (None, "", "not a code", 99, {}, 500, 502, 503):
        assert history_api_actions.outcome_for(value) == "unknown", value
    assert history_api_actions.OUTCOMES == ("applied", "refused", "unknown")


def test_a_refused_write_keeps_its_kind_and_carries_what_it_did(history_env, auth_env) -> None:
    """It stays a change, because somebody attempted it and that is the thing a
    person reading this surface came for. What changed is that the entry now says
    it did not happen."""
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    assert viewer.post(f"/api/versions/{version_id}/restore", json={}).status_code == 403

    rows = _changes(viewer)
    refused = [row for row in rows if row["facts"].get("status") == 403]
    assert refused, "the refused write is on the timeline"
    assert refused[0]["kind"] == "change", "still a change, because it was attempted"
    assert refused[0]["facts"]["action"] == "restore"
    assert refused[0]["facts"]["outcome"] == "refused"

    admin = _as(history_env, auth_env, "admin", "admin")
    assert admin.post("/api/versions/snapshot", json={"label": "after"}).status_code == 200
    landed = [row for row in _changes(admin) if row["facts"].get("action") == "restore_point_saved"]
    assert landed and landed[0]["facts"]["outcome"] == "applied"


def test_every_change_entry_carries_an_outcome_from_the_closed_set(history_env, auth_env) -> None:
    admin = _as(history_env, auth_env, "admin", "admin")
    admin.post("/api/versions/snapshot", json={})
    admin.post("/api/definitely-not-a-route")
    for row in _changes(admin):
        assert row["facts"]["outcome"] in history_api_actions.OUTCOMES


# --- what a figure that attests may count ---------------------------------------

def test_the_attested_figure_counts_only_what_landed(history_env, auth_env) -> None:
    """The critic's third consequence, in one arithmetic. Two acts are attempted
    and one is refused, so the attestation says one change happened and one
    attempt was refused, and the two are never one number."""
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    assert viewer.post(f"/api/versions/{version_id}/restore", json={}).status_code == 403

    admin = _as(history_env, auth_env, "admin", "admin")
    assert admin.post("/api/versions/snapshot", json={"label": "the good one"}).status_code == 200

    body = admin.get("/api/history").json()
    attestation = body["attestation"]
    counts = attestation["counts"]
    assert attestation["outcomes"]["refused"] == 1, "one attempt was refused"
    assert attestation["outcomes"]["applied"] >= 1, "and one landed"
    assert attestation["refused"] == 1
    assert attestation["changed"] == attestation["attempted"] - 1, (
        "the refusal is out of the attested figure and nothing else moved")
    assert attestation["changed"] == (
        attestation["outcomes"]["applied"] + counts["restore"] + counts["restore_point"])
    assert attestation["verdict"] == "changed"


def test_a_window_of_nothing_but_refusals_attests_as_unchanged(history_env, auth_env) -> None:
    """The sharpest state: every attempt refused and nothing else on the record.
    Nothing changed, the verdict says so, and every attempt is still listed,
    because hiding them would be the other lie."""
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    for _ in range(3):
        assert viewer.post("/api/definitely-not-a-route").status_code == 404

    since = viewer.get("/api/history/since", params={"day": "2000-01-01"}).json()
    assert since["counts"]["change"] == 3, "the attempts are all on the record"
    assert since["counts"]["restore_point"] == 0 and since["counts"]["restore"] == 0
    assert since["refused"] == 3 and since["attempted"] == 3
    assert since["changed"] == 0, "and not one of them is inside the attested figure"
    assert since["verdict"] == "unchanged"


def test_an_act_with_no_recorded_result_is_neither_changed_nor_refused(history_env, auth_env) -> None:
    """Tri-state, and the third state is the one that cannot be attested to. The
    store holds no such line today, which is exactly why it is carried: the day
    one appears it must not read as a change that happened."""
    path = activity_log.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts": "2026-08-03T09:00:00.000+00:00", "user": "admin", "role": "admin",
        "event": "request", "method": "PUT", "path": "/api/settings",
        "status": None, "duration_ms": 3.0, "via": "dashboard",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    activity_log.reset_runtime_state()

    admin = _as(history_env, auth_env, "admin", "admin")
    body = admin.get("/api/history", params={"limit": 500}).json()
    entry = next(row for row in body["entries"]
                 if row["kind"] == "change" and row["facts"]["path"] == "/api/settings")
    assert entry["facts"]["outcome"] == "unknown"

    since = admin.get("/api/history/since", params={"day": "2026-08-01"}).json()
    assert since["outcomes"]["unknown"] >= 1
    assert since["changed"] == 0, "an act nobody recorded the result of changed nothing anybody can attest to"
    assert since["refused"] == 0, "and it is not called a refusal either"
    assert since["verdict"] == "unknown", "unchanged is a claim, and this record cannot make it"


# --- the figure beside the filter ------------------------------------------------

def test_the_windowed_tally_says_how_many_of_those_changes_were_refused(history_env, auth_env) -> None:
    """The critic's second consequence. The Change filter counts attempts, which
    is right, so the payload carries what share of them the server refused, taken
    over the same day window the tally is taken over."""
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    viewer.post(f"/api/versions/{version_id}/restore", json={})
    viewer.post("/api/definitely-not-a-route")

    body = viewer.get("/api/history", params={"limit": 500}).json()
    assert body["outcomes"]["refused"] == 2, "a 403 and a 404, both refusals"
    assert body["outcomes"]["applied"] + body["outcomes"]["refused"] + body["outcomes"]["unknown"] == (
        body["counts"]["change"]), "every change entry is in exactly one outcome"

    # It is a window figure, exactly like the tally it sits beside, so a day the
    # reader is not looking at cannot contribute to it.
    empty = viewer.get("/api/history", params={"limit": 500, "until": "2020-01-01"}).json()
    assert empty["counts"]["change"] == 0 and empty["outcomes"]["refused"] == 0


def test_the_since_route_and_the_landing_attestation_still_agree(history_env, auth_env) -> None:
    """The landing verdict rides the timeline read, so the two bodies have to be
    the same body, including the new figures."""
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    viewer.post("/api/definitely-not-a-route")
    landing = viewer.get("/api/history").json()
    assert landing["attestation"] == viewer.get("/api/history/since").json()


def test_the_settings_activity_log_reads_the_same_decision() -> None:
    """One classification, two surfaces. The panel on the rules page had the same
    defect from the same cause, and it is closed from the same module."""
    entry = activity_log._with_action({"method": "POST", "path": "/api/versions/x/restore", "status": 403})
    assert entry["action"] == "restore" and entry["outcome"] == "refused" and entry["saved"] is True
    assert activity_log._with_action({"method": "PUT", "path": "/api/settings", "status": 200})["outcome"] == "applied"


def test_the_unit_under_all_of_it_counts_only_the_change_kind() -> None:
    entries = [
        {"kind": "change", "facts": {"outcome": "applied"}},
        {"kind": "change", "facts": {"outcome": "refused"}},
        {"kind": "change", "facts": {"outcome": "unknown"}},
        {"kind": "preview", "facts": {"outcome": "refused"}},
        {"kind": "run", "facts": {}},
        {"kind": "restore_point", "facts": {}},
    ]
    assert history_api_timeline.outcome_counts(entries) == {"applied": 1, "refused": 1, "unknown": 1}


# --- the sentence a person reads, executed rather than grepped -------------------

MODULE_PROBE = """
import { REFUSED_LABELS, actLabel, doorLabel, outcomeNote, outcomeOf, outcomeWord, refusedSinceLine, refusedTabLine }
  from './tv-break-dashboard/src/history/history-refused.js';
import { ACTION_LABELS } from './tv-break-dashboard/src/history/history-labels.js';
const codes = Object.keys(ACTION_LABELS);
console.log(JSON.stringify({
  codes: codes.length,
  refused: Object.keys(REFUSED_LABELS).length,
  missing: codes.filter((code) => !REFUSED_LABELS[code]),
  same: codes.filter((code) => REFUSED_LABELS[code][0] === ACTION_LABELS[code][0]
    || REFUSED_LABELS[code][1] === ACTION_LABELS[code][1]),
  blank: codes.filter((code) => !REFUSED_LABELS[code][0].trim() || !REFUSED_LABELS[code][1].trim()),
  guardrailHe: [actLabel('guardrail_change', 'applied', 'he'), actLabel('guardrail_change', 'refused', 'he')],
  guardrailEn: [actLabel('guardrail_change', 'applied', 'en'), actLabel('guardrail_change', 'refused', 'en')],
  switchHe: actLabel('model_activation_change', 'refused', 'he'),
  unknownCode: [actLabel('nothing-like-this', 'applied', 'en'), actLabel('nothing-like-this', 'refused', 'en')],
  read: [outcomeOf({facts: {outcome: 'refused'}}), outcomeOf({facts: {status: 403}}), outcomeOf({status: 200}),
    outcomeOf({facts: {status: 200}}), outcomeOf({}), outcomeOf({facts: {status: null}}),
    outcomeOf({facts: {status: 500}}), outcomeOf({status: 503})],
  doors: [doorLabel('applied', false, 'he'), doorLabel('refused', false, 'he'), doorLabel('refused', true, 'he'),
    doorLabel('unknown', false, 'he'), doorLabel('applied', true, 'he')],
  chips: [outcomeWord('refused', 'he'), outcomeWord('unknown', 'he'), outcomeWord('applied', 'he')],
  notes: [outcomeNote('refused', 'he'), outcomeNote('unknown', 'en'), outcomeNote('applied', 'en')],
  tab: [refusedTabLine(680), refusedTabLine(1)],
  since: [refusedSinceLine(743, 'all'), refusedSinceLine(1, 'all')],
  sinceSelf: [refusedSinceLine(160, 'self'), refusedSinceLine(1, 'self'), refusedSinceLine(2, 'self')],
}));
"""


def _run_module() -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", MODULE_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-600:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_every_act_the_product_can_record_has_a_word_for_being_refused() -> None:
    """Executed rather than read, because the defect was a word that existed and
    was chosen wrongly. A code with no refused word of its own would fall back to
    the word for the act that happened, which is the defect itself."""
    measured = _run_module()
    assert measured["codes"] == measured["refused"] == 43
    assert measured["missing"] == [], "a code with no refused word falls back to the word for the act"
    assert measured["same"] == [], "and a refused word identical to the applied one is the same failure"
    assert measured["blank"] == []
    assert measured["guardrailHe"] == ["מגבלת רגולציה נשמרה", "שמירת מגבלת רגולציה נדחתה"]
    assert measured["guardrailEn"] == ["Regulatory limit saved", "Regulatory limit save refused"]
    assert measured["switchHe"] == "שינוי מתג מודל הקהל נדחה", (
        "the switch is thrown in two directions, so a refusal of it names the change and not the direction")
    assert measured["unknownCode"] == ["Change", "Change refused"], (
        "an unclassified act still says whether it happened")


def test_the_outcome_is_read_from_the_payload_and_from_either_shape() -> None:
    """The timeline nests the fields under facts and the settings activity log
    carries them at the top level, and one rule reads both. A record with neither
    a stated outcome nor a status is unknown, never the happy answer, and a
    server failure is unknown on both sides of the wire rather than a refusal on
    one and a refusal on the other."""
    measured = _run_module()
    assert measured["read"] == [
        "refused", "refused", "applied", "applied", "unknown", "unknown", "unknown", "unknown"]
    for code in (500, 503):
        assert history_api_actions.outcome_for(code) == measured["read"][-1], (
            "the surface and the endpoint draw the same two lines")


def test_a_refused_act_is_offered_no_door_to_the_surface_it_did_not_change() -> None:
    measured = _run_module()
    doors = measured["doors"]
    assert doors[0] == "פתחו את המסך שהשתנה"
    assert doors[1] == "" and doors[2] == "", "a refusal opens nothing, whichever kind it was"
    assert doors[3] == "פתחו את המסך שהפעולה נגעה בו", (
        "an act nobody recorded the result of does not claim to have changed the surface either")
    assert doors[4] == "פתחו את המסך שעבורו זה חושב", "a preview keeps its own words"

    chips = measured["chips"]
    assert chips[0] == "נדחתה" and chips[1] == "התוצאה לא ידועה"
    assert chips[2] == "", "an act that happened needs no word saying so; every other row on the list is one"
    notes = measured["notes"]
    assert "לא נשמר דבר" in notes[0] and "אין כאן מה להחזיר" in notes[0]
    assert notes[1].startswith("The recorded result does not say"), (
        "true whether the recorder wrote no status at all or the server failed on it")
    assert "was refused" not in notes[1] and "nothing was saved" not in notes[1]
    assert notes[2] == ""


def test_the_two_sentences_that_carry_a_refusal_count_carry_only_that_figure() -> None:
    """Both languages, both plurals, and no figure the payload did not give."""
    measured = _run_module()
    tab_many, tab_one = measured["tab"]
    assert tab_many == ["680 of these were refused and changed nothing.", "נדחו 680 מהן ולא שינו דבר."]
    assert tab_one == ["One of these was refused and changed nothing.", "אחת מהן נדחתה ולא שינתה דבר."]
    since_many, since_one = measured["since"]
    assert since_many == ["743 attempts were refused and changed nothing.", "נדחו 743 ניסיונות שלא שינו דבר."]
    assert since_one == ["One attempt was refused and changed nothing.", "ניסיון אחד נדחה ולא שינה דבר."]
    for sentence in (*tab_many, *tab_one, *since_many, *since_one):
        assert not any(char.isdigit() for char in sentence) or any(
            figure in sentence for figure in ("680", "743")), sentence
    # The Hebrew opens on a Hebrew word, which is the round-nine lesson: dir=auto
    # takes its direction from the first strong character and a page of digits
    # would flip the sentence inside a right-to-left page.
    for hebrew in (tab_many[1], tab_one[1], since_many[1], since_one[1]):
        assert hebrew[0].isalpha() and not hebrew[0].isascii()


def test_the_since_refusal_sentence_names_its_own_scope_and_never_the_stores() -> None:
    """The defect a blind critic measured on 2026-08-07: the same store, the same
    minutes, read 'נדחו 160 ניסיונות שלא שינו דבר' as an admin and 'נדחו 2 ניסיונות
    שלא שינו דבר' as a self-scoped reader, with no scope word on either, even
    though `refusedSinceLine`'s figure is drawn from `outcome_counts`, which is
    taken over the `change` kind alone and is already filtered to the caller's
    own account before it reaches this function whenever `scope` is 'self'
    (`history_api._assemble` narrows `activity` to `self_user` ahead of the
    merge). `sinceCountLine` and `sinceEmptyLine` already carry this argument;
    this is the third sentence on the strip that did not, one line lower and
    otherwise untouched."""
    measured = _run_module()
    self_many, self_one, self_two = measured["sinceSelf"]
    since_many, _since_one = measured["since"]

    # Every figure is still the payload's own, in both languages.
    assert "160" in self_many[0] and "160" in self_many[1]

    # A self-scoped figure can never read as the identical, unqualified claim an
    # all-scoped figure makes: the two sentences differ, plural and singular, in
    # both languages, even when the count would otherwise line up as identical
    # wording (743 vs any self-scoped many-count is already a different number,
    # so the singular case is the one that would have collided silently).
    assert self_one != ["One attempt was refused and changed nothing.", "ניסיון אחד נדחה ולא שינה דבר."]
    assert self_many[0] != since_many[0] and self_many[1] != since_many[1]

    # The self-scoped sentence names the set it covers, in both languages.
    for pair_ in (self_many, self_one, self_two):
        assert "your own" in pair_[0], f"the self-scoped refusal line names its own set: {pair_[0]}"
        assert "שלכם" in pair_[1], f"the self-scoped refusal line names its own set: {pair_[1]}"

    # The all-scope sentence is untouched: identical to what it always was.
    assert since_many == ["743 attempts were refused and changed nothing.", "נדחו 743 ניסיונות שלא שינו דבר."]


# --- the surfaces ask for it -----------------------------------------------------

def test_the_row_and_the_opened_entry_both_read_the_outcome() -> None:
    row = _read("HistoryRow.jsx")
    assert "actLabel(facts.action, outcomeOf(entry), locale)" in row, "the row says what the act did"
    assert "ACTION_LABELS" not in row, "and never the word for the act that happened, whatever happened"
    assert 'data-outcome={outcome || undefined}' in row, "so the row can be drawn as what it was"
    assert 'className="hist-chip refused"' in row, "and carries the word, because colour is not a signal on its own"

    detail = _read("HistoryDetail.jsx")
    assert "const door = doorLabel(outcome, preview, locale) ? ACTION_DOORS[facts.action] : '';" in detail, (
        "the door is withheld from an act that changed nothing")
    assert "{actLabel(facts.action, outcome, locale)}" in detail
    assert "{outcomeNote(outcome, locale)}" in detail, "and the opened entry says so in words"
    css = _read("history.css")
    assert '.hist-row[data-outcome="refused"] .hist-row-title' in css


def test_the_attestation_strip_reads_the_endpoints_own_applied_figure() -> None:
    """The strip summed the tabs, and the tabs count attempts. It now reads the
    figure the endpoint computed, and prints the refusals as their own sentence."""
    since = _read("HistorySince.jsx")
    assert "const changeCount = Number((body && body.changed) || 0);" in since
    assert "const refusedCount = Number((body && body.refused) || 0);" in since
    assert "counts.change" not in since, "the tab tally is never summed into an attestation again"
    assert "refusedSinceLine(refusedCount, scope)" in since, (
        "the refusal sentence reads the same scope the count and empty sentences already do")
    assert "(changeCount || refusedCount) && onShow" in since, (
        "and a window holding only refusals can still be opened in the list")

    page = _read("HistoryPage.jsx")
    assert "const refused = Number((((body && body.outcomes) || {}).refused) || 0);" in page
    assert "{name === 'change' && refused ? (" in page, "the Change tab says how many of them were refused"
    assert "title={pageText(locale, ...refusedTabLine(refused))}" in page, "with its sentence on the control"


def test_the_search_and_the_fold_both_follow_the_word_the_row_prints() -> None:
    """Two smaller instances of the same defect. A reader searching for what was
    refused types the word they can see, and a folded row prints one title, so an
    act that happened and the same act refused may not fold into one."""
    fold = _read("history-fold.js")
    assert "const action = actLabel(facts.action, outcomeOf(entry), locale);" in fold
    assert "&& outcomeOf(a) === outcomeOf(b)" in fold


def test_the_word_for_a_kind_no_longer_claims_every_one_of_them_saved_something() -> None:
    labels = _read("history-labels.js")
    assert "change: ['Something was saved, or the attempt was refused', 'משהו נשמר, או שהניסיון נדחה']," in labels
    panel = _read("ActivityLogPanel.jsx")
    assert "return actLabel(action, outcomeOf(entry), locale) || null;" in panel, (
        "the settings activity log had the same defect from the same cause")
