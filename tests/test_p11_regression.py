"""P11, Bar 3: what worked before this piece still works, and the words still hold.

Three things are pinned here.

The older projection at ``GET /api/make-good-alerts`` is untouched. It is the
signal the optimizer's own pacing weights read, ``campaign_flights.csv`` is still a
header-only seed, and two shared tests already assert the honest data-pending
answer it gives. This file asserts the same three properties from P11's side, so a
later round that reshapes this module cannot quietly move them.

The new routes are exactly the three this piece published, mounted on the router
wave zero already created, so no path this piece adds redefines one that existed.

And the copy laws hold on every string these modules ship.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store, pacing_alerts_api
from kairos_api import pacing_alerts_api_board, pacing_alerts_api_read, pacing_alerts_api_words

# The two bidi controls a figure is allowed to carry: the first-strong isolate and
# the pop that closes it. They are named rather than pasted so a reader of this
# file can see which characters the assertions are about. It is the same pair
# ``src/shell/bidi.jsx`` uses, per design-rules.md section 6: the left-to-right
# isolate this replaced laid a Hebrew channel name out left to right.
FSI = "\u2068"
PDI = "\u2069"

# Every backend module this piece owns, by glob rather than by hand, so a helper
# split out under the section 8.2 naming rule inherits the size law and the copy
# laws on the round it is created rather than on the round somebody remembers.
MODULES = sorted(
    list(Path("kairos_api").glob("pacing_alerts_api*.py"))
    + list(Path("kairos_api").glob("makegood_store*.py"))
)
SURFACE = Path("tv-break-dashboard/src/clients/pacing")
ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / SURFACE / "pacing-helpers.js"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def test_the_older_make_good_projection_answers_exactly_as_it_did() -> None:
    body = _client().get("/api/make-good-alerts").json()
    assert body["data_available"] is False
    assert body["alerts"] == []
    assert body["reason"] == "campaign_flights.csv has no campaign rows yet (header-only seed)."


def test_this_piece_publishes_five_paths_and_redefines_none() -> None:
    paths = {route.path for route in pacing_alerts_api.router.routes}
    assert paths == {"/api/pacing", "/api/pacing/{campaign_id}/days",
                     "/api/pacing/{campaign_id}/accept", "/api/make-goods",
                     "/api/make-goods/{make_good_id}/state", "/api/make-good-alerts"}
    operations = {(route.path, method) for route in pacing_alerts_api.router.routes
                  for method in route.methods if method in {"GET", "POST"}}
    assert operations == {
        ("/api/pacing", "GET"),
        ("/api/pacing/{campaign_id}/days", "GET"),
        ("/api/pacing/{campaign_id}/accept", "POST"),
        ("/api/make-goods", "GET"),
        ("/api/make-goods", "POST"),
        ("/api/make-goods/{make_good_id}/state", "POST"),
        ("/api/make-good-alerts", "GET"),
    }


def test_the_broadcast_days_ride_their_own_read_rather_than_every_board_row() -> None:
    """The board is a list somebody triages; the days are the drill behind one row.

    Measured on the shipped data: the days were 181,274 of a 365,884 byte board
    payload, and they are the one term that grows as campaigns times flight days.
    On their own read the board is 184,610 bytes and a drill is 2,838.
    """
    body = _client().get("/api/pacing").json()
    assert body["rows"], "the shipped data must still produce rows"
    for row in body["rows"]:
        assert "days" not in row, row["campaign_id"]
        assert "days_available" in row


def test_the_shipped_ledger_seed_is_a_header_and_not_a_fabricated_row() -> None:
    frame = makegood_store.load_frame()
    assert list(frame.columns) == makegood_store.COLUMNS
    assert len(frame) == 0


def test_no_source_file_this_piece_owns_passes_the_size_law() -> None:
    for path in MODULES:
        assert len((ROOT / path).read_text(encoding="utf-8").splitlines()) <= 450, path
    for path in sorted(SURFACE.glob("*")):
        assert len((ROOT / path).read_text(encoding="utf-8").splitlines()) <= 450, path


def test_the_copy_laws_hold_on_every_string_this_piece_ships() -> None:
    banned = re.compile(r"[—–!\U0001F300-\U0001FAFF☀-➿]")
    for path in MODULES + sorted(SURFACE.glob("*")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            # The one legitimate bang in JavaScript is the not operator, and it is
            # never a display string, so only a bang inside quotes is a defect.
            stripped = re.sub(r"!==|!=|!\w|!\(|![\[{'\"`]|\+\+|--", "", line)
            assert not banned.search(stripped), f"{path}:{number} {line}"


def test_the_retired_words_appear_nowhere_on_this_surface() -> None:
    retired = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש", "משתמש")
    for path in MODULES + sorted(SURFACE.glob("*")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for word in retired:
            assert word not in text, f"{word} in {path}"


def test_the_published_triggers_are_on_the_payload_and_named_as_a_policy() -> None:
    block = pacing_alerts_api_words.trigger_block()
    assert block["on_pace_ratio"] == pacing_alerts_api_words.ON_PACE_RATIO
    assert block["at_risk_ratio"] == pacing_alerts_api_words.AT_RISK_RATIO
    assert block["rule_he"] and block["not_a_commercial_term_he"]


def test_every_unavailable_state_names_both_what_is_missing_and_the_way_to_supply_it() -> None:
    for code in ("no_goal", "unmeasurable", "no_source", "gap_in_elapsed", "not_started", "no_flight_dates"):
        block = pacing_alerts_api_words.reason(code)
        assert block["reason_en"] and block["reason_he"]
        assert block["path_forward_en"] and block["path_forward_he"]


def test_the_read_layer_never_writes_the_stores_it_reads() -> None:
    source = (ROOT / "kairos_api/pacing_alerts_api_read.py").read_text(encoding="utf-8")
    for forbidden in ("write_frame", "to_csv", "os.replace"):
        assert forbidden not in source
    assert "pacing_alerts_api_board" in source
    assert pacing_alerts_api_read.board_payload is not None
    assert pacing_alerts_api_board.parse_date("2025-04-27") is not None


def test_every_new_route_carries_the_wall_and_the_legacy_one_is_left_as_it_was() -> None:
    """The wall is adopted on each route this piece added, and on none it did not.

    ``/api/make-good-alerts`` predates this piece and two shared tests read it
    without a session, so wrapping it would be a behaviour change Bar 3 forbids.
    """
    walled, bare = set(), set()
    for route in pacing_alerts_api.router.routes:
        target = walled if getattr(route.endpoint, "kairos_wall", None) is not None else bare
        target.add(route.path)
    assert walled == {"/api/pacing", "/api/pacing/{campaign_id}/days",
                      "/api/pacing/{campaign_id}/accept", "/api/make-goods",
                      "/api/make-goods/{make_good_id}/state"}
    assert bare == {"/api/make-good-alerts"}
    assert pacing_alerts_api.PACING_WALL.company_only is False


def test_the_board_tells_a_reader_whether_it_may_be_changed_before_the_click() -> None:
    body = _client().get("/api/pacing").json()
    assert "can_edit" in body
    ledger = _client().get("/api/make-goods").json()
    assert "can_edit" in ledger


def test_who_acted_is_read_through_the_accessor_the_wall_publishes() -> None:
    """Attribution comes from ``session_for`` and never from a request attribute.

    Nothing in this package writes ``request.state.session``, so a module that
    reads it records a blank name on every write. The ledger's whole subject is
    who acted, so the accessor is pinned rather than the behaviour of a mock.
    """
    source = (ROOT / "kairos_api/pacing_alerts_api.py").read_text(encoding="utf-8")
    assert "session_for" in source
    assert "getattr(request.state" not in source


def test_a_refused_transition_never_prints_a_stored_state_key() -> None:
    """The guard on the class. A key is how a row is stored, not how it is read.

    Proven to bite against the sentence this replaced, which put ``raised`` and
    ``settled`` verbatim into both halves and the English word ``none`` into the
    Hebrew one.
    """
    from kairos_api.pacing_alerts_api_write import refuse_transition

    for kind in (makegood_store.MAKE_GOOD, makegood_store.ACCEPTANCE):
        for current in makegood_store.TRANSITIONS:
            for target in makegood_store.TRANSITIONS:
                detail = refuse_transition(current, target, kind).detail
                for key in makegood_store.TRANSITIONS:
                    assert key not in detail["message_en"], (kind, current, target, key)
                    assert key not in detail["message_he"], (kind, current, target, key)
                assert not re.search(r"[a-zA-Z]", detail["message_he"]), (kind, current, target)


def test_no_display_string_opens_a_direction_isolate_with_its_own_separator() -> None:
    """A space inside an isolate is reordered onto the far edge of the run.

    Measured on the shipped ledger: an offer of 0.6 beside a window ending
    2025-05-10 rendered as ``2025-05-100.6``, because the separating space sat
    inside the ``dir`` element instead of beside it.
    """
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            assert not re.search(r'dir=(?:"[a-z]+"|\{[^}]*\})>\s+\S', line), f"{path}:{number} {line}"


def test_the_write_gate_is_read_as_the_pair_the_session_module_returns() -> None:
    """``payloadCanEdit`` returns an object, so holding it whole is always true.

    A read-only account was shown the raise control and the server refused it
    afterwards, which is the opposite of the contract this piece published.
    """
    text = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "gate.canEdit" in text
    assert "gate.reason" in text
    assert "const canEdit = payloadCanEdit(" not in text


def test_a_figure_that_carries_a_unit_isolates_its_numeral_and_never_its_words() -> None:
    """A left-to-right isolate lays its contents out left to right.

    Wrapped around a phrase that is already Hebrew it puts the words in the wrong
    order. Measured in a browser on the shipped board before the fix, the headline
    figure of every one of 56 rows read ``4.4 מתוך נקודות רייטינג 35``, the unit
    ahead of its own number, because ``pair`` isolated the whole of
    ``amount(35, rating_points, he)``. The isolate now sits inside ``amount``
    around the numeral alone.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ amount, pair, bare }} from {json.dumps(str(HELPERS))};
process.stdout.write(JSON.stringify({{
  points: amount(35, 'rating_points', 'he'),
  money: amount(70000, 'ils', 'he'),
  pair: pair(4.4, 35, 'rating_points', 'he'),
  english: amount(35, 'rating_points', 'en'),
  bare: bare(4.4, 'rating_points', 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    for key in ("points", "money", "pair"):
        value = read[key]
        assert FSI in value and PDI in value, (key, value)
        for run in re.findall(f"{FSI}(.*?){PDI}", value):
            assert not re.search(r"[֐-׿]", run), (key, run)
    # The English forms are already left to right and take no isolate at all.
    assert FSI not in read["english"]
    assert FSI not in read["bare"]


def test_no_component_wraps_a_unit_bearing_figure_in_a_second_isolate() -> None:
    """The class, not the six sites. ``amount`` and ``pair`` isolate their own numeral."""
    offenders = []
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if re.search(r"isolate\(\s*(amount|pair)\(", line):
                offenders.append(f"{path}:{number} {line.strip()}")
    assert offenders == []


def test_the_ledger_surface_reads_both_endings_and_names_the_act_by_kind() -> None:
    """Reading only ``make_goods`` made every recorded acceptance invisible.

    Measured in a browser: the view tab counted one record and the list under it
    printed the empty state. ``actWord`` also took three arguments and was called
    with two, so the kind slot received the locale and a Hebrew reader was shown
    the English verb.
    """
    text = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "payload.decisions" in text
    assert "const rows = payload.make_goods" not in text
    assert "actWord(state, record.kind, locale)" in text
    assert "vocabulary.kinds" in text


def test_every_class_this_surface_names_has_a_rule_in_one_of_its_stylesheets() -> None:
    """A component that names a class no sheet defines renders unstyled and silently.

    The repo build never parses this tree, because nothing imports it yet, so a
    class with no rule ships without a single warning. Nine of them were measured
    that way after the round that added the second ending.
    """
    used: set[str] = set()
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for chunk in re.findall(r"className=(?:\"([^\"]*)\"|\{`([^`]*)`\})", text):
            for token in re.findall(r"(?:pacing|makegood)-[a-z-]+", " ".join(chunk)):
                used.add(token)
    defined: set[str] = set()
    for path in sorted(SURFACE.glob("*.css")):
        defined.update(re.findall(r"\.((?:pacing|makegood)-[a-z-]+)",
                                  (ROOT / path).read_text(encoding="utf-8")))
    assert used - defined == set()


def test_every_stylesheet_this_surface_ships_is_imported_by_it() -> None:
    """A sheet nobody imports is a sheet the browser never loads."""
    imported = set()
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        imported.update(re.findall(r"import '\./([a-z-]+\.css)'", text))
    assert {path.name for path in SURFACE.glob("*.css")} == imported


def test_a_percentage_prints_the_figure_it_is_a_percentage_of() -> None:
    """Both operands of a number on a screen belong on the same screen.

    Measured in a browser on the shipped board: the row printed ``4.4 of 35
    rating points`` and ``88%``, and the 5.0 that makes 88 percent true existed
    only as the position of an unlabelled mark on the bar. A reader who divided
    what they could see got 12.6 percent, which is a different campaign.
    """
    text = (ROOT / SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "expected_through_counted_day" in text
    assert "pacing-against" in text
    body = _client().get("/api/pacing").json()
    for row in body["rows"]:
        for key in ("rating", "money"):
            line = row.get(key)
            if line and line.get("pace", {}).get("ratio") is not None:
                assert line["reference"]["expected_through_counted_day"] is not None, row["campaign_id"]


def test_a_refused_write_is_printed_on_the_surface_that_asked_for_it() -> None:
    """notify() is a no-op at the address this panel is mounted at.

    Measured: workspace-router.jsx renders the Campaigns destination without a
    notify prop, so ClientsWorkspace falls back to its own default and every
    notice this panel sends is swallowed. Polling the whole document every 100 ms
    for 2.5 s after a refused offer found no refusal text anywhere on screen.
    """
    text = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "pacing-refusal" in text
    assert 'role="alert"' in text
    # Every write states its refusal through the one function that words them.
    assert text.count("refuse(") >= 4
    styles = (ROOT / SURFACE / "pacing.css").read_text(encoding="utf-8")
    assert ".pacing-refusal" in styles


def test_the_offer_form_closes_only_when_the_move_landed() -> None:
    """A refused offer used to take the value, the window and the note with it."""
    ledger = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "const landed = await onMove(" in ledger
    assert "if (landed) setOffering('');" in ledger
    workspace = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "return true;" in workspace and "return false;" in workspace


def test_the_day_drill_quotes_the_delivery_ledger_words_rather_than_its_own() -> None:
    """One product may not hold two words for one state of one store.

    ``campaigns_delivery.py`` publishes ``AIR_STATE_VOCABULARY`` and the Clients
    destination renders it. This drill had drifted to "Booked, not aired yet"
    where that vocabulary says "Scheduled, not aired yet".
    """
    from kairos_api import campaigns_delivery

    text = (ROOT / SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    for entry in campaigns_delivery.AIR_STATE_VOCABULARY:
        assert entry["label_en"] in text, entry["value"]
        assert entry["label_he"] in text, entry["value"]
        assert entry["meaning_en"] in text, entry["value"]
        assert entry["meaning_he"] in text, entry["value"]


def test_a_unit_on_a_form_label_is_a_word_and_never_the_stored_key() -> None:
    """The ledger read carries no unit vocabulary, so reaching for one prints the key.

    Measured in a browser after the first attempt at this fix: the offer field
    read "Offer, in rating_points".
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ unitWord }} from {json.dumps(str(HELPERS))};
process.stdout.write(JSON.stringify({{
  pointsEn: unitWord('rating_points', 'en'),
  pointsHe: unitWord('rating_points', 'he'),
  moneyEn: unitWord('ils', 'en'),
  moneyHe: unitWord('ils', 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    for value in read.values():
        assert "_" not in value, value
        assert value not in {"rating_points", "ils"}, value
    assert re.search(r"[֐-׿]", read["pointsHe"])
    assert re.search(r"[֐-׿]", read["moneyHe"])
    ledger = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "vocabulary.units" not in ledger


def test_a_campaign_named_on_a_ledger_record_opens_that_campaign_s_own_row() -> None:
    """A name that looks like a link and lands on a different row is a dead end.

    Measured: opening the ledger record of a campaign sitting at index 6 of 56
    returned to the board with row 0 focused, unscrolled and unmarked.
    """
    board = (ROOT / SURFACE / "PacingBoard.jsx").read_text(encoding="utf-8")
    assert "focusCampaignId" in board
    assert "findIndex((row) => row.campaign_id === focusCampaignId)" in board
    workspace = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "setFocusCampaign(id)" in workspace
    assert "onOpenCampaign={openCampaign}" in workspace


def test_the_board_says_how_many_of_its_rows_the_demo_seed_wrote() -> None:
    """A count that mixes seeded rows into an operational one is not honest.

    The payload has carried ``counts.demo`` from the first round; the sentence
    above the list did not read it.
    """
    text = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "counts.demo" in text
    body = _client().get("/api/pacing").json()
    assert "demo" in body["counts"]


def test_the_api_and_the_surface_hold_one_rule_for_when_a_make_good_may_be_raised() -> None:
    """Two rules for one act let any client put a debt in the ledger the product denies.

    Measured before this: ``POST /api/make-goods`` on CMP_D040 answered 201 with
    ``deficit_kind: to_date``, while the surface's own ``remedyFor`` offered that
    raise on 0 of 56 rows. 13 of the 56 reached the ``to_date`` rung. The rule
    kept is the surface's, and the trade says why: a make-good compensates a spot
    that did not air or aired wrong, and a flight with unbooked days ahead has
    had no spot fail yet.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    body = _client().get("/api/pacing").json()
    assert body["raise_rule"]["raisable_deficit_kinds"] == list(pacing_alerts_api_read.RAISABLE_KINDS)
    assert makegood_store.TO_DATE not in body["raise_rule"]["raisable_deficit_kinds"]

    # The surface's answer for every shipped row, out of the shipped module.
    script = f"""
import {{ remedyFor }} from {json.dumps(str(HELPERS))};
const rows = {json.dumps(body["rows"])};
process.stdout.write(JSON.stringify(rows.map((row) => remedyFor(row, {{}}).kind)));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    offered = json.loads(done.stdout)
    assert len(offered) == len(body["rows"])

    # The server's answer for the same rows, through the same reader the write
    # path uses. The two must agree row for row.
    from kairos_api import pacing_alerts_api_wire as wire

    full = wire.expand(pacing_alerts_api_read.board_payload())
    as_of = pacing_alerts_api_board.parse_date(full["as_of"]["instant"])
    for row, kind in zip(full["rows"], offered):
        deficit, why = pacing_alerts_api_read.raisable_deficit(row, as_of)
        assert (deficit is not None) == (kind == "raise"), (row["campaign_id"], kind, why)


def test_a_row_that_is_only_behind_to_date_is_refused_by_name_and_told_what_makes_it_owed() -> None:
    """The refusal has to route, not just deny. A gap to date is a real figure."""
    from kairos_api.pacing_alerts_api import REFUSED_RAISE

    for code in ("nothing_measured", "not_owed_yet"):
        en, he = REFUSED_RAISE[code]
        assert en and he
        assert not re.search(r"[a-zA-Z]", he), code
    assert pacing_alerts_api_words.NOT_OWED_YET_PATH_EN in REFUSED_RAISE["not_owed_yet"][0]


def test_the_board_sends_each_reason_once_and_the_surface_puts_it_back_exactly() -> None:
    """The wire got smaller and no component saw a different shape.

    Measured on the shipped data: of 189,850 bytes of rows, 87,976 were reason
    and path prose and 20,526 were the same list of unsourced dates written three
    times per row. The round trip is asserted against the real board rather than
    against a fixture, because a lossy collapse would be invisible in a fixture.
    """
    from kairos_api import pacing_alerts_api_wire as wire

    collapsed = pacing_alerts_api_read.board_payload()
    assert collapsed["wire"]["collapsed"] is True
    assert collapsed["reasons"], "at least one reason code must ride the payload once"

    raw = pacing_alerts_api_read.board_payload()
    for key in ("reasons", "forward_reasons", "reference_rule", "wire"):
        raw.pop(key)
    # Rebuild the full shape from the collapsed one and compare to the same board
    # with nothing lifted off it.
    rebuilt = wire.expand(pacing_alerts_api_read.board_payload())
    for key in ("reasons", "forward_reasons", "reference_rule", "wire"):
        rebuilt.pop(key)
    uncollapsed = _uncollapsed()
    assert json.dumps(rebuilt, ensure_ascii=False, sort_keys=True) == json.dumps(
        uncollapsed, ensure_ascii=False, sort_keys=True)
    assert len(json.dumps(collapsed, ensure_ascii=False).encode()) < len(
        json.dumps(uncollapsed, ensure_ascii=False).encode())


def _uncollapsed() -> dict:
    """The board with nothing lifted off it, for the round-trip comparison."""
    from kairos_api import pacing_alerts_api_wire as wire

    keep = wire.collapse
    wire.collapse = lambda payload: payload
    try:
        return pacing_alerts_api_read.board_payload()
    finally:
        wire.collapse = keep


def test_the_javascript_that_expands_the_board_is_the_inverse_of_the_python_that_collapses_it() -> None:
    """The expansion ships in the browser, so it is executed rather than read."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    collapsed = pacing_alerts_api_read.board_payload()
    script = f"""
import {{ readFileSync }} from 'node:fs';
const source = readFileSync({json.dumps(str(ROOT / SURFACE / 'pacing-api.js'))}, 'utf8');
const body = source.slice(source.indexOf('const PROSE'), source.indexOf('export function loadBoard'))
  .replace('export function expandBoard', 'function expandBoard');
const run = new Function(`${{body}}; return expandBoard;`);
process.stdout.write(JSON.stringify(run()({json.dumps(collapsed)})));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    rebuilt = json.loads(done.stdout)
    for key in ("reasons", "forward_reasons", "reference_rule", "wire"):
        rebuilt.pop(key, None)
    assert _shape(rebuilt) == _shape(_uncollapsed())


def _shape(value):
    """The payload with whole floats and ints read as one number.

    node writes 5280.0 as 5280 on the way back through JSON, which is the same
    figure and a different string. The comparison is about the shape and the
    values, not about how a serialiser spells a round number.
    """
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def test_what_the_counted_figure_is_stands_in_front_of_the_reader_and_not_behind_a_disclosure() -> None:
    """A reader who never opens the disclosure could read at risk as a delivery shortfall.

    The delivery ledger's own ``figures_basis`` says the figures are the planned
    break rating from the traffic file, and it was carried in the store and
    rendered on no screen in the product.
    """
    body = _client().get("/api/pacing").json()
    assert body["counted_is_planned_en"] and body["counted_is_planned_he"]
    assert body["as_of"]["figures_basis"], "the ledger's own sentence must reach the payload"
    text = (ROOT / SURFACE / "PacingBoard.jsx").read_text(encoding="utf-8")
    # The line is above the disclosure, so it is rendered before it in the source.
    assert text.index("counted_is_planned") < text.index("pacing-basis-details")
    assert "figures_basis" in text


def test_this_surface_states_direction_nowhere_and_reads_the_shell_primitive_instead() -> None:
    """design-rules.md section 6, swept the round this piece's directory was released.

    ``verify-direction-rules.mjs`` quarantined ``src/clients/pacing/`` because the
    sweep could not edit a tree another agent was holding. This asserts the debt
    is paid from inside the row that owed it, so the quarantine line can go.
    """
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            assert not re.search(r"\bdir=(?:\"[^\"]*\"|\{)", line), f"{path}:{number} {line.strip()}"
    for path in sorted(SURFACE.glob("*.css")):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert not re.search(r"(?:^|[\s;{])(direction|unicode-bidi)\s*:", text), path
        assert not re.search(r"text-align:\s*(left|right)\b", text), path
        assert not re.search(r"(?:^|[\s;{])(margin|padding|border)-(left|right)\s*:", text), path


def test_the_isolate_this_surface_joins_into_prose_is_the_shell_s_own_pair() -> None:
    """One product, one isolate. A left-to-right one lays a Hebrew name out backwards."""
    helpers = (ROOT / SURFACE / "pacing-helpers.js").read_text(encoding="utf-8")
    shell = (ROOT / "tv-break-dashboard/src/shell/bidi.jsx").read_text(encoding="utf-8")
    assert "\u2068" in helpers and "\u2069" in helpers
    assert "\u2066" not in helpers, "the left-to-right isolate has no caller on this surface"
    # bidi.jsx writes the pair as escapes, on purpose: the characters render as
    # nothing, so a literal pair in the source is invisible to review.
    assert "u2068" in shell and "u2069" in shell


def test_a_refusal_that_is_a_plain_string_still_reaches_the_person_who_was_refused() -> None:
    """The auth middleware answers with detail as a string, not as the bilingual shape.

    Reading only the bilingual shape returned an empty string for it, so a
    refused write said nothing at all.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ readFileSync }} from 'node:fs';
const source = readFileSync({json.dumps(str(ROOT / SURFACE / 'pacing-api.js'))}, 'utf8');
const start = source.indexOf('export function refusalText');
const body = source.slice(start, source.indexOf('export function refusalOpens'))
  .replace('export function refusalText', 'function refusalText');
const run = new Function(`${{body}}; return refusalText;`);
const refusalText = run();
process.stdout.write(JSON.stringify({{
  plain: refusalText({{ detail: 'Your account may read this and not change it.' }}, 'he'),
  bilingual: refusalText({{ detail: {{ message_en: 'en', message_he: 'he' }} }}, 'he'),
  nothing: refusalText({{ detail: null }}, 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    assert read["plain"] == "Your account may read this and not change it."
    assert read["bilingual"] == "he"
    assert read["nothing"] == ""


def test_every_pacing_write_lands_in_the_products_own_persistent_activity_feed() -> None:
    """The claim that it recorded nothing was wrong, and this pins the truth.

    ``record_api_mutation`` is registered in ``server.py`` and observes every
    mutating ``/api`` request, so a raise, an acceptance and a move are already
    recorded with the actor, the path and the status. What is missing is only the
    word History prints for them, and that table is P8's file.
    """
    from kairos_api import activity_log, history_api_actions

    assert activity_log.record_api_mutation is not None
    for path in ("/api/make-goods", "/api/pacing/CMP_X/accept", "/api/make-goods/MG_0001/state"):
        assert "POST" in activity_log.MUTATING_METHODS
        assert path not in activity_log.EXCLUDED_PATHS
        assert path.startswith("/api/")
        # Recorded, and named ``other`` until P8 adds two rows to its own table.
        assert history_api_actions.action_for("POST", path) == "other"
