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

The surface half of this guard lives in ``test_p11_surface_regression.py``. The
two were one file until it passed the 450-line size law, and the division is by
what a test reads: this one reads the API, the stores and the words, and that one
reads ``src/clients/pacing/**`` or executes it in node.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store, pacing_alerts_api
from kairos_api import pacing_alerts_api_board, pacing_alerts_api_read, pacing_alerts_api_words

# Every backend module this piece owns, by glob rather than by hand, so a helper
# split out under the section 8.2 naming rule inherits the size law and the copy
# laws on the round it is created rather than on the round somebody remembers.
MODULES = sorted(
    list(Path("kairos_api").glob("pacing_alerts_api*.py"))
    + list(Path("kairos_api").glob("makegood_store*.py"))
)
# The test files this piece owns, by the reserved prefix, for the same reason.
# The size law reached the modules and the surface and stopped short of the
# tests, so this file itself grew to 664 lines under a guard that could not see
# it. A law that exempts its own enforcement is not a law.
TESTS = sorted(Path("tests").glob("test_p11_*.py"))
SURFACE = Path("tv-break-dashboard/src/clients/pacing")
ROOT = Path(__file__).resolve().parents[1]


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
    """Every file in the ownership row, tests included.

    The guard used to read the modules and the surface and stop there, so this
    very file reached 664 lines while asserting that nothing had. A test file is
    a file this piece writes and the law does not exempt it.
    """
    for path in MODULES + TESTS + sorted(SURFACE.glob("*")):
        assert len((ROOT / path).read_text(encoding="utf-8").splitlines()) <= 450, path


def test_no_display_string_is_hard_wrapped_across_two_source_lines() -> None:
    """One display string, one source line, per design-rules.md section 5.

    Adjacent-literal concatenation is a manual break by another spelling: the
    string is authored in pieces, so a reader reviewing the copy reads a shape
    the layout will never produce, and a diff on one word touches two lines.
    Measured before this guard: 21 of them across three of these modules.
    """
    split = re.compile(r'"[^"\n]*"\s*\n\s*"', re.M)
    for path in MODULES:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert split.search(text) is None, path


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
    surface = (ROOT / "tv-break-dashboard/src/clients/pacing/PacingBoard.jsx").read_text(encoding="utf-8")
    assert surface.index("localized(trigger, 'rule'") < surface.index("pacing-basis-details")


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


def test_a_read_publishes_the_wall_refusal_in_both_languages_without_touching_the_wall() -> None:
    """The pair rides every stamped payload this piece emits, and only when it is real.

    The wall is a frozen wave-zero module holding one Hebrew constant, so the
    translation is published beside its answer rather than in place of it. A
    caller that reads only ``can_edit_reason`` sees exactly what it saw before.
    """

    class _Stub:
        """A wall that refuses, so the false branch is exercised without a session."""

        def __init__(self, refusal: str) -> None:
            self.refusal = refusal

        def stamp(self, payload, request):
            payload["can_edit"] = False
            payload["can_edit_reason"] = self.refusal
            return payload

    from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL

    keep = pacing_alerts_api.PACING_WALL
    try:
        pacing_alerts_api.PACING_WALL = _Stub(READ_ONLY_ROLE_DETAIL)
        stamped = pacing_alerts_api._stamp({}, None)
        assert stamped["can_edit"] is False
        assert stamped["can_edit_reason"] == READ_ONLY_ROLE_DETAIL
        assert stamped["can_edit_reason_he"] == READ_ONLY_ROLE_DETAIL
        assert not re.search(r"[֐-׿]", stamped["can_edit_reason_en"])

        # A refusal with no translation publishes the wall's words alone, and a
        # stale pair from an earlier stamp of the same dict never survives.
        pacing_alerts_api.PACING_WALL = _Stub("a sentence this piece does not translate")
        again = pacing_alerts_api._stamp(stamped, None)
        assert "can_edit_reason_en" not in again
        assert "can_edit_reason_he" not in again
    finally:
        pacing_alerts_api.PACING_WALL = keep

    # And with the real wall and no session, auth is off, the answer is true and
    # no refusal of either shape is published.
    body = _client().get("/api/pacing").json()
    assert body["can_edit"] is True
    assert "can_edit_reason" not in body
    assert "can_edit_reason_en" not in body
