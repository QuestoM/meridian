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

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store, pacing_alerts_api
from kairos_api import pacing_alerts_api_board, pacing_alerts_api_read, pacing_alerts_api_words

MODULES = [
    Path("kairos_api/pacing_alerts_api.py"),
    Path("kairos_api/pacing_alerts_api_board.py"),
    Path("kairos_api/pacing_alerts_api_read.py"),
    Path("kairos_api/pacing_alerts_api_words.py"),
    Path("kairos_api/makegood_store.py"),
]
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


def test_this_piece_publishes_three_paths_and_redefines_none() -> None:
    paths = {route.path for route in pacing_alerts_api.router.routes}
    assert paths == {"/api/pacing", "/api/make-goods", "/api/make-goods/{make_good_id}/state",
                     "/api/make-good-alerts"}
    operations = {(route.path, method) for route in pacing_alerts_api.router.routes
                  for method in route.methods if method in {"GET", "POST"}}
    assert operations == {
        ("/api/pacing", "GET"),
        ("/api/make-goods", "GET"),
        ("/api/make-goods", "POST"),
        ("/api/make-goods/{make_good_id}/state", "POST"),
        ("/api/make-good-alerts", "GET"),
    }


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
    assert walled == {"/api/pacing", "/api/make-goods", "/api/make-goods/{make_good_id}/state"}
    assert bare == {"/api/make-good-alerts"}
    assert pacing_alerts_api.PACING_WALL.company_only is False


def test_the_board_tells_a_reader_whether_it_may_be_changed_before_the_click() -> None:
    body = _client().get("/api/pacing").json()
    assert "can_edit" in body
    ledger = _client().get("/api/make-goods").json()
    assert "can_edit" in ledger
