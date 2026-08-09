"""P11, Bar 3: the acts half of the surface regression guard.

Split out of ``test_p11_surface_regression.py`` when that file passed the 450-line
size law on 2026-08-09. The division is by what a test is ABOUT rather than by
where the line fell: everything here is about an ACT and what the surface says
after it, and everything left behind is about how the surface reads before one.

Both halves read ``src/clients/pacing/**`` as text. The guards that execute the
shipped module in node live in ``test_p11_surface_javascript.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import pacing_alerts_api

SURFACE = Path("tv-break-dashboard/src/clients/pacing")
ROOT = Path(__file__).resolve().parents[1]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def test_a_write_that_landed_says_so_on_this_panel_and_not_only_through_the_shell() -> None:
    """The shell swallows every notice this panel sends, so it prints its own.

    Measured in a browser: ``workspace-router.jsx`` renders the Campaigns
    destination with no ``notify`` prop and ``CampaignsPage.jsx`` holds the word
    zero times, so a successful acceptance produced no toast at all. Both files
    are outside this piece. The panel already printed its refusals for the same
    reason and said nothing when a write landed, which is the worse half: a
    refusal leaves the screen unchanged and a silent success leaves the reader
    guessing whether they wrote a record.
    """
    panel = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert 'className="pacing-notice"' in panel
    assert 'role="status"' in panel
    # All three writes announce through one function, so none of them can land
    # silently by being written a fourth way.
    assert panel.count("announce(") == 4, "one definition and one call per write"
    css = (ROOT / SURFACE / "pacing.css").read_text(encoding="utf-8")
    assert ".pacing-notice" in css


def test_the_read_only_refusal_reaches_an_english_reader_in_english() -> None:
    """The wall holds one Hebrew constant and it is a frozen wave-zero module.

    Measured: a viewer account reading this board in English met
    לחשבון צפייה אין הרשאת עריכה with every other word on the screen in English.
    The pair is published by this piece's own reads, with the Hebrew taken from
    the wall's constant rather than copied, so the sentence a Hebrew reader meets
    is unchanged and the two cannot drift.
    """
    from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL
    from kairos_api import pacing_alerts_api_words as words

    block = words.edit_refusal_block(READ_ONLY_ROLE_DETAIL)
    assert block["can_edit_reason_he"] == READ_ONLY_ROLE_DETAIL
    assert block["can_edit_reason_en"]
    assert not re.search(r"[֐-׿]", block["can_edit_reason_en"])
    # A refusal this piece holds no translation for is published in the wall's
    # own words alone rather than paraphrased.
    assert words.edit_refusal_block("something the wall never said") == {}

    panel = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "localized(board.payload, 'can_edit_reason', locale)" in panel


def test_the_headline_count_says_how_much_of_it_the_seed_wrote() -> None:
    """A count of what needs deciding reads as a morning's work.

    Measured on the shipped board: every row the board asks a decision about is
    one the demo seed wrote, because the seed sets each goal by scaling the
    observed figures over the flight, so a reader taking the headline for an
    operational figure would be reading the seed's own arithmetic back to itself.
    Nothing here is fabricated and nothing is fixable without real flights, so the
    count states its own provenance.
    """
    counts = _client().get("/api/pacing").json()["counts"]
    asking = counts["behind"] + counts["at_risk"]
    assert counts["demo_needing_a_decision"] <= asking
    assert asking > 0, "the shipped data no longer exercises this, so re-measure before trusting the guard"
    summary = (ROOT / SURFACE / "pacing-summary.js").read_text(encoding="utf-8")
    assert "demo_needing_a_decision" in summary


def test_a_form_for_an_act_the_record_no_longer_allows_leaves_the_screen() -> None:
    """A refusal keeps what the reader typed; a fresh read that closed the record does not.

    Measured in a browser. With the close form open I withdrew the record over
    the API, the submit was refused with the right sentence and the typed values
    were kept, which is correct. Taking the reload the refusal now offers
    corrected the row to Withdrawn and Closed, and the form under it went on
    offering to revoke a decision that was already revoked. The two cases differ
    by whether the record still allows the act, so that is what the form is
    gated on rather than on whether a write failed.
    """
    ledger = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "closing.state) >= 0" in ledger
    assert "record.next_states.indexOf('offered') >= 0" in ledger
