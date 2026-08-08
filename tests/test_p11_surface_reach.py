"""P11: what a reader can reach from a row, and what the board carried and hid.

The fourth guard file on this surface, under the reserved ``tests/test_p11_*``
prefix and split from the others by subject rather than by size. Every assertion
below is about the same class of defect: the payload already held the thing, the
board already computed it, and there was no way for the person in front of the
screen to get to it.

Six of them, each measured before it was closed.

* The row printed one of the campaign's two goal lines. 48 of the 56 shipped rows
  carry both, and on 10 of those the two verdicts disagree.
* A control that named a record opened the ledger and dropped the record.
* A row whose pace could not be stated named the path forward and offered no
  control that took it.
* The figure was inert text and the only way into the days behind it was a
  separately labelled button.
* A decision written by one keystroke offered no way back.
* The keyboard was the fastest path on the board and nothing on screen said the
  list had to be focused first.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store, pacing_alerts_api

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "tv-break-dashboard" / "src" / "clients" / "pacing"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def _headline_and_other(row: dict) -> tuple[dict | None, dict | None]:
    """The line the row leads with and the one beside it, the surface's own rule."""
    rating = row.get("rating") or {}
    money = row.get("money") or {}
    if rating.get("goal") is not None:
        return rating, (money if money.get("goal") is not None else None)
    if money.get("goal") is not None:
        return money, None
    return None, None


def test_the_row_states_the_campaigns_other_goal_and_not_only_the_one_it_leads_with() -> None:
    """A campaign here is booked against two goals and the row printed one.

    Measured on the shipped board rather than argued: 48 of the 56 rows carry
    both a rating goal and a money goal, and on 10 of those 48 the two verdicts
    disagree. Every one of the 10 reads at risk on rating and on pace on money,
    CMP_D040 at 0.88 against 0.9989, which is the difference between a campaign
    that is spending to plan and under-delivering audience and one that is behind
    on both. The second line existed on the payload from the first round and
    reached the screen only as a bare pair under the day drill, with no verdict,
    no reference and no ratio.
    """
    body = _client().get("/api/pacing").json()
    both = 0
    divergent = 0
    for row in body["rows"]:
        headline, other = _headline_and_other(row)
        if headline is None or other is None:
            continue
        both += 1
        if headline["pace"]["verdict"] != other["pace"]["verdict"]:
            divergent += 1
    assert both > 0, "the shipped data no longer carries two goal lines, so re-measure"
    assert divergent > 0, "the shipped data no longer diverges, so re-measure before trusting this"

    # One component states it, on the row and again under the drill, so this
    # product cannot come to hold two statements of one goal.
    line = (SURFACE / "PacingGoalLine.jsx").read_text(encoding="utf-8")
    for field in ("counted.through_counted_day", "expected_through_counted_day", "pace.verdict"):
        assert field in line, field
    row_text = (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "<PacingGoalLine line={second}" in row_text
    days = (SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    assert "<PacingGoalLine line={second}" in days
    css = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    assert ".pacing-goal-line > * + *" in css, "four facts on one line take a rule between them"


def test_a_control_that_names_a_record_opens_that_record() -> None:
    """Both seams passed an id into a handler that dropped it.

    ``Remedy`` and ``Acceptance`` both call ``onOpenMakeGood`` with a make-good
    id and the panel answered ``() => setView(LEDGER)``, so "Risk taken on, open
    the record" and "Open make-good MG_0001" landed on an unscrolled, unmarked
    ledger. The opposite direction was already exact: a name in the ledger
    focuses its own board row through ``focusCampaignId``.
    """
    panel = (SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "onOpenMakeGood={openMakeGood}" in panel
    assert "onOpenMakeGood={() => setView(LEDGER)}" not in panel
    assert "focusMakeGoodId={focusMakeGood}" in panel
    # The refusal that names a record goes to the same place, rather than to the
    # top of the ledger the record is somewhere in.
    assert "openMakeGood(opens.id)" in panel

    ledger_surface = (SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "focusMakeGoodId" in ledger_surface
    assert "makegood-focused" in ledger_surface
    assert "scrollIntoView" in ledger_surface
    assert ".makegood-focused" in (SURFACE / "makegood.css").read_text(encoding="utf-8")


def test_a_row_that_cannot_be_paced_offers_the_control_its_path_forward_names() -> None:
    """Five rows read "Open the campaign and set a goal on its flight" and offered no door.

    Measured on the shipped board: all five unknown rows carry the code
    ``no_goal``, whose path forward names the campaign, and the remedy for a
    supply block returned null. Every at-risk row got a named remedy control and
    these got none.

    ``unmeasurable`` deliberately still gets none: its path forward is to supply
    a panel breakdown for an audience, and no screen in this product does that.
    """
    body = _client().get("/api/pacing").json()
    unknown = [row for row in body["rows"] if row["headline"]["verdict"] == "unknown"]
    assert unknown, "the shipped data no longer exercises this, so re-measure"
    codes = {str(row["headline"].get("code") or "") for row in unknown}
    assert codes <= {"no_goal", "no_flight_dates", "no_source", "not_started", "unmeasurable"}, codes

    acts = (SURFACE / "PacingActs.jsx").read_text(encoding="utf-8")
    assert "remedy.kind === 'supply'" in acts
    assert "OPENS_THE_CAMPAIGN" in acts and "no_goal" in acts
    assert "unmeasurable" not in acts.split("const OPENS_THE_UPLOAD", 1)[1].split("\n\n", 1)[0]
    row_text = (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "onOpenCampaign={onOpenCampaign ? () => onOpenCampaign(row.campaign_id) : null}" in row_text


def test_the_figure_is_the_way_into_the_days_it_was_summed_from() -> None:
    """Stripe's transferable mechanic is that the amount is the link to the rows behind it.

    This board reached its drill only from a separately labelled button below the
    row, while ``4.4 of 35 rating points`` sat inert above it. The labelled
    control stays, because a control discoverable only by hovering is not
    discoverable, and the accessible name of the figure carries the act as well
    as the figure so the two read as one control to a screen reader.
    """
    row_text = (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "pacing-figure-open" in row_text
    assert "aria-expanded={expanded} aria-label={opens}" in row_text
    # A row the delivery ledger holds no day for offers no drill and no control.
    assert "{days && onToggle ? (" in row_text
    assert ".pacing-figure-open" in (SURFACE / "pacing-row.css").read_text(encoding="utf-8")


def test_the_decision_one_keystroke_writes_can_be_taken_back_from_the_banner() -> None:
    """Pressing a on a focused row recorded MG_0001 with no confirmation and no undo.

    The reference is Google Ads, which writes every applied recommendation to its
    change history and can undo it there. The record could always be revoked, but
    only by finding it in the ledger.

    The undo is not a fourth act. It is the withdrawal the ledger already holds,
    carrying the one published reason that says the record should not have been
    opened, and the surface holds no copy of either: both ride the answer to the
    write.
    """
    undo = makegood_store.undo_block()
    assert makegood_store.reason_allowed(undo["state"], undo["reason"]), "the store would refuse its own undo"
    assert undo["state"] in makegood_store.REASON_REQUIRED
    for entry_state in makegood_store.ENTRY_STATE.values():
        assert undo["state"] in makegood_store.TRANSITIONS[entry_state], entry_state
    assert not re.search(r"[֐-׿]", undo["label_en"])
    assert re.search(r"[֐-׿]", undo["label_he"])

    panel = (SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "pending.undo.state" in panel and "pending.undo.reason" in panel
    assert "answer.undo" in panel
    # The surface never spells the transition or the reason itself.
    assert "'withdrawn'" not in panel and "opened_in_error" not in panel
    assert ".pacing-undo" in (SURFACE / "pacing.css").read_text(encoding="utf-8")


def test_the_keyboard_legend_names_what_every_one_of_its_keys_needs_first() -> None:
    """Measured: pressing j with focus on the body did nothing at all.

    Only after focusing ``.pacing-list`` did the marker move from row 0 to row 1.
    The legend read as a claim that the keys worked wherever the reader was
    standing. The list already carried a focus ring; it now says so, and points
    at the legend so a reader who never sees it is told.
    """
    board = (SURFACE / "PacingBoard.jsx").read_text(encoding="utf-8")
    assert "with this list focused, j and k step" in board
    assert 'id="pacing-keys"' in board
    assert 'aria-describedby="pacing-keys"' in board
    assert ".pacing-list:focus-visible" in (SURFACE / "pacing.css").read_text(encoding="utf-8")


def test_the_row_a_reader_left_this_board_by_is_the_row_they_come_back_to() -> None:
    """Leaving by a campaign name unmounts this panel and its state goes with it.

    Measured: clicking the name on row 0 switched the destination tab from pacing
    to campaigns, and returning cost a click on the Pacing tab and a rescroll
    through 56 rows. The outward trip was built a round ago and the return trip
    was not.
    """
    place = (SURFACE / "pacing-place.js").read_text(encoding="utf-8")
    assert "sessionStorage" in place
    # Storage is not always there, and a place marker is never worth a blank
    # screen, so every access is guarded.
    assert place.count("catch (error)") >= 3
    # Reading it clears it, or a reader is dragged back to one row all sitting.
    assert "removeItem" in place

    panel = (SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "rememberCampaign(id)" in panel
    assert "takeRememberedCampaign()" in panel
    assert "setFocusCampaign(remembered)" in panel


def test_the_drill_agrees_with_its_own_number_when_one_spot_was_left_out() -> None:
    """The majority case of this sentence read as if it were about several spots.

    Measured by the round-three critic on ``data/campaign_delivery.csv``: 32 day
    rows carry a dropped count and 21 of them carry exactly 1, so the sentence a
    reader met most often was ``1 מתוכם הושמטו``, a plural verb on a single
    spot, and ``1 of them left out by a booking rule``, which is not a sentence.
    The numeral is isolated in the Hebrew for the reason every other numeral on
    this surface is: it opens a right-to-left run and its direction is its own.
    """
    days = (SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    assert "function droppedSentence" in days
    # One clause, chosen by the count, and no template that agrees with nothing.
    assert "אחד מתוכם הושמט בגלל כלל הזמנה" in days
    assert "of them was left out by a booking rule" in days
    assert "of them were left out by a booking rule" in days
    assert "מתוכם הושמטו בגלל כלל הזמנה" in days
    assert "of them left out by a booking rule" not in days
    assert "isolate(count)" in days
    # And the ledger still exercises the singular, or the guard is about nothing.
    payload = _client().get("/api/pacing/CMP_D001/days").json()
    singles = [day for day in payload["days"] if (day.get("spots_dropped_by_rule") or 0) == 1]
    assert singles, "no day on this campaign carries exactly one dropped spot any more"


def test_the_row_the_keyboard_is_on_is_a_fact_and_not_only_a_colour() -> None:
    """j and k moved a ring and told assistive technology nothing at all.

    Measured by the round-three critic in a browser: after the place marker sent
    the reader back to row 13 of 56, the marked row carried ``aria-current``
    null, ``id`` null and ``tabindex`` null, and the list it sits in carries
    ``role="list"`` with no ``aria-activedescendant``. The only signal that the
    keyboard was on that row was the 2 px ring ``pacing-row.css`` draws.

    ``aria-current`` is valid on a listitem and moves no focus, so the list keeps
    the key handler it owns. A roving tabindex would announce the move as it
    happens and is left open in the state file, because it changes which element
    the keydown starts from.
    """
    board = (SURFACE / "PacingBoard.jsx").read_text(encoding="utf-8")
    assert "aria-current={index === focused ? 'true' : undefined}" in board
    # The visible ring stays exactly as it was, so nothing about this is a swap.
    sheet = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    assert ".pacing-focused .pacing-row" in sheet
    assert "--focus-ring" in sheet
