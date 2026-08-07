"""P11, Bar 3: the surface half of the regression guard.

Split out of ``test_p11_regression.py`` when that file passed the 450-line size
law. The division is by what a test reads rather than by age: everything here
reads ``src/clients/pacing/**`` as text, and everything left behind reads the API,
the stores and the words.

The guards that execute the shipped module in node live in
``test_p11_surface_javascript.py``, because a defect about syntax or about
arithmetic is not one a text guard can see at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import pacing_alerts_api

SURFACE = Path("tv-break-dashboard/src/clients/pacing")
ROOT = Path(__file__).resolve().parents[1]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


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
