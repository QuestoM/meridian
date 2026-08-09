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
import subprocess
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
    # A sheet is reached either by a component that imports it or by another
    # sheet that does. The second path exists because pacing-row.css passed the
    # 450 line cap and was split, and the half that moved is loaded by the half
    # it came out of rather than by the component, since adding one import line
    # to PacingWorkspace.jsx would have pushed THAT file over the same cap.
    #
    # What this test is actually about is reachability, so it asks about
    # reachability rather than about one mechanism. A sheet reached by neither is
    # still a sheet the browser never loads.
    imported = set()
    for path in sorted(SURFACE.glob("*.jsx")):
        text = (ROOT / path).read_text(encoding="utf-8")
        imported.update(re.findall(r"import '\./([a-z-]+\.css)'", text))
    for path in sorted(SURFACE.glob("*.css")):
        text = (ROOT / path).read_text(encoding="utf-8")
        imported.update(re.findall(r"@import '\./([a-z-]+\.css)'", text))
    unreachable = {path.name for path in SURFACE.glob("*.css")} - imported
    assert unreachable == set(), f"{sorted(unreachable)} ships and nothing loads it"


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

    The two counting sentences moved to ``pacing-summary.js`` when the panel
    reached the size law, and they moved as whole functions with their prose
    unchanged. This reads the file that now holds them; the sentences themselves
    are executed against the shipped board in
    ``test_p11_surface_javascript.py::test_the_two_counting_sentences_count_the_board_they_are_about``,
    which is a stronger guard than either grep.
    """
    text = (ROOT / SURFACE / "pacing-summary.js").read_text(encoding="utf-8")
    assert "counts.demo" in text
    panel = (ROOT / SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "seededSentence(board, locale)" in panel
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
    """One product, one isolate. A left-to-right one lays a Hebrew name out backwards.

    The guard named the surface and read one file of it, so four hand-typed
    left-to-right isolates lived in ``PacingWorkspace.jsx`` under an assertion
    that said they could not. Measured in a browser: the Hebrew acceptance notice
    wrapped a Hebrew campaign name in U+2066, which is the exact layout the row
    heading beside it had already been fixed for. It now reads every file on the
    surface, which is the class the sentence always claimed.
    """
    helpers = (ROOT / SURFACE / "pacing-helpers.js").read_text(encoding="utf-8")
    shell = (ROOT / "tv-break-dashboard/src/shell/bidi.jsx").read_text(encoding="utf-8")
    assert "\u2068" in helpers and "\u2069" in helpers
    offenders = []
    for path in sorted(list(SURFACE.glob("*.js")) + list(SURFACE.glob("*.jsx"))):
        text = (ROOT / path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "\u2066" in line or "\u2067" in line:
                offenders.append(f"{path}:{number} {line.strip()}")
    assert offenders == [], "the directional isolates have no caller on this surface"
    # bidi.jsx writes the pair as escapes, on purpose: the characters render as
    # nothing, so a literal pair in the source is invisible to review.
    assert "u2068" in shell and "u2069" in shell


def test_a_counted_figure_states_how_much_of_it_has_not_aired_yet() -> None:
    """There is no delivery feed, so aired against scheduled is all this board has.

    The delivery ledger splits every day into aired, scheduled and unknown, and
    every goal line on the payload carries ``counted.delivered`` and
    ``counted.booked_not_aired`` separately. The row printed only their sum.
    Measured on the shipped board: of the 51 rows that carry a goal, 18 count
    spots that have not aired and on 7 of them nothing has aired at all, five of
    those reading at risk. On those five the board said at risk about a campaign
    that had aired nothing and the only way to learn it was to open the drill and
    read a state column.
    """
    row = (ROOT / SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "counted.booked_not_aired" in row, "the row never reads the half that has not aired"
    assert "pacing-not-aired" in row
    css = (ROOT / SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    assert ".pacing-not-aired" in css

    payload = _client().get("/api/pacing").json()
    split = 0
    for board_row in payload["rows"]:
        line = board_row.get("rating") if (board_row.get("rating") or {}).get("goal") is not None else board_row.get("money")
        if line and line.get("goal") is not None and line["counted"]["booked_not_aired"] > 0:
            split += 1
    assert split > 0, "the shipped data no longer exercises this, so re-measure before trusting the guard"


def test_the_day_drill_closes_on_the_figure_the_row_states() -> None:
    """Zero derivation by the reader. Seven day rows and no total is arithmetic.

    The total is the server's own ``through_counted_day`` and is never summed in
    the browser, so the drill cannot disagree with the row it was opened from.
    """
    days = (ROOT / SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    assert "tfoot" in days
    assert "line.counted.through_counted_day" in days
    row = (ROOT / SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "line={line}" in row, "the drill cannot state the total it was not handed"


def test_the_drill_says_when_a_rule_left_spots_out_of_a_day() -> None:
    """A money figure of nought with nothing beside it is a figure nobody can check.

    ``data/campaign_delivery.csv`` carries ``spots_dropped_by_rule`` on every day
    row and no screen in this product read it. Measured: 32 of the 62 sourced day
    rows carry one, and all three days that price at zero are among them, so the
    drill printed ILS 0 next to a real rating figure and said nothing about why.
    The rule is named by its engine key only, so the count is stated and the key
    is not.
    """
    days = (ROOT / SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    assert "spots_dropped_by_rule" in days
    # The id may be READ as a lookup key and must never be RENDERED. Asserting
    # the string is absent from the source confused those two: the drill has to
    # read the key to find the rule's own sentence, and it is the sentence that
    # reaches the screen. So this asserts what a reader sees.
    assert "{line.rule}" in days, "the drill no longer renders the rule's own sentence"
    # key={line.id} is a React key and never reaches a reader, so the check is
    # about the id in a TEXT position rather than about the string appearing.
    rendered_id = re.search(r">\s*\{\s*line\.id\s*\}|\{\s*line\.id\s*\}\s*<", days)
    assert rendered_id is None, (
        "the engine key reaches the screen. A reader who cannot act on "
        "DEFAULT_ONE_PER_BREAK is not helped by being shown it."
    )
    # And the third state, which the code's own comment promised and its filter
    # removed: a cause the rule file cannot name is COUNTED and said, not dropped.
    assert "named: false" in days, (
        "a drop whose rule the file does not carry is filtered out of the drill, so an "
        "unnameable cause reads as no cause at all"
    )
    css = (ROOT / SURFACE / "pacing-days.css").read_text(encoding="utf-8")
    assert ".pacing-day-dropped" in css

    payload = _client().get("/api/pacing/CMP_D030/days").json()
    carried = [day for day in payload["days"] if (day.get("spots_dropped_by_rule") or 0) > 0]
    assert carried, "the shipped ledger no longer exercises this, so re-measure before trusting the guard"


def test_no_guard_on_this_surface_stands_on_a_file_the_repository_does_not_have() -> None:
    """A guard that stands on an untracked file is not a guard at HEAD.

    Measured, and the finding that failed a previous round. ``verify-parses.mjs``
    sat beside the surface and was never committed, while the test that shelled
    out to it by path was, so the committed tree carried a guard that could not
    run and a component that could not compile. The driver now lives inside
    ``test_p11_surface_javascript.py`` and the script is gone.

    Two assertions, and between them they are the defect. Every file of this
    surface that a ``tests/test_p11_*`` guard names by path exists on disk, which
    is how that defect showed at HEAD: the test named a script that was not
    there. And no file of this surface is an executable script and no guard
    mentions one, because a driver that runs from beside the surface is only as
    tracked as its weakest half, and the driver belongs in the guard that runs it.

    The wider sentence a previous round asserted, that nothing under
    ``src/clients/pacing`` may exist only in a working tree, cannot be made to
    pass by the round that adds a file to the surface: a builder may not commit,
    so a new component is untracked on the day it is written and that assertion
    fails for the one reason that is nobody's defect. Which files are waiting for
    a commit is recorded as a blocker in the state file instead, which is where a
    thing that needs a commit belongs.
    """
    on_disk = {path.name for path in (ROOT / SURFACE).glob("*") if path.is_file()}
    named = re.compile(r"SURFACE\s*/\s*\"([^\"]+)\"")
    seen = 0
    for guard in sorted((ROOT / "tests").glob("test_p11_*.py")):
        text = guard.read_text(encoding="utf-8")
        for name in named.findall(text):
            seen += 1
            assert name in on_disk, f"{guard.name} names {name}, which is not on this surface"
    assert seen > 5, "no guard names a file of this surface, which is a mis-invocation"
    # And there is no script beside the surface for a guard to shell out to. The
    # only reference any of them makes to a .mjs is to the repo's own date and
    # direction sweeps, which live in the frontend package and are run by npm.
    assert [path.name for path in (ROOT / SURFACE).glob("*.mjs")] == []


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
