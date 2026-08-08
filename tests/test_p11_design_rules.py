"""P11: the three defects the owner reported on the pacing card, held closed.

Reported 2026-08-07 against `?clients=pacing#Campaigns` with two screenshots and
recorded in `docs/ux-gauntlet/owner-reported.md`. The first is a class rather
than a site: a sweep of the whole dashboard found 26 one-sided accent bars across
17 files, 23 were replaced in `33239036`, and three were left standing because
live agents held those files at that moment. `clients/pacing/pacing-row.css` was
the one he actually photographed.

These assertions are about the rules in `docs/ux-gauntlet/design-rules.md` rather
than about a screenshot, because a rule that lives in one file and nowhere else
is the accident that produced 26 of these in the first place.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "tv-break-dashboard" / "src" / "clients" / "pacing"
RULES = ROOT / "docs" / "ux-gauntlet" / "design-rules.md"

# The one legitimate one-sided declaration, by section 1: a structural divider
# between two things, at a single pixel in the line colour. Anything thicker, or
# in a state colour, is the banned accent.
DIVIDER = re.compile(r"border-inline-(?:start|end):\s*1px solid var\(--line\);")
ONE_SIDED = re.compile(r"border-inline-(?:start|end):\s*([^;]+);")


def test_no_stylesheet_on_this_surface_draws_a_one_sided_accent_bar() -> None:
    offenders = []
    for path in sorted(SURFACE.glob("*.css")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            found = ONE_SIDED.search(line)
            if not found:
                continue
            if DIVIDER.search(line) or "transparent" in found.group(1):
                continue
            offenders.append(f"{path.name}:{number} {line.strip()}")
    assert offenders == [], offenders


def test_a_verdict_colours_the_whole_frame_and_never_one_edge_of_it() -> None:
    """Section 1 and section 2 together: a full border in the state colour."""
    text = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    for verdict, token in (("behind", "--red"), ("at_risk", "--amber"), ("on_pace", "--teal")):
        assert f".pacing-row.{verdict} {{ border-color: var({token}); }}" in text, verdict
    assert "border-inline-start-color" not in text


def test_the_facts_on_a_row_are_separated_by_a_rule_and_not_by_a_space() -> None:
    """Section 3. Three facts run together read as one broken sentence.

    Worse in Hebrew, where the eye has no capital letters to find the boundaries,
    which is the language the owner read it in.
    """
    text = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    assert ".pacing-forward > * + *" in text
    block = text.split(".pacing-forward > * + *", 1)[1].split("}", 1)[0]
    assert "border-inline-start: 1px solid var(--line);" in block
    assert "padding-inline-start" in block


def test_the_acts_on_a_row_share_one_group_one_height_and_one_primary() -> None:
    """Section 4. A filled button, an outlined one and a bare link is three weights."""
    markup = (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    styles = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    assert "pacing-row-acts" in markup
    assert "pacing-row-disclosure" in markup
    # The disclosure lives outside the act group, in its own element.
    acts = markup.split('className="pacing-row-acts"', 1)[1].split("pacing-row-disclosure", 1)[0]
    assert "pacing-days-toggle" not in acts
    assert "pacing-remedy-days" not in acts
    # One filled act, and every act the same height.
    assert ".pacing-remedy.open" not in styles
    assert "min-block-size: 28px;" in styles
    assert "flex-wrap: nowrap;" in styles.split(".pacing-row-acts", 1)[1].split("}", 1)[0]


def test_no_reader_facing_date_on_this_surface_is_built_by_hand() -> None:
    """Section 7, dates row. dd/mm/yyyy, decided in shell/dates.js and nowhere else.

    Measured in a browser on the shipped surface, and it is the owner's own
    report: the days with no source printed
    ``2025-04-28, 2025-04-29, 2025-04-30, 2025-05-01, 2025-05-02, 2025-05-03``,
    six spellings of one unbroken run in a machine format, which is the exact
    string quoted in ``shell/dates.js``'s header as the reason that module
    exists. The flight window on a row and on a ledger record and the day column
    in the drill printed raw ISO too, while the Campaigns tab of the same
    destination printed 27/04/2025-03/05/2025 for the same field.

    ``verify-date-rules.mjs`` cannot see any of it: this directory is on its
    QUARANTINED list, because it was held by a live agent when that guard was
    written. This test holds the same line from inside the piece until the
    quarantine entry is deleted, and it fails on the same four shapes.
    """
    # A payload's calendar-day field interpolated straight into a string.
    raw_iso = re.compile(r"\$\{(?:(?!format)[^}])*\b(?:starts_on|ends_on|broadcast_date|window_start|window_end)\b[^}]*\}")
    # A list of days joined by hand, which collapses no run and separates with a
    # comma a reader cannot tell from a range joiner.
    hand_join = re.compile(r"\bdays\s*\.join\s*\(")
    # Calendar parts read off a Date, the disguise that produces a fourth format.
    parts = re.compile(r"\.get(?:UTC)?(?:Date|Month|FullYear|Day)\s*\(")
    # A React key and a class name are machine-facing, exactly as the upstream
    # guard's own exception list says a dedupe key and a filename are. No reader
    # sees either, and dd/mm/yyyy in a key would be the bug.
    machine_facing = re.compile(r"\b(?:key|className)=")
    offenders = []
    for path in sorted(SURFACE.glob("*.js")) + sorted(SURFACE.glob("*.jsx")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith(("//", "*", "/*")) or machine_facing.search(line):
                continue
            for rule in (raw_iso, hand_join, parts):
                if rule.search(line):
                    offenders.append(f"{path.name}:{number} {line.strip()}")
    assert offenders == []
    # And the four sites read the home rather than re-deriving it.
    assert "from '../../shell/dates'" in (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "formatDayList(remedy.days, locale)" in (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "from '../../shell/dates'" in (SURFACE / "PacingDays.jsx").read_text(encoding="utf-8")
    assert "from '../../shell/dates'" in (SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")


def test_a_notice_names_the_record_by_its_own_kind() -> None:
    """The ledger holds two kinds and the notice named one of them for both.

    Measured in a browser: revoking a recorded risk acceptance printed
    "Make-good MG_0001 is now Withdrawn" over a row whose own chip read Risk
    acceptance, so the notice and the ledger contradicted each other about the
    same act. The kind vocabulary is what the row is labelled from, so it is
    what the notice reads too.
    """
    panel = (SURFACE / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    assert "vocabularyLabel(words.kinds, noun, 'en')" in panel
    assert "vocabularyLabel(words.kinds, noun, 'he')" in panel
    assert "Make-good ${makeGoodId} is now" not in panel


def test_a_closed_record_states_when_it_was_closed() -> None:
    """Closing is the one act here that cannot be undone and it had no time on it.

    The store has written ``closed_at`` on every closed row from the first
    round and no screen read it, so the ledger could say who closed a record
    and why and never when.
    """
    ledger = (SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "instant(record.closed_at)" in ledger


def test_the_rules_this_file_enforces_are_the_written_ones() -> None:
    """A guard that cites a rule nobody wrote down is the accident, not the fix."""
    text = RULES.read_text(encoding="utf-8")
    assert "No one-sided accent bar" in text
    assert "A row of facts needs a separator, not whitespace" in text
    assert "Buttons in one group share one size and one baseline" in text
    assert "dd/mm/yyyy" in text


def test_a_value_joined_into_prose_is_isolated_in_both_languages() -> None:
    """Direction is a property of the value, never of the sentence around it.

    The counted-basis line applied two rules to the same two values: its Hebrew
    branch isolated them and its English branch did not.

    Measured with Range rectangles on both branches rather than read off a
    screenshot, because the first reading of that screenshot was wrong. The
    channel name renders identically isolated or bare on the shipped value, so
    nothing was painting backwards; it is isolated because the next channel may
    not begin with a Hebrew letter. The instant is the one that moves: in a
    Hebrew line the bare form put 27/04/2025 at x=453 and the isolated form at
    x=401, a 52 px reorder around the comma and the full stop.
    """
    board = (SURFACE / "PacingBoard.jsx").read_text(encoding="utf-8")
    assert "const named = isolate(channel);" in board
    assert "const when = isolate(instant(asOf.instant));" in board
    assert "covers ${channel}" not in board


def test_the_instant_this_surface_prints_reads_dd_mm_yyyy_like_every_other_date() -> None:
    """Two shapes of one thing on one screen, and one of them a machine format.

    design-rules.md asks for dd/mm/yyyy in both locales. The flight window and
    the day drill already printed it through shell/dates.js while the counted
    instant printed the payload's raw yyyy-mm-dd beside them. The reshape is on
    the string: shell/dates.js reads every stamp in Asia/Jerusalem, and the
    delivery ledger's counted_as_of declares no offset, so a zone would be
    invented rather than read.
    """
    helpers = (SURFACE / "pacing-helpers.js").read_text(encoding="utf-8")
    assert "${parts[2]}/${parts[1]}/${parts[0]}" in helpers
    assert "dd/mm/yyyy" in RULES.read_text(encoding="utf-8")


def test_the_two_write_banners_stay_in_the_readers_view() -> None:
    """A shortcut fires from a row anywhere in a list of 56, and the banner is the answer.

    Measured before this: with the list scrolled 547 px and the acceptance
    shortcut pressed on a focused row, the notice recording the decision painted
    at y=-175 in a 907 px viewport, entirely above the fold. notify() is a no-op
    at the published mount, so that banner is the only confirmation the panel
    has.
    """
    css = (SURFACE / "pacing.css").read_text(encoding="utf-8")
    block = css.split(".pacing-refusal,")[1].split("}")[0]
    assert ".pacing-notice" in block
    assert "position: sticky;" in block


def test_a_reading_on_a_row_is_not_gated_on_the_write_permission() -> None:
    """A viewer could see that a campaign was at risk and not what would fix it.

    The remedy sentence names the quantity to book and the disclosure lists the
    broadcast days it applies to. Neither is an act, and both sat behind
    ``canEdit``, so a read-only account lost the diagnosis along with the
    controls. The acts below them are still gated.
    """
    row = (SURFACE / "PacingRow.jsx").read_text(encoding="utf-8")
    assert "canEdit ? <RemedySentence" not in row
    assert "canEdit ? <RemedyDays" not in row
    assert "<RemedySentence remedy={remedy} locale={locale} />" in row
    assert "canEdit ? (" in row, "the acts themselves are still behind the gate"


def test_the_name_on_a_board_row_looks_like_the_control_it_is() -> None:
    """It carried a cursor and nothing else, and a cursor is not on the screen.

    Measured: colour rgb(17,24,39), no underline, no border, no background,
    identical to the heading beside it, while the same seam in the ledger paints
    in the control colour. A reader scanning the board had no way to know the
    name opened the campaign.
    """
    styles = (SURFACE / "pacing-row.css").read_text(encoding="utf-8")
    block = styles.split(".pacing-name-open {")[1].split("}")[0]
    assert "color: var(--blue);" in block
    ledger = (SURFACE / "makegood.css").read_text(encoding="utf-8")
    assert "color: var(--blue);" in ledger.split(".makegood-campaign {")[1].split("}")[0]
