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


def test_the_rules_this_file_enforces_are_the_written_ones() -> None:
    """A guard that cites a rule nobody wrote down is the accident, not the fix."""
    text = RULES.read_text(encoding="utf-8")
    assert "No one-sided accent bar" in text
    assert "A row of facts needs a separator, not whitespace" in text
    assert "Buttons in one group share one size and one baseline" in text
