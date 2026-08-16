"""P2: the laws, checked against the source of Plan, the week.

These are the four Bar 4 rules a critic runs on any surface, made mechanical for
the one tree this piece owns: the retired vocabulary is absent, the Israeli week
is Sunday-first, no file is over the cap, no token is defined outside the frozen
sheet, and no display string is hard-wrapped across source lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEEK = ROOT / "tv-break-dashboard" / "src" / "plan" / "week"
SOURCES = sorted(list(WEEK.glob("*.jsx")) + list(WEEK.glob("*.js")))
# The cap is a law about files, and a stylesheet is a file. The first version of
# this module globbed only the two script extensions, so two sheets sat 60 lines
# over the cap with every check in here green.
EVERY_FILE = sorted(SOURCES + list(WEEK.glob("*.css")))

# Section 4.8: retired from both activities, because they were the collision.
RETIRED = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _stylesheets() -> str:
    """Every sheet of the destination, which the 450-line law split by component."""
    return "\n".join(_text(path) for path in sorted(WEEK.glob("*.css")))


def _without_comments(text: str) -> str:
    """Strip line and block comments before looking for copy.

    An apostrophe inside a comment opens a false quote that a naive extractor
    then closes many lines later, swallowing code into what it believes is a
    display string. That produces a check which passes or fails on the
    punctuation of a comment, which is worse than no check at all.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def _copy(text: str) -> list[str]:
    """Every quoted run, matched inside one line, comments removed first."""
    found: list[str] = []
    for line in _without_comments(text).splitlines():
        for single, double in re.findall(r"'([^'\\]*)'|\"([^\"\\]*)\"", line):
            found.append(single or double)
    return [item for item in found if item]


def test_the_tree_has_sources_to_check():
    assert len(SOURCES) >= 15, [path.name for path in SOURCES]


@pytest.mark.parametrize("path", EVERY_FILE, ids=lambda path: path.name)
def test_no_file_in_the_tree_is_over_the_cap(path):
    lines = len(_text(path).splitlines())
    assert lines <= 450, f"{path.name} is {lines} lines"


def test_the_cap_check_actually_covers_the_stylesheets():
    """The rule that failed last time was the coverage, not the cap, so the
    coverage is asserted rather than assumed."""
    covered = {path.name for path in EVERY_FILE}
    sheets = {path.name for path in WEEK.glob("*.css")}
    assert sheets, "the destination has stylesheets"
    assert sheets <= covered, sheets - covered


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_no_retired_word_reaches_a_display_string(path):
    """The critic's own grep, narrowed to the strings a person reads.

    Prop and handler names are code, not copy, so the check is on quoted text.
    """
    # Section 4.8 is explicit that the HTTP paths do not change, so an endpoint
    # string is not copy and is exempt. Everything a person reads is not.
    copy = [item for item in _copy(_text(path)) if not item.startswith("/api/")]
    flat = " ".join(copy).lower()
    hits = [word for word in RETIRED if word.lower() in flat]
    assert hits == [], f"{path.name}: {hits}"


def test_the_week_is_presented_sunday_first_with_friday_and_saturday_as_the_weekend():
    from_model = _text(WEEK / "plan-week-model.js")
    assert "export const SUNDAY_FIRST = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];" in from_model
    assert "export const WEEKEND_DAYS = ['Fri', 'Sat'];" in from_model


def test_no_surface_in_this_tree_orders_a_week_with_the_shell_monday_first_array():
    """The shell's dayKeys is Monday-first and frozen, so this tree orders itself."""
    offenders = [
        path.name for path in SOURCES
        if "dayKeys" in _text(path)
    ]
    assert offenders == [], offenders


def test_no_token_is_defined_outside_the_frozen_sheet():
    """W0-2 froze src/tokens.css as the only place a token may be defined."""
    stylesheet = _stylesheets()
    defined = re.findall(r"^\s*(--[a-z0-9-]+)\s*:", stylesheet, flags=re.MULTILINE)
    assert defined == [], defined
    # And it does use them, or the rule would be vacuous.
    assert stylesheet.count("var(--") > 40


def test_the_stylesheet_is_written_in_logical_properties():
    """RTL correctness by construction rather than by a mirrored copy."""
    stylesheet = _stylesheets()
    physical = re.findall(
        r"^\s*(margin-left|margin-right|padding-left|padding-right|left|right|text-align:\s*(?:left|right))\s*:",
        stylesheet,
        flags=re.MULTILINE,
    )
    assert physical == [], physical
    assert "margin-inline-start" in stylesheet
    assert "text-align: start" in stylesheet


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_no_display_string_is_hard_wrapped_across_source_lines(path):
    """One display string per source line: a quoted run never ends in a backslash."""
    for number, line in enumerate(_text(path).splitlines(), start=1):
        assert not re.search(r"['\"]\s*\\$", line), f"{path.name}:{number}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_no_em_dash_no_emoji_and_no_exclamation_in_copy(path):
    text = _text(path)
    assert "—" not in text, f"{path.name} carries an em dash"
    flat = " ".join(_copy(text))
    assert "!" not in flat, f"{path.name} carries an exclamation mark in copy"
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", flat), path.name


def test_every_entrance_lands_on_the_step_it_was_named_for():
    model = _text(WEEK / "plan-week-model.js")
    for entrance, section in (
        ("Optimizer", "objective"),
        ("Schedule", "board"),
        ("Inventory", "supply"),
        ("Forecasts", "compare"),
    ):
        assert f"{entrance}: '{section}'" in model, entrance


def test_all_four_entrances_render_the_one_destination():
    for name in ("OptimizerWorkspace", "SchedulePage", "InventoryPage", "ForecastsPage"):
        text = _text(WEEK / f"{name}.jsx")
        assert "import PlanWeek from './PlanWeek';" in text, name
        assert "<PlanWeek" in text, name


def test_the_command_list_teaches_the_keyboard_it_actually_binds():
    """Linear's rule: the shortcut printed on a row is the shortcut that fires."""
    commands = _text(WEEK / "plan-week-commands.js")
    keyboard = _text(WEEK / "use-plan-keyboard.js")
    assert "shortcut: ['mod', 'k']" in commands
    assert "key === 'k'" in keyboard
    # One list feeds both, so a palette row and a key press cannot disagree.
    assert "planCommands" in _text(WEEK / "PlanWeek.jsx")
    assert "usePlanKeyboard({ commands" in _text(WEEK / "PlanWeek.jsx")


def test_a_shortcut_never_fires_while_somebody_is_typing():
    keyboard = _text(WEEK / "use-plan-keyboard.js")
    assert "isTypingTarget" in keyboard
    assert "if (typing" in keyboard


def test_the_target_is_read_from_the_store_and_never_computed_here():
    """Every target figure and the verdict come from one call, so no screen
    invents a number and no second implementation of the threshold exists."""
    strip = _text(WEEK / "GoalStrip.jsx")
    assert "progress.projected?.revenue" in strip
    assert "target.amount_ils" in strip
    assert "verdict.variance_ils" in strip
    # The three-state word is a lookup on the server's own state, never a
    # comparison written in the browser.
    assert "STATE_WORDS" in strip
    for banned in ("amount", "variance", "at_risk_band"):
        assert f"{banned} =" not in strip.replace("const amount", ""), banned


def test_an_unset_target_is_an_honest_empty_state_with_a_control():
    """Owner decision 3 is unanswerable, so an unset window prints no figure and
    offers the route that supplies one, rather than a sentence."""
    strip = _text(WEEK / "GoalStrip.jsx")
    assert "No target is set for this window" in strip
    assert "לא נקבע יעד לחלון הזה" in strip
    assert "Set a target on Today" in strip
    assert "window.location.hash = 'Overview'" in strip
    # The question is answered in exactly one place on the destination.
    assert "Is this week on plan" not in _text(WEEK / "RunPanel.jsx")


def test_an_unscoped_payload_is_refused_rather_than_captioned():
    """The competitor boundary is not satisfied by a caption.

    Measured live on 2026-08-01, when another piece's partial settings write
    cleared the operator channel: the route could not scope the payload, the
    board drew three rival channels by name with their programme titles and
    their revenue, and the strip printed 54,650,165.39, the whole four-channel
    market, under the operator's own revenue label. Both now decline, name the
    input that is missing and offer the door that supplies it.
    """
    board = _text(WEEK / "BoardPanel.jsx")
    assert "const unscoped = schedule?.scope?.plan?.scoped === false;" in board
    assert "{unscoped && (" in board
    assert "{!unscoped && (" in board
    assert "window.location.hash = 'Settings'" in board
    assert "לא הוגדר ערוץ מפעיל" in board

    strip = _text(WEEK / "GoalStrip.jsx")
    assert "const scoped = Boolean(channel);" in strip
    assert "{scoped ? formatCurrency(progress.projected?.revenue, locale) : EMPTY_VALUE}" in strip
    assert "window.location.hash = 'Settings'" in strip


def test_the_board_opens_on_the_real_day_workbench_and_keeps_analysis_secondary():
    """The editable day is the board's operating path, not a fourth chart.

    The week grid used to be the default and the full day editor was hidden one
    view-switch away.  The completed workbench migration intentionally reverses
    that hierarchy: the existing DayBoard is the default, while the grid,
    daypart and source-timeline views are explicitly read-only analysis.  This
    assertion follows the component seam instead of pinning presentation copy.
    """
    week = _text(WEEK / "PlanWeek.jsx")
    board = _text(WEEK / "BoardPanel.jsx")
    workbench = _text(WEEK / "PlanBoardWorkbench.jsx")

    assert "const [boardView, setBoardView] = useState('day');" in week
    assert "const VIEWS = ['day', 'grid', 'strip', 'timeline'];" in board
    assert "{view === 'day' && (" in board and "<PlanBoardWorkbench" in board
    assert "{view !== 'day' && (" in board and "plan-analysis-label" in board

    # The workbench composes the already-safe daily editor and the real version
    # rail.  It must not create a decorative parallel plan or a synthetic
    # checkpoint merely to fill the new layout.
    assert "<DayBoard" in workbench and "day={day}" in workbench
    assert "<PlanVersionRail" in workbench and "versions={versions}" in workbench
    assert "morning" not in workbench.lower()


def test_the_published_threshold_is_printed_beside_the_state():
    """Google Ads' device: the rule that decided the state is on the strip."""
    strip = _text(WEEK / "GoalStrip.jsx")
    assert "verdict.threshold_he" in strip
    assert "verdict.threshold_en" in strip


def test_the_remedy_sits_in_the_same_strip_as_the_diagnosis():
    strip = _text(WEEK / "GoalStrip.jsx")
    for state, control in (
        ("behind", "onGo('compare')"),
        ("on_plan", "onGo('publish')"),
        ("no_projection", "onGo('run')"),
    ):
        assert state in strip, state
        assert control in strip, control


def test_the_worth_of_a_second_is_read_not_recomputed():
    """JS-13's figure is computed once by the piece that owns the rate card."""
    api = _text(WEEK / "plan-week-api.js")
    assert "/api/yield-per-second" in api
    block = _text(WEEK / "YieldBlock.jsx")
    assert "totals.yield_per_second" in block
    # The basis travels with the figure and is printed, not hidden in a tooltip.
    assert "basisFormula(basis)" in block
    assert "retention_cost_basis" in block
    # No arithmetic on the money in the browser.
    assert "/ " not in block.replace("</", "").replace("//", "")


def test_the_basis_reaches_the_operator_in_the_operator_words():
    """Measured on the Hebrew Supply step before this was fixed: seven engine
    identifiers and one untranslated English paragraph on screen, because the
    server states its basis in field names and the surface printed the string.

    The words are now the surface's and the engine's own line is one disclosure
    away, so nothing is hidden and nothing is unreadable.
    """
    block = _text(WEEK / "YieldBlock.jsx")
    basis = _text(WEEK / "plan-week-basis.js")
    engine_names = (
        "risk_lambda",
        "ci_low",
        "ci_high",
        "retention_cost_ils",
        "base_rate",
        "baseline_tvr",
        "unit_seconds",
        "retention_share",
        "ad_seconds",
    )
    # Not in a rendered string on the block. A table column's `key` is the
    # payload field it reads, which is code and never reaches a label, so it is
    # removed before the copy is extracted. The one other place these names may
    # appear is the constant the words are gated on, which is compared and never
    # printed.
    rendered = " ".join(_copy(re.sub(r"key:\s*'[a-z_]+'", " ", block)))
    for name in engine_names:
        assert name not in rendered, name
    gate = basis.split("const KNOWN_FORMULA = ")[1].splitlines()[0]
    for name in ("retention_cost_ils", "base_rate", "baseline_tvr", "unit_seconds"):
        assert name in gate, name

    # The words render only when the formula is the one they describe, which is
    # what stops a translation from outliving the thing it translated.
    assert "if (basisFormula(basis) !== KNOWN_FORMULA) return null;" in basis
    assert "export function unfamiliarBasisWords(locale)" in basis
    assert "עלות השימור של ברייק" in basis
    assert "The retention cost of a break" in basis
    # The band claims no interval width, because the width is the model's.
    assert "95" not in _copy_text(basis)


def test_the_band_reads_in_the_language_of_the_page():
    """A hard-coded English join word printed 2.18M to 6.03M on a Hebrew page
    while every other string in the file went through pageText."""
    block = _text(WEEK / "YieldBlock.jsx")
    assert "{pageText(locale, ' to ', ' עד ')}" in block
    assert "{pageText(locale, 'band ', 'טווח ')}" in block
    # Each figure carries its own direction; the sentence around it does not.
    assert 'dir="ltr" className="numeric">\n              {formatCurrency(data.retention_cost_low' not in block


def _copy_text(text: str) -> str:
    return " ".join(_copy(text))


def test_the_net_after_retention_cost_is_rendered_from_the_payload_not_from_a_constant():
    compare = _text(WEEK / "ComparePanel.jsx")
    assert "summary.revenue_net" in compare
    assert "payload.delta?.revenue_net" in compare
    # The string the old panel printed instead of the figure is gone from the
    # copy, which is where it was rendered from.
    rendered = " ".join(_copy(compare))
    assert "Not exposed" not in rendered
    assert "לא נחשף" not in rendered


def test_the_palette_groups_are_contiguous_so_no_group_appears_twice():
    """Measured in a browser before this was fixed: the actions group rendered
    twice because the palette's own row sat after the zooms, and the list is
    grouped by contiguous runs."""
    commands = _text(WEEK / "plan-week-commands.js")
    order = commands.split("return [")[1].split("]")[0]
    assert order.replace(" ", "") == "...actions,...navigation,...questions,...zooms"
    assert "actions.push({" in commands
    assert "id: 'palette'" in commands


def test_an_open_version_keeps_its_place_in_the_set():
    """Linear's 1 / 31 with two arrows, on the one set this destination has."""
    panel = _text(WEEK / "PublishPanel.jsx")
    assert "selectedIndex + 1} / {versions.length}" in panel
    assert "walkTo(selectedIndex - 1)" in panel
    assert "walkTo(selectedIndex + 1)" in panel
    # Walking selects and opens in one move rather than moving a highlight.
    assert "onSelect(next.version_id);" in panel
    assert "onDiff(next.version_id);" in panel


def test_a_collapsed_plan_needs_a_separate_deliberate_act_before_freeze():
    panel = _text(WEEK / "PublishPanel.jsx")
    model = _text(WEEK / "plan-week-model.js")
    api = _text(WEEK / "plan-week-api.js")

    assert "collapseWarning(live)" in panel
    assert "setCollapseConfirmed(true)" in panel
    assert "collapse.collapsed && !collapseConfirmed" in panel
    assert "onPublish(collapseConfirmed)" in panel
    assert "live?.collapse" in model
    assert "confirm_collapse: Boolean(confirmCollapse)" in api


def test_a_channel_name_inside_a_sentence_carries_its_own_direction():
    """RTL law: a Hebrew name inside an English sentence reorders the
    punctuation around it unless it is isolated."""
    assert "<bdi>{channel}</bdi>" in _text(WEEK / "GoalStrip.jsx")
    assert "<bdi>{scopeChannel}</bdi>" in _text(WEEK / "SupplyPanel.jsx")
    # Both places the comparison names the channel: the finished scope line and
    # the line that says what is running while the week is still arriving.
    compare = _text(WEEK / "ComparePanel.jsx")
    assert "<bdi>{scope.channel}</bdi>" in compare
    assert "<bdi>{channel}</bdi>" in compare
    model = _text(WEEK / "plan-week-model.js")
    # isolate moved to src/shell/bidi.jsx, the single home for isolation, so this
    # surface imports it instead of defining its own. That is the better answer
    # and this assertion follows it: the law was never "defined here", it was
    # "one definition", and a second copy is how two of them drift.
    assert "from '../../shell/bidi'" in model, (
        "plan-week-model.js no longer reads isolate from the primitive"
    )
    assert "export function isolate(value)" not in model, (
        "plan-week-model.js defines its own isolate again, so there are two"
    )
    assert "isolate(note.scope_channel)" in model


def test_the_worth_of_a_second_is_printed_at_the_grain_it_is_asked_at():
    """The shell formatter compacts to whole shekels, which turns 142.7044 into
    143 and hides the difference between two rate cards."""
    block = _text(WEEK / "YieldBlock.jsx")
    assert "minimumFractionDigits: 2" in block
    assert "perSecond(totals.yield_per_second, locale)" in block
    assert "perSecond(row.yield_per_second, locale)" in block


def test_the_one_proven_dead_component_is_gone():
    """Section 3.5 of the specification names it as the single deletion, and its
    proof was the fourteen lines themselves: a hard-coded empty state with no
    data path and, after the merge, no call site."""
    assert not (WEEK / "InventoryHeatmap.jsx").exists()
    for path in SOURCES:
        assert "InventoryHeatmap" not in _text(path), path.name


def test_the_blended_score_is_printed_at_the_grain_the_comparison_is_made_at():
    """The shell formatter keeps one decimal, which prints 0.5405 and 0.4611 as
    the same 0.5, so two scenarios that differ only in the score would read as
    identical on the surface whose whole job is to separate them."""
    compare = _text(WEEK / "ComparePanel.jsx")
    assert "function blendedScore(value, locale)" in compare
    assert "minimumFractionDigits: 3" in compare
    assert "blendedScore(summary.objective, locale)" in compare
