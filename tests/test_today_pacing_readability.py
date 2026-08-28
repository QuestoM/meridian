"""Regression guards for the dense Today and pacing readouts.

These checks pin the information architecture behind the reported screenshots:
time and yield figures must name their context, and a collapsed pacing record
must remain a complete, operable summary rather than a chevron-only disclosure.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TODAY = ROOT / "tv-break-dashboard" / "src" / "today"
PACING = ROOT / "tv-break-dashboard" / "src" / "clients" / "pacing"


def _block(text: str, selector: str) -> str:
    return text.rsplit(selector, 1)[1].split("}", 1)[0]


def test_transmission_clock_names_its_context_and_protects_edge_ticks() -> None:
    markup = (TODAY / "TransmissionRibbon.jsx").read_text(encoding="utf-8")
    styles = (TODAY / "transmission-ribbon.css").read_text(encoding="utf-8")

    assert "Programme times · 24-hour plan clock" in markup
    assert "שעות התוכניות · שעון תוכנית של 24 שעות" in markup
    assert 'aria-describedby="transmission-clock-context"' in markup
    stage = _block(styles, ".transmission-stage {")
    assert "padding:" in stage and "calc(var(--space-4) + var(--space-5))" in stage


def test_today_scope_and_decision_descriptions_wrap_instead_of_ellipsising() -> None:
    styles = (TODAY / "studio-ledger-today.css").read_text(encoding="utf-8")
    scope = _block(styles, ".today-answer-decisions .today-decision-scope {")
    description = _block(
        styles,
        ".today-decision-list .decision-row > div:first-of-type strong,",
    )

    for block in (scope, description):
        assert "white-space: normal;" in block
        assert "overflow-wrap: anywhere;" in block
        assert "text-overflow: ellipsis" not in block


def test_yield_rows_are_labelled_rtl_columns_with_a_named_relative_scale() -> None:
    markup = (TODAY / "YieldView.jsx").read_text(encoding="utf-8")
    styles = (TODAY / "studio-ledger-today.css").read_text(encoding="utf-8")

    assert 'className="yield-table" role="table"' in markup
    assert "Yield / ad second" in markup and "Projected revenue" in markup
    assert "Daypart" in markup and "Programme" in markup
    assert "relative scale: highest value in this group = 100%" in markup
    assert 'className="yield-relative-meter"' in markup and 'role="meter"' in markup
    assert "chart-ltr" not in markup
    columns = _block(styles, ".yield-view .yield-column-head,")
    assert "grid-template-columns:" in columns
    label = _block(styles, ".yield-view .yield-bar-label {")
    assert "white-space: normal;" in label and "overflow-wrap: anywhere;" in label


def test_collapsed_pacing_row_is_a_full_clickable_fact_summary() -> None:
    row = (PACING / "PacingRow.jsx").read_text(encoding="utf-8")
    board = (PACING / "PacingBoard.jsx").read_text(encoding="utf-8")
    workspace = (PACING / "PacingWorkspace.jsx").read_text(encoding="utf-8")
    styles = (PACING / "pacing-row-collapsed.css").read_text(encoding="utf-8")

    # The closed row carries four figures, not three, and the fourth is the
    # DENOMINATOR of the percent beside it. Measured on a first-run study: with
    # counted 2.6 next to a commitment of 20 next to 91%, a media buyer read
    # "91% of 20" and reported the campaign healthy, because the reference the
    # 91% is really against - what is due by the counted day - was named only
    # in the open row. So "Pace / risk", a label that named neither the
    # numerator nor the denominator, is gone: the percent now says what it is a
    # percent OF, and what it is of is printed beside it.
    for label in ("Counted delivery", "Due by the counted day",
                  "Flight commitment", "Of what is due"):
        assert label in row
    assert "Pace / risk" not in row, "a percent must name what it is a percent of"
    assert "amount(line.counted.through_counted_day, line.unit, locale)" in row
    assert "amount(line.reference.expected_through_counted_day, line.unit, locale)" in row
    assert "amount(line.goal, line.unit, locale)" in row
    assert '<Button type="button" className="pacing-row-hit"' in row
    assert "aria-describedby={expanded ? undefined : compactId}" in row
    assert "event.stopPropagation(); onOpenCampaign" in row
    assert "fromControl" in board and "event.key === 'Enter' && !fromControl" in board
    assert "import './pacing-row-collapsed.css';" in workspace
    assert ".pacing-row:not(.open) .pacing-name-facts" in styles
    assert "display: flex;" in styles
    assert ".pacing-row-hit.MuiButton-root" in styles and "position: absolute;" in styles
    assert "pointer-events: auto;" in styles
