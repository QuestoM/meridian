"""Wave three: the owner's four reported interface defect classes stay closed."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"


def _text(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def _block(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start:css.index("}", start) + 1]


def test_folded_rules_pages_are_not_duplicate_rail_destinations() -> None:
    nav = _text("shell/nav.js")
    rail = nav.split("export const DOMAIN_DEFINITIONS = [", 1)[1].split("];", 1)[0]
    removed = nav.split("export const LEGACY_TARGETS = {", 1)[1].split("\n};", 1)[0]
    assert "'Calendar'" not in rail and "'Pricing'" not in rail
    assert "Calendar: { view: 'Governance', params: { rules: 'calendar' } }" in removed
    assert "Pricing: { view: 'Governance', params: { rules: 'rate_card' } }" in removed
    for legacy in ("Calendar", "Pricing"):
        assert f"{legacy}:" in removed
    shell = _text("shell/TVBreakDashboard.jsx")
    assert "window.history.replaceState" in shell


def test_workspace_gutter_is_logical_and_pages_do_not_double_it() -> None:
    shell = _text("shell/studio-shell.css")
    workspaces = _text("shell/styles-workspaces.css")
    workspace = _block(shell, ".workspace {")
    top_bar = _block(shell, ".top-bar {")
    top_bar_primary = _block(shell, ".top-bar-primary {")
    direct_page = _block(workspaces, ".workspace > .page-workspace")
    assert "padding-inline:" in workspace or "padding: 0 var(--space-6)" in workspace
    assert "margin-inline:" in top_bar and "padding-inline:" in top_bar_primary
    assert "padding-inline: 0" in direct_page
    for block in (workspace, top_bar, direct_page):
        assert "padding-left:" not in block and "padding-right:" not in block


def test_header_labels_never_wrap_and_duplicate_controls_yield_first() -> None:
    foundations = _text("shell/styles.css")
    shell = _text("shell/studio-shell.css")
    controls = _block(foundations, ".date-control,")
    status = _block(foundations, ".freshness,")
    assert "white-space: nowrap;" in controls
    assert "white-space: nowrap;" in status
    assert "@media (max-width: 1399px) and (min-width: 1200px)" in shell
    responsive = shell.split("@media (max-width: 1399px) and (min-width: 1200px)", 1)[1]
    assert ".top-bar .freshness" in responsive and "display: none;" in responsive
    assert ".top-bar .locale-toggle .locale-toggle-label" in responsive
    assert "clip-path: inset(50%);" in responsive


def test_timeline_breaks_keep_duration_width_and_hide_partial_text() -> None:
    css = _text("shell/styles-timeline.css")
    chip = _block(css, ".timeline-break.MuiButton-root")
    assert "min-width: 0;" in chip
    assert "container-type: inline-size;" in chip
    assert "overflow: hidden;" in chip
    narrow = css.split("@container (max-width: 55px)", 1)[1].split("}", 1)[0]
    for line in ("clock", "detail", "meta", "gold"):
        assert f".break-chip-{line}" in narrow
    assert "display: none;" in narrow
    timeline = _text("plan/week/TimelineView.jsx")
    assert "const position = breakPositionStyle(breakItem)" in timeline
    assert "exactStart / 60" in timeline
    assert "durationSeconds / 60" in timeline
    assert "programmeSpanMinutes(item)" in timeline
    assert "style={position}" in timeline
    assert "aria-label={accessibleLabel}" in timeline
    assert "title={accessibleLabel}" in timeline
    assert "aria-pressed={selected}" in timeline


def test_cross_midnight_programmes_and_breaks_keep_broadcast_day_seconds() -> None:
    from kairos_api.dashboard_api import _build_break_operations

    programmes = pd.DataFrame([{
        "Channel": "רשת 13",
        "Title": "Late show",
        "Start_datetime": "2024-11-01 23:30:00",
        "End_datetime": "2024-11-02 00:30:00",
        "Duration": 3600,
        "TVR": 4.0,
    }])
    schedule = pd.DataFrame([{
        "channel": "רשת 13",
        "date": "2024-11-01",
        "day": "Fri",
        "program_type": "Other",
        "start_time": "23:30",
        "num_breaks": 2,
        "break_length": 120,
        "predicted_revenue": 1000,
        "predicted_retention": 0.9,
        "base_rate": 100,
        "baseline_tvr": 4,
    }])

    result = _build_break_operations(programmes, schedule)
    assert result["programs"][0]["start_seconds"] == 23 * 3600 + 30 * 60
    assert result["programs"][0]["end_seconds"] == 24 * 3600 + 30 * 60
    starts = [item["start_seconds"] for item in result["breaks"]]
    assert starts == [23 * 3600 + 50 * 60, 24 * 3600 + 10 * 60]
