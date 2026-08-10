"""Wave three: the owner's four reported interface defect classes stay closed."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"


def _text(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def _block(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start:css.index("}", start) + 1]


def test_folded_rules_pages_are_not_duplicate_rail_destinations() -> None:
    nav = _text("shell/nav.js")
    rail = nav.split("export const navItems = [", 1)[1].split("];", 1)[0]
    removed = nav.split("export const removedRoutes = [", 1)[1].split("];", 1)[0]
    assert "'Calendar'" not in rail and "'Pricing'" not in rail
    assert {"Calendar", "Pricing"} <= set(re.findall(r"'([^']+)'", removed))
    router = _text("shell/workspace-router.jsx")
    for legacy in ("Calendar", "Pricing"):
        assert f"'{legacy}'" in router
        assert "replaceState" in router


def test_workspace_gutter_is_logical_and_pages_do_not_double_it() -> None:
    css = _text("shell/styles.css")
    workspace = _block(css, ".workspace {")
    top_bar = _block(css, ".top-bar {")
    direct_page = _block(css, ".workspace > .page-workspace")
    assert "padding-inline:" in workspace
    assert "margin-inline:" in top_bar and "padding-inline:" in top_bar
    assert "padding-inline: 0" in direct_page
    for block in (workspace, top_bar, direct_page):
        assert "padding-left:" not in block and "padding-right:" not in block


def test_header_labels_never_wrap_and_duplicate_controls_yield_first() -> None:
    css = _text("shell/styles.css")
    controls = _block(css, ".date-control,")
    status = _block(css, ".freshness,")
    assert "white-space: nowrap;" in controls
    assert "white-space: nowrap;" in status
    assert "@media (max-width: 1500px)" in css
    assert ".status-group > .freshness" in css
    assert ".command-group > .secondary-button" in css


def test_timeline_breaks_keep_a_legible_fixed_width_not_the_120s_width() -> None:
    css = _text("shell/styles.css")
    chip = _block(css, ".timeline-break.MuiButton-root")
    assert "min-width: 68px;" in chip
    assert "width: auto;" in chip
    timeline = _text("plan/week/TimelineView.jsx")
    assert "const { left } = positionStyle" in timeline
    assert "style={{ left }}" in timeline
    assert "style={{ left," not in timeline
    assert "aria-pressed={selected}" in timeline
