"""The docked assistant is one bounded column, and the field never leaves it.

Owner report with a screenshot, 2026-08-28: "אין כמעט מקום לראות את השיחה וגם
אם המסך נמוך אז לא רואים את הלמטה, השורת הקלדה וכפתור השליחה". Measured at a
900px window before the fix: of 774px of dock body the conversation held 239,
and with a real conversation the thread's own viewport cap pushed the composer
below the fold - so the field and the send button, the two controls the panel
exists for, were unreachable without scrolling a panel nobody expects to
scroll.

The layout law this pins: the chain from the dock down to the thread is flex
with min-height 0 at every link, the THREAD is the only scroll region, and the
composer is pinned below it. A viewport cap anywhere on that chain re-introduces
a second height authority and brings the defect back, so its absence is
asserted rather than assumed. Geometry itself was verified in a browser at 700,
900 and 1010 pixels: composer visible, send button visible, dock never clipped,
dock body never scrolling.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "kai"
LAYOUT = ROOT / "assistant-dock-layout.css"
CONSOLE = ROOT / "assistant-console.css"
PANEL = ROOT / "AssistantPanel.jsx"
COMPOSER = ROOT / "AssistantComposer.jsx"


def _layout() -> str:
    return LAYOUT.read_text(encoding="utf-8")


def test_the_thread_is_the_only_scroller_and_carries_no_viewport_cap():
    css = _layout()
    assert ".asst-in-dock .asst-thread { flex: 1;" in css
    assert "max-height: none" in css, "a vh cap on the thread is the defect itself"
    # And the dock body must have stopped being a scroller.
    assert re.search(r"\.asst-dock-body\.asst-dock-body \{[^}]*overflow: hidden", css)


def test_every_link_in_the_chain_can_actually_shrink():
    """min-height: 0 at each level, or a flex child refuses to shrink below its
    content and the column overflows exactly as it did before."""
    css = _layout()
    for selector in (".asst-in-dock {", ".asst-in-dock .asst-layout {", ".asst-in-dock .asst-chat {"):
        block = css[css.index(selector):css.index("}", css.index(selector))]
        assert "min-height: 0" in block, f"{selector} must be able to shrink"


def test_the_composer_and_its_attachments_are_pinned_not_scrolled():
    css = _layout()
    assert ".asst-in-dock .asst-composer, .asst-in-dock .asst-upload { flex: none; }" in css


def test_the_attachment_controls_ride_inside_the_composer_row():
    """They were a band of their own above the field; in a narrow dock every
    band is taken from the conversation."""
    panel = PANEL.read_text(encoding="utf-8")
    composer = COMPOSER.read_text(encoding="utf-8")
    assert "attachments={(" in panel, "the upload is handed to the composer"
    assert "<AssistantUpload" in panel
    assert "{attachments}" in composer, "and rendered inside the composer row"
    # The old standalone row must not come back.
    assert not re.search(r"/>\s*\n\s*\n\s*<AssistantComposer", panel)
    console = CONSOLE.read_text(encoding="utf-8")
    upload = console[console.index(".asst-upload {"):console.index("}", console.index(".asst-upload {"))]
    assert "padding: 0" in upload, "an inline control carries no band padding"


def test_the_keyboard_hint_costs_no_height_until_the_field_is_focused():
    """A sentence you need on your first day and never again wrapped to two
    permanent lines under the field. It is revealed on focus in the dock, and
    stays always-on for the full page where there is room."""
    css = _layout()
    assert ".asst-in-dock .asst-hint { display: none; }" in css
    assert ".asst-in-dock .asst-chat:focus-within .asst-hint { display: block; }" in css
    console = CONSOLE.read_text(encoding="utf-8")
    assert ".asst-hint {" in console, "the full-page hint keeps its own always-on rule"


def test_the_dock_repeats_neither_its_title_nor_its_subtitle_in_the_body():
    """The tab already says Conversation and the status line already says the
    rest; a heading and a sentence repeating them cost a band for nothing."""
    css = _layout()
    assert ".asst-in-dock .asst-chat > .panel-head h2" in css
    assert ".asst-dock-title small { display: none; }" in css
