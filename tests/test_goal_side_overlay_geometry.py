"""The dashboard's side surfaces start below the persistent operator header."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src"


def _rule(path: str, selector: str) -> str:
    source = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", source, re.S)
    assert match, f"{path} no longer publishes {selector}"
    return match.group("body")


def test_the_shell_header_height_is_one_named_layout_token():
    tokens = (ROOT / "tokens.css").read_text(encoding="utf-8")
    assert "--shell-header-height: 57px;" in tokens


def test_every_custom_side_surface_starts_below_that_header():
    rules = {
        "kai/assistant-console.css": _rule("kai/assistant-console.css", ".asst-dock"),
        "history/activity-feed.css": _rule("history/activity-feed.css", ".activity-feed"),
        "sources/sources-tables.css": _rule("sources/sources-tables.css", ".rows-drawer"),
        "plan/day/schedule-inspector.css": _rule("plan/day/schedule-inspector.css", ".schedule-inspector"),
        "plan/break/break-inspector.css": _rule("plan/break/break-inspector.css", ".break-inspector"),
    }
    for path, body in rules.items():
        assert "var(--shell-header-height)" in body, path
        assert "height: 100vh" not in body, path
        assert not re.search(r"(?:^|;)\s*top:\s*0\b", body), path


def test_custom_side_surfaces_are_bounded_to_the_remaining_viewport():
    flush = {
        "history/activity-feed.css": _rule("history/activity-feed.css", ".activity-feed"),
        "sources/sources-tables.css": _rule("sources/sources-tables.css", ".rows-drawer"),
        "plan/day/schedule-inspector.css": _rule("plan/day/schedule-inspector.css", ".schedule-inspector"),
    }
    for path, body in flush.items():
        assert "inset-block: var(--shell-header-height) 0;" in body, path

    floating = _rule("plan/break/break-inspector.css", ".break-inspector")
    assert "inset-block: calc(var(--shell-header-height) + var(--space-4)) var(--space-4);" in floating


def test_dialogs_share_the_header_boundary_and_stay_viewport_clamped():
    """Modal dialogs are the same citizens as the side surfaces: the operator
    header stays visible above every overlay, and every frame width is clamped
    to the viewport so a large dialog on a small screen keeps its margins."""
    body = _rule("studio/studio-workspaces.css", ".studio-dialog")
    assert "inset-block: var(--shell-header-height) 0;" in body

    studio = (ROOT / "studio/studio-workspaces.css").read_text(encoding="utf-8")
    frames = re.findall(
        r"\.studio-dialog[^{]*\.studio-modal__frame\s*\{[^}]*inline-size:\s*([^;]+);",
        studio,
    )
    assert frames, "the dialog frame no longer declares its widths"
    for width in frames:
        assert width.strip().startswith("min(100%"), width


def test_popup_shells_come_from_the_modal_primitives():
    """One overlay system: a popup's shell is the studio Dialog or Sheet, never
    a hand-rolled aria-modal aside with its own geometry. Docked side panels
    (MUI drawers, the record rail) are side surfaces, covered above."""
    for path in ROOT.rglob("*.jsx"):
        if path.name == "modal-primitives.jsx":
            continue
        assert 'aria-modal="true"' not in path.read_text(encoding="utf-8"), path.name
    for css in (ROOT / "clients").glob("*.css"):
        assert "clients-onboard-layer" not in css.read_text(encoding="utf-8"), css.name


def test_sheets_and_mui_drawers_use_the_same_header_boundary():
    studio = (ROOT / "studio/studio-workspaces.css").read_text(encoding="utf-8")
    theme = (ROOT / "shell/theme-form-overrides.js").read_text(encoding="utf-8")
    assert "inset-block: var(--shell-header-height) 0;" in studio
    assert "top: 'var(--shell-header-height)'" in theme
    assert "calc(100dvh - var(--shell-header-height))" in theme


def test_focused_outlined_fields_change_one_quiet_pixel_not_a_thick_frame():
    theme = (ROOT / "shell/theme-form-overrides.js").read_text(encoding="utf-8")
    studio = (ROOT / "studio/studio.css").read_text(encoding="utf-8")
    assert "borderWidth: 1" in theme
    focused = _rule("studio/studio.css", ".kairos-shell .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline,\n.studio-modal .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline")
    assert "border-width: 1px" in focused


def test_global_keyboard_focus_is_one_sober_high_contrast_boundary():
    selectors = {
        "shell/styles.css": ":where(button, [role='button'], a, [tabindex]):focus-visible",
        "studio/studio.css": ".kairos-shell :where(button, a, input, select, textarea, [tabindex]):focus-visible,\n.studio-modal :where(button, a, input, select, textarea, [tabindex]):focus-visible",
    }
    for path, selector in selectors.items():
        body = _rule(path, selector)
        assert "outline: 1px solid var(--focus-ring);" in body, path
        assert "outline-offset: 3px;" in body, path
        assert "box-shadow" not in body, path

    tokens = (ROOT / "tokens.css").read_text(encoding="utf-8")
    assert "--focus-ring: var(--accent);" in tokens
