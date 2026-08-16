"""Mabat shares the page body without shortening the persistent shell header."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src"


def _rule(selector: str, sheet: str = "studio-ledger-kai.css") -> str:
    source = (ROOT / "kai" / sheet).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", source, re.S)
    assert match, f"missing layout contract: {selector}"
    return match.group("body")


def test_mabat_adds_a_real_body_column_without_overlaying_the_workspace():
    shell = _rule(".kairos-shell.assistant-open")
    dock = _rule(".asst-dock", "assistant-console.css")

    assert "grid-template-columns: var(--rail-expanded) minmax(0, 1fr) auto" in shell
    assert "top: var(--shell-header-height)" in dock
    assert "height: calc(100dvh - var(--shell-header-height))" in dock


def test_open_mabat_keeps_the_header_full_width_and_the_body_in_place():
    workspace = _rule(".kairos-shell.assistant-open .workspace")
    header = _rule(".kairos-shell.assistant-open .top-bar")

    assert "padding-block-start: var(--shell-header-height)" in workspace
    assert "position: fixed" in header
    assert "inset-block-start: 0" in header
    assert "inset-inline: var(--rail-expanded) 0" in header
    assert "inline-size: auto" in header
    assert "margin-inline: 0" in header
