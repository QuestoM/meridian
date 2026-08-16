"""P5: the calendar's rate-card control always renders, and Rules follows a
query it did not write itself.

Two measured defects, one cause each, both inside this piece's own row.

The calendar tab's banner named a live pricing layer and the page that
controls it, but the control was gated on ``typeof setActiveView ===
'function'``, a prop RulesWorkspace never supplied, so it compiled out of
every render: a dead end, live. The fix hands the banner the workspace's own
navigation directly (``onOpenRateCard``), renders the control every time, and
names it after the tab it actually opens (the rate card is a tab in this
workspace now, not a separate Pricing page).

Wiring the old prop instead would have produced a second dead end: the shell's
legacy Pricing bookmark rewrites the ``?rules`` query and flips the top-level
view without ever remounting RulesWorkspace, and RulesWorkspace read that
query only once, in ``useState``'s initializer, so the address bar would claim
the rate card while the calendar tab stayed open. ``nextRulesSection`` in
``rules-lib.js`` is the piece that now reconciles a later query against the
section already showing, checked on every render rather than only at mount.

Both are measured against the shipped modules, bundled with the product's own
bundler, not a restatement of them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
PROBE = Path(__file__).with_name("test_p5_calendar_route_probe.mjs")


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped module cannot be run here")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    if not PROBE.exists():
        pytest.skip("the calendar route probe is missing")
    return found


@pytest.fixture(scope="module")
def result() -> dict:
    node = _node()
    proc = subprocess.run(
        [node, str(PROBE)], capture_output=True, text=True, check=False, cwd=str(APP),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


LOCALES = ("he", "en")
PRICING_STATES = ("None", "True", "False")
CALLBACK_STATES = ("callback", "no_callback")
STALE_STRINGS = ("Pricing page", "עמוד התמחור")


def test_the_banner_control_renders_on_every_combination(result):
    """The measured defect: the control was gated on a prop nothing supplied,
    so it never rendered. It must render exactly once, every time, whether or
    not a callback happens to be present, because the gate itself is the bug."""
    banner = result["banner"]
    assert len(banner) == len(LOCALES) * len(PRICING_STATES) * len(CALLBACK_STATES)
    for key, markup in banner.items():
        controls = re.findall(r'<button\b[^>]*class="[^"]*\bcal-banner-link\b[^"]*"', markup)
        assert len(controls) == 1, f"{key} renders no control"


def test_the_banner_never_names_the_page_that_was_renamed(result):
    """Pricing is a tab in this workspace now (the rate card); a sentence
    that still says "the Pricing page" sends a reader looking for a
    destination that no longer exists under that name."""
    for key, markup in result["banner"].items():
        for stale in STALE_STRINGS:
            assert stale not in markup, f"{key} still names the renamed destination"


def test_the_banner_names_the_rate_card_in_both_languages(result):
    banner = result["banner"]
    assert "כרטיס התעריפים" in banner["he_true_callback"]
    assert "rate card" in banner["en_true_callback"]


def test_a_query_changed_by_something_else_after_mount_is_followed(result):
    """The exact scenario the critic measured live: the calendar tab open,
    then something outside this component's own clicks rewrites ?rules to
    rate_card. The section must follow it rather than staying on the value it
    mounted with."""
    assert result["route"]["external_change_is_followed"] == "rate_card"


def test_a_query_that_already_matches_the_current_section_is_a_no_op(result):
    """Guards against a fix that keeps forcing a re-render loop once the
    section and the query agree."""
    assert result["route"]["unchanged_query_is_a_no_op"] == "rate_card"


def test_a_missing_or_invalid_query_never_clears_a_real_section(result):
    """sectionFromLocation already returns '' for a query naming no section
    this workspace has; nextRulesSection must treat that as no change rather
    than as an instruction to blank the section a person is looking at."""
    assert result["route"]["empty_query_keeps_the_current_section"] == "licence"
