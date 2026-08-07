"""P5: Today's compliance card never prints a market-wide verdict as its own.

The measured defect. Today's ``ComplianceLedger.jsx`` was fed ``overview.compliance``
and printed it verbatim: ``GET /api/overview`` returns a compliance block with no
scope key at all, ``retention_floor`` violations 3,584 and ``break_spacing``
observed 7.0, drawn from all four channels in the plan, 6,635 of the 9,026 graded
breaks a rival's. P5's own ``GET /api/compliance`` scopes the same seven checks to
the declared operator channel, violations 736, spacing 7.01, and its violations
list names no rival. The landing screen disagreed with the licence page one click
away by 4.9x on the same check with nothing on screen to say which basis either
figure was drawn from.

The fix moved the card's data source: it now fetches its own scoped route
directly and keeps the prop its parent hands it as a fallback used only once its
own fetch has failed. This file tests the pure decision function the component
renders from, ``complianceViewState`` in ``rules-lib.js``, run through the real
shipped module rather than a restatement of it, because that function is exactly
where the earlier defect lived: it decided nothing, and the component simply drew
whatever prop it was given.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
PROBE = Path(__file__).with_name("test_p5_compliance_view_probe.mjs")


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped module cannot be run here")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    if not PROBE.exists():
        pytest.skip("the compliance view probe is missing")
    return found


@pytest.fixture(scope="module")
def cases() -> dict:
    node = _node()
    result = subprocess.run(
        [node, str(PROBE)], capture_output=True, text=True, check=False, cwd=str(APP),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_a_market_wide_fallback_with_no_scope_key_never_becomes_the_verdict(cases):
    """The exact shape GET /api/overview returns: checks, no scope key at all."""
    view = cases["unscoped_fallback_is_basis_missing"]
    assert view["kind"] == "basis_missing"
    assert "data" not in view, "the market-wide checks must not ride along on this state"


def test_the_cards_own_fetch_wins_even_when_a_fallback_prop_is_also_supplied(cases):
    view = cases["own_wins_over_fallback"]
    assert view["kind"] == "scoped"
    assert view["data"]["disclaimer"] == "own", "the own-fetch payload, not the fallback, must render"
    assert view["scope"]["scope_channel"] == "ערוץ 13"


def test_a_still_loading_card_never_paints_the_unscoped_fallback_meanwhile(cases):
    """Before the own fetch has answered or failed, nothing is printed as a verdict."""
    view = cases["still_loading"]
    assert view["kind"] == "loading"
    assert "data" not in view


def test_a_properly_scoped_fallback_is_still_shown_once_the_own_fetch_fails(cases):
    """The fallback is not banned outright, only an unscoped one is refused."""
    view = cases["scoped_fallback_is_shown"]
    assert view["kind"] == "scoped"
    assert view["scope"]["rows_out"] == 736


def test_a_fallback_naming_no_operator_channel_reads_as_no_population_not_compliant(cases):
    view = cases["no_channel_fallback"]
    assert view["kind"] == "no_channel"
    assert view["reasonHe"] and view["reasonEn"]


FIRST_STRONG_ISOLATE = "⁨"
POP_DIRECTIONAL_ISOLATE = "⁩"


def _first_strong_char(text: str) -> str:
    return next(ch for ch in text if ch.isalpha() or ch.isdigit())


def test_the_english_scope_sentence_leads_with_a_latin_character(cases):
    """The measured defect: the English sentence led with the Hebrew channel
    name, so dir="auto" on the paragraph resolved right to left, threw the TV
    icon to the wrong edge and reordered the channel name's own digit and
    punctuation. The first strong character a bidi algorithm finds decides
    the paragraph's direction, so the fix is verified at that character."""
    sentence = cases["scope_sentence_en"]
    assert _first_strong_char(sentence).isascii(), f"expected a Latin lead character in {sentence!r}"
    assert sentence.startswith("This operator's own channel")


def test_the_english_scope_sentence_isolates_the_channel_name(cases):
    """Leading with English words fixed the paragraph's own direction, but the
    channel name still sat unisolated next to a colon and a digit run of its
    own, and the measured render was still wrong once the paragraph direction
    was fixed alone: "is 2391 :13 רשת breaks judged". The channel name must be
    wrapped in the isolate pair this product already uses for every other
    foreign-script figure, so the neutrals beside it never borrow its script's
    direction."""
    sentence = cases["scope_sentence_en"]
    isolated = f"{FIRST_STRONG_ISOLATE}רשת 13{POP_DIRECTIONAL_ISOLATE}"
    assert isolated in sentence
    assert sentence == (
        "This operator's own channel is "
        f"{isolated}: 2391 breaks judged, 6635 on other channels left out."
    )


def test_the_hebrew_scope_sentence_still_leads_with_the_channel_name(cases):
    """The Hebrew card was already correct; this guards against a fix that
    reorders the Hebrew sentence to chase the English one instead of leaving
    it alone. The channel name is isolated here too, for the same reason,
    which does not change its reading, only fences its own script off."""
    sentence = cases["scope_sentence_he"]
    assert _first_strong_char(sentence) == "ר"
    isolated = f"{FIRST_STRONG_ISOLATE}רשת 13{POP_DIRECTIONAL_ISOLATE}"
    assert sentence.startswith(isolated)
