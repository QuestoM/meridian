"""P5: the price-slot tester offers the operator's own week, Sunday first.

The measured defect. The rate card and the price-any-slot tester sit on one
rendered page. The card reads its day-of-week premiums through the ordered
reader and puts Sunday first; the tester's own weekday control held a literal
``[1, 2, 3, 4, 5, 6, 7]`` and opened on 1, so three centimetres below a card
that read the Israeli week correctly, the one control an operator actually
drives served ``שני, שלישי, רביעי, חמישי, שישי, שבת, ראשון`` in Hebrew and
``Mon`` through ``Sun`` in English, with Monday selected. One screen, two weeks.

A source reader cannot catch that and neither can the payload: the store is ISO
and stays ISO, and both readers are handed the same keys. Only rendering both in
the same run compares them. So this file bundles the shipped component with the
bundler the product builds with, renders it to static markup in both locales,
reads the control's options in the order a person meets them, and asks the rate
card's own reader for the week it renders directly above. The last test restores
the literal week in the source and asserts the control goes back to reading
Monday first, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
TESTER = APP / "src" / "rules" / "PricingSlotTester.jsx"
PROBE = Path(__file__).with_name("test_p5_tester_probe.mjs")

# The Israeli week as the reader meets it. ISO values, because that is what the
# engine prices by and what a save has to send back.
SUNDAY_FIRST_VALUES = ["7", "1", "2", "3", "4", "5", "6"]
SUNDAY_FIRST_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
SUNDAY_FIRST_EN = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# The line the mutant replaces, asserted present before it is cut so the mutation
# can never be a silent no-op. Restoring the literal restores both symptoms at
# once, because the control's opening day is the first day of the week it reads.
ORDERED_WEEK = "const WEEKDAY_OPTIONS = DAY_ORDER.map(Number);"
LITERAL_WEEK = "const WEEKDAY_OPTIONS = [1, 2, 3, 4, 5, 6, 7];"

OPTION = re.compile(r'<option value="(?P<value>[^"]*)"(?P<selected> selected="")?>(?P<label>[^<]*)</option>')


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    if not (APP / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules is not installed, so nothing can be rendered")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    if not PROBE.exists():
        pytest.skip("the render probe is missing")
    return found


@pytest.fixture(scope="module")
def pricing() -> dict:
    """The rate card the product serves, which is the state the tester renders."""
    from kairos_api.pricing_api import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/pricing")
    assert response.status_code == 200, response.text
    return response.json()


def _render(tmp_path: Path, payload: dict, source: str) -> dict:
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    under_test = work / "tester-under-test.jsx"
    under_test.write_text(source, encoding="utf-8")
    pricing_file = work / "pricing.json"
    pricing_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [node, str(PROBE), str(work / "bundle"), str(out), str(under_test), str(pricing_file)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _weekday_options(rendered: dict, locale: str) -> list[dict]:
    """Every option of the rendered weekday control, in the order it renders."""
    html = rendered["html"][locale]
    selects = re.findall(r"<select.*?</select>", html, re.S)
    assert len(selects) == 1, "the tester renders one weekday control, and this reads that one"
    return [match.groupdict() for match in OPTION.finditer(selects[0])]


@pytest.fixture(scope="module")
def shipped(tmp_path_factory, pricing) -> dict:
    source = TESTER.read_text(encoding="utf-8")
    assert ORDERED_WEEK in source, "the ordering under test is not in the shipped component any more"
    return _render(tmp_path_factory.mktemp("shipped"), pricing, source)


def test_the_weekday_control_offers_the_week_sunday_first_in_both_locales(shipped):
    for locale, words in (("he", SUNDAY_FIRST_HE), ("en", SUNDAY_FIRST_EN)):
        options = _weekday_options(shipped, locale)
        assert [option["label"] for option in options] == words
        assert [option["value"] for option in options] == SUNDAY_FIRST_VALUES
        assert options[0]["label"] == words[0], "the Israeli week starts on Sunday"
        assert [option["label"] for option in options][-2:] == words[-2:], "the weekend is Friday and Saturday, and it is last"


def test_the_control_opens_on_the_day_the_operators_week_starts_on(shipped):
    """The default a person meets before touching anything, not just the order."""
    for locale in ("he", "en"):
        options = _weekday_options(shipped, locale)
        chosen = [option for option in options if option["selected"]]
        assert len(chosen) == 1, "exactly one day is selected"
        assert chosen[0]["value"] == "7", "ISO weekday 7 is Sunday"
        assert chosen[0] is options[0], "the day it opens on is the first day of the week it reads"


def test_the_tester_and_the_rate_card_above_it_read_the_same_week(shipped):
    """One screen, one week. This is the comparison the defect failed."""
    card = shipped["card_day_keys"]
    assert card == SUNDAY_FIRST_VALUES, "the card's own reader is the reference, and it is Sunday first"
    assert shipped["card_day_labels_he"] == SUNDAY_FIRST_HE
    for locale in ("he", "en"):
        assert [option["value"] for option in _weekday_options(shipped, locale)] == card


def test_reordering_the_reading_moved_no_key_the_engine_prices_by(shipped):
    """The store is ISO and stays ISO: the control still posts the key it held."""
    values = [option["value"] for option in _weekday_options(shipped, "he")]
    assert sorted(values) == ["1", "2", "3", "4", "5", "6", "7"], "no day was invented, dropped or renumbered"
    source = TESTER.read_text(encoding="utf-8")
    assert "weekday_iso: Number(slot.weekday_iso) || FIRST_WEEKDAY," in source, (
        "the posted body carries the ISO value the option holds"
    )


def test_with_the_literal_week_the_same_control_reads_monday_first(tmp_path, pricing):
    """The mutant, which is exactly what a critic measured on the shipped bundle."""
    source = TESTER.read_text(encoding="utf-8")
    mutant = source.replace(ORDERED_WEEK, LITERAL_WEEK)
    assert mutant != source
    rendered = _render(tmp_path, pricing, mutant)
    assert [option["label"] for option in _weekday_options(rendered, "he")] == [
        "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון",
    ]
    assert [option["label"] for option in _weekday_options(rendered, "en")] == [
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    ]
    chosen = [option for option in _weekday_options(rendered, "he") if option["selected"]]
    assert [option["value"] for option in chosen] == ["1"], "and it opened on Monday"
    assert rendered["card_day_keys"] == SUNDAY_FIRST_VALUES, (
        "while the rate card three centimetres above went on reading Sunday first, which is the defect"
    )
