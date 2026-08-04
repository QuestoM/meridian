"""The blocked register is a set of paths forward, measured in a real browser.

**The defect this file exists to keep dead.** Coverage is the first screen of
JS-16 and it was the only screen on the console that ended in prose. Measured on
the shipped tree with the critic's own probe,
``document.querySelectorAll('.mc-body button, .mc-body a').length`` per section
read: gates 21, drift 1, candidates 15, training 18, versions 10, provenance 2,
coverage 0. Five factors were reported blocked and the screen offered nothing to
do about any of them, while the payload it was already holding carried the
answer four times over:

1. **The date arrived with no name.** The row printed ``row.earliest.start``
   alone. The payload carries ``earliest.name_he`` for all five rows and
   ``earliest.end`` for the range, so a screen that knew the block ends with
   חנוכה, over the eight days to 2025-01-02, showed a bare 2024-12-26.
2. **The source was a path, not an address.** Two of the five name
   ``data/calendar_events.csv``, which is the store behind ``/api/events``, which
   is what the calendar section of Rules reads and writes. It was rendered as a
   ``<code>`` for somebody to go and find. The other three name files checked in
   with the product, where the honest answer is that nobody supplies them and
   only time ends the block, and the screen said neither thing.
3. **The evidence was raw keys.** ``days_in_window: 30
   event_free_days_in_window: 0``, which is the inside of the program on the
   outside of it.
4. **The variances were raw floats.** ``tau^2 0.00017294779183852117 /
   0.06510336856729029``, seventeen significant digits on the one line of the
   panel that did not go through the console's own number bit.

**Why this is a browser test.** Every one of the four is a rendered consequence,
and the second is a rendered consequence with a destination on the other side of
it. Asserting that the source file contains a ``<button>`` proves nothing about
whether pressing it leaves the console. So the real bridge is built by the
product's own bundler, run in a real headless Chrome, handed the body the real
route serves, and asked what is on screen and where a press lands.

**What is real here and what is a stand-in.** The payload is real: it is the body
of ``GET /api/model/coverage`` taken from the running application through a
client, artifacts, calendar and event store and all. The shell around the console
is the same stand-in the other console measurements use, and the address the
press lands on is checked against the frozen shell's own resolver rather than
against a line of its source.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_p7_console_bridge_harness import (
    BRIDGE,
    build_harness,
    resolve_shell_views,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

ROOT = Path(__file__).resolve().parents[1]
CONSOLE_DIR = ROOT / "tv-break-dashboard" / "src" / "model" / "console"

COVERAGE_ROUTE = "/api/model/coverage"

# The address the source control lands on. It is the frozen shell's own label for
# the calendar section of Rules, kept as a known route by ``nav.js`` and turned
# into the Rules workspace with that section open by the frozen router.
EVENTS_HASH = "Calendar"

# How long the console gets to appear, and how long the press that leaves it gets
# before the measurement gives up. The second is the one a person feels.
OPEN_BUDGET_MS = 4000
PRESS_BUDGET_MS = 400

# The section the rail opens second. Asserted through the route the body prints
# rather than trusted, so a reordered rail fails here instead of measuring the
# wrong screen.
COVERAGE_RAIL_INDEX = 1

HARNESS_JS = """
import { mountModelConsole } from '%(bridge)s';

const ROW = '.mc-blocked';
const BUDGET = %(open)d;

function settle(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

async function waitFor(read, budgetMs) {
  const started = performance.now();
  for (;;) {
    const value = read();
    if (value) return value;
    if (performance.now() - started > budgetMs) return null;
    await settle(20);
  }
}

// The stand-in operator shell, in the product's own language and direction.
function shellChild() {
  const node = document.createElement('div');
  node.className = 'kairos-shell rtl';
  node.setAttribute('dir', 'rtl');
  node.setAttribute('lang', 'he');
  node.append(document.createElement('main'));
  return node;
}

function text(node) {
  return node ? node.textContent.trim().replace(/\\s+/g, ' ') : '';
}

// One blocked row, read from the document and nothing else.
function readRows() {
  return Array.from(document.querySelectorAll(ROW)).map((row) => {
    const source = row.querySelector('.mc-blocked-source');
    const control = source ? source.querySelector('button') : null;
    const path = control || source;
    return {
      label: text(row.querySelector('strong')),
      earliest: text(row.querySelector('.mc-blocked-date')),
      earliestName: text(row.querySelector('.mc-earliest-name')),
      earliestSpan: text(row.querySelector('.mc-earliest-span')),
      counted: text(row.querySelector('.mc-blocked-evidence')),
      source: text(source),
      sourceIsControl: Boolean(control),
      sourcePath: text(path ? path.querySelector('code') : null),
      supply: text(row.querySelector('.mc-blocked-supply')),
      controls: row.querySelectorAll('button, a').length,
      whole: text(row),
    };
  });
}

async function run() {
  const result = { phases: {}, rows: [] };

  mountModelConsole();
  await fetch('/testctl/session?as=company');
  document.getElementById('root').replaceChildren(shellChild());
  const switcher = await waitFor(() => document.querySelector('.mc-switcher'), BUDGET);
  if (!switcher) throw new Error('the switcher never appeared, so nothing was measured');
  switcher.click();

  const view = await waitFor(() => document.querySelector('.mc-console'), BUDGET);
  if (!view) throw new Error('the console never rendered');
  const rail = await waitFor(() => document.querySelectorAll('.mc-rail-item')[%(rail)d], BUDGET);
  if (!rail) throw new Error('the rail never rendered');
  rail.click();
  await waitFor(() => document.querySelectorAll(ROW).length, BUDGET);

  result.phases.coverage = {
    route: text(document.querySelector('.mc-route code')),
    dir: view.getAttribute('dir'),
    lang: view.getAttribute('lang'),
    bodyControls: document.querySelectorAll('.mc-body button, .mc-body a').length,
    variances: text(document.querySelector('.mc-variances')),
    body: text(document.querySelector('.mc-body')),
  };
  result.rows = readRows();

  // The press a person makes on the source of a blocked row, and where it lands.
  const control = document.querySelector('.mc-blocked-source button');
  result.phases.press = { control: Boolean(control), label: text(control) };
  if (control) {
    const before = performance.now();
    control.click();
    const gone = await waitFor(() => !document.querySelector('.mc-console'), BUDGET);
    result.phases.press.leftMs = Math.round(performance.now() - before);
    result.phases.press.consoleGone = Boolean(gone);
    result.phases.press.hash = window.location.hash;
    result.phases.press.switcher = Boolean(document.querySelector('.mc-switcher'));
  }
  return result;
}

run().then((result) => {
  return fetch('/testctl/result', { method: 'POST', body: JSON.stringify(result) });
}).catch((error) => {
  return fetch('/testctl/result', {
    method: 'POST',
    body: JSON.stringify({ failed: String(error && error.stack ? error.stack : error) }),
  });
});
"""


@pytest.fixture(scope="module")
def coverage(tmp_path_factory) -> "dict":
    """The body ``GET /api/model/coverage`` serves, from the application itself."""
    os.environ["KAIROS_MODEL_RELEASES_DIR"] = str(tmp_path_factory.mktemp("releases"))
    from kairos_api.model_console_api import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get(COVERAGE_ROUTE)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["blocked"], "no factor is blocked on this tree, so this measurement would prove nothing"
    return payload


@pytest.fixture(scope="module")
def scenario(tmp_path_factory, coverage) -> "dict":
    skip_unless_a_real_browser_is_available()
    work = tmp_path_factory.mktemp("p7-coverage")
    harness_js = HARNESS_JS % {
        "bridge": os.path.relpath(BRIDGE, work.resolve() / "src"),
        "open": OPEN_BUDGET_MS,
        "rail": COVERAGE_RAIL_INDEX,
    }
    dist = build_harness(work, harness_js)
    return run_scenario(dist, work, {COVERAGE_ROUTE: coverage})


def _supply_table() -> "dict[str, str]":
    """The panel's own classification of a source, read out of the panel."""
    source = (CONSOLE_DIR / "CoveragePanel.jsx").read_text(encoding="utf-8")
    body = re.search(r"const SUPPLY = \{(.*?)\};", source, re.S)
    assert body is not None, "the coverage panel no longer classifies its sources"
    return dict(re.findall(r"'([^']+)':\s*'([^']+)'", body.group(1)))


def _dated(coverage: "dict") -> "list[dict]":
    return [row for row in coverage["blocked"] if row["earliest_state"] == "dated"]


# ---------------------------------------------------------------------------
# The premise, and the two guards that hold without a browser
# ---------------------------------------------------------------------------


def test_the_register_is_worth_measuring(coverage) -> None:
    """Without these four facts every assertion below would be vacuously true."""
    rows = coverage["blocked"]
    table = _supply_table()
    stores = [row for row in rows if table.get(row["source"]) == "store"]
    checked_in = [row for row in rows if table.get(row["source"]) == "time"]
    assert stores, "no blocked row names a store somebody can open"
    assert checked_in, "no blocked row names a file only time can end"
    assert _dated(coverage), "no blocked row carries a date, so the name and the range are untestable"
    assert any(row["earliest"]["end"] != row["earliest"]["start"] for row in _dated(coverage)), (
        "no blocked row carries a range, so rendering only the start would still pass"
    )


def test_every_source_the_register_emits_is_classified_by_the_panel(coverage) -> None:
    """A source the panel does not know reads unknown, so a new one fails here.

    This is the seam between the module that names the source and the screen that
    turns it into an address. The panel renders an honest unknown rather than
    guessing, which means a new source would degrade the screen in silence. It
    fails this instead.
    """
    table = _supply_table()
    unknown = sorted({row["source"] for row in coverage["blocked"] if row["source"] not in table})
    assert unknown == [], f"the register names sources the panel cannot classify: {unknown}"
    assert set(table.values()) <= {"store", "time"}, f"the panel grew a state nothing renders: {table}"


def test_every_counted_figure_the_register_emits_has_a_word(coverage) -> None:
    """The other half of the raw-key defect: a new key would print raw again."""
    words = (CONSOLE_DIR / "console-words.js").read_text(encoding="utf-8")
    known = set(re.findall(r"'coverage\.evidence\.([a-z_]+)'", words))
    keys = {key for row in coverage["blocked"] for key in (row.get("evidence") or {})}
    assert keys, "the register counts nothing, so this guard would prove nothing"
    assert keys <= known, f"these counted figures have no word and would render as keys: {sorted(keys - known)}"


def test_the_payload_this_screen_renders_names_no_rival_channel(coverage) -> None:
    """The competitor boundary, on the one body this measurement puts on a screen."""
    from kairos_api import model_console_artifacts as artifacts

    payload = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    base = payload.get("base") or {}
    owned = str(base.get("owned_channel") or "")
    rivals = sorted(name for name in (base.get("hist_channel") or {}) if name != owned)
    assert rivals, "the artifact holds no rival channels, so this test would prove nothing"
    body = json.dumps(coverage, ensure_ascii=False)
    for rival in rivals:
        assert rival not in body, f"the coverage payload names the rival channel {rival}"


# ---------------------------------------------------------------------------
# The four defects, measured on screen
# ---------------------------------------------------------------------------


def test_the_console_painted_the_real_register_in_hebrew(scenario, coverage) -> None:
    """The premise. Without it every assertion below would be about a blank screen."""
    assert "failed" not in scenario, scenario.get("failed")
    phase = scenario["phases"]["coverage"]
    assert phase["dir"] == "rtl" and phase["lang"] == "he"
    assert phase["route"] == COVERAGE_ROUTE, (
        f"the rail opened {phase['route']}, so some other screen was measured"
    )
    assert len(scenario["rows"]) == len(coverage["blocked"]), (
        f"the register renders {len(scenario['rows'])} of {len(coverage['blocked'])} blocked factors"
    )


def test_the_screen_that_ended_in_prose_now_carries_a_control_for_every_openable_row(scenario, coverage) -> None:
    """The measurement the critic took, taken again: it read zero.

    The count is not asserted as "more than zero". It is asserted as exactly the
    number of blocked rows whose source is a store a person can open, so a
    decorative control added anywhere on this screen fails here too.
    """
    table = _supply_table()
    openable = [row for row in coverage["blocked"] if table.get(row["source"]) == "store"]
    assert scenario["phases"]["coverage"]["bodyControls"] == len(openable), (
        f"coverage carries {scenario['phases']['coverage']['bodyControls']} controls"
        f" and {len(openable)} of its rows name a store somebody can open"
    )
    for measured, row in zip(scenario["rows"], coverage["blocked"]):
        expected = 1 if table.get(row["source"]) == "store" else 0
        assert measured["controls"] == expected, (
            f"the row for {row['gate_id']} carries {measured['controls']} controls and its source is {row['source']}"
        )


def test_every_dated_row_names_the_thing_that_ends_it_and_the_range_it_runs(scenario, coverage) -> None:
    """Defect one, dead: the date arrived alone.

    The name and both ends of the range are compared against the payload the
    browser was handed, so this cannot pass on a hard-coded string.
    """
    for measured, row in zip(scenario["rows"], coverage["blocked"]):
        if row["earliest_state"] != "dated":
            assert measured["earliestName"] == "", f"{row['gate_id']} renders a name with no date"
            continue
        earliest = row["earliest"]
        assert measured["earliestName"] == earliest["name_he"], (
            f"{row['gate_id']} shows the name {measured['earliestName']!r} and the payload says {earliest['name_he']!r}"
        )
        assert earliest["start"] in measured["earliestSpan"], (
            f"{row['gate_id']} shows the span {measured['earliestSpan']!r} without its first date"
        )
        if earliest["end"] != earliest["start"]:
            assert earliest["end"] in measured["earliestSpan"], (
                f"{row['gate_id']} runs to {earliest['end']} and the screen shows {measured['earliestSpan']!r}"
            )


def test_the_source_of_a_blocked_row_is_the_address_it_names(scenario, coverage) -> None:
    """Defect two, dead: the store was printed and not opened.

    The path stays on screen either way, because it is what the steward checks
    the console against. What changes is that where the path is a page in this
    product it is the control that opens it, and where it is a file checked in
    with the product the screen says plainly that nobody supplies it.
    """
    table = _supply_table()
    for measured, row in zip(scenario["rows"], coverage["blocked"]):
        supply = table[row["source"]]
        assert measured["sourcePath"] == row["source"], (
            f"{row['gate_id']} shows the source {measured['sourcePath']!r} and the payload says {row['source']!r}"
        )
        assert measured["sourceIsControl"] is (supply == "store"), (
            f"{row['gate_id']} names {row['source']} and its source control state is {measured['sourceIsControl']}"
        )
        assert measured["supply"], f"{row['gate_id']} says nothing about who supplies its source"
    stated = {measured["supply"] for measured in scenario["rows"]}
    assert len(stated) == 2, f"the register states one answer for both kinds of source: {stated}"


def test_pressing_that_source_leaves_the_console_for_the_page_that_owns_the_store(scenario) -> None:
    """The other half of defect two: the control goes where its words say.

    A control that renders and does nothing is the dead end one level down, so
    the press is made the way a person makes it and the address it lands on is
    read off the window afterwards.
    """
    phase = scenario["phases"]["press"]
    assert phase["control"] is True, "no blocked row carries a control to press"
    assert phase["consoleGone"] is True, "the console is still on screen after the control was pressed"
    assert phase["switcher"] is True, "the way back into the console did not return"
    assert phase["hash"] == f"#{EVENTS_HASH}", (
        f"the control landed on {phase['hash']} and the store it names lives on #{EVENTS_HASH}"
    )
    assert phase["leftMs"] <= PRESS_BUDGET_MS, f"leaving for the store took {phase['leftMs']} ms"


def test_the_frozen_shell_still_knows_the_address_that_control_lands_on(tmp_path) -> None:
    """The premise the press rests on, driven rather than read.

    A hash is a destination only because the shell turns it into one. The
    resolver lives in a file this piece may not write, so it is bundled by the
    product's own bundler and asked, in a browser, what this address resolves to.
    The unknown address is asked with it, because a resolver that echoed whatever
    it was handed would satisfy the first assertion and prove nothing.
    """
    unknown = "#NoPageIsFiledUnderThisName"
    views = resolve_shell_views(tmp_path, [f"#{EVENTS_HASH}", unknown, ""])
    assert "failed" not in views, views.get("failed")
    resolved = views["resolved"]
    assert resolved[f"#{EVENTS_HASH}"] == EVENTS_HASH, (
        f"the shell resolves the register's address for the event store to {resolved[f'#{EVENTS_HASH}']}"
    )
    assert resolved[unknown] == resolved[""], (
        "the resolver returns any address it is handed, so the assertion above proves nothing"
    )


def test_the_counted_figures_are_words_and_not_the_keys_they_are_stored_under(scenario, coverage) -> None:
    """Defect three, dead: the evidence read days_in_window: 30 on screen."""
    body = scenario["phases"]["coverage"]["body"]
    keys = {key for row in coverage["blocked"] for key in (row.get("evidence") or {})}
    for key in keys:
        assert key not in body, f"the register still prints the raw key {key}"
    for measured, row in zip(scenario["rows"], coverage["blocked"]):
        evidence = row.get("evidence") or {}
        if not evidence:
            assert measured["counted"] == "", f"{row['gate_id']} counts nothing and renders a count line"
            continue
        for value in evidence.values():
            assert str(value) in measured["counted"], (
                f"{row['gate_id']} counted {value} and the screen reads {measured['counted']!r}"
            )


def test_the_two_variances_print_at_the_precision_of_the_rest_of_the_surface(scenario, coverage) -> None:
    """Defect four, dead: seventeen significant digits beside a rounded ratio."""
    measured = scenario["phases"]["coverage"]["variances"]
    assert measured, "the contrast ratio no longer shows what it is made of"
    retention = coverage["retention"]
    for key in ("between_cell_variance_tau2", "pooled_within_variance"):
        value = retention[key]
        assert abs(value) < 1000, "these variances grew past the grouping threshold; re-measure this assertion"
        assert repr(value) not in measured, f"{key} still prints as the raw float {value!r}"
        assert f"{value:.6f}" in measured, (
            f"{key} reads {measured!r} and rounds to {value:.6f}"
        )
    for fraction in re.findall(r"\d+\.(\d+)", measured):
        assert len(fraction) <= 6, f"a figure on this line carries {len(fraction)} decimal digits: {measured}"
