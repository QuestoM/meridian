"""The gate filter, measured in a real browser against the real gate ledger.

**The defect this file exists to keep dead.** The five state chips at the top of
the console's first screen are the filter, so a chip is a number that opens the
rows behind it. Measured in a browser on the shipped tree, three things were
wrong at once and all three were invisible to every source assertion:

1. Pressing the chip whose count is zero rendered no rows, no groups and no
   sentence of any kind. The screen went from the chip legend straight to the
   next panel's heading. A number that promises rows opened a silent hole, which
   is section 3.6's dead end exactly.
2. The panel subtitle read "13 of 13" under every chip, including the one that
   rendered three rows, because it was composed from the unfiltered array. A
   count that says thirteen while three rows are on screen is the figure
   contradicting itself on the same screen.
3. The chips carried no ``aria-pressed``. Which one was active was carried by a
   CSS class alone, while the console's own rail sets ``aria-current`` correctly,
   so the surface was inconsistent with itself and a screen reader was told
   nothing about the filter it was inside.

**Why this is a browser test.** Every one of the three is a rendered
consequence: rows that exist in an array and not in the document, a string
composed at render time, and an attribute the browser reports. Asserting the
source contains a template proves none of them. So the real bridge is built by
the product's own bundler, run in a real headless Chrome, handed the payload the
real route serves, and asked what is on screen after each press.

**What is real here and what is a stand-in, stated plainly.** The payload is
real: it is the body of ``GET /api/model/gates`` taken from the running
application through a client, artifacts and all, so the counts pressed in the
browser are the product's own measured counts and not a fixture. The shell
around the console is the same stand-in the bridge measurements already use, for
the same reason and with the same premise asserted separately from source.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_p7_console_bridge_harness import (
    BRIDGE,
    build_harness,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

# The route whose body the browser is handed, and the section it paints.
GATES_ROUTE = "/api/model/gates"

# How long the console gets to appear after the switcher is pressed, and how
# long one filter press gets before the measurement gives up on it. The second
# is the one a person feels.
OPEN_BUDGET_MS = 4000
PRESS_BUDGET_MS = 400

HARNESS_JS = """
import { mountModelConsole } from '%(bridge)s';

const PANEL = '.mc-panel';
const CHIP = '.mc-legend-item';

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

// The stand-in operator shell, in the product's own language and direction, so
// the console the bridge mounts settles on Hebrew the way the product does.
function shellChild() {
  const node = document.createElement('div');
  node.className = 'kairos-shell rtl';
  node.setAttribute('dir', 'rtl');
  node.setAttribute('lang', 'he');
  const main = document.createElement('main');
  node.append(main);
  return node;
}

function gatesPanel() {
  return document.querySelectorAll(PANEL)[0] || null;
}

function chips() {
  return Array.from(document.querySelectorAll(CHIP));
}

// Which state a chip filters to, taken from the chip itself rather than from
// its position, so a reordered legend cannot make this measurement lie.
function chipState(chip) {
  const token = Array.from(chip.classList)
    .filter((name) => name.startsWith('mc-') && name !== 'mc-legend-item')
    .find((name) => name !== 'on');
  return token ? token.slice(3) : 'all';
}

function text(node) {
  return node ? node.textContent.trim() : '';
}

// Everything on screen after a press, read from the document and nothing else.
function screen() {
  const panel = gatesPanel();
  const empty = panel ? panel.querySelector('.mc-gate-empty') : null;
  return {
    rows: panel ? panel.querySelectorAll('.mc-gate').length : -1,
    groups: panel ? panel.querySelectorAll('.mc-gate-group').length : -1,
    subtitle: text(panel ? panel.querySelector('.mc-panel-head p') : null),
    emptyStates: panel ? panel.querySelectorAll('.mc-gate-empty').length : -1,
    emptyText: text(empty),
    emptyControls: empty ? empty.querySelectorAll('a[href],button').length : 0,
    pressed: chips().map((chip) => chip.getAttribute('aria-pressed')),
    roles: chips().map((chip) => chip.getAttribute('role')),
    panelText: text(panel),
  };
}

async function pressChip(index) {
  const chip = chips()[index];
  const before = performance.now();
  chip.click();
  await new Promise((resolve) => { requestAnimationFrame(() => resolve()); });
  const pressMs = Math.round(performance.now() - before);
  return {
    state: chipState(chip),
    count: Number(text(chip.querySelector('.mc-legend-count'))),
    label: text(chip.querySelector('.mc-legend-label')),
    pressMs,
    ...screen(),
  };
}

async function run() {
  const result = { phases: {}, chips: [] };

  mountModelConsole();
  await fetch('/testctl/session?as=company');
  document.getElementById('root').replaceChildren(shellChild());
  const switcher = await waitFor(() => document.querySelector('.mc-switcher'), %(open)d);
  if (!switcher) throw new Error('the switcher never appeared, so nothing was measured');
  switcher.click();

  const view = await waitFor(() => document.querySelector('.mc-console'), %(open)d);
  if (!view) throw new Error('the console never rendered');
  const legend = await waitFor(() => document.querySelectorAll(CHIP).length, %(open)d);
  result.phases.opened = {
    dir: view.getAttribute('dir'),
    lang: view.getAttribute('lang'),
    chips: legend,
    ...screen(),
  };

  // Every chip in turn, in the order the legend renders them.
  for (let index = 0; index < chips().length; index += 1) {
    result.chips.push(await pressChip(index));
  }

  // The way back out of a chip that opened an empty state: the control the
  // empty state itself carries, pressed the way a person presses it.
  const zero = result.chips.findIndex((row) => row.count === 0);
  result.phases.zeroIndex = zero;
  if (zero >= 0) {
    await pressChip(zero);
    const control = gatesPanel().querySelector('.mc-gate-empty button');
    result.phases.wayBack = { control: Boolean(control), label: text(control) };
    if (control) {
      control.click();
      await new Promise((resolve) => { requestAnimationFrame(() => resolve()); });
      Object.assign(result.phases.wayBack, screen());
    }
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
def ledger(tmp_path_factory) -> "dict":
    """The body ``GET /api/model/gates`` serves, from the application itself."""
    os.environ["KAIROS_MODEL_RELEASES_DIR"] = str(tmp_path_factory.mktemp("releases"))
    from kairos_api.model_console_api import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get(GATES_ROUTE)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["gates"], "the gate ledger is empty, so this measurement would prove nothing"
    return payload


@pytest.fixture(scope="module")
def scenario(tmp_path_factory, ledger) -> "dict":
    skip_unless_a_real_browser_is_available()
    work = tmp_path_factory.mktemp("p7-gates")
    harness_js = HARNESS_JS % {
        "bridge": os.path.relpath(BRIDGE, work.resolve() / "src"),
        "open": OPEN_BUDGET_MS,
    }
    dist = build_harness(work, harness_js)
    return run_scenario(dist, work, {GATES_ROUTE: ledger})


def _numbers(sentence: str) -> "list[int]":
    return [int(figure) for figure in re.findall(r"\d+", sentence)]


def test_the_console_painted_the_real_ledger_in_hebrew(scenario, ledger) -> None:
    """The premise. Without it every assertion below would be about a blank screen."""
    assert "failed" not in scenario, scenario.get("failed")
    phase = scenario["phases"]["opened"]
    assert phase["dir"] == "rtl" and phase["lang"] == "he"
    assert phase["chips"] == len(ledger["states"]) + 1, "the legend lost a chip"
    assert phase["rows"] == len(ledger["gates"]), (
        f"the unfiltered table renders {phase['rows']} of {len(ledger['gates'])} gates"
    )


def test_every_chip_opens_exactly_the_rows_it_counts(scenario, ledger) -> None:
    """A number that opens rows opens as many as it says, and never zero of them.

    The count comes off the chip in the browser and the rows are counted in the
    document, so this compares what a person reads with what a person gets.
    """
    assert scenario["chips"], "no chip was pressed"
    for row in scenario["chips"]:
        expected = len(ledger["gates"]) if row["state"] == "all" else ledger["counts"][row["state"]]
        assert row["count"] == expected, f"{row['state']} chip reads {row['count']}, the ledger says {expected}"
        assert row["rows"] == row["count"], (
            f"the {row['state']} chip counts {row['count']} and opens {row['rows']} rows"
        )
        assert row["groups"] <= 2, f"{row['state']} rendered {row['groups']} family groups"


def test_the_subtitle_counts_the_rows_on_screen_and_not_the_payload(scenario, ledger) -> None:
    """The defect, dead: the subtitle moved with the filter under every chip.

    It read "13 of 13" under all five, which is the same figure disagreeing with
    the rows beneath it. Both numbers are asserted, and the fact that the string
    changes at all is asserted too, because a subtitle that never moves is
    exactly what shipped.
    """
    total = len(ledger["gates"])
    seen = set()
    for row in scenario["chips"]:
        figures = _numbers(row["subtitle"])
        assert len(figures) == 2, f"the subtitle stopped carrying two figures: {row['subtitle']}"
        assert figures[0] == row["rows"], (
            f"the {row['state']} chip shows {row['rows']} rows under a subtitle that says {figures[0]}"
        )
        assert figures[1] == total, f"the subtitle's total moved: {row['subtitle']}"
        seen.add(row["subtitle"])
    assert len(seen) > 1, f"the subtitle never changed across five filters: {seen}"


def test_the_chip_with_no_members_opens_an_empty_state_that_names_the_state(scenario, ledger) -> None:
    """The dead end, closed, in the state's own recorded words.

    The zero chip is real on this tree: the ledger records no gate that has not
    been asked. Pressing it used to render nothing at all. It now renders one
    empty state, which names the state, carries the sentence that state means
    from the payload itself, and holds the control back to all of them.
    """
    zero = [row for row in scenario["chips"] if row["count"] == 0]
    assert zero, "no state on this tree is empty, so this measurement would prove nothing"
    for row in zero:
        state = next(entry for entry in ledger["states"] if entry["id"] == row["state"])
        assert row["rows"] == 0 and row["groups"] == 0
        assert row["emptyStates"] == 1, (
            f"the {row['state']} chip opened {row['emptyStates']} empty states, so a number opened a hole"
        )
        assert state["he"] in row["emptyText"], (
            f"the empty state does not name the state that is empty: {row['emptyText']}"
        )
        assert state["meaning_he"] in row["emptyText"], (
            "the empty state names the state and not what that state means"
        )
        assert row["emptyControls"] == 1, (
            f"the empty state carries {row['emptyControls']} ways forward"
        )


def test_no_chip_that_has_members_renders_an_empty_state(scenario) -> None:
    """The other direction: the empty state appears only where it is true."""
    for row in scenario["chips"]:
        if row["count"] > 0:
            assert row["emptyStates"] == 0, f"the {row['state']} chip renders rows and an empty state"


def test_the_empty_state_carries_the_way_back_and_it_works(scenario, ledger) -> None:
    """An empty state whose control does nothing is the dead end one level down."""
    phase = scenario["phases"].get("wayBack")
    assert phase, "the zero chip was never pressed, so the way back was not measured"
    assert phase["control"] is True, "the empty state carries no control"
    assert phase["label"] == "הצגת כל השערים", f"the control's own words moved: {phase['label']}"
    assert phase["rows"] == len(ledger["gates"]), (
        f"pressing the way back left {phase['rows']} rows on screen"
    )
    assert phase["emptyStates"] == 0
    assert phase["pressed"][0] == "true", "the way back did not move the filter to all"


def test_each_chip_reports_its_own_state_to_a_screen_reader(scenario) -> None:
    """The active chip is an attribute, not a colour.

    Measured on the shipped tree: ``aria-pressed`` was null on all five and the
    only mark of the active one was the CSS class ``on``, while the console's
    own rail sets ``aria-current`` on the section it is showing. Exactly one chip
    is pressed at a time, and it is the one that was pressed.
    """
    for index, row in enumerate(scenario["chips"]):
        assert None not in row["pressed"], f"a chip carries no aria-pressed: {row['pressed']}"
        assert row["pressed"].count("true") == 1, (
            f"{row['pressed'].count('true')} chips report themselves pressed at once"
        )
        assert row["pressed"][index] == "true", (
            f"chip {index} was pressed and chip {row['pressed'].index('true')} reports it"
        )


def test_the_filter_answers_inside_the_budget_a_person_feels(scenario) -> None:
    """A filter is a press, so it is measured as one rather than assumed."""
    slowest = max(row["pressMs"] for row in scenario["chips"])
    assert slowest <= PRESS_BUDGET_MS, f"the slowest filter press took {slowest} ms"


def test_the_ledger_the_browser_was_handed_is_the_one_the_route_serves(ledger) -> None:
    """The payload above is the product's, so this pins what it has to carry.

    Every assertion in this file reads the chip counts, the state labels and the
    meanings out of that body. If the route stopped carrying one of them the
    browser measurements would silently become weaker instead of failing.
    """
    assert set(ledger) >= {"gates", "counts", "states", "total", "layers"}
    assert ledger["total"] == len(ledger["gates"])
    assert sum(ledger["counts"].values()) == len(ledger["gates"])
    for state in ledger["states"]:
        assert state["he"] and state["meaning_he"], f"a state lost its words: {state}"
    families = {gate["model"] for gate in ledger["gates"]}
    assert families <= {"retention", "audience"}, (
        f"the ledger grew a family the panel does not name: {sorted(families)}"
    )


def test_the_payload_the_browser_was_handed_names_no_rival_channel(ledger) -> None:
    """The competitor boundary, on the one body this measurement puts on a screen."""
    from kairos_api import model_console_artifacts as artifacts

    payload = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    base = payload.get("base") or {}
    owned = str(base.get("owned_channel") or "")
    rivals = sorted(name for name in (base.get("hist_channel") or {}) if name != owned)
    assert rivals, "the artifact holds no rival channels, so this test would prove nothing"
    body = json.dumps(ledger, ensure_ascii=False)
    for rival in rivals:
        assert rival not in body, f"the gate ledger names the rival channel {rival}"
