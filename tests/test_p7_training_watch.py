"""The training panel follows the run it started, measured in a real browser.

**The defect this file exists to keep dead.** A blind measurement of the shipped
tree, twice: the steward pressed train, the panel read ``/api/model/training``
once 41 ms later, got ``state: running``, and never asked again. The server
recorded ``finished_at`` 7.05 s after ``started_at`` with ``exit_code: 0``, and
179 s later the screen still read the word for training, with that trainer's own
control still disabled, so the steward could not start another run either.
Leaving the section and returning read the truth, which proved the store was
right and only the screen was wrong. The console stated a false state during the
job story it exists for.

**Why this is a browser test.** A run ends on the server with no click behind it,
so the whole defect lives in the gap between two reads that nobody triggers.
Nothing about the source of a component can show whether the screen learns
about an event that happens while it is idle. So the real bridge is built with
the product's own bundler, run in a real headless Chrome, handed the bodies the
route really serves, and asked what is on screen after doing nothing at all.

**What is real here and what is a stand-in, stated plainly.** The three bodies
are real and they come from one real training run: the audience trainer is
started through ``model_console_training.start`` into a temporary releases
store, and ``payload()`` is taken before it, while it runs and after it ends. So
the browser is handed the product's own words for its own run, including the
duration the run really took and the exit code it really returned. The shell
around the console is the same stand-in the other two browser measurements use,
for the same reason and with the same premise asserted separately from source.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import pytest

from test_p7_console_bridge_harness import (
    BRIDGE,
    build_harness,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

ROOT = Path(__file__).resolve().parents[1]
WORDS = ROOT / "tv-break-dashboard" / "src" / "model" / "console" / "console-words.js"

# The artifact the measurement trains, and the shipped file it must not touch.
ARTIFACT = "audience"
SHIPPED = ROOT / "models" / "audience_model.json"

# The route the panel reads, and the rail position that opens it. The position
# is checked against the route the panel prints for itself, so a reordered rail
# fails loudly instead of measuring the wrong section.
TRAINING_ROUTE = "/api/model/training"
TRAINING_RAIL_INDEX = 4

# How long a real run gets before this measurement gives up on it. Measured on
# this tree: the audience trainer takes about 11 s.
RUN_BUDGET_S = 240

# The three browser budgets. The middle one is the defect: how long the screen
# gets to learn, with nobody touching it, that the run it started has ended.
OPEN_BUDGET_MS = 4000
FINISH_BUDGET_MS = 12000
SETTLE_MS = 5000

HARNESS_JS = """
import { mountModelConsole } from '%(bridge)s';

const TRAINING = '%(route)s';
const OPEN = %(open)d;
const FINISH = %(finish)d;
const SETTLE = %(settle)d;
const RAIL = %(rail)d;
const TRAINER = %(trainer)d;

// Every press this scenario makes, counted. The claim under test is that the
// screen moves with no press between the start and the end, so the presses are
// counted rather than asserted about in prose.
let clicks = 0;

function press(node) {
  clicks += 1;
  node.click();
}

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
  node.append(document.createElement('main'));
  return node;
}

function text(node) {
  return node ? node.textContent.trim() : '';
}

// How many times the browser really asked for the route, read from the
// browser's own resource timeline rather than from anything the page counts.
function reads() {
  return performance.getEntriesByType('resource')
    .filter((entry) => entry.name.indexOf(TRAINING) >= 0).length;
}

// Each run row, with the state the panel believes and the word a person reads.
// The state comes off the row's own class, so the two can be compared and a row
// that says one thing in its markup and another in its words fails.
function runRows() {
  return Array.from(document.querySelectorAll('.mc-run')).map((row) => {
    const token = Array.from(row.classList).find((name) => name.indexOf('mc-run-') === 0);
    return {
      id: text(row.querySelector('.mc-run-id')),
      state: token ? token.slice('mc-run-'.length) : '',
      verdict: text(row.querySelector('.mc-verdict')),
    };
  });
}

function trainerRows() {
  return Array.from(document.querySelectorAll('.mc-trainer')).map((node) => {
    const control = node.querySelector('button');
    return {
      label: text(node.querySelector('strong')),
      control: text(control),
      disabled: Boolean(control && control.disabled),
    };
  });
}

function snapshot() {
  return {
    reads: reads(),
    clicks,
    rows: runRows(),
    trainers: trainerRows(),
    note: text(document.querySelector('.mc-run-watch')),
    route: text(document.querySelector('.mc-route code')),
  };
}

function anyRunning() {
  return runRows().some((row) => row.state === 'running');
}

async function run() {
  const result = { phases: {} };

  mountModelConsole();
  await fetch('/testctl/session?as=company');
  document.getElementById('root').replaceChildren(shellChild());
  const switcher = await waitFor(() => document.querySelector('.mc-switcher'), OPEN);
  if (!switcher) throw new Error('the switcher never appeared, so nothing was measured');
  press(switcher);

  const view = await waitFor(() => document.querySelector('.mc-console'), OPEN);
  if (!view) throw new Error('the console never rendered');
  const rail = await waitFor(() => document.querySelectorAll('.mc-rail-item')[RAIL], OPEN);
  if (!rail) throw new Error('the rail never rendered the training section');
  press(rail);
  const opened = await waitFor(() => document.querySelector('.mc-trainer'), OPEN);
  if (!opened) throw new Error('the training section never rendered a trainer');
  result.phases.opened = snapshot();

  const trainer = document.querySelectorAll('.mc-trainer')[TRAINER];
  if (!trainer) throw new Error('the trainer this measurement starts is not on screen');
  press(trainer.querySelector('button'));

  const appeared = await waitFor(anyRunning, OPEN);
  result.phases.started = { ...snapshot(), appeared: Boolean(appeared) };

  // The measurement itself. Nothing is pressed from here until the screen has
  // either learned that the run ended or run out of time.
  const from = performance.now();
  const ended = await waitFor(() => runRows().length > 0 && !anyRunning(), FINISH);
  result.phases.finished = {
    ...snapshot(),
    reached: Boolean(ended),
    waitedMs: Math.round(performance.now() - from),
  };

  // And the other half: the reads stop when no run is open. Three watch
  // intervals of doing nothing, then the same counts again.
  await settle(SETTLE);
  result.phases.settled = { ...snapshot(), afterMs: SETTLE };
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


def _word(key: str, locale: str = "he") -> str:
    """The product's own word for a key, read from the module that defines it."""
    pattern = rf"'{re.escape(key)}':\s*\{{[^}}]*\b{locale}: '([^']+)'"
    found = re.search(pattern, WORDS.read_text(encoding="utf-8"))
    assert found, f"the console defines no {locale} word for {key}"
    return found.group(1)


@pytest.fixture(scope="module")
def bodies(tmp_path_factory) -> "dict":
    """One real training run, and the three payloads the route serves around it."""
    from kairos_api import model_console_training as training

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path_factory.mktemp("releases")))
        shipped_before = hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
        before = training.payload()
        record = training.start(ARTIFACT, actor="steward1")
        running = training.payload()
        deadline = time.time() + RUN_BUDGET_S
        while training.in_flight() and time.time() < deadline:
            time.sleep(0.2)
        finished = training.payload()
        yield {
            "before": before,
            "running": running,
            "finished": finished,
            "record": record,
            "trainer_index": [row["artifact"] for row in before["trainers"]].index(ARTIFACT),
            "shipped_before": shipped_before,
            "shipped_after": hashlib.sha256(SHIPPED.read_bytes()).hexdigest(),
        }


@pytest.fixture(scope="module")
def scenario(tmp_path_factory, bodies) -> "dict":
    """The browser, handed those three bodies in the order the route would serve them.

    The running body is served twice on purpose. One repeat proves the panel
    keeps asking rather than getting lucky with a single retry, and it makes the
    measurement independent of exactly which read follows the write.
    """
    skip_unless_a_real_browser_is_available()
    if bodies["finished"]["in_flight"]:
        pytest.fail("the run never ended, so there is nothing for the screen to learn")
    work = tmp_path_factory.mktemp("p7-training")
    harness_js = HARNESS_JS % {
        "bridge": os.path.relpath(BRIDGE, work.resolve() / "src"),
        "route": TRAINING_ROUTE,
        "open": OPEN_BUDGET_MS,
        "finish": FINISH_BUDGET_MS,
        "settle": SETTLE_MS,
        "rail": TRAINING_RAIL_INDEX,
        "trainer": bodies["trainer_index"],
    }
    dist = build_harness(work, harness_js)
    sequence = [bodies["before"], bodies["running"], bodies["running"], bodies["finished"]]
    return run_scenario(dist, work, {TRAINING_ROUTE: sequence},
                        {TRAINING_ROUTE: bodies["record"]})


def _terminal_word(bodies: "dict") -> str:
    """The word the screen must end on, taken from the run's own recorded state."""
    state = bodies["finished"]["runs"][0]["state"]
    return _word("training.done") if state == "done" else _word("training.failed")


# ---------------------------------------------------------------------------
# The run the browser was handed, before anything about the screen
# ---------------------------------------------------------------------------


def test_the_run_the_browser_was_handed_really_started_and_really_ended(bodies) -> None:
    """The premise. Without it the measurement below would be about a fixture."""
    assert bodies["running"]["in_flight"].get(ARTIFACT), (
        "the run was not in flight after it was started, so nothing was measured"
    )
    started = bodies["running"]["runs"][0]
    assert started["state"] == "running"
    assert not bodies["finished"]["in_flight"], (
        f"the run never left the flight register: {bodies['finished']['in_flight']}"
    )
    ended = bodies["finished"]["runs"][0]
    assert ended["run_id"] == started["run_id"]
    assert ended["state"] in {"done", "failed"}, f"the run ended in {ended['state']}"
    assert ended["duration_seconds"] > 0 and ended["finished_at"], (
        "the store recorded no duration, so the run's end was never recorded either"
    )
    assert ended.get("summary_error") is None, ended.get("summary_error")


def test_the_run_wrote_into_the_releases_store_and_not_over_the_shipped_artifact(bodies) -> None:
    """The training law, re-measured on a real run rather than asserted again.

    The shipped artifact is hashed either way, because a run that failed may
    still have written something on its way there.
    """
    ended = bodies["finished"]["runs"][0]
    assert ended["writes_shipped_artifact"] is False
    assert bodies["shipped_after"] == bodies["shipped_before"], (
        "the shipped audience artifact moved while a training run was measured"
    )
    produced = ended.get("produced") or {}
    if ended["state"] != "done":
        assert not produced, f"a run that ended in {ended['state']} claims an artifact"
        return
    assert produced.get("sha256"), "the run ended done and produced no artifact"
    assert "training_runs" in produced["path"], (
        f"the run wrote outside the releases store: {produced['path']}"
    )
    assert produced["sha256"] != bodies["shipped_after"], (
        "the run reports the shipped artifact as its own output"
    )


# ---------------------------------------------------------------------------
# The screen, in a real browser
# ---------------------------------------------------------------------------


def test_the_training_section_opened_on_its_own_route_with_nothing_running(scenario) -> None:
    """The premise of every assertion below: the right panel, in its rest state."""
    assert "failed" not in scenario, scenario.get("failed")
    phase = scenario["phases"]["opened"]
    assert phase["route"] == TRAINING_ROUTE, (
        f"the rail position opened {phase['route']} instead of the training section"
    )
    assert phase["rows"] == [], f"the panel already showed a run: {phase['rows']}"
    assert phase["note"] == "", "the panel says it is following a run before one was started"
    assert [row["disabled"] for row in phase["trainers"]] == [False, False], (
        "a trainer was disabled before anything was started"
    )


def test_starting_a_run_puts_it_on_the_screen_as_running(scenario, bodies) -> None:
    """One row, in the running state, in the product's own word for it."""
    phase = scenario["phases"]["started"]
    assert phase["appeared"] is True, "the started run never appeared on the screen"
    assert len(phase["rows"]) == 1, f"{len(phase['rows'])} run rows for one run"
    row = phase["rows"][0]
    assert row["state"] == "running"
    assert row["verdict"] == _word("training.running"), (
        f"the running row reads {row['verdict']}"
    )
    assert row["id"] == bodies["record"]["run_id"], "the row is about another run"
    assert phase["trainers"][bodies["trainer_index"]]["disabled"] is True, (
        "the trainer that is running can be started again"
    )
    assert phase["note"] == _word("training.watching"), (
        f"the panel does not say it is following the run: {phase['note']}"
    )


def test_the_panel_learns_the_run_ended_with_no_press_at_all(scenario, bodies) -> None:
    """The defect, dead.

    Between the start and this snapshot the scenario presses nothing, so every
    change on the screen came from the panel asking again. The word asserted is
    the one the run's own recorded state earns, not a word chosen here.
    """
    started = scenario["phases"]["started"]
    finished = scenario["phases"]["finished"]
    assert finished["reached"] is True, (
        f"the screen still read {started['rows'][0]['verdict']} after {finished['waitedMs']} ms"
    )
    assert finished["clicks"] == started["clicks"], (
        f"{finished['clicks'] - started['clicks']} presses happened during the measurement"
    )
    row = finished["rows"][0]
    assert row["id"] == bodies["record"]["run_id"]
    assert row["state"] == bodies["finished"]["runs"][0]["state"]
    assert row["verdict"] == _terminal_word(bodies), f"the ended row reads {row['verdict']}"
    assert finished["waitedMs"] <= FINISH_BUDGET_MS
    assert finished["reads"] > started["reads"], (
        "the screen changed without the route being read, which cannot be true"
    )


def test_the_trainer_can_be_started_again_the_moment_the_run_ends(scenario, bodies) -> None:
    """The second half of the same defect, and the one that blocks the next run."""
    finished = scenario["phases"]["finished"]
    trainer = finished["trainers"][bodies["trainer_index"]]
    assert trainer["disabled"] is False, "the trainer is still disabled after its run ended"
    assert trainer["control"] == _word("training.start"), (
        f"the control still reads {trainer['control']}"
    )
    assert finished["note"] == "", (
        f"the panel still says it is following a run that ended: {finished['note']}"
    )


def test_the_panel_stops_reading_the_route_once_no_run_is_open(scenario) -> None:
    """A watch that never ends is a second defect wearing the first one's fix."""
    finished = scenario["phases"]["finished"]
    settled = scenario["phases"]["settled"]
    assert settled["reads"] == finished["reads"], (
        f"the panel read the route {settled['reads'] - finished['reads']} more times"
        f" in {settled['afterMs']} ms with no run open"
    )
    assert settled["clicks"] == finished["clicks"]
    assert settled["rows"] == finished["rows"], "the screen moved after the run had ended"


def test_the_browser_and_the_server_agree_on_how_many_calls_there_were(scenario) -> None:
    """Two independent counts of the same thing, because one of them is the page's.

    The browser's timeline carries no method, so the write is in its number and
    the server's two counters are added to meet it.
    """
    settled = scenario["phases"]["settled"]
    served = scenario["server_reads"]
    reads = served[TRAINING_ROUTE]
    writes = served[f"POST {TRAINING_ROUTE}"]
    assert writes == 1, f"the scenario wrote to the route {writes} times"
    assert reads + writes == settled["reads"], (
        f"the server answered {reads} reads and one write, and the browser"
        f" recorded {settled['reads']} calls"
    )
    assert reads >= 4, (
        f"the route was read {reads} times, which is fewer than opening the section,"
        " the read after the write and two watch reads"
    )
