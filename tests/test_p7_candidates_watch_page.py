"""The page the candidate-shelf measurement runs, and the constants it needs.

Split out of ``test_p7_candidates_watch.py`` when that file reached the 450-line
law, and nothing changed on the way out: the same script, the same selectors, the
same budgets. It defines no test of its own. The scenario and every assertion
stayed in the file that owns them.

The script does four things and reports what it saw after each: open the shelf,
press one candidate's measurement control, wait with zero presses until the
screen either learns the measurement ended or runs out of time, then wait again
to prove the reads stop. Every count it reports is taken from the browser itself,
so the scenario can compare it against the server's own count of the same calls.

The press is a flag rather than a fixed step, because the defect was measured on
the entry that has no press in it: a steward opening the shelf while a
measurement somebody else started is already running. With the flag off the
script opens the shelf and touches nothing, and the phases mean the same thing,
so both entries are measured by one page and read by one set of selectors.
"""

from __future__ import annotations

# The shelf's own route and the rail position that opens it. The position is
# checked against the route the panel prints for itself, so a reordered rail
# fails loudly instead of measuring the wrong section.
LIST_ROUTE = "/api/model/candidates"
DETAIL_PREFIX = "/api/model/candidates/"
CANDIDATES_RAIL_INDEX = 3

# The candidate this measurement measures.
SUBJECT = "afterwindow"

# The three browser budgets. The middle one is the defect: how long the screen
# gets to learn, with nobody touching it, that the measurement it started ended.
OPEN_BUDGET_MS = 4000
FINISH_BUDGET_MS = 12000
SETTLE_MS = 5000

HARNESS_JS = """
import { mountModelConsole } from '%(bridge)s';

const LIST = '%(route)s';
const SUBJECT = '%(subject)s';
const RAIL = %(rail)d;
const OPEN = %(open)d;
const FINISH = %(finish)d;
const SETTLE = %(settle)d;
const PRESS = %(press)d;

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

// How many times the browser really asked for the shelf, read from the browser's
// own resource timeline. Every per-candidate read carries the shelf's path as a
// prefix, so the pathname is compared whole rather than searched for, and the
// write to one candidate is excluded by the same comparison.
function reads() {
  return performance.getEntriesByType('resource')
    .filter((entry) => new URL(entry.name).pathname === LIST).length;
}

// Each card, with the money state the panel believes and the figures a person
// reads. The state comes off the card's own markup rather than from anything
// this script remembers, so a card that says one thing in its classes and
// another in its words fails.
function cards() {
  return Array.from(document.querySelectorAll('.mc-candidate')).map((node) => ({
    id: text(node.querySelector('.mc-candidate-head strong')),
    measuring: Boolean(node.querySelector('.mc-measuring')),
    measuringWords: text(node.querySelector('.mc-measuring')),
    pastRuns: text(node.querySelector('.mc-measuring small')),
    stale: Boolean(node.querySelector('.mc-money-block.mc-stale')),
    block: Boolean(node.querySelector('.mc-money-block')),
    moneyRows: node.querySelectorAll('.mc-money-block .mc-money').length,
    owned: text(node.querySelector('.mc-money-block .mc-money')),
    scope: text(node.querySelector('.mc-money-block .mc-money-scope small')),
    controls: Array.from(node.querySelectorAll('.mc-candidate-grid .mc-button')).map(text),
  }));
}

function snapshot() {
  return {
    reads: reads(),
    clicks,
    cards: cards(),
    watch: text(document.querySelector('.mc-candidate-watch')),
    route: text(document.querySelector('.mc-route code')),
  };
}

function anyMeasuring() {
  return cards().some((card) => card.measuring);
}

function subjectNode() {
  return Array.from(document.querySelectorAll('.mc-candidate'))
    .find((node) => text(node.querySelector('.mc-candidate-head strong')) === SUBJECT) || null;
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
  if (!rail) throw new Error('the rail never rendered the candidates section');
  press(rail);
  const opened = await waitFor(() => document.querySelector('.mc-candidate'), OPEN);
  if (!opened) throw new Error('the candidates section never rendered a candidate');
  result.phases.opened = snapshot();

  if (PRESS) {
    const node = subjectNode();
    if (!node) throw new Error('the candidate this measurement starts is not on screen');
    const control = node.querySelector('.mc-candidate-grid .mc-button');
    if (!control) throw new Error('the candidate offers no measurement control to press');
    press(control);
  }

  const appeared = await waitFor(anyMeasuring, OPEN);
  result.phases.started = { ...snapshot(), appeared: Boolean(appeared) };

  // The measurement itself. Nothing is pressed from here until the screen has
  // either learned that the measurement ended or run out of time.
  const from = performance.now();
  const ended = await waitFor(() => cards().length > 0 && !anyMeasuring(), FINISH);
  result.phases.finished = {
    ...snapshot(),
    reached: Boolean(ended),
    waitedMs: Math.round(performance.now() - from),
  };

  // And the other half: the reads stop when nothing is open. Three watch
  // intervals of doing nothing, then the same counts again.
  await settle(SETTLE);
  result.phases.settled = {
    ...snapshot(),
    afterMs: SETTLE,
    consoleText: text(document.querySelector('.mc-console')),
  };
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


def page_script(bridge: str, press: bool = True) -> str:
    """The script, with the bridge import path and the budgets already in it."""
    return HARNESS_JS % {
        "bridge": bridge,
        "route": LIST_ROUTE,
        "subject": SUBJECT,
        "rail": CANDIDATES_RAIL_INDEX,
        "open": OPEN_BUDGET_MS,
        "finish": FINISH_BUDGET_MS,
        "settle": SETTLE_MS,
        "press": 1 if press else 0,
    }
