"""The model console's door, and the shell it covers, measured in a real browser.

**The defect this file exists to keep dead.** The bridge that mounts the console
asked ``GET /api/auth/me`` exactly once, inside an effect with empty
dependencies, on a React root that is created at boot and never remounts. So on
the most common way into the product, signed out, the bridge was answered 401
before the person had typed anything, latched "not company", and never asked
again. Measured on the running app: after signing in through the form,
``document.getElementById('kairos-model-console-root').innerHTML.length`` stayed
0 and ``document.querySelector('.mc-switcher')`` stayed null for the whole
session, with no message of any kind. A manual page reload produced the switcher
instantly for the same account. The destination existed and had no door.

**The second defect, measured after the first was fixed.** With the console open,
the operator shell underneath it was still fully live: on the live DOM, 61
focusable controls outside the console root, two ``main`` landmarks, and the
first focusable element in the whole document was the shell's own ``סקירה``
control, painted over by a fixed overlay. The first Tab put a keyboard steward on
an invisible operator control. The console is a different shell, and a different
shell that leaves the one underneath it reachable is a lie to anyone who does not
use a mouse.

**Why this is a browser test and not a source assertion.** The first fix is a
``MutationObserver`` on the application root's child list and the second is
``inert`` plus ``aria-hidden`` on that same root. Asserting that the source
contains either proves nothing about whether the browser honours it. So the
bridge is built by the product's own bundler and run in a real Chrome, the
switcher is asserted as a control on screen, and inertness is asserted by asking
each shell control to take focus and measuring that it cannot.

**What is a stand-in here, stated plainly.** The shell is not in this page. The
harness reproduces the one thing the bridge depends on: the application mounts
into ``#root`` and the shell returns the sign-in card INSTEAD of the workspace,
so an auth transition swaps ``#root``'s only child. Its stand-in shell carries
its own nav control and its own ``main`` landmark, so the focus measurement has
something to be wrong about. Both premises live in files this piece does not own,
so they are asserted separately, from source, in
``test_the_frozen_shell_still_makes_the_transition_the_bridge_watches``. If the
shell ever stops making that transition, that test fails and says so, rather than
this one passing against a premise that has quietly stopped being true.

The scenario is one page load, five phases, no reload:

1. Signed out. Nothing renders, and nothing tells anyone the console exists.
2. Signed in as a channel account. Still nothing, because affiliation decides.
3. Signed out, then signed in as a company account. The switcher appears, and
   the time it takes is measured rather than assumed.
4. The console is opened from that switcher. The shell behind it is measured.
5. The console is left by its own control. The shell is measured again, because
   a surface that disables the product and does not give it back is worse than
   the defect it fixed.
"""

from __future__ import annotations

import os

import pytest

from test_p7_console_bridge_harness import (
    BRIDGE,
    FRONTEND,
    build_harness,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

# The switcher's own class and copy, which are the thing a person looks for.
SWITCHER_CLASS = "mc-switcher"
SWITCHER_HE = "קונסולת המודל (חברה)"
SWITCHER_EN = "Model console (company)"

# How long the switcher gets to appear after an authentication. This is the one
# that matters: it is a person waiting.
SWITCHER_BUDGET_MS = 4000

# The stand-in shell's own first control, which is what the first Tab used to
# land on while the console covered it.
SHELL_FIRST_CONTROL = "סקירה"

# The scenario, in the browser. It reproduces the shell's own transition: the
# server starts answering for the new account, and #root's single child is
# replaced. Nothing here reloads, and a reload would restart this script and
# lose the phases already recorded, so a single posted result is itself the
# proof that one page load did all of it.
HARNESS_JS = """
import { mountModelConsole } from '%(bridge)s';

const CONSOLE_NODE = 'kairos-model-console-root';
const FOCUSABLE = 'a[href],button,input,select,textarea,[tabindex]';

const SHELL_CONTROLS = [
  '%(shellControl)s',
  'תוכנית',
];

function switcher() {
  return document.querySelector('.%(switcher)s');
}

function consoleRoot() {
  return document.getElementById(CONSOLE_NODE);
}

function consoleHtmlLength() {
  const node = consoleRoot();
  return node ? node.innerHTML.length : -1;
}

// The stand-in operator shell: a workspace root, one nav control and one
// landmark, which is the smallest thing that can be wrongly reachable.
function shellChild() {
  const node = document.createElement('div');
  node.className = 'kairos-shell rtl';
  node.setAttribute('dir', 'rtl');
  node.setAttribute('lang', 'he');
  const nav = document.createElement('nav');
  SHELL_CONTROLS.forEach((label) => {
    const control = document.createElement('button');
    control.type = 'button';
    control.textContent = label;
    nav.appendChild(control);
  });
  const main = document.createElement('main');
  main.appendChild(document.createElement('input'));
  node.append(nav, main);
  return node;
}

function loginChild() {
  const node = document.createElement('div');
  node.className = 'login-screen';
  return node;
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

async function waitForSwitcher(budgetMs) {
  const started = performance.now();
  for (;;) {
    if (switcher()) return Math.round(performance.now() - started);
    if (performance.now() - started > budgetMs) return null;
    await settle(20);
  }
}

async function waitForLabel(text, budgetMs) {
  const started = performance.now();
  for (;;) {
    const control = switcher();
    if (control && control.textContent === text) return Math.round(performance.now() - started);
    if (performance.now() - started > budgetMs) return null;
    await settle(20);
  }
}

// Whether a control can actually take focus, asked of the browser rather than
// inferred from an attribute. An inert subtree answers no to every one of them.
function takesFocus(node) {
  node.focus();
  return document.activeElement === node;
}

function shellState() {
  const shell = document.getElementById('root');
  const inConsole = (node) => Boolean(consoleRoot() && consoleRoot().contains(node));
  const controls = Array.from(document.querySelectorAll(FOCUSABLE));
  const outside = controls.filter((node) => !inConsole(node));
  const first = controls.find(takesFocus) || null;
  const mains = Array.from(document.querySelectorAll('main'));
  return {
    inert: shell.hasAttribute('inert'),
    ariaHidden: shell.getAttribute('aria-hidden'),
    outsideControls: outside.length,
    outsideTakingFocus: outside.filter(takesFocus).length,
    firstFocusLabel: first ? first.textContent.trim() : '',
    firstFocusInConsole: Boolean(first && inConsole(first)),
    mainsInDom: mains.length,
    mainsExposed: mains.filter((node) => !node.closest('[aria-hidden="true"]')).length,
  };
}

async function authenticateAs(mode, child) {
  await fetch('/testctl/session?as=' + mode);
  document.getElementById('root').replaceChildren(child);
}

async function run() {
  const result = { navigationType: '', phases: {} };
  const nav = performance.getEntriesByType('navigation');
  result.navigationType = nav.length ? nav[0].type : 'unknown';

  mountModelConsole();
  await settle(700);
  result.phases.signedOut = {
    switcher: Boolean(switcher()),
    consoleHtmlLength: consoleHtmlLength(),
  };

  await authenticateAs('channel', shellChild());
  result.phases.channel = {
    switcherMs: await waitForSwitcher(1200),
    consoleHtmlLength: consoleHtmlLength(),
  };

  await authenticateAs('out', loginChild());
  await settle(300);
  await authenticateAs('company', shellChild());
  const companyMs = await waitForSwitcher(%(budget)d);
  const control = switcher();
  result.phases.company = {
    switcherMs: companyMs,
    consoleHtmlLength: consoleHtmlLength(),
    label: control ? control.textContent : '',
    tag: control ? control.tagName : '',
  };

  // The shell settles its language onto the workspace root after that root has
  // already appeared, because the language comes from the settings read. So the
  // switcher has to follow the attribute, not read it once.
  const shell = document.querySelector('.kairos-shell');
  if (shell) shell.setAttribute('lang', 'en');
  result.phases.language = {
    englishMs: await waitForLabel('%(english)s', 1500),
    label: switcher() ? switcher().textContent : '',
  };

  // The console, opened the way a person opens it, and the shell underneath it.
  result.phases.beforeOpen = shellState();
  switcher().click();
  const opened = await waitFor(() => document.querySelector('.mc-console'), %(budget)d);
  result.phases.consoleOpen = { opened: Boolean(opened), ...shellState() };

  // And left the way a person leaves it, which has to give the shell back.
  const back = await waitFor(() => document.querySelector('.mc-header .mc-button'), %(budget)d);
  result.phases.afterClose = { backControl: Boolean(back) };
  if (back) {
    back.click();
    result.phases.afterClose.switcherMs = await waitForSwitcher(%(budget)d);
    Object.assign(result.phases.afterClose, shellState());
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
def scenario(tmp_path_factory) -> "dict":
    skip_unless_a_real_browser_is_available()
    work = tmp_path_factory.mktemp("p7-bridge")
    harness_js = HARNESS_JS % {
        "bridge": os.path.relpath(BRIDGE, work.resolve() / "src"),
        "switcher": SWITCHER_CLASS,
        "budget": SWITCHER_BUDGET_MS,
        "english": SWITCHER_EN,
        "shellControl": SHELL_FIRST_CONTROL,
    }
    return run_scenario(build_harness(work, harness_js), work)


def test_the_scenario_ran_in_one_page_load(scenario) -> None:
    """No reload, which is the whole point: a reload always fixed this defect."""
    assert "failed" not in scenario, scenario.get("failed")
    assert scenario["navigationType"] == "navigate", (
        f"the page was loaded as {scenario['navigationType']}, so a reload could explain the result"
    )


def test_signed_out_the_console_shows_nothing_at_all(scenario) -> None:
    """Absence, not a disabled control and not a message.

    A control that says the console exists would tell an account with no right
    to it that the other side of the line is there.
    """
    phase = scenario["phases"]["signedOut"]
    assert phase["switcher"] is False
    assert phase["consoleHtmlLength"] == 0


def test_a_channel_account_that_signs_in_still_sees_nothing(scenario) -> None:
    """The re-ask must not have turned into a way in.

    This is the failure direction that matters: the fix makes the bridge ask
    again on every auth transition, so a channel account's transition is asked
    about too, and the answer has to keep being no.
    """
    phase = scenario["phases"]["channel"]
    assert phase["switcherMs"] is None, "a channel account was given the console's door"
    assert phase["consoleHtmlLength"] == 0


def test_a_company_account_that_signs_in_gets_the_door_without_a_reload(scenario) -> None:
    """The defect, dead. Measured from the transition to the control on screen."""
    phase = scenario["phases"]["company"]
    assert phase["switcherMs"] is not None, (
        "the switcher never appeared after signing in, which is the shipped defect"
    )
    assert phase["switcherMs"] <= SWITCHER_BUDGET_MS
    assert phase["consoleHtmlLength"] > 0
    assert phase["tag"] == "BUTTON"
    assert phase["label"] == SWITCHER_HE, f"the switcher's own words moved: {phase['label']}"


def test_the_switcher_follows_the_language_the_shell_settles_on(scenario) -> None:
    """One defect, two symptoms, and this is the second one.

    The shell writes its language onto the workspace root after that root has
    already rendered, because it comes from the settings read. Measured on the
    running app before the fix: a shell in English kept a switcher in Hebrew for
    the whole session, for the same reason the switcher was missing entirely,
    which is a value read once and never asked about again.
    """
    phase = scenario["phases"]["language"]
    assert phase["englishMs"] is not None, (
        f"the switcher kept its old language after the shell changed: {phase['label']}"
    )
    assert phase["label"] == SWITCHER_EN


def test_the_shell_is_reachable_until_the_console_covers_it(scenario) -> None:
    """The baseline, so the next test measures a change and not a constant.

    Without this, a harness whose stand-in shell had no controls at all would
    make the inertness assertion pass by having nothing to assert about.
    """
    phase = scenario["phases"]["beforeOpen"]
    assert phase["inert"] is False
    assert phase["ariaHidden"] is None
    assert phase["outsideControls"] >= 3, "the stand-in shell carries no controls to be wrong about"
    assert phase["outsideTakingFocus"] == phase["outsideControls"]
    assert phase["firstFocusLabel"] == SHELL_FIRST_CONTROL
    assert phase["firstFocusInConsole"] is False
    assert phase["mainsExposed"] == 1


def test_the_operator_shell_is_out_of_reach_while_the_console_covers_it(scenario) -> None:
    """The second defect, dead, asked of the browser rather than of the source.

    Every shell control is still in the document, which is the honest state of a
    thing that is covered rather than unmounted, and not one of them can take
    focus. The first focus in the whole document is inside the console, so the
    first Tab lands where the reader is looking.
    """
    phase = scenario["phases"]["consoleOpen"]
    assert phase["opened"] is True, "the console never rendered, so nothing here was measured"
    assert phase["inert"] is True, "the operator shell is still in the tab order under the console"
    assert phase["ariaHidden"] == "true", "the operator shell is still in the accessibility tree"
    assert phase["outsideControls"] >= 3, "the shell was unmounted, which is not what was asked for"
    assert phase["outsideTakingFocus"] == 0, (
        f"{phase['outsideTakingFocus']} operator controls under the console can still take focus"
    )
    assert phase["firstFocusInConsole"] is True, (
        f"the first focus in the document is outside the console, on {phase['firstFocusLabel']}"
    )
    assert phase["mainsInDom"] == 2, "the console and the shell no longer both carry a landmark"
    assert phase["mainsExposed"] == 1, "two landmarks are exposed at once, so neither names the page"


def test_leaving_the_console_gives_the_operator_shell_back(scenario) -> None:
    """A console that disables the product and keeps it disabled is worse.

    The way out is the console's own control, and the attributes are removed
    rather than left behind, so the shell is exactly as reachable as it was
    before the console was ever opened.
    """
    phase = scenario["phases"]["afterClose"]
    assert phase["backControl"] is True, "the console rendered no way out to press"
    assert phase["switcherMs"] is not None, "the console never closed"
    assert phase["inert"] is False, "the operator shell was left inert after the console closed"
    assert phase["ariaHidden"] is None, "the operator shell was left hidden after the console closed"
    assert phase["outsideTakingFocus"] == phase["outsideControls"]
    assert phase["firstFocusLabel"] == SHELL_FIRST_CONTROL


def test_the_frozen_shell_still_makes_the_transition_the_bridge_watches() -> None:
    """The harness's premise, asserted against the two frozen files that hold it.

    The bridge watches ``#root``'s child list because the shell returns an auth
    screen INSTEAD of the workspace. Both halves of that sentence live in files
    this piece may not write, so both are pinned here. A shell that starts
    rendering the sign-in card inside a permanent wrapper would break the fix
    silently; this test is what makes that noisy instead.
    """
    entry = (FRONTEND / "src" / "index.jsx").read_text(encoding="utf-8")
    assert "document.getElementById('root')" in entry, (
        "the application no longer mounts into #root; the bridge watches the wrong node"
    )
    markup = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert 'id="root"' in markup

    shell = (FRONTEND / "src" / "shell" / "TVBreakDashboard.jsx").read_text(encoding="utf-8")
    assert "if (authScreen) {" in shell and "return authScreen;" in shell, (
        "the shell no longer returns the auth screen instead of the workspace"
    )
    assert "className={`kairos-shell" in shell, "the workspace root is no longer .kairos-shell"

    bridge = BRIDGE.read_text(encoding="utf-8")
    assert "new MutationObserver" in bridge and "childList: true" in bridge, (
        "the bridge no longer watches for the transition it was fixed to watch"
    )
