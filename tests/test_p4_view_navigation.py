"""P4: a navigation entry beats the view this destination stored itself.

Three navigation entries mount the Clients destination, each naming the view it
opens on, and every view tab click stores the view in the query string so a
supplied address resolves. Those two facts collided. ``initialView`` read the
stored view before the view the entry asked for, and because the three entries
mount three different components the state was rebuilt from that stored value on
every navigation. Measured on the built app: after one tab click, קמפיינים,
מפרסמים and סוכנויות all left the previous panel on screen while the side rail
and the breadcrumb moved, so the chrome named a place the operator was not in.
Three of the destination's own entries were dead and a reload was the only
recovery.

This file runs the shipped helper. The module is copied verbatim into a
temporary ``.mjs`` file so node parses it as an ES module, its one import is
resolved to a stub, and the browser's ``window`` is replaced with a location and
a history that behave like the real ones. The sequence is the one the workspace
performs: ``initialView`` once per mount and ``writeParams`` on every settled
view.

The last test mutates the shipped rule back to what it was and asserts this
sequence fails, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tv-break-dashboard" / "src" / "clients" / "clients-money-helpers.js"
WORKSPACE = ROOT / "tv-break-dashboard" / "src" / "clients" / "ClientsWorkspace.jsx"

# The guard this file exists to hold. Removing it is exactly the rule that
# shipped before, so it is the mutation the last test applies.
GUARD = "  if (VIEWS.includes(fromUrl) && fromUrl !== writtenView) {"
WITHOUT_GUARD = "  if (VIEWS.includes(fromUrl)) {"

STUB = """
export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}
"""

HARNESS = """
import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';

const HELPERS = pathToFileURL(process.argv[2]).href;
const STUB = pathToFileURL(process.argv[3]).href;

// The helper imports the shell's text formatter, which is another piece's file
// and is not under test here.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('shell/format')) {
      return { url: STUB, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

// A page load. Module state resets with it, which is what makes an address
// supplied in the bar authoritative again after a reload.
let session = 0;
async function open(url) {
  session += 1;
  const parsed = new URL(url, 'http://127.0.0.1');
  const location = { pathname: parsed.pathname, search: parsed.search, hash: parsed.hash };
  globalThis.window = {
    location,
    history: {
      // Both verbs, mirroring the browser: the write engine pushes when a
      // place changes (Back walks places) and replaces only for arrival
      // normalization. These tests assert ADDRESSES, which push and replace
      // shape identically, so the sequences below hold for both verbs.
      pushState(state, title, next) {
        const now = new URL(next, 'http://127.0.0.1');
        location.pathname = now.pathname;
        location.search = now.search;
        location.hash = now.hash;
      },
      replaceState(state, title, next) {
        const now = new URL(next, 'http://127.0.0.1');
        location.pathname = now.pathname;
        location.search = now.search;
        location.hash = now.hash;
      },
    },
  };
  const helpers = await import(`${HELPERS}?session=${session}`);
  // What the workspace does on mount, and what it does when a view settles.
  const mount = (requested) => {
    const active = helpers.initialView(requested);
    helpers.writeParams({ [helpers.VIEW_PARAM]: active, client: '' });
    return { active, url: location.search + location.hash };
  };
  const tab = (view) => {
    helpers.writeParams({ [helpers.VIEW_PARAM]: view, client: '' });
    return { active: view, url: location.search + location.hash };
  };
  return { mount, tab };
}

const steps = [];

const first = await open('/#Campaigns');
steps.push(['A entry campaigns', first.mount('campaigns')]);
steps.push(['B tab agencies', first.tab('agencies')]);
steps.push(['C entry advertisers', first.mount('advertisers')]);
steps.push(['D entry campaigns', first.mount('campaigns')]);
steps.push(['E tab money', first.tab('money')]);
steps.push(['F entry agencies', first.mount('agencies')]);

const second = await open('/?clients=money#Campaigns');
steps.push(['G supplied address', second.mount('campaigns')]);
steps.push(['H entry advertisers', second.mount('advertisers')]);

process.stdout.write(JSON.stringify(steps));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped helper cannot be executed here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the helper's import cannot be stubbed")
    return found


def _run(tmp_path: Path, source: str) -> dict[str, dict[str, str]]:
    """Run the navigation sequence against one version of the helper."""
    module = tmp_path / "helpers.mjs"
    module.write_text(source, encoding="utf-8")
    stub = tmp_path / "format.mjs"
    stub.write_text(STUB, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run(
        [
            _node(),
            # the shell moved bidi.jsx and dates.js under src/shell; this hook
            # resolves both to the real modules so the harness under test can import them.
            "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
            str(harness), str(module), str(stub),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {label: state for label, state in json.loads(result.stdout)}


@pytest.fixture(scope="module")
def shipped() -> str:
    source = HELPERS.read_text(encoding="utf-8")
    assert GUARD in source, "the rule under test is not in the shipped helper any more"
    return source


def test_every_navigation_entry_opens_its_own_view_after_a_tab_click(tmp_path, shipped):
    """The measured defect, as the sequence that produced it."""
    steps = _run(tmp_path, shipped)
    assert steps["A entry campaigns"]["active"] == "campaigns"
    assert steps["B tab agencies"]["active"] == "agencies"
    assert steps["C entry advertisers"]["active"] == "advertisers", "the entry the operator pressed was discarded"
    assert steps["D entry campaigns"]["active"] == "campaigns"
    assert steps["E tab money"]["active"] == "money"
    assert steps["F entry agencies"]["active"] == "agencies"


def test_the_stored_view_follows_the_entry_so_the_address_stays_true(tmp_path, shipped):
    """The address bar names the view on screen, whichever way it was reached."""
    steps = _run(tmp_path, shipped)
    assert steps["C entry advertisers"]["url"] == "?clients=advertisers#Campaigns"
    assert steps["D entry campaigns"]["url"] == "?clients=campaigns#Campaigns"
    assert steps["F entry agencies"]["url"] == "?clients=agencies#Campaigns"


def test_a_supplied_address_still_opens_the_view_it_names(tmp_path, shipped):
    """The bookmark case, which is why the stored view is read at all."""
    steps = _run(tmp_path, shipped)
    assert steps["G supplied address"]["active"] == "money", "a supplied ?clients=money must win over the entry"
    assert steps["H entry advertisers"]["active"] == "advertisers", "and the next entry must win over it"


def test_the_rule_that_shipped_before_fails_this_sequence(tmp_path, shipped):
    """Proof that the three tests above bite. Remove the guard and C goes dead."""
    mutant = shipped.replace(GUARD, WITHOUT_GUARD)
    assert mutant != shipped
    steps = _run(tmp_path, mutant)
    assert steps["C entry advertisers"]["active"] == "agencies"
    assert steps["D entry campaigns"]["active"] == "agencies"
    assert steps["H entry advertisers"]["active"] == "money"


def test_the_workspace_resets_its_view_when_the_entry_changes_under_it():
    """The other half of the fix, which needs React and is pinned in the source.

    The three entries mount three components today, so the reset that matters is
    the one at mount, which the tests above execute. This assertion holds the
    live case shut too: one component kept mounted while the entry prop changes.
    """
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "const opened = useRef(view);" in source
    assert "if (opened.current === view) {" in source, "the mount run must be skipped or a supplied address is lost"
    assert "setActive(requestedView(view));" in source
    assert "}, [view]);" in source


def test_commercial_data_load_does_not_repeat_when_only_locale_changes():
    """The four locale-neutral reads rerun only for an explicit refresh/reload."""
    source = WORKSPACE.read_text(encoding="utf-8")
    effects = re.findall(
        r"  useEffect\(\(\) => \{\n(?P<body>.*?)\n  \}, \[(?P<deps>[^\]]*)\]\);",
        source,
        flags=re.DOTALL,
    )
    data_effects = [(body, deps) for body, deps in effects if "Promise.allSettled" in body]
    assert len(data_effects) == 1, "the Commercial resource loader must remain one identifiable effect"

    body, dependency_source = data_effects[0]
    load_group = re.search(r"Promise\.allSettled\(\[(?P<calls>[^\]]+)\]\)", body)
    assert load_group, "the Commercial resources must still load as one settled group"
    assert re.findall(r"\b(load\w+)\(\)", load_group.group("calls")) == [
        "loadClients",
        "loadMoney",
        "loadCampaigns",
        "loadAdvertiserRules",
    ]

    dependencies = [dependency.strip() for dependency in dependency_source.split(",") if dependency.strip()]
    assert dependencies == ["refreshKey", "reloadKey"]
    assert "locale" not in dependencies, "a language-only render must not repeat locale-neutral API reads"
