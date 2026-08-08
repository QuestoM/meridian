"""P4: the client's pricing rule card, rendered, in all four of its states.

The section this file measures replaced two empty properties whose controls led
to a grid of forty five rows that could not contain the client. A helper-level
assertion cannot guard what a person reads, so this file bundles the shipped
section where it lives, with the bundler the product builds with, renders it with
React, and reads the sentences and the controls out of the markup.

Four states, because a surface that shows one state's copy in another state's
situation is the defect this whole destination exists to end: bound, unbound,
read only, and the store not read yet. The fifth case, the mutation at the foot
of the file, removes the join the bound state depends on and asserts the card
states the disagreement rather than quietly claiming the client is unpriced, so a
pass here can never be vacuous.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"
SECTION = CLIENTS / "ClientRuleCard.jsx"
HELPERS = CLIENTS / "clients-rule-helpers.js"

# The client the critic measured, and its figure in the shipped priced ledger.
CLIENT = "פריסבי"
CLIENT_GROSS = 56034.0

# The two sentences the two dead controls carried, which may not come back.
DEAD_SPELLING = "הוסיפו כתיב בכרטיס הלקוח"
DEAD_RULE = "קשרו כלל בכרטיס הלקוח"

# The copy the fixed section carries in each of its states.
CREATE_CONTROL = "צרו כלל תמחור ללקוח הזה"
OPEN_RULE_CONTROL = "פתחו את כרטיס הכלל המלא"
ADD_SPELLING_CONTROL = "הוסיפו כתיב על הכלל"
UNBOUND_REASON = "אף כלל שמור אינו נושא את שם הלקוח הזה"
NEEDS_RULE_FIRST = "ללקוח הזה דרוש קודם כלל תמחור"

# The join the bound state depends on, and the mutation that removes it.
SHIPPED_JOIN = """  return (rows || []).find((row) => {
    if (isUnboundRow(row)) {
      return false;
    }
    return rowTokens(row).some((token) => wanted.has(token));
  }) || null;
"""
NO_JOIN = """  return null;
"""


ENTRY = """
export {{ default as ClientRuleCard }} from '{section}';
"""

RENDER = """
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, casesFile, outFile, helpersSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom/server', 'lucide-react', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) {
      return { url: MAP[specifier], shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  external: ['react', 'react-dom', 'react/jsx-runtime', 'lucide-react'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'rule-card-under-test',
    // The bundler the product builds with refuses to bundle css, and the
    // stylesheet is not what is under test, so the import resolves to an empty
    // module under a virtual id that carries no css extension.
    resolveId(source) {
      return source.endsWith('.css') ? { id: `\\0stylesheet-${source}.mjs` } : null;
    },
    load(id) {
      if (id.startsWith('\\0stylesheet-')) {
        return 'export default {};';
      }
      return id === 'HELPERS_PATH' ? fs.readFileSync(helpersSource, 'utf8') : null;
    },
  }],
});

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));

const rendered = {};
for (const [name, props] of Object.entries(cases)) {
  rendered[name] = renderToStaticMarkup(
    React.createElement(surface.ClientRuleCard, { locale: 'he', ...props }),
  );
}
fs.writeFileSync(outFile, JSON.stringify(rendered), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped section cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True, text=True, check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    if not (APP / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules is not installed, so nothing can be rendered")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    return found


def _client(**extra) -> dict:
    base = {
        "advertiser": CLIENT, "shown_name": CLIENT, "aliases": [],
        "bound_to_rules_row": False, "effective_premium": 1.0,
        "gross": CLIENT_GROSS, "net": CLIENT_GROSS, "spots": 6,
    }
    base.update(extra)
    return base


def _cases() -> dict:
    bound_row = {
        "advertiser_id": "ADV_03", "name": CLIENT, "display_name": "",
        "aliases": 'פריסבי בע"מ', "default_premium": 1.15, "conditions": [],
    }
    unbound = [{"advertiser_id": f"ADV_{i:02d}", "name": "", "display_name": "",
                "aliases": "", "default_premium": 1.0, "conditions": []} for i in range(1, 46)]
    return {
        "unbound": {"client": _client(), "rows": unbound, "canEdit": True},
        "read_only": {"client": _client(), "rows": unbound, "canEdit": False,
                      "refusal": "החשבון הזה אינו רשאי לשנות דבר כאן."},
        "bound": {
            "client": _client(bound_to_rules_row=True, effective_premium=1.15),
            "rows": [*unbound, bound_row],
            "canEdit": True,
        },
        "loading": {"client": _client(), "rows": None, "canEdit": True},
    }


def _render(tmp_path: Path, helpers_source: str) -> dict:
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "helpers-under-test.js"
    source.write_text(helpers_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(ENTRY.format(section=SECTION.as_posix()), encoding="utf-8")
    script = work / "render.mjs"
    script.write_text(
        RENDER.replace("APP_PACKAGE", (APP / "package.json").as_posix()).replace("HELPERS_PATH", HELPERS.as_posix()),
        encoding="utf-8",
    )
    cases = work / "cases.json"
    cases.write_text(json.dumps(_cases(), ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [
            node,
            # the shell moved bidi.jsx and dates.js under src/shell; this hook
            # resolves both to the real modules so the bundle under test can import them.
            "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
            str(script), str(entry), str(work / "bundle"), str(cases), str(out), str(source),
        ],
        capture_output=True, text=True, check=False, cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> str:
    source = HELPERS.read_text(encoding="utf-8")
    assert SHIPPED_JOIN in source, "the join under test is not in the shipped helper any more"
    return source


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, shipped) -> dict:
    return _render(tmp_path_factory.mktemp("client-rule-card"), shipped)


def test_the_unbound_client_gets_a_control_it_can_complete_here(rendered) -> None:
    """No rule exists, so the section offers to create one, on this record."""
    markup = rendered["unbound"]
    assert CREATE_CONTROL in markup
    assert UNBOUND_REASON in markup
    assert NEEDS_RULE_FIRST in markup
    assert DEAD_SPELLING not in markup and DEAD_RULE not in markup
    assert OPEN_RULE_CONTROL not in markup, "there is no rule card to open yet"


def test_the_unbound_client_reads_as_rate_card_and_never_as_a_blank(rendered) -> None:
    """The honest reading of an unbound client is 1.00, not an empty property."""
    markup = rendered["unbound"]
    assert '<span class="numeric" dir="ltr">1.00x</span>' in markup
    assert "מחיר מחירון" in markup


def test_the_bound_client_shows_its_premium_and_opens_the_row_that_holds_it(rendered) -> None:
    """The bound state: the real premium, the row id, and the way into it."""
    markup = rendered["bound"]
    assert '<span class="numeric" dir="ltr">1.15x</span>' in markup
    assert "+15%" in markup
    assert 'dir="ltr">ADV_03<' in markup
    assert OPEN_RULE_CONTROL in markup
    assert ADD_SPELLING_CONTROL in markup
    assert CREATE_CONTROL not in markup


def test_the_bound_client_lists_its_spellings_with_where_each_came_from(rendered) -> None:
    """A spelling typed on the rule and one seen in the data are not one thing."""
    markup = rendered["bound"]
    assert html.escape('פריסבי בע"מ') in markup
    assert "הוקלד על הכלל" in markup
    listed = markup.split("clients-spellings")[1].split("</ul>")[0]
    assert listed.count("<li>") == 1, "the client's own name is not one of its other spellings"


def test_a_read_only_account_meets_the_refusal_and_no_control(rendered) -> None:
    """The endpoint is the authority, so the control is not shown to be refused."""
    markup = rendered["read_only"]
    assert "החשבון הזה אינו רשאי לשנות דבר כאן." in markup
    assert CREATE_CONTROL not in markup
    assert UNBOUND_REASON in markup, "the state is still stated, only the control is withheld"


def test_a_store_that_has_not_been_read_says_so_rather_than_claiming_unbound(rendered) -> None:
    """Tri-state: real, unavailable, unknown, and never one wearing another's copy."""
    markup = rendered["loading"]
    assert "קורא את מאגר התמחור" in markup
    assert UNBOUND_REASON not in markup
    assert CREATE_CONTROL not in markup


def test_without_the_join_the_bound_card_states_the_disagreement(tmp_path, shipped) -> None:
    """Mutation: remove the join and the bound state cannot be produced at all.

    What replaces it is the fourth state rather than the unbound one, which is
    the point: the client read still says a rule prices this client, so claiming
    it is unpriced would be the same lie in the other direction.
    """
    mutated = _render(tmp_path, shipped.replace(SHIPPED_JOIN, NO_JOIN))
    markup = mutated["bound"]
    assert OPEN_RULE_CONTROL not in markup
    assert "1.15x" not in markup
    assert CREATE_CONTROL not in markup
    assert UNBOUND_REASON not in markup
    assert "שתי הקריאות סותרות" in markup
