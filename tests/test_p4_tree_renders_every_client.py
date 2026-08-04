"""P4: every client the payload carries is on screen and opens its record.

The measured defect. ``client_tree`` computed a third group of clients, the ones
reachable only through a booked campaign, and the surface rendered two. On the
shipped bundle a client booked with no priced spot was a name on a campaign row
that opened nothing: the tree header counted 41 while 42 clients were on file,
the search could not find it, and clicking it set the address and opened zero
record panels.

The two tests that already existed could not catch it. One measured the payload,
which was right all along, and one measured the helper that flattens the tree.
Neither rendered the component, and the component was where the group was
dropped.

So this file renders the shipped components. ``ClientTree.jsx`` is bundled with
the bundler the product builds with, rendered by React to static markup, and
read for the rows a person would see. Then the same payload is flattened by the
shipped helper and the row it finds is rendered as ``ClientRecord.jsx``, which is
the panel the click opens. The last test removes the third group from the source
and asserts the client disappears from both, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"
TREE = CLIENTS / "ClientTree.jsx"

# The client this file books. It exists only inside a temporary store, so
# ``data/campaigns.csv`` is never written by the suite.
BOOKED_CLIENT = "חברת בדיקה חדשה"
CAMPAIGN_NAME = "קמפיין בדיקה"

# The group the defect dropped, as the shipped component renders it.
GROUP_TITLE = "הוזמנו, טרם שודרו"
GROUP_NOTE = "יש להם קמפיין רשום ואין תשדיר מתומחר ביום הנקרא"
UNLINKED_TITLE = "לקוחות ללא סוכנות"

# The mutation: the third group, cut out of the shipped source exactly as it was
# missing when this was measured.
THIRD_GROUP = """      <FlatGroup
        title={pageText(locale, 'Booked, nothing aired yet', 'הוזמנו, טרם שודרו')}
        note={pageText(locale, 'they have a campaign on file and no priced spot in the day being read', 'יש להם קמפיין רשום ואין תשדיר מתומחר ביום הנקרא')}
        clients={booked}
        locale={locale}
        onOpen={onOpenClient}
      />
"""

ENTRY = """
export {{ default as ClientTree }} from '{tree}';
export {{ default as ClientRecord }} from '{record}';
export * as helpers from '{helpers}';
"""

# One node run: bundle the shipped source, render it, report what is on screen.
# React and the icon set are resolved from the application's own install, which
# is the same react the browser bundle uses.
RENDER = """
import { createRequire, isBuiltin, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, treeFile, name, outFile, treeSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
// Every package the bundle leaves external resolves from the application's own
// install, because the bundle is written to a temporary directory that has no
// node_modules of its own. A fixed list of five names stood here, and it broke
// the day a component on this destination imported a sixth: the render died on
// a missing package rather than on anything this file measures.
const MAP = new Map();
function fromApp(specifier) {
  if (!MAP.has(specifier)) {
    try {
      const found = require_.resolve(specifier);
      MAP.set(specifier, found.startsWith('/') ? pathToFileURL(found).href : '');
    } catch {
      MAP.set(specifier, '');
    }
  }
  return MAP.get(specifier);
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!isBuiltin(specifier) && !/^[./]|^node:|^file:/.test(specifier)) {
      const found = fromApp(specifier);
      if (found) {
        return { url: found, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
});

// The component is bundled where it lives, so its own imports resolve exactly as
// they do in the shipped build. The version under test is supplied through the
// loader rather than by copying the file, which is what keeps the mutant honest.
const { build } = await import('rolldown');
await build({
  input: entry,
  // Every package stays external and resolves from the application's install
  // through the hook above, so the bundle holds this destination's own source
  // and nothing else, and one module instance serves both the bundle and the
  // renderer below.
  external: (id) => !/^[./]/.test(id),
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'client-tree-under-test',
    load(id) {
      return id === 'TREE_PATH' ? fs.readFileSync(treeSource, 'utf8') : null;
    },
  }],
});

const React = (await import('react')).default;
const { renderToStaticMarkup: markup } = await import('react-dom/server');
// The design system renders through Emotion, which needs a cache to write into.
// Without one the first styled component on the surface throws and the render
// reports nothing, which reads as a defect on the surface rather than a missing
// provider in the harness.
const { CacheProvider } = await import('@emotion/react');
const cacheModule = await import('@emotion/cache');
const createCache = cacheModule.default.default || cacheModule.default;
const cache = createCache({ key: 'kairos-test' });
const renderToStaticMarkup = (element) => markup(React.createElement(CacheProvider, { value: cache }, element));
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const tree = JSON.parse(fs.readFileSync(treeFile, 'utf8'));

const treeHtml = renderToStaticMarkup(React.createElement(surface.ClientTree, {
  tree,
  locale: 'he',
  onOpenClient: () => {},
  onOnboard: () => {},
}));

const rows = surface.helpers.flattenClients(
  tree.agencies || [],
  tree.unlinked || [],
  tree.clients_booked_without_spots || [],
);
const client = rows.find((row) => row.advertiser === name) || null;
const recordHtml = client
  ? renderToStaticMarkup(React.createElement(surface.ClientRecord, {
    client,
    rows,
    locale: 'he',
    basis: tree.basis,
    delivery: null,
    statuses: [{ value: 'active', label_en: 'Active', label_he: 'פעיל' }],
    goalWords: [],
    onClose: () => {},
    onStep: () => {},
    onOpenMoney: () => {},
    onOpenRecords: () => {},
    onBookCampaign: () => {},
  }))
  : '';

fs.writeFileSync(outFile, JSON.stringify({
  treeHtml,
  recordHtml,
  rows: rows.length,
  found: Boolean(client),
}), 'utf8');
"""


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
    return found


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> dict:
    """The real tree, with one campaign booked for a client with no spot.

    The campaign store is redirected to a temporary file, so the state this file
    measures is created by the store's own writer and the tracked one is never
    touched.
    """
    from kairos_api import campaigns_api_store as store
    from kairos_api.campaigns_read_clients import client_tree

    tmp = tmp_path_factory.mktemp("campaign-store")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store, "CAMPAIGNS_PATH", tmp / "campaigns.csv")
        patch.setattr(store, "BACKUP_DIR", tmp / "_backups")
        row = store.blank_row()
        row.update({
            "record_type": store.CAMPAIGN,
            "campaign_id": "CMP_0001",
            "name": CAMPAIGN_NAME,
            "advertiser": BOOKED_CLIENT,
            "agency_id": "",
            "status": "active",
            "starts_on": "2026-09-01",
            "ends_on": "2026-09-30",
        })
        store.write_frame(store.append(store.load_frame(), row))
        return client_tree()


def _render(tmp_path: Path, payload: dict, tree_source: str) -> dict:
    """Bundle and render the shipped surface against one version of the tree."""
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "tree-under-test.jsx"
    source.write_text(tree_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(
        ENTRY.format(
            tree=TREE.as_posix(),
            record=(CLIENTS / "ClientRecord.jsx").as_posix(),
            helpers=(CLIENTS / "clients-money-helpers.js").as_posix(),
        ),
        encoding="utf-8",
    )
    script = work / "render.mjs"
    script.write_text(
        RENDER.replace("APP_PACKAGE", (APP / "package.json").as_posix()).replace("TREE_PATH", TREE.as_posix()),
        encoding="utf-8",
    )
    tree_json = work / "tree.json"
    tree_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [
            node,
            str(script),
            str(entry),
            str(work / "bundle"),
            str(tree_json),
            BOOKED_CLIENT,
            str(out),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> str:
    source = TREE.read_text(encoding="utf-8")
    assert THIRD_GROUP in source, "the group under test is not in the shipped component any more"
    return source


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, payload, shipped) -> dict:
    return _render(tmp_path_factory.mktemp("shipped"), payload, shipped)


def test_the_payload_carries_a_client_no_agency_and_no_spot_can_reach(payload):
    """The state under test exists, or everything below would pass vacuously."""
    assert [client["advertiser"] for client in payload["clients_booked_without_spots"]] == [BOOKED_CLIENT]
    # The header counts every row it renders, priced or not. It is stated as the
    # rows rather than as one more than the priced ones, because an operator who
    # onboards a client adds a row with no money and the count must follow the
    # rows, not a number this file remembers.
    rows = [
        *[client for agency in payload["agencies"] for client in agency["clients"]],
        *payload["unlinked"],
        *payload["clients_booked_without_spots"],
    ]
    unpriced = [row for row in rows if row["gross"] is None]
    assert payload["counts"]["clients"] == len(rows)
    assert payload["counts"]["clients"] == payload["counts"]["clients_with_money"] + len(unpriced)
    assert BOOKED_CLIENT in {row["advertiser"] for row in unpriced}
    assert BOOKED_CLIENT not in {
        client["advertiser"] for agency in payload["agencies"] for client in agency["clients"]
    }
    assert BOOKED_CLIENT not in {client["advertiser"] for client in payload["unlinked"]}


def test_the_booked_client_is_a_row_on_screen_under_its_own_group(rendered):
    """The critic's first measurement: the name is nowhere in the rendered tree."""
    html = rendered["treeHtml"]
    assert GROUP_TITLE in html, "the third group must render beside the unlinked one"
    assert GROUP_NOTE in html, "the group states why its rows have no money"
    assert BOOKED_CLIENT in html
    assert html.index(GROUP_TITLE) < html.index(BOOKED_CLIENT), "the name belongs inside that group"
    row = html[html.index(GROUP_TITLE):]
    opener = f'<button type="button" class="clients-link">{BOOKED_CLIENT}</button>'
    assert opener in row, "the name is a control that opens the record, not text"


def test_the_header_counts_every_client_the_component_renders(payload, rendered):
    """A header that argues with the rows beneath it is the same defect again."""
    html = rendered["treeHtml"]
    rows = (
        sum(agency["client_count"] for agency in payload["agencies"])
        + len(payload["unlinked"])
        + len(payload["clients_booked_without_spots"])
    )
    assert payload["counts"]["clients"] == rows
    assert f"⁦{rows}⁩ לקוחות" in html
    assert rendered["rows"] == rows, "the flattened set the record walks is the same set"


def test_the_group_states_its_own_count_in_the_right_singular(rendered):
    """One client is a client, and Hebrew has a singular the count must respect."""
    group = rendered["treeHtml"][rendered["treeHtml"].index(GROUP_TITLE):]
    assert '<span class="numeric" dir="ltr">1</span><small>לקוח</small>' in group
    assert "<small>לקוחות</small>" not in group.split("</article>")[0], "one row never reads לקוחות"


def test_clicking_that_name_opens_a_record_with_its_campaign_and_its_reason(rendered):
    """The critic's second measurement: the click opened zero record panels."""
    assert rendered["found"], "the flattened rows must contain the booked client"
    record = rendered["recordHtml"]
    assert record, "a client in the rows must render a record"
    assert BOOKED_CLIENT in record
    assert CAMPAIGN_NAME in record, "the record shows what was booked"
    assert "CMP_0001" in record
    assert "ללא שיוך לסוכנות" in record, "no agency link is a stated state, not a blank"
    assert "אין תשדיר מתומחר ללקוח הזה, ולכן אין שורה לפתוח" in record
    assert "2025-04-27" in record, "the day being read is named beside the missing money"


def test_the_record_knows_it_is_the_last_of_the_set_it_was_opened_from(payload, rendered):
    """The position counter counts the whole tree, so it proves the set it walks."""
    total = payload["counts"]["clients"]
    assert f"{total} / {total}" in rendered["recordHtml"]


def test_no_money_is_invented_for_a_client_that_delivered_nothing(payload, rendered):
    """A booking is not revenue. The figures are dashes and the totals are the ledger."""
    booked = payload["clients_booked_without_spots"][0]
    assert booked["gross"] is None and booked["net"] is None and booked["spots"] is None
    assert payload["totals"]["gross"] == 699450.0
    assert payload["totals"]["net"] == 669978.0
    record = rendered["recordHtml"]
    assert "699,450" not in record and "669,978" not in record


def test_without_the_third_group_the_client_vanishes_from_the_screen(tmp_path, payload, shipped):
    """Proof the tests above bite: the defect, restored, fails them."""
    mutant = shipped.replace(THIRD_GROUP, "")
    assert mutant != shipped
    html = _render(tmp_path, payload, mutant)["treeHtml"]
    assert GROUP_TITLE not in html
    assert BOOKED_CLIENT not in html, "this is exactly what was measured on the shipped bundle"
    total = payload["counts"]["clients"]
    assert f"⁦{total}⁩ לקוחות" in html, "and the header went on counting a client with no row, which is the defect"
    assert UNLINKED_TITLE not in html, "no other group covers for it: the unlinked group is empty on this data"
