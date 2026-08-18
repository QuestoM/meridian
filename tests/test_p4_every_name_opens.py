"""P4: a name that carries a figure opens the thing it names, measured by rendering.

Two cells on this destination printed a name that opened nothing while the same
object opened one view away, and both source-pinning suites passed over them.
That is the class of defect this file exists to catch, so it does not read the
source: it transforms the shipped components, renders them to markup with the
real payloads, finds the cell by the value a person would click, and asks
whether that one cell holds an interactive element.

The two measured before the fix, rendered:

  * the campaign cell inside the money drill was a bare ``<td>`` holding the
    campaign name, while the break chip in the same row was a button and the
    identical name was a button in the campaign ranking one tab away, and
  * the agency cell on the campaign board was a ``span`` reading ``ישירים``
    over ``AGY_04``, while the client name beside it was a button and the client
    record already opened that very agency record.

Nothing that an assertion depends on is stubbed. ``pageText`` is the two-line
language pick, the icons are empty components, the stylesheet is an empty
module, and the two panels the board opens under a row are not rendered because
no row is open. Everything else is the shipped module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "tv-break-dashboard"
CLIENTS = DASH / "src" / "clients"

# Render the shipped components. The transform is the dashboard's own (vite's
# oxc), so the JSX these tests execute is the JSX the browser executes.
HARNESS = """
import fs from 'node:fs';
import path from 'node:path';
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';

const [, , DASH, SRC, OUT, PAYLOAD] = process.argv;
const req = createRequire(pathToFileURL(path.join(DASH, 'package.json')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);

// Every module in the directory, rather than a list somebody has to remember to
// extend. Twice now a component gained one import and this harness died on
// ERR_MODULE_NOT_FOUND — a failure that says nothing about the component and
// everything about the list. Compiling the whole directory costs milliseconds
// and takes the class of failure away; anything unreachable from the entry
// modules is simply never imported.
const MODULES = fs.readdirSync(SRC).filter((name) => /\.jsx?$/.test(name));
const built = new Map();
const icons = new Set();

fs.mkdirSync(OUT, { recursive: true });
for (const name of MODULES) {
  const out = await transformWithOxc(fs.readFileSync(path.join(SRC, name), 'utf8'), name, { jsx: { runtime: 'automatic' } });
  for (const match of out.code.matchAll(/import\\s*\\{([^}]*)\\}\\s*from\\s*"lucide-react"/g)) {
    match[1].split(',').map((part) => part.trim()).filter(Boolean).forEach((icon) => icons.add(icon));
  }
  const file = path.join(OUT, `${name.replace(/\\.jsx?$/, '')}.mjs`);
  fs.writeFileSync(file, out.code, 'utf8');
  built.set(name.replace(/\\.jsx?$/, ''), file);
}

// shell/bidi and shell/dates are real primitives, not scaffolding, and the
// modules above now name them for the same reason MoneyBoard's campaign cell
// and CampaignBoard's agency cell became buttons: a figure or a name that used
// to print as bare text now goes through Figure/Code/Name, and DeliveryState's
// dates go through the same file's formatters. Compiling a stub would let the
// modules resolve while testing nothing about what they actually render, so
// these build the shipped files, from their own directory next to src/clients.
// dates.js itself imports isolate from './bidi', so shell/bidi builds first.
const SHELL = path.join(DASH, 'src', 'shell');
for (const name of ['bidi.jsx', 'dates.js']) {
  const out = await transformWithOxc(fs.readFileSync(path.join(SHELL, name), 'utf8'), name, { jsx: { runtime: 'automatic' } });
  const file = path.join(OUT, `shell-${name.replace(/\\.jsx?$/, '')}.mjs`);
  fs.writeFileSync(file, out.code, 'utf8');
  built.set(`shell/${name.replace(/\\.jsx?$/, '')}`, file);
}

function stub(name, body) {
  const file = path.join(OUT, name);
  fs.writeFileSync(file, body, 'utf8');
  return file;
}

const FORMAT = stub('format.mjs', "export function pageText(locale, en, he) { return locale === 'he' ? he : en; }\\n");
const CSS = stub('css.mjs', 'export default {};\\n');
const LUCIDE = stub('lucide.mjs', [...icons].map((icon) => `export function ${icon}() { return null; }`).join('\\n'));
const API = stub('api.mjs', 'export async function endCampaign() { return {}; }\\nexport async function loadOnboardingOptions() { return {}; }\\n');
const PANEL = stub('panel.mjs', 'export default function Panel() { return null; }\\n');

const ACTIONS = stub('actions.mjs', `
import React from 'react';
function action(tag) {
  return React.forwardRef(function Action({ children, loading, loadingIndicator, ...props }, ref) {
    return React.createElement(tag, { ...props, ref }, children);
  });
}
export const Button = action('button');
export const ButtonBase = action('button');
export const IconButton = action('button');
`);

// Every package the transformed modules import, resolved before the hooks are
// registered. Resolving inside the hook re-enters it, which is a recursion.
const bare = new Map();
const wanted = new Set(['react', 'react/jsx-runtime', 'react-dom/server']);
for (const file of built.values()) {
  for (const match of fs.readFileSync(file, 'utf8').matchAll(/from\\s*"([^".][^"]*)"/g)) {
    if (!match[1].startsWith('.') && match[1] !== 'lucide-react') wanted.add(match[1]);
  }
}
for (const name of wanted) bare.set(name, req.resolve(name));

registerHooks({
  resolve(specifier, context, next) {
    const hit = (url) => ({ url: pathToFileURL(url).href, shortCircuit: true });
    if (specifier.endsWith('.css')) return hit(CSS);
    if (specifier.endsWith('shell/format')) return hit(FORMAT);
    if (specifier.endsWith('/bidi')) return hit(built.get('shell/bidi'));
    if (specifier.endsWith('shell/dates')) return hit(built.get('shell/dates'));
    if (specifier.endsWith('studio/actions')) return hit(ACTIONS);
    if (specifier.endsWith('clients-api')) return hit(API);
    if (specifier === 'lucide-react') return hit(LUCIDE);
    if (specifier === './CampaignDetail' || specifier === './CampaignTerms') return hit(PANEL);
    const stem = specifier.startsWith('./') ? specifier.slice(2).replace(/\\.jsx?$/, '') : '';
    if (stem && built.has(stem)) return hit(built.get(stem));
    if (bare.has(specifier)) return hit(bare.get(specifier));
    return next(specifier, context);
  },
});

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const MoneyBoard = (await import(pathToFileURL(built.get('MoneyBoard')).href)).default;
const MoneyDetail = (await import(pathToFileURL(built.get('MoneyDetail')).href)).default;
const CampaignBoard = (await import(pathToFileURL(built.get('CampaignBoard')).href)).default;
const helpers = await import(pathToFileURL(built.get('clients-money-helpers')).href);
const payload = JSON.parse(fs.readFileSync(PAYLOAD, 'utf8'));

// The one cell that holds this value, so a control anywhere else on the page
// cannot answer for it. The row is cut at its own boundaries first.
function cell(html, value) {
  const at = html.indexOf(value);
  if (at < 0) return null;
  const rowStart = html.lastIndexOf('<tr', at);
  const rowEnd = html.indexOf('</tr>', at);
  if (rowStart >= 0 && rowEnd >= at) {
    const row = html.slice(rowStart, rowEnd);
    const found = row.split('</td>').map((part) => `${part}</td>`).find((part) => part.includes(value));
    if (found !== undefined) return found;
  }
  const candidates = [['<button', '</button>'], ['<dd', '</dd>'], ['<span', '</span>']]
    .map(([open, close]) => ({ open, close, start: html.lastIndexOf(open, at) }))
    .filter((part) => part.start >= 0)
    .sort((a, b) => b.start - a.start);
  for (const part of candidates) {
    const end = html.indexOf(part.close, at);
    if (end >= at) return html.slice(part.start, end + part.close.length);
  }
  return null;
}

// The one removed-spot row that holds this value. The same break id is printed
// in the table of priced spots above, so the search starts inside the removed
// block or it would measure the control that already worked.
function droppedItem(html, value) {
  const block = html.indexOf('clients-dropped');
  if (block < 0) return null;
  const rest = html.slice(block);
  const at = rest.indexOf(value);
  if (at < 0) return null;
  const start = rest.lastIndexOf('<li', at);
  const end = rest.indexOf('</li>', at);
  return start < 0 || end < 0 ? null : rest.slice(start, end + 5);
}

const opened = { group: 'advertisers', key: payload.advertiser };
const money = renderToStaticMarkup(React.createElement(MoneyBoard, {
  money: payload.money,
  locale: 'he',
  drill: opened,
  onDrill: () => {},
  onOpenClient: () => {},
}));
const board = (props) => renderToStaticMarkup(React.createElement(CampaignBoard, {
  board: payload.board,
  locale: 'he',
  notify: () => {},
  gate: { canEdit: true, reason: '' },
  agencies: helpers.agencyIndex(payload.tree),
  onOnboard: () => {},
  onOpenClient: () => {},
  onReload: () => {},
  ...props,
}));

const wired = board({ openCampaignId: payload.campaign_id, onOpenAgency: () => {} });
const unwired = board({ openCampaignId: payload.campaign_id });

// The drill of a client whose day really had spots removed by a rule, and the
// same drill with the ledger holding no row for the break one of them names,
// which is the state a break made only of removed spots is in.
const removedRow = payload.money.advertisers.find((row) => row.advertiser === payload.dropped_advertiser);
const removed = renderToStaticMarkup(React.createElement(MoneyBoard, {
  money: payload.money,
  locale: 'he',
  drill: { group: 'advertisers', key: payload.dropped_advertiser },
  onDrill: () => {},
  onOpenClient: () => {},
}));
const thinned = renderToStaticMarkup(React.createElement(MoneyDetail, {
  money: { ...payload.money, breaks: payload.money.breaks.filter((row) => String(row.break_id) !== payload.dropped_break) },
  row: removedRow,
  field: 'advertiser',
  locale: 'he',
  position: null,
  onStep: () => {},
  onOpenBreak: () => {},
  onOpenCampaign: () => {},
}));

process.stdout.write(JSON.stringify({
  drill_open: money.includes('clients-detail'),
  campaign_cell: cell(money, payload.campaign),
  break_chip: cell(money, payload.break_id),
  agency_cell: cell(wired, payload.agency_id),
  agency_name_cell: cell(wired, payload.agency_name),
  client_cell: cell(wired, payload.client),
  agency_cell_unwired: cell(unwired, payload.agency_id),
  dropped_item: droppedItem(removed, payload.dropped_break),
  dropped_item_unranked: droppedItem(thinned, payload.dropped_break),
}));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped components cannot be rendered here")
    if not (DASH / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules are not installed, so nothing can be rendered")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the imports cannot be redirected")
    return found


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> dict:
    """The real ledger, the real client tree and the real campaign board."""
    from kairos_api import campaigns_api
    from kairos_api.campaigns_read_clients import client_tree
    from kairos_api.campaigns_read_money import board

    money = board()
    if not money["available"]:
        pytest.skip("no priced daily file on disk, so there is no money board to render")
    booked = campaigns_api.list_campaigns(None)
    with_agency = next(
        (row for row in booked["campaigns"] if row["agency_id"]),
        None,
    )
    assert with_agency is not None, "no campaign on the board carries an agency, so the cell cannot be measured"
    tree = client_tree()
    names = {agency["agency_id"]: agency["name"] for agency in tree["agencies"]}
    advertiser = money["advertisers"][0]
    ranked_breaks = {str(row["break_id"]) for row in money["breaks"]}
    removed = None
    for row in money["advertisers"]:
        keys = set(row.get("dropped_keys") or [])
        hit = next(
            (entry for entry in money["dropped"]
             if entry["spot_key"] in keys and str(entry["break_id"]) in ranked_breaks),
            None,
        )
        if hit:
            removed = (row["advertiser"], str(hit["break_id"]))
            break
    if removed is None:
        pytest.skip("no client on this ledger had a spot removed by a rule, so there is no row to measure")
    return {
        "money": money,
        "tree": tree,
        "board": booked,
        "advertiser": advertiser["advertiser"],
        "campaign": advertiser["campaigns"][0]["campaign"],
        "campaign_id": with_agency["campaign_id"],
        "break_id": str(advertiser["campaigns"][0]["breaks"][0]),
        "agency_id": with_agency["agency_id"],
        "agency_name": names.get(with_agency["agency_id"], ""),
        "client": with_agency["advertiser"],
        "dropped_advertiser": removed[0],
        "dropped_break": removed[1],
    }


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, payload) -> dict:
    tmp = tmp_path_factory.mktemp("clients-render")
    harness = tmp / "render.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    data = tmp / "payload.json"
    data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [_node(), str(harness), str(DASH), str(CLIENTS), str(tmp / "build"), str(data)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_campaign_that_carries_a_figure_opens_its_rows(rendered, payload):
    """The named gap: the campaign cell inside the money drill was plain text."""
    assert rendered["drill_open"], "the client's drill did not render, so no cell was measured"
    cell = rendered["campaign_cell"]
    assert cell, f"no cell holds {payload['campaign']}"
    assert "<button" in cell, f"the campaign cell opens nothing: {cell}"
    assert payload["campaign"] in cell


def test_the_break_chip_beside_it_still_opens_its_break(rendered):
    """The control that already worked in the same row, which must not regress."""
    assert "<button" in (rendered["break_chip"] or "")


def test_the_agency_on_the_campaign_board_opens_the_agency_record(rendered, payload):
    """The second half of the gap: the agency cell was inert text with an id."""
    cell = rendered["agency_cell"]
    assert cell, f"no cell holds {payload['agency_id']}"
    assert "<button" in cell, f"the agency cell opens nothing: {cell}"
    assert payload["agency_name"] and payload["agency_name"] in cell, "the name leads, not the key"
    assert payload["agency_id"] in cell, "the storage key stays findable under the name"


def test_the_client_name_on_the_same_row_still_opens_the_client(rendered):
    """The control that already worked on that board, which must not regress."""
    assert "<button" in (rendered["client_cell"] or "")


def test_an_agency_cell_with_no_opener_stays_a_label(rendered):
    """Proof the assertion above is not vacuous, and the rule it encodes.

    A control that opens nothing is worse than a label, so the cell becomes a
    control only where a caller supplied the opener. Rendering the same board
    without one must therefore produce no button in that cell, which is also
    what proves the button above came from the wiring and not from the markup.
    """
    cell = rendered["agency_cell_unwired"]
    assert cell, "the cell must still render its name and its id"
    assert "<button" not in cell


def test_a_removed_spot_opens_the_break_it_would_have_sat_in(rendered, payload):
    """The third cell of the same class, found by sweeping rather than by report.

    A rule removed the spot, the row states which break it names, and that break
    is a chip in the table of priced spots directly above it. Measured in the
    browser before this: three removed rows on one client, zero controls.
    """
    item = rendered["dropped_item"]
    assert item, f"no removed row holds break {payload['dropped_break']}"
    assert "<button" in item, f"the break behind a removed spot opens nothing: {item}"
    assert payload["dropped_break"] in item


def test_a_break_the_ledger_does_not_rank_stays_a_label(rendered, payload):
    """Proof the assertion above is not vacuous, and the rule it encodes.

    A break holding nothing but removed spots has no row on the break grouping,
    so there is nothing behind its id. Rendering the same drill against a ledger
    without that row must therefore print the id and no control.
    """
    item = rendered["dropped_item_unranked"]
    assert item, "the removed row must still state the break it names"
    assert payload["dropped_break"] in item
    assert "<button" not in item


def test_the_workspace_supplies_both_openers(payload):
    """The other half: the two callbacks are handed down by the surface itself.

    Two hops since the file-size split: the workspace binds openAgencyRecord
    into the `on` bundle, and ClientsPanels hands it to the board.
    """
    workspace = (CLIENTS / "ClientsWorkspace.jsx").read_text(encoding="utf-8")
    assert "openAgency: openAgencyRecord" in workspace
    panels = (CLIENTS / "ClientsPanels.jsx").read_text(encoding="utf-8")
    board = panels.split("<CampaignBoard")[1].split("/>")[0]
    assert "onOpenAgency={on.openAgency}" in board
    assert "setActive('agencies')" in workspace, "the opener has to reach the agency records view"
    money = (CLIENTS / "MoneyBoard.jsx").read_text(encoding="utf-8")
    assert "onOpenCampaign={(name) => onDrill({ group: 'campaigns', key: name })}" in money
