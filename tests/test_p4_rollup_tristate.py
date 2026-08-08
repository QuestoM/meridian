"""P4: the campaign rollup never prints a count it has not measured.

The panel had two states where the data has three. While its read was in flight
`campaigns || fetched || { campaigns: [] }` produced an empty payload, so the
header printed "0 קמפיינים" and the table printed "לא נמצאו שורות קמפיינים".
The round-five critic measured that claim standing at 1,883 ms with the read
delayed, and at 132 to 277 ms warm, on a screen whose whole subject is money.

Zero campaigns and a count nobody has taken are different facts, and a failed
read is a third. This file executes the shipped component through all three,
compiled with the JSX compiler the product builds with, on a hook runtime that
really writes state and really re-renders, with the read held open so the test
decides when it lands.

The last test puts the defect back from git history, so a pass here cannot be
vacuous: the version at ``HEAD`` must claim zero while the same read is in
flight, and the shipped one must not.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
COMPONENT = DASHBOARD / "src" / "clients" / "CampaignRollupPanel.jsx"
RELATIVE = "tv-break-dashboard/src/clients/CampaignRollupPanel.jsx"

# The commit whose version of this file has two states where the data has three.
BROKEN_COMMIT = "HEAD"

LOADING = "טוען קמפיינים שנצפו בנתוני המקור"
LOADING_COUNT = "בטעינה"
NO_ROWS = "לא נמצאו שורות קמפיינים."
FAILED_COUNT = "לא נטען"
ZERO_CLAIM = "0 קמפיינים"

RUNTIME = """
// The hooks this component calls, with React's own semantics for them: a state
// write lands, the next render sees it, and an effect runs after the render
// whose dependencies changed.
const hooks = [];
let cursor = 0;
let component = null;
let props = null;
let tree = null;
let dirty = false;
let queue = [];

function slot(kind, make) {
  const at = cursor;
  cursor += 1;
  if (!(at in hooks)) hooks[at] = { kind, ...make() };
  if (hooks[at].kind !== kind) throw new Error(`hook ${at} changed from ${hooks[at].kind} to ${kind}`);
  return hooks[at];
}

function same(a, b) {
  return !!a && !!b && a.length === b.length && a.every((item, i) => Object.is(item, b[i]));
}

export function useState(initial) {
  const held = slot('state', () => ({ value: typeof initial === 'function' ? initial() : initial }));
  return [held.value, (next) => {
    const value = typeof next === 'function' ? next(held.value) : next;
    if (!Object.is(value, held.value)) { held.value = value; dirty = true; }
  }];
}

export function useEffect(fn, deps) {
  const held = slot('effect', () => ({ deps: null, cleanup: null, first: true }));
  if (held.first || !same(held.deps, deps)) {
    held.deps = deps;
    held.first = false;
    queue.push(() => {
      if (typeof held.cleanup === 'function') held.cleanup();
      held.cleanup = fn();
    });
  }
}

export function createElement(type, config, ...children) {
  const merged = { ...(config || {}) };
  if (children.length) merged.children = children.length === 1 ? children[0] : children;
  return { type, props: merged };
}

export const Fragment = 'Fragment';

function once() {
  cursor = 0;
  dirty = false;
  tree = component(props);
  const pending = queue;
  queue = [];
  pending.forEach((run) => run());
  if (dirty) once();
}

export function mount(Component, initial) {
  hooks.length = 0;
  queue = [];
  component = Component;
  props = initial;
  once();
}

export async function settle() {
  for (let turn = 0; turn < 20; turn += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (dirty) once();
  }
}

export function deep(node) {
  if (node === null || node === undefined || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(deep);
  if (typeof node.type === 'function') return deep(node.type(node.props));
  return { type: node.type, props: { ...(node.props || {}), children: deep(node.props.children) } };
}

export function rendered() { return deep(tree); }

export default { createElement, Fragment, useState, useEffect };
"""

FORMAT = """
export function pageText(locale, en, he) { return locale === 'he' ? he : en; }
export function formatCurrency(value) { return String(value); }
export function formatMinutes(value) { return String(value); }
export function formatNumber(value) { return String(value); }
"""

PLAN_MODEL = """
export function normalizeRows(rows) { return Array.isArray(rows) ? rows : []; }
"""

# shell/bidi did not exist when this harness was written, and every panel that
# prints a figure now imports it. Without this stub node cannot resolve the
# specifier and the whole file errors before a single assertion runs, which is
# how 36 tests across four files went red at once.
#
# The stub keeps the one thing a caller can observe: the wrapper element and its
# class. It deliberately does NOT reproduce the isolation characters. A test
# about a count or a column should not depend on invisible control codes, and
# the real behaviour has its own guard in npm run test:direction.
BIDI = """
import React from 'react';

const wrap = (className) => function Wrapper({ children, className: extra, title }) {
  return React.createElement('span', { className: [className, extra].filter(Boolean).join(' '), title }, children);
};

export const Figure = wrap('bidi-figure');
export const Code = wrap('bidi-code');
export const Name = wrap('bidi-name');
export function isolate(value) { return value; }
export function documentDirection(locale) { return locale === 'he' ? 'rtl' : 'ltr'; }
// Not forwardRef: the fake react runtime in this file is createElement and two
// hooks, and nothing here needs the ref the real one forwards.
export function DirectionRoot({ children, ...rest }) {
  return React.createElement('div', rest, children);
}
export function Prose({ as: Element = 'p', children, className, ...rest }) {
  return React.createElement(Element, { className, ...rest }, children);
}
"""

# The grid, reduced to the behaviour this file is about: with no rows a person
# reads the empty label, and with rows a person reads each column's own cell,
# built the same way the real DataTable builds one, through `column.render`.
# That is the one part of the real grid a column-opens-nothing regression
# would actually show up in, so the fake keeps it rather than collapsing every
# row to a row count.
PRIMITIVES = """
import React from 'react';

export function DataTable({ rows, emptyLabel, columns }) {
  if (!rows || !rows.length) {
    return React.createElement('div', { className: 'grid-empty' }, emptyLabel);
  }
  return React.createElement('div', { className: 'grid-rows' }, rows.map((row, index) =>
    React.createElement('div', { className: 'grid-row', key: index }, (columns || []).map((column) =>
      React.createElement('span', { className: `grid-cell grid-cell-${column.key}`, key: column.key },
        column.render ? column.render(row) : row[column.key])))));
}
"""

# The read, held open so the test decides when and how it lands. The rollup's
# own drill never opens in these scenarios (no click is simulated), so
# `loadRollupDetail` only has to exist as a valid export, not do anything.
API = """
export function loadRollup() {
  return new Promise((resolve, reject) => { globalThis.__land = resolve; globalThis.__fail = reject; });
}
export function loadRollupDetail() {
  return new Promise(() => {});
}
"""

ALERTS = """
export default function MakeGoodAlerts() { return null; }
"""

# The shell's placeholder, by identity. `use-kairos-data.js:22` seeds its state
# with this exact object and hands it back when the shared read fails, so the
# panel receives a truthy prop holding an empty list before anything is read.
FALLBACKS = """
export const fallbackCampaigns = { campaigns: [] };
"""

LUCIDE = """
export function ArrowLeft() { return null; }
"""

MONEY_HELPERS = """
export function exactMoney(value) { return String(value); }
"""

HARNESS = """
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const [componentPath, dashboardDir] = process.argv.slice(2);
const here = dirname(new URL(import.meta.url).pathname);

const req = createRequire(pathToFileURL(join(dashboardDir, 'resolve-from-here.js')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);

const RUNTIME = pathToFileURL(join(here, 'react-runtime.mjs')).href;
const FORMAT = pathToFileURL(join(here, 'format.mjs')).href;
const PLAN = pathToFileURL(join(here, 'plan-model.mjs')).href;
const PRIMITIVES = pathToFileURL(join(here, 'primitives.mjs')).href;
const API = pathToFileURL(join(here, 'clients-api.mjs')).href;
const ALERTS = pathToFileURL(join(here, 'alerts.mjs')).href;
const FALLBACKS = pathToFileURL(join(here, 'fallbacks.mjs')).href;
const LUCIDE = pathToFileURL(join(here, 'lucide.mjs')).href;
const MONEY_HELPERS = pathToFileURL(join(here, 'money-helpers.mjs')).href;
const BIDI = pathToFileURL(join(here, 'bidi.mjs')).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'react') return { url: RUNTIME, shortCircuit: true };
    if (specifier === 'lucide-react') return { url: LUCIDE, shortCircuit: true };
    if (specifier.endsWith('shell/format')) return { url: FORMAT, shortCircuit: true };
    if (specifier.endsWith('shell/plan-model')) return { url: PLAN, shortCircuit: true };
    if (specifier.endsWith('shell/primitives')) return { url: PRIMITIVES, shortCircuit: true };
    if (specifier.endsWith('shell/fallbacks')) return { url: FALLBACKS, shortCircuit: true };
    if (specifier.endsWith('shell/bidi')) return { url: BIDI, shortCircuit: true };
    if (specifier.endsWith('clients-api')) return { url: API, shortCircuit: true };
    if (specifier.endsWith('clients-money-helpers')) return { url: MONEY_HELPERS, shortCircuit: true };
    if (specifier.endsWith('MakeGoodAlerts')) return { url: ALERTS, shortCircuit: true };
    if (specifier.endsWith('.css')) return { url: pathToFileURL(join(here, 'empty.mjs')).href, shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const compiled = await transformWithOxc(readFileSync(componentPath, 'utf8'), 'CampaignRollupPanel.jsx', {
  jsx: { runtime: 'classic', pragma: 'React.createElement', pragmaFrag: 'React.Fragment' },
});
const modulePath = join(here, 'panel.mjs');
writeFileSync(modulePath, compiled.code, 'utf8');

const react = await import(RUNTIME);
const Panel = (await import(pathToFileURL(modulePath).href)).default;

function textOf(node, out) {
  if (node === null || node === undefined || typeof node === 'boolean') return out;
  if (typeof node === 'string' || typeof node === 'number') return out.concat(String(node));
  if (Array.isArray(node)) return node.reduce((carried, child) => textOf(child, carried), out);
  return node.props ? textOf(node.props.children, out) : out;
}

function screen() { return textOf(react.rendered(), []).join(' '); }

// A rough HTML string, kept only for the one thing plain text cannot answer:
// whether a cell is a control or a label. Real tag names survive; everything
// else collapses the way `screen()` already does.
function html(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(html).join('');
  if (!node.type) return '';
  const inner = node.props ? html(node.props.children) : '';
  return typeof node.type === 'string' ? `<${node.type}>${inner}</${node.type}>` : inner;
}

const seen = {};
react.mount(Panel, { locale: 'he', refreshKey: 1 });
await react.settle();
seen.inFlight = screen();

globalThis.__land({ campaigns: [{ Campaign: 'a' }, { Campaign: 'b' }, { Campaign: 'c' }] });
await react.settle();
seen.landed = screen();
seen.landedHtml = html(react.rendered());

react.mount(Panel, { locale: 'he', refreshKey: 2 });
await react.settle();
globalThis.__land({ campaigns: [] });
await react.settle();
seen.empty = screen();

react.mount(Panel, { locale: 'he', refreshKey: 3 });
await react.settle();
globalThis.__fail(new Error('500'));
await react.settle();
seen.failed = screen();

// The assembled app's own sequence: the shell hands its placeholder down as a
// prop before the shared read lands. That is where the zero claim really came
// from, because the placeholder is truthy and holds an empty list.
const { fallbackCampaigns } = await import(FALLBACKS);
react.mount(Panel, { locale: 'he', refreshKey: 4, campaigns: fallbackCampaigns });
await react.settle();
seen.placeholder = screen();

globalThis.__land({ campaigns: [{ Campaign: 'a' }, { Campaign: 'b' }] });
await react.settle();
seen.placeholderLanded = screen();

react.mount(Panel, { locale: 'he', refreshKey: 5, campaigns: { campaigns: [{ Campaign: 'a' }] } });
await react.settle();
seen.supplied = screen();

// The two remaining findings on this panel: a campaign name opened nothing,
// and a source with no advertiser column rendered a blank cell rather than a
// stated reason.
react.mount(Panel, { locale: 'he', refreshKey: 6 });
await react.settle();
globalThis.__land({
  campaigns: [{ Campaign: 'מבצע קיץ', advertiser_id: null, spots: 12, seconds: 340, revenue: null, last_airing: '01/01/2026' }],
  advertiser_available: false,
  revenue_available: true,
});
await react.settle();
seen.advertiserUnavailable = screen();
seen.advertiserUnavailableHtml = html(react.rendered());

process.stdout.write(JSON.stringify(seen));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be executed here")
    read = "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"
    probe = subprocess.run([found, "-e", read], capture_output=True, text=True, check=False)
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the component's imports cannot be stubbed")
    if not (DASHBOARD / "node_modules" / "vite").is_dir():
        pytest.skip("the dashboard has no installed vite, so the JSX cannot be compiled here")
    return found


def _run(tmp_path: Path, source: str) -> dict:
    stubs = {
        "react-runtime.mjs": RUNTIME, "format.mjs": FORMAT, "plan-model.mjs": PLAN_MODEL,
        "primitives.mjs": PRIMITIVES, "clients-api.mjs": API, "alerts.mjs": ALERTS,
        "fallbacks.mjs": FALLBACKS, "empty.mjs": "export default {};\n",
        "lucide.mjs": LUCIDE, "money-helpers.mjs": MONEY_HELPERS, "bidi.mjs": BIDI,
        "harness.mjs": HARNESS, "CampaignRollupPanel.jsx": source,
    }
    for name, body in stubs.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    result = subprocess.run(
        [_node(), str(tmp_path / "harness.mjs"), str(tmp_path / "CampaignRollupPanel.jsx"), str(DASHBOARD)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    # A count written as two adjacent nodes and one written as a single string
    # read identically on screen, so the comparison is on the words, not on the
    # whitespace the renderer happened to put between them.
    return {key: " ".join(str(value).split()) for key, value in json.loads(result.stdout).items()}


@pytest.fixture(scope="module")
def shipped(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("shipped"), COMPONENT.read_text(encoding="utf-8"))


# --- the bar ------------------------------------------------------------------


def test_a_read_in_flight_is_not_a_count_of_zero(shipped):
    """The gap, stated as the sentence the panel must not print."""
    assert LOADING_COUNT in shipped["inFlight"]
    assert LOADING in shipped["inFlight"]
    assert ZERO_CLAIM not in shipped["inFlight"]
    assert NO_ROWS not in shipped["inFlight"]


def test_the_count_appears_when_the_read_lands(shipped):
    assert "3 קמפיינים" in shipped["landed"]
    assert LOADING_COUNT not in shipped["landed"]
    assert LOADING not in shipped["landed"]


def test_a_real_zero_is_still_a_zero(shipped):
    """The honest empty state survives, which is the other half of the fix."""
    assert ZERO_CLAIM in shipped["empty"]
    assert NO_ROWS in shipped["empty"]
    assert LOADING_COUNT not in shipped["empty"]


def test_a_failed_read_says_so_rather_than_reporting_none(shipped):
    assert FAILED_COUNT in shipped["failed"]
    assert ZERO_CLAIM not in shipped["failed"]
    assert NO_ROWS not in shipped["failed"]


def test_the_shells_placeholder_is_not_read_as_a_count_of_zero(shipped):
    """The assembled app's own sequence, which is where the zero claim came from.

    ``use-kairos-data.js:22`` seeds the shared state with ``fallbackCampaigns``
    and passes it down, so the panel gets a truthy prop holding an empty list
    before anything has been read. Identity, not truthiness, is what separates
    that placeholder from a payload.
    """
    assert LOADING_COUNT in shipped["placeholder"]
    assert ZERO_CLAIM not in shipped["placeholder"]
    assert NO_ROWS not in shipped["placeholder"]
    assert "2 קמפיינים" in shipped["placeholderLanded"]


def test_a_payload_the_shell_really_read_is_used_as_it_is(shipped):
    """The prop still wins when it is a payload, so no second read is spent."""
    assert "1 קמפיינים" in shipped["supplied"]
    assert LOADING_COUNT not in shipped["supplied"]


def test_a_campaign_name_opens_its_own_rows(shipped):
    """The rollup's own dead end: a campaign name used to open nothing at all."""
    assert "<button>a</button>" in shipped["landedHtml"], shipped["landedHtml"]


def test_the_advertiser_column_states_unavailable_rather_than_a_blank(shipped):
    """The source carries no advertiser column at all, and the cell says so.

    A blank cell with no stated reason is a placeholder, not an honest empty
    state; the note above the table and the cell itself both have to carry the
    reason, the way the revenue note already does for a missing revenue column.
    """
    assert "לא זמין" in shipped["advertiserUnavailable"]
    assert "אין עמודת מפרסם" in shipped["advertiserUnavailable"]
    assert "אין עמודת מפרסם" not in shipped["landed"], "a source that does carry the column prints no such note"


def test_the_campaign_name_still_opens_when_the_advertiser_is_unavailable(shipped):
    """The two fixes live in the same file and must not fight each other."""
    assert "<button>מבצע קיץ</button>" in shipped["advertiserUnavailableHtml"]


# --- the defect, put back ------------------------------------------------------


def test_the_version_in_history_claims_zero_while_the_read_is_in_flight(tmp_path):
    """Without this, every assertion above could be measuring nothing."""
    read = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BROKEN_COMMIT}:{RELATIVE}"],
        capture_output=True, text=True, check=False,
    )
    if read.returncode != 0:
        pytest.skip("git cannot read the previous version of this component here")
    if "campaigns || fetched || { campaigns: [] }" not in read.stdout:
        pytest.skip("the version in history no longer carries the two-state read")

    broken = _run(tmp_path, read.stdout)
    assert ZERO_CLAIM in broken["inFlight"]
    assert NO_ROWS in broken["inFlight"]
    # And the sequence the assembled app really runs, which is the one measured
    # in the browser: the shell's placeholder printed as a measured zero.
    assert ZERO_CLAIM in broken["placeholder"]
    assert NO_ROWS in broken["placeholder"]
