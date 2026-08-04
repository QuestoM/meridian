"""P4: the onboarding form keeps the agency the operator entered.

JS-5 starts by naming the agency. The form asks for it before its own options
read has landed, and the build that shipped in ``8f0c78c9`` applied the read's
default the moment it arrived, unconditionally. Measured in a browser by the
round-4 critic: "a new agency" chosen at t+0 reverted to "an agency we already
work with" at t+424, t+360 and t+361 ms, and a submit after that revert booked
the campaign under an agency nobody picked, with that agency's rebate.

This file executes the shipped component through that exact sequence. It is
compiled with the JSX compiler the product builds with, ``react`` is a hook
runtime that really writes state and really re-renders, the api module is a read
the test lands when it chooses, and the payload is the one the real backend
builds. Two tests then put the defect back, once from git history and once by
removing the guard from the current file, so a pass here cannot be vacuous. The
other half is the default the guard must not lose: an untouched form still lands
on the first agency, and one opened from a client record on that client's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
COMPONENT = DASHBOARD / "src" / "clients" / "OnboardClientFlow.jsx"
HELPERS = DASHBOARD / "src" / "clients" / "clients-money-helpers.js"

# The commit that carries the defect, and the guard that fixes it.
BROKEN_COMMIT = "8f0c78c9"
GUARD = """        if (!chosen.current.agency) {
          setAgencyId(wanted ? wanted.agency_id : (payload.agencies.length ? payload.agencies[0].agency_id : ''));
        }
        if (!chosen.current.mode) {
          setAgencyMode(payload.agencies.length ? 'existing' : 'new');
        }
"""
UNGUARDED = """        setAgencyId(wanted ? wanted.agency_id : (payload.agencies.length ? payload.agencies[0].agency_id : ''));
        setAgencyMode(payload.agencies.length ? 'existing' : 'new');
"""

NEW_AGENCY = "סוכנות חדשה"
EXISTING_AGENCY = "סוכנות שאנחנו כבר עובדים איתה"
AGENCY_NAME = "שם הסוכנות"
CLIENT_NAME = "שם הלקוח"
AGENCY_SELECT = "סוכנות"
TYPED_AGENCY = "סוכנות הבדיקה"
AGENCY_REBATE = "אחוז רבייט"
TYPED_CLIENT = "לקוח הבדיקה"

RUNTIME = """
// React's semantics for the hooks this component calls, small enough to read
// and real enough that a state write lands and the next render sees it. useMemo
// recomputes every render, which React permits because a memo is a cache hint,
// and nothing in this component uses one as an effect dependency.
const hooks = [];
let cursor = 0;
let component = null;
let props = null;
let tree = null;
let dirty = false;
let queue = [];
let locked = false;

function slot(kind, make) {
  if (locked) throw new Error(`a nested component called ${kind}`);
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

export function useRef(value) { return slot('ref', () => ({ ref: { current: value } })).ref; }

export function useMemo(factory) { slot('memo', () => ({})); return factory(); }

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
}

export function mount(Component, initial) {
  hooks.length = 0;
  queue = [];
  component = Component;
  props = initial;
  once();
}

// React re-renders when the handler that wrote the state returns, which is not
// asynchronous, so one event is one write and one render.
export function flush() { if (dirty) once(); }

export async function settle() {
  for (let turn = 0; turn < 20; turn += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (dirty) once();
  }
}

export function rendered() { return tree; }

export function deep(node) {
  if (node === null || node === undefined || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(deep);
  if (typeof node.type === 'function') {
    locked = true;
    try { return deep(node.type(node.props)); } finally { locked = false; }
  }
  return { type: node.type, props: { ...(node.props || {}), children: deep(node.props.children) } };
}

export default { createElement, Fragment, useState, useRef, useMemo, useEffect };
"""

ICONS = """
const icon = () => null;
export const Check = icon;
export const Plus = icon;
export const Trash2 = icon;
export const X = icon;

"""

FORMAT = """
export function pageText(locale, en, he) { return locale === 'he' ? he : en; }
"""

# The api module, with the options read held open so the test decides when it
# lands. That is the whole point: the operator acts while it is in flight.
API = """
export function loadOnboardingOptions() { return globalThis.__options; }

export function onboardClient(payload) {
  globalThis.__submitted = payload;
  return Promise.resolve(globalThis.__result);
}
"""

HARNESS = """
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const [componentPath, helpersPath, payloadPath, dashboardDir] = process.argv.slice(2);
const here = dirname(new URL(import.meta.url).pathname);

const req = createRequire(pathToFileURL(join(dashboardDir, 'resolve-from-here.js')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);

const RUNTIME = pathToFileURL(join(here, 'react-runtime.mjs')).href;
const ICONS = pathToFileURL(join(here, 'icons.mjs')).href;
const FORMAT = pathToFileURL(join(here, 'format.mjs')).href;
const API = pathToFileURL(join(here, 'clients-api.mjs')).href;
const HELPERS = pathToFileURL(helpersPath).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'react') return { url: RUNTIME, shortCircuit: true };
    if (specifier === 'lucide-react') return { url: ICONS, shortCircuit: true };
    if (specifier.endsWith('shell/format')) return { url: FORMAT, shortCircuit: true };
    if (specifier.endsWith('clients-api')) return { url: API, shortCircuit: true };
    if (specifier.endsWith('clients-money-helpers')) return { url: HELPERS, shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const compiled = await transformWithOxc(readFileSync(componentPath, 'utf8'), 'OnboardClientFlow.jsx', {
  jsx: { runtime: 'classic', pragma: 'React.createElement', pragmaFrag: 'React.Fragment' },
});
const modulePath = join(here, 'flow.mjs');
writeFileSync(modulePath, compiled.code, 'utf8');

const react = await import(RUNTIME);
const Flow = (await import(pathToFileURL(modulePath).href)).default;
const payload = JSON.parse(readFileSync(payloadPath, 'utf8'));

function textOf(node, out) {
  if (node === null || node === undefined || typeof node === 'boolean') return out;
  if (typeof node === 'string' || typeof node === 'number') return out.concat(String(node));
  if (Array.isArray(node)) return node.reduce((carried, child) => textOf(child, carried), out);
  return node.props ? textOf(node.props.children, out) : out;
}

// The label a control sits under: the field's own span when there is one, and
// otherwise every word inside the label, which is how the radios are written.
function labelOf(node) {
  const children = [].concat(node.props.children || []);
  const span = children.find((child) => child && child.type === 'span');
  return textOf(span || node.props.children, []).join(' ').trim();
}

function collect(node, label, found) {
  if (node === null || node === undefined || typeof node !== 'object') return found;
  if (Array.isArray(node)) {
    node.forEach((child) => collect(child, label, found));
    return found;
  }
  const here = node.type === 'label' ? labelOf(node) : label;
  if (node.type === 'input' && node.props.type === 'radio') {
    found.radios.push({ label: here, checked: !!node.props.checked, onChange: node.props.onChange });
  } else if (node.type === 'input') {
    found.fields.push({ label: here, value: node.props.value, onChange: node.props.onChange });
  } else if (node.type === 'select') {
    const options = [].concat(node.props.children || []).flat().filter(Boolean);
    found.selects.push({ label: here, value: node.props.value, options: options.map((o) => o.props.value) });
  } else if (node.type === 'form') {
    found.form = node.props.onSubmit;
  }
  if (node.props) collect(node.props.children, here, found);
  return found;
}

function controls() {
  return collect(react.deep(react.rendered()), '', { radios: [], fields: [], selects: [], form: null });
}

function shape() {
  const found = controls();
  const values = {};
  found.fields.forEach((field) => { values[field.label] = field.value; });
  const selects = {};
  found.selects.forEach((entry) => { selects[entry.label] = { value: entry.value, options: entry.options }; });
  return { checked: found.radios.filter((r) => r.checked).map((r) => r.label), values, selects };
}

// One operator act: the event, then the render React performs when the handler
// that wrote the state returns.
function fire(kind, label, value) {
  const found = controls()[kind].find((entry) => entry.label.startsWith(label));
  if (!found) throw new Error(`no ${kind} labelled ${label}`);
  found.onChange({ target: { value } });
  react.flush();
}

globalThis.__result = {
  agency: { agency_id: payload.next_agency_id, outcome: 'created' },
  advertiser: { advertiser: 'x', outcome: 'linked' },
  campaign: { campaign_id: payload.next_campaign_id, advertiser: 'x' },
  flights: [], discount: { outcome: 'stored', note: '' },
};

function open(prefill) {
  let land = null;
  globalThis.__options = new Promise((resolve) => { land = resolve; });
  globalThis.__submitted = null;
  react.mount(Flow, { locale: 'he', prefill, onClose: () => {}, onDone: () => {} });
  return () => land(payload);
}

const out = {};

// The operator opens the panel, chooses a new agency and types, all while the
// options read is still in flight. Then the read lands.
let land = open({});
fire('radios', '%(NEW_AGENCY)s', 'new');
fire('fields', '%(AGENCY_NAME)s', '%(TYPED_AGENCY)s');
fire('fields', '%(AGENCY_REBATE)s', '7');
fire('fields', '%(CLIENT_NAME)s', '%(TYPED_CLIENT)s');
await react.settle();
out.chose = { before: shape() };
land();
await react.settle();
out.chose.after = shape();
controls().form({ preventDefault: () => {} });
out.chose.submitted = globalThis.__submitted;

// Nobody touches anything, so the read's default is what the form should show.
land = open({});
await react.settle();
land();
await react.settle();
out.untouched = shape();

// Opened from a client record, which names the agency that client buys through.
const wanted = payload.agencies[payload.agencies.length - 1].agency_id;
land = open({ agencyId: wanted });
land();
await react.settle();
out.prefilled = { wanted, ...shape() };

process.stdout.write(JSON.stringify(out));
""" % {"NEW_AGENCY": NEW_AGENCY, "AGENCY_NAME": AGENCY_NAME, "CLIENT_NAME": CLIENT_NAME,
       "TYPED_AGENCY": TYPED_AGENCY, "TYPED_CLIENT": TYPED_CLIENT, "AGENCY_REBATE": AGENCY_REBATE}


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


@pytest.fixture(scope="module")
def payload_path(tmp_path_factory) -> Path:
    """The options the real backend builds, which is what the form reads."""
    from kairos_api.campaigns_api_onboarding import options

    path = tmp_path_factory.mktemp("onboarding") / "options.json"
    payload = options()
    assert payload["agencies"], "the options carry no agencies, so this file is measuring nothing"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, source: str, payload_path: Path) -> dict:
    """Drive one version of the form through the three sequences."""
    stubs = {"react-runtime.mjs": RUNTIME, "icons.mjs": ICONS, "format.mjs": FORMAT,
             "clients-api.mjs": API, "harness.mjs": HARNESS, "OnboardClientFlow.jsx": source}
    for name, body in stubs.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    component = tmp_path / "OnboardClientFlow.jsx"
    result = subprocess.run(
        [_node(), str(harness), str(component), str(HELPERS), str(payload_path), str(DASHBOARD)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def shipped() -> str:
    source = COMPONENT.read_text(encoding="utf-8")
    assert GUARD in source, "the guard under test is not in the shipped component any more"
    return source


@pytest.fixture(scope="module")
def run(tmp_path_factory, shipped, payload_path) -> dict:
    return _run(tmp_path_factory.mktemp("shipped"), shipped, payload_path)


def test_the_agency_the_operator_chose_survives_the_read(run):
    """The defect, as the sequence that produced it."""
    before = run["chose"]["before"]
    assert before["checked"] == [NEW_AGENCY]
    assert before["values"][AGENCY_NAME] == TYPED_AGENCY
    after = run["chose"]["after"]
    assert after["checked"] == [NEW_AGENCY], "the read reverted the agency the operator chose"
    assert after["values"][AGENCY_NAME] == TYPED_AGENCY, "the typed agency name was thrown away"
    assert after["values"][CLIENT_NAME] == TYPED_CLIENT
    assert AGENCY_SELECT not in after["selects"], "the existing-agency picker came back over the new agency"


def test_the_submit_after_a_late_read_books_the_agency_the_operator_typed(run, payload_path):
    """What the revert cost: the campaign went to an agency nobody picked."""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    submitted = run["chose"]["submitted"]
    assert submitted["agency"]["name"] == TYPED_AGENCY
    assert submitted["agency"]["agency_id"] == payload["next_agency_id"]
    assert submitted["agency"]["agency_id"] != payload["agencies"][0]["agency_id"]
    assert submitted["agency"]["rebate_percent"] == 7, "the rebate booked was not the one typed"
    assert submitted["advertiser"] == TYPED_CLIENT


def test_an_untouched_form_still_opens_on_the_first_agency_when_the_read_lands(run, payload_path):
    """The default the guard must not lose."""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    untouched = run["untouched"]
    assert untouched["checked"] == [EXISTING_AGENCY]
    assert untouched["selects"][AGENCY_SELECT]["value"] == payload["agencies"][0]["agency_id"]
    assert untouched["selects"][AGENCY_SELECT]["options"] == [a["agency_id"] for a in payload["agencies"]]


def test_a_form_opened_from_a_client_record_still_opens_on_that_client_agency(run):
    """The other default: booking from a record lands on that record's agency."""
    prefilled = run["prefilled"]
    assert prefilled["checked"] == [EXISTING_AGENCY]
    assert prefilled["selects"][AGENCY_SELECT]["value"] == prefilled["wanted"]


def test_the_file_that_shipped_reverts_the_choice_and_books_the_wrong_agency(tmp_path, payload_path):
    """Proof the four tests above bite, run against the tree that actually shipped."""
    read = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BROKEN_COMMIT}:tv-break-dashboard/src/clients/OnboardClientFlow.jsx"],
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        pytest.skip(f"{BROKEN_COMMIT} is not in this clone, so the shipped defect cannot be replayed")
    assert UNGUARDED in read.stdout, "the replayed file does not carry the unguarded write"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    broken = _run(tmp_path, read.stdout, payload_path)
    after = broken["chose"]["after"]
    assert after["checked"] == [EXISTING_AGENCY]
    assert AGENCY_NAME not in after["values"]
    assert broken["chose"]["submitted"]["agency"] == {"agency_id": payload["agencies"][0]["agency_id"]}


def test_removing_the_guard_from_the_current_file_reproduces_the_revert(tmp_path, shipped, payload_path):
    """The same proof against the current file, so it holds when history is shallow."""
    mutant = shipped.replace(GUARD, UNGUARDED)
    assert mutant != shipped
    after = _run(tmp_path, mutant, payload_path)["chose"]["after"]
    assert after["checked"] == [EXISTING_AGENCY]
    assert AGENCY_NAME not in after["values"]
