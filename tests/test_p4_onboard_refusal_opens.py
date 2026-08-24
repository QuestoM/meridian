"""P4: a refusal that names a record hands over that record's address.

Two refusals on the onboarding flow name something that already exists and tell
the reader what to do about it. ``'X' already has a campaign named 'Y', as
CMP_07. Open that one instead of booking a second.`` and ``'X' already has a
manual link to agency 'AGY_09'. Remove it before booking through another.`` Both
arrived as a sentence and nothing else: the account manager was told to open a
record and given no way to it, on the one flow whose promise is that they never
have to leave and come back.

Everything except the two ends was already built. ``clients-api.js`` reads
``detail.opens`` off a refusal and carries it onto the error, ``RefusalNotice``
grows the control, and ``ClientsWorkspace`` passes ``onOpenRefused`` down. The
endpoint never sent an address and the flow rendered the sentence as a bare
paragraph, so the component was exported, wired at both ends, and dead.

Nothing here is inferred from source text. The refusal comes from the real
router over temporary stores, that exact JSON is handed to the shipped
``clients-api`` module through a stubbed ``fetch``, the shipped
``OnboardClientFlow`` is compiled with the dashboard's own JSX compiler and
executed, its form is really submitted, and the control that appears is really
pressed. The last two tests take the address away, once at the endpoint and once
at the component's own prop, and assert the control disappears, so a pass here
can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "tv-break-dashboard"
CLIENTS = DASH / "src" / "clients"

ORDER = {
    "advertiser": "טורנדו מוצרי צריכה",
    "campaign_name": "קיץ 2026",
    "campaign_starts_on": "2026-08-02",
    "campaign_ends_on": "2026-08-29",
    "flights": [],
}
NEW_AGENCY = {"name": "סוכנות חדשה", "rebate_percent": 4.0}

# React, as much of it as one component instance needs, with hook slots that
# survive a re-render so a state change really changes what the next render
# returns. Elements are plain objects, which is what lets the walker below find
# one component inside another's tree by identity rather than by markup.
REACT_STUB = """
const hooks = [];
let cursor = 0;

export function reset() { cursor = 0; }

export function useState(initial) {
  const at = cursor;
  cursor += 1;
  if (hooks.length <= at) hooks[at] = typeof initial === 'function' ? initial() : initial;
  const set = (next) => { hooks[at] = typeof next === 'function' ? next(hooks[at]) : next; };
  return [hooks[at], set];
}

export function useRef(value) {
  const at = cursor;
  cursor += 1;
  if (hooks.length <= at) hooks[at] = { current: value };
  return hooks[at];
}

export function useMemo(factory) { cursor += 1; return factory(); }
export function useCallback(fn) { cursor += 1; return fn; }
export function useEffect() { cursor += 1; }

export function createElement(type, props, ...children) {
  const merged = { ...(props || {}) };
  if (children.length === 1) merged.children = children[0];
  if (children.length > 1) merged.children = children;
  return { type, props: merged };
}

export const Fragment = 'Fragment';
// shell/bidi.jsx calls React.forwardRef at module scope for DirectionRoot, and
// this stub stands in for react before that module evaluates, so its absence is
// a TypeError at import time rather than anything this file is about. Identity
// is enough: nothing here forwards a ref.
export function forwardRef(render) { return render; }

export default { createElement, Fragment, useState, useRef, useMemo, useCallback, useEffect, forwardRef };
"""

JSX_STUB = """
import { createElement, Fragment } from 'react';
export function jsx(type, props) { return { type, props: props || {} }; }
export const jsxs = jsx;
export { Fragment, createElement };
"""

HARNESS = """
import fs from 'node:fs';
import path from 'node:path';
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';

const [, , DASH, SRC, OUT, PAYLOAD] = process.argv;
const req = createRequire(pathToFileURL(path.join(DASH, 'package.json')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);
const plan = JSON.parse(fs.readFileSync(PAYLOAD, 'utf8'));

const MODULES = ['OnboardClientFlow.jsx', 'clients-api.js', 'clients-money-helpers.js', 'weekday-scope-helpers.js'];
const built = new Map();
const icons = new Set();

fs.mkdirSync(OUT, { recursive: true });
for (const name of MODULES) {
  let text = fs.readFileSync(path.join(SRC, name), 'utf8');
  if (name === 'OnboardClientFlow.jsx' && plan.cut) {
    if (!text.includes(plan.cut)) throw new Error(`the line this mutation removes is not in ${name}`);
    text = text.replace(plan.cut, plan.instead);
  }
  const out = await transformWithOxc(text, name, { jsx: { runtime: 'automatic' } });
  for (const match of out.code.matchAll(/import\\s*\\{([^}]*)\\}\\s*from\\s*"lucide-react"/g)) {
    match[1].split(',').map((part) => part.trim()).filter(Boolean).forEach((icon) => icons.add(icon));
  }
  const file = path.join(OUT, `${name.replace(/\\.jsx?$/, '')}.mjs`);
  fs.writeFileSync(file, out.code, 'utf8');
  built.set(name.replace(/\\.jsx?$/, ''), file);
}

function stub(name, body) {
  const file = path.join(OUT, name);
  fs.writeFileSync(file, body, 'utf8');
  return file;
}

const REACT = stub('react.mjs', plan.react);
const JSX = stub('jsx.mjs', plan.jsx);
const FORMAT = stub('format.mjs', "export function pageText(locale, en, he) { return locale === 'he' ? he : en; }\\n");
const ACTIONS = stub('actions.mjs', `
  export const Button = 'button';
  export const ButtonBase = 'button';
  export const IconButton = 'button';
`);
const CONTROLS = stub('controls.mjs', `
  import React from 'react';
  export function InputControl(props) { return React.createElement('input', props); }
  export function SelectControl({ children, ...props }) { return React.createElement('select', props, children); }
  export function TextAreaControl(props) { return React.createElement('textarea', props); }
  export function Pressable({ children, type = 'button', ...props }) {
    return React.createElement('button', { ...props, type }, children);
  }
`);
const CSS = stub('css.mjs', 'export default {};\\n');
const MODAL = stub('modal.mjs', `
  import React from 'react';
  export function Dialog({ title, description, footer, children, className }) {
    return React.createElement('section', { className, role: 'dialog' },
      React.createElement('header', null,
        React.createElement('h2', null, title),
        description ? React.createElement('p', null, description) : null),
      children,
      footer ? React.createElement('footer', null, footer) : null);
  }
  export const Sheet = Dialog;
`);
const LUCIDE = stub('lucide.mjs', [...icons].map((icon) => `export function ${icon}() { return null; }`).join('\\n'));
const BASE = stub('base.mjs', "export const API_BASE = '';\\n");

registerHooks({
  resolve(specifier, context, next) {
    const hit = (url) => ({ url: pathToFileURL(url).href, shortCircuit: true });
    if (specifier === 'react') return hit(REACT);
    if (specifier === 'react/jsx-runtime') return hit(JSX);
    if (specifier.endsWith('.css')) return hit(CSS);
    if (specifier.endsWith('shell/format')) return hit(FORMAT);
    if (specifier.endsWith('studio/actions')) return hit(ACTIONS);
    if (specifier.endsWith('studio/dom-controls')) return hit(CONTROLS);
    if (specifier.endsWith('studio/modal')) return hit(MODAL);
    if (specifier.endsWith('shell/api')) return hit(BASE);
    if (specifier === 'lucide-react') return hit(LUCIDE);
    const stem = specifier.startsWith('./') ? specifier.slice(2).replace(/\\.jsx?$/, '') : '';
    if (stem && built.has(stem)) return hit(built.get(stem));
    return next(specifier, context);
  },
});

// The refusal exactly as the router sent it, answered to the shipped api module
// so the error the component sees is the error the product builds.
globalThis.fetch = async () => ({
  ok: false,
  status: plan.status,
  statusText: 'Refused',
  json: async () => plan.body,
});

const React = await import(pathToFileURL(REACT).href);
const flow = await import(pathToFileURL(built.get('OnboardClientFlow')).href);
const OnboardClientFlow = flow.default;
const opened = [];
const props = {
  locale: 'he',
  prefill: null,
  onClose: () => {},
  onDone: () => {},
  onOpenRefused: (target) => opened.push(target),
};

function walk(node, found) {
  if (!node || typeof node !== 'object') return found;
  if (Array.isArray(node)) { node.forEach((child) => walk(child, found)); return found; }
  if (node.type && node.props) {
    if (node.type === flow.RefusalNotice) found.push(node);
    if (typeof node.type === 'string' && node.props.onSubmit) found.onSubmit = node.props.onSubmit;
    walk(node.props.children, found);
    return found;
  }
  return found;
}

function render() {
  React.reset();
  const found = [];
  walk(OnboardClientFlow(props), found);
  return found;
}

const first = render();
if (!first.onSubmit) throw new Error('the flow rendered no form to submit');
await first.onSubmit({ preventDefault() {} });
const after = render();
const notice = after[0];
const drawn = notice ? flow.RefusalNotice(notice.props) : null;

const buttons = [];
walk(drawn, buttons);
function controls(node, into) {
  if (!node || typeof node !== 'object') return into;
  if (Array.isArray(node)) { node.forEach((child) => controls(child, into)); return into; }
  if (node.type === 'button') into.push(node);
  if (node.props) controls(node.props.children, into);
  return into;
}
const pressable = controls(drawn, []);
for (const button of pressable) if (button.props.onClick) button.props.onClick();

process.stdout.write(JSON.stringify({
  notice_rendered: Boolean(notice),
  sentence: notice ? notice.props.error : '',
  address: notice ? notice.props.opens : null,
  opener_supplied: Boolean(notice && notice.props.onOpen),
  controls: pressable.map((button) => (button.props.children || '')),
  opened,
}));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be executed")
    if not (DASH / "node_modules" / "vite").is_dir():
        pytest.skip("the dashboard's node_modules are not installed, so nothing can be compiled")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the imports cannot be redirected")
    return found


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """The real routers over temporary stores, so no operator file is written."""
    from kairos_api import agencies, agency_conditions, campaigns_api, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)
    monkeypatch.setattr(agency_conditions, "_latest_daily_pairs", lambda: ([], None))
    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(agency_conditions.router)
    app.include_router(campaigns_api.router)
    return TestClient(app)


def _duplicate_campaign(client: TestClient) -> tuple[int, dict]:
    """Book the order, book it again, and return the second answer."""
    first = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    assert first.status_code == 201, first.text
    repeat = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    return repeat.status_code, repeat.json()


def _linked_elsewhere(client: TestClient) -> tuple[int, dict]:
    """A client already linked to another agency, which is the second refusal."""
    assert client.post("/api/agencies", json={"agency_id": "AGY_09", "name": "סוכנות אחרת"}).status_code == 201
    linked = client.post("/api/agencies/AGY_09/advertisers", json={"advertiser": ORDER["advertiser"]})
    assert linked.status_code in {200, 201}
    refused = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    return refused.status_code, refused.json()


def _render(tmp_path: Path, status: int, body: dict, cut: str = "", instead: str = "") -> dict:
    """Execute the shipped flow against one refusal and report what it drew."""
    harness = tmp_path / "flow.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": status,
                "body": body,
                "react": REACT_STUB,
                "jsx": JSX_STUB,
                "cut": cut,
                "instead": instead,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            _node(),
            # the shell moved bidi.jsx and dates.js under src/shell; this hook
            # resolves both to the real modules so the harness under test can import them.
            "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
            str(harness), str(DASH), str(CLIENTS), str(tmp_path / "build"), str(plan),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_endpoint_sends_the_address_of_the_campaign_it_names(client):
    """The first half: the refusal carries the id the sentence tells you to open."""
    status, body = _duplicate_campaign(client)
    assert status == 409
    detail = body["detail"]
    assert "Open that one instead" in detail["message_en"]
    assert detail["opens"]["kind"] == "campaign"
    assert detail["opens"]["id"], "the id is the address, so it cannot be blank"
    assert detail["opens"]["id"] in detail["message_en"], "the address is the record the sentence names"
    assert detail["opens"]["id"] in detail["message_he"]


def test_the_endpoint_sends_the_address_of_the_agency_it_names(client):
    """The second refusal names an agency record, and now says which one."""
    status, body = _linked_elsewhere(client)
    assert status == 409
    detail = body["detail"]
    assert detail["opens"] == {"kind": "agency", "id": "AGY_09"}


def test_a_refusal_that_names_no_record_carries_no_address(client):
    """The honesty half: no address means no control, so the key is absent."""
    refused = client.post("/api/clients/onboarding", json={
        "agency": NEW_AGENCY, **ORDER, "campaign_ends_on": "29/08/2026",
    })
    assert refused.status_code == 400
    detail = refused.json()["detail"]
    assert detail["message_en"] == "The end date must be an ISO date, YYYY-MM-DD"
    assert "opens" not in detail, "a refusal that names nothing must not offer a way to nothing"


def test_the_refused_flow_draws_a_control_that_opens_the_record(tmp_path, client):
    """The measured gap: submit the form, be refused, and press what appears."""
    status, body = _duplicate_campaign(client)
    drawn = _render(tmp_path, status, body)
    assert drawn["notice_rendered"], "the refusal drew nothing at all"
    assert drawn["sentence"] == body["detail"]["message_he"], "a Hebrew flow reads the Hebrew half"
    assert drawn["address"] == body["detail"]["opens"]
    assert drawn["opener_supplied"], "the flow must hand the notice the workspace's opener"
    assert drawn["controls"] == ["פתחו את הקמפיין הזה"]
    assert drawn["opened"] == [body["detail"]["opens"]], "pressing it must open that exact record"


def test_the_agency_refusal_draws_the_agency_control(tmp_path, client):
    """The same, for the refusal whose record is an agency card."""
    status, body = _linked_elsewhere(client)
    drawn = _render(tmp_path, status, body)
    assert drawn["controls"] == ["פתחו את כרטיס הסוכנות"]
    assert drawn["opened"] == [{"kind": "agency", "id": "AGY_09"}]


def test_a_refusal_with_no_record_behind_it_stays_a_sentence(tmp_path, client):
    """Proof the control is grown by the address and not by the markup."""
    refused = client.post("/api/clients/onboarding", json={
        "agency": NEW_AGENCY, **ORDER, "campaign_ends_on": "29/08/2026",
    })
    drawn = _render(tmp_path, refused.status_code, refused.json())
    assert drawn["notice_rendered"], "the sentence must still be shown"
    assert drawn["sentence"], "and it must still be the endpoint's own sentence"
    assert drawn["address"] is None
    assert drawn["controls"] == []
    assert drawn["opened"] == []


def test_without_the_address_on_the_notice_the_dead_end_comes_back(tmp_path, client):
    """The mutation: cut the address out of the prop and the way out disappears."""
    status, body = _duplicate_campaign(client)
    drawn = _render(
        tmp_path,
        status,
        body,
        cut="opens={state.opens}",
        instead="opens={null}",
    )
    assert drawn["notice_rendered"], "the sentence still renders, which is the old behaviour"
    assert drawn["controls"] == [], "the control must come from the wiring under test"
    assert drawn["opened"] == []
