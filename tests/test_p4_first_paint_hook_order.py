"""P4: the clients tree survives its own first paint.

The workspace mounts ``ClientTree`` with ``tree`` null and sets the tree when
``GET /api/clients`` resolves, so the component renders at least twice per visit.
The build that shipped in ``cdc3935f`` added a sixth hook, the demo tally, below
the loading return, so the first paint called five hooks and the render after the
read called six. React refuses that: it throws error #310, "Rendered more hooks
than during the previous render", and it unmounts the whole application, not the
panel. Measured in a browser at that commit, a reload on the clients view dropped
the document to a blank page. The blast radius was every destination, because one
component took the shell down with it.

Neither ``test_p4_client_tree.py`` nor ``test_p4_view_navigation.py`` could catch
it. The first measures the payload the server builds and never renders anything;
the second executes the navigation helper, which has no hooks at all.

This file executes the shipped component. The source is compiled with the same
JSX compiler the product builds with, ``react`` is replaced by a recorder that
enforces React's own two conditions on hook order, the icon set and the shell's
text formatter are stubbed, and the money helpers are the real module. The
payload is the one the real backend builds, so the second render is the render an
operator gets. Two tests then put the defect back, once by taking the file out of
git history and once by moving the statement in the current file, and assert the
sequence fails, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
COMPONENT = DASHBOARD / "src" / "clients" / "ClientTree.jsx"
HELPERS = DASHBOARD / "src" / "clients" / "clients-money-helpers.js"

# The commit that introduced the defect, and the statement that carried it.
BROKEN_COMMIT = "cdc3935f"
TALLY = """  const { demo: demoCampaignCount } = useMemo(
    () => demoTally(
      tree && tree.agencies,
      tree && tree.unlinked,
      tree && tree.clients_booked_without_spots,
    ),
    [tree],
  );
"""
LOADING_RETURN = """  if (!tree) {
    return <div className="clients-loading">{pageText(locale, 'Loading clients', 'טוען לקוחות')}</div>;
  }
"""
TOO_MANY_HOOKS = "Rendered more hooks than during the previous render."
LOADING_TEXT = "טוען לקוחות"

RECORDER = """
// React's hook contract, enforced as the reconciler enforces it across two
// renders of one component instance: hook N of this render is hook N of the
// render before it, and a render may not call more hooks than that render did.
let current = null;
let previous = null;
let cursor = 0;

export function startRender() {
  current = [];
  cursor = 0;
}

export function endRender() {
  if (previous && current.length < previous.length) {
    throw new Error('Rendered fewer hooks than expected.');
  }
  previous = current;
  return current.slice();
}

function record(kind) {
  const at = cursor;
  cursor += 1;
  if (previous) {
    if (at >= previous.length) {
      throw new Error('Rendered more hooks than during the previous render.');
    }
    if (previous[at] !== kind) {
      throw new Error(`Hook ${at} changed from ${previous[at]} to ${kind}.`);
    }
  }
  current.push(kind);
}

export function useState(initial) {
  record('useState');
  return [typeof initial === 'function' ? initial() : initial, () => {}];
}

export function useMemo(factory) {
  record('useMemo');
  return factory();
}

export function useCallback(fn) {
  record('useCallback');
  return fn;
}

export function useRef(value) {
  record('useRef');
  return { current: value };
}

export function useEffect() {
  record('useEffect');
}

export function useReducer(reducer, initial) {
  record('useReducer');
  return [initial, () => {}];
}

// Not recorded, because it is not a hook and it is not called during a render.
// It is here because the surface under test now reaches shell/bidi.jsx, which
// calls React.forwardRef at module scope for DirectionRoot. This recorder is
// substituted for the real react BEFORE that module evaluates, so a missing
// forwardRef is a TypeError at import time rather than anything about hooks.
// The identity passthrough is enough: nothing in these tests forwards a ref.
export function forwardRef(render) {
  return render;
}

export function createElement(type, props, ...children) {
  const merged = { ...(props || {}) };
  if (children.length === 1) merged.children = children[0];
  if (children.length > 1) merged.children = children;
  return { type, props: merged };
}

export const Fragment = 'Fragment';

export default {
  createElement,
  forwardRef,
  Fragment,
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
  useReducer,
};
"""

ICONS = """
const icon = () => null;
export const Building2 = icon;
export const ChevronDown = icon;
export const ChevronUp = icon;
export const Plus = icon;
export const Search = icon;
export const UserPlus = icon;
"""

FORMAT = """
export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}
"""

HARNESS = """
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const [componentPath, helpersPath, payloadPath, dashboardDir] = process.argv.slice(2);
const here = dirname(new URL(import.meta.url).pathname);

// The JSX compiler the product builds with, resolved from the dashboard, which
// is the only directory here that has node_modules.
const req = createRequire(pathToFileURL(join(dashboardDir, 'resolve-from-here.js')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);

const RECORDER = pathToFileURL(join(here, 'react-recorder.mjs')).href;
const ICONS = pathToFileURL(join(here, 'icons.mjs')).href;
const FORMAT = pathToFileURL(join(here, 'format.mjs')).href;
const HELPERS = pathToFileURL(helpersPath).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'react') return { url: RECORDER, shortCircuit: true };
    if (specifier === 'lucide-react') return { url: ICONS, shortCircuit: true };
    if (specifier.endsWith('shell/format')) return { url: FORMAT, shortCircuit: true };
    if (specifier.endsWith('clients-money-helpers')) return { url: HELPERS, shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const compiled = await transformWithOxc(readFileSync(componentPath, 'utf8'), 'ClientTree.jsx', {
  jsx: { runtime: 'classic', pragma: 'React.createElement', pragmaFrag: 'React.Fragment' },
});
const modulePath = join(here, 'component.mjs');
writeFileSync(modulePath, compiled.code, 'utf8');

const recorder = await import(RECORDER);
const module = await import(pathToFileURL(modulePath).href);
const Component = module.default;
const payload = JSON.parse(readFileSync(payloadPath, 'utf8'));

// Every string the render put on screen, in order, without a DOM.
function text(node, out) {
  if (node === null || node === undefined || typeof node === 'boolean') return out;
  if (typeof node === 'string' || typeof node === 'number') {
    out.push(String(node));
    return out;
  }
  if (Array.isArray(node)) {
    node.forEach((child) => text(child, out));
    return out;
  }
  if (node.props) text(node.props.children, out);
  return out;
}

const renders = [];
function render(label, props) {
  recorder.startRender();
  let error = '';
  let body = [];
  try {
    body = text(Component(props), []);
  } catch (failure) {
    error = failure.message;
  }
  renders.push({ label, error, hooks: recorder.endRender(), text: body.join(' | ') });
}

const props = { locale: 'he', canEdit: true, onOpenClient: () => {}, onOnboard: () => {} };
render('first paint', { ...props, tree: null });
render('after the read', { ...props, tree: payload });
render('steady state', { ...props, tree: payload });

process.stdout.write(JSON.stringify(renders));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be executed here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the component's imports cannot be stubbed")
    if not (DASHBOARD / "node_modules" / "vite").is_dir():
        pytest.skip("the dashboard has no installed vite, so the JSX cannot be compiled here")
    return found


@pytest.fixture(scope="module")
def payload_path(tmp_path_factory) -> Path:
    """The tree the real backend builds, which is what the second render reads."""
    from kairos_api.campaigns_read_clients import client_tree

    path = tmp_path_factory.mktemp("clients") / "payload.json"
    path.write_text(json.dumps(client_tree(), ensure_ascii=False), encoding="utf-8")
    return path


def _render(tmp_path: Path, source: str, payload_path: Path) -> dict[str, dict]:
    """Mount one version of the component and return every render it performed."""
    (tmp_path / "react-recorder.mjs").write_text(RECORDER, encoding="utf-8")
    (tmp_path / "icons.mjs").write_text(ICONS, encoding="utf-8")
    (tmp_path / "format.mjs").write_text(FORMAT, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    component = tmp_path / "ClientTree.jsx"
    component.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            _node(),
            # the shell moved bidi.jsx and dates.js under src/shell; this hook
            # resolves both to the real modules so the harness under test can import them.
            "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
            str(harness), str(component), str(HELPERS), str(payload_path), str(DASHBOARD),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {render["label"]: render for render in json.loads(result.stdout)}


@pytest.fixture(scope="module")
def shipped() -> str:
    source = COMPONENT.read_text(encoding="utf-8")
    assert TALLY in source, "the statement under test is not in the shipped component any more"
    assert source.index(TALLY) < source.index(LOADING_RETURN), "the tally is below the loading return again"
    return source


def test_the_first_paint_and_the_render_after_it_call_the_same_hooks(tmp_path, shipped, payload_path):
    """The defect, as the sequence that produced it. Same count, same order."""
    renders = _render(tmp_path, shipped, payload_path)
    first = renders["first paint"]["hooks"]
    assert first, "the component called no hooks at all, so this test is measuring nothing"
    for label in ("first paint", "after the read", "steady state"):
        assert renders[label]["error"] == "", f"{label}: {renders[label]['error']}"
        assert renders[label]["hooks"] == first, f"{label} changed the hook sequence"


def test_the_first_paint_says_it_is_loading_rather_than_rendering_an_empty_tree(tmp_path, shipped, payload_path):
    """The reason the early return exists survives the fix."""
    renders = _render(tmp_path, shipped, payload_path)
    assert renders["first paint"]["text"] == LOADING_TEXT


def test_the_render_after_the_read_carries_the_counts_the_backend_measured(tmp_path, shipped, payload_path):
    """The hoisted tally still counts, and it counts the payload rather than a guess."""
    import json as _json

    tree = _json.loads(payload_path.read_text(encoding="utf-8"))
    rendered = _render(tmp_path, shipped, payload_path)["after the read"]["text"]
    for count in ("agencies", "clients", "campaigns"):
        assert str(tree["counts"][count]) in rendered, count
    demo = sum(
        1
        for group in (tree["agencies"], tree["unlinked"], tree["clients_booked_without_spots"])
        for holder in group
        for client in (holder.get("clients") or [holder])
        for campaign in (client.get("campaigns") or [])
        if campaign.get("is_demo")
    )
    if demo:
        assert str(demo) in rendered, "the demo split vanished from the header"


def test_the_file_that_shipped_before_takes_the_application_down(tmp_path, payload_path):
    """Proof the three tests above bite, run against the tree that actually shipped."""
    read = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BROKEN_COMMIT}:tv-break-dashboard/src/clients/ClientTree.jsx"],
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        pytest.skip(f"{BROKEN_COMMIT} is not in this clone, so the shipped defect cannot be replayed")
    renders = _render(tmp_path, read.stdout, payload_path)
    assert renders["first paint"]["error"] == ""
    assert len(renders["first paint"]["hooks"]) == 5
    assert renders["after the read"]["error"] == TOO_MANY_HOOKS
    assert renders["after the read"]["text"] == "", "the render that throws puts nothing on screen"


def test_moving_the_tally_below_the_loading_return_reproduces_the_crash(tmp_path, shipped, payload_path):
    """The same proof against the current file, so it holds when history is shallow."""
    mutant = shipped.replace(TALLY, "").replace(LOADING_RETURN, LOADING_RETURN + "\n" + TALLY)
    assert mutant != shipped
    renders = _render(tmp_path, mutant, payload_path)
    assert renders["after the read"]["error"] == TOO_MANY_HOOKS
