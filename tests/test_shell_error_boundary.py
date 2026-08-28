"""A failed screen must say so; a blank page says nothing.

Without a boundary React unmounts the entire tree on any render error, so a
single bad screen takes the navigation rail with it and the operator is left
with white. The realistic trigger is not exotic: deploying while sessions are
open makes the running page ask for code chunks that no longer exist, and this
product deploys while people work in it.

These tests pin three things about the boundary, executing the SHIPPED
component with the dashboard's own JSX compiler rather than reading its source:
a stale-build failure is named as one and promises the data is untouched, any
other failure is reported honestly instead of blanking, and a healthy child is
passed through unchanged. The fourth property - that the boundary is an
ANCESTOR of the workspace rather than its sibling - is asserted structurally,
because that is the mistake this fix was first written with: renderWorkspace
was called inline, so its throw belonged to the parent's render and escaped the
boundary entirely. Verified in a browser before and after.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
BOUNDARY = DASHBOARD / "src" / "shell" / "ErrorBoundary.jsx"
SHELL = DASHBOARD / "src" / "shell" / "TVBreakDashboard.jsx"


def _node() -> str:
    for candidate in ("/opt/homebrew/bin/node", "/usr/local/bin/node", "node"):
        if candidate == "node" or Path(candidate).exists():
            return candidate
    return "node"


HARNESS = r"""
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { registerHooks } from 'node:module';

const [, , boundaryPath, dashboardDir, outPath] = process.argv;
const here = dirname(outPath);
const req = createRequire(pathToFileURL(join(dashboardDir, 'resolve.js')));
const { transformWithOxc } = await import(pathToFileURL(req.resolve('vite')).href);

// React, small and real enough: a class component with getDerivedStateFromError
// and a render pass that re-renders with the derived state when a child throws.
const REACT = join(here, 'react.mjs');
writeFileSync(REACT, `
export function createElement(type, config, ...children) {
  const props = { ...(config || {}) };
  if (children.length) props.children = children.length === 1 ? children[0] : children;
  return { type, props };
}
export const Fragment = 'Fragment';
export class Component {
  constructor(props) { this.props = props || {}; this.state = {}; }
  setState(next) { this.state = { ...this.state, ...next }; }
}
// The component extends React.Component off the DEFAULT import, so the default
// must carry the same class the named export does.
export default { createElement, Fragment, Component };
`, 'utf8');

const ACTIONS = join(here, 'actions.mjs');
writeFileSync(ACTIONS, `
import React from 'react';
export function Button({ children, ...props }) { return React.createElement('button', props, children); }
export const IconButton = Button;
`, 'utf8');

const FORMAT = join(here, 'format.mjs');
writeFileSync(FORMAT, "export function pageText(locale, en, he) { return locale === 'he' ? he : en; }\n", 'utf8');

const CSS = join(here, 'style.mjs');
writeFileSync(CSS, 'export default {};\n', 'utf8');

registerHooks({
  resolve(specifier, context, next) {
    const hit = (p) => ({ url: pathToFileURL(p).href, shortCircuit: true });
    if (specifier.endsWith('.css')) return hit(CSS);
    if (specifier === 'react') return hit(REACT);
    if (specifier.endsWith('studio/actions')) return hit(ACTIONS);
    if (specifier.endsWith('./format') || specifier.endsWith('shell/format')) return hit(FORMAT);
    return next(specifier, context);
  },
});

const compiled = await transformWithOxc(readFileSync(boundaryPath, 'utf8'), 'ErrorBoundary.jsx', {
  jsx: { runtime: 'classic', pragma: 'React.createElement', pragmaFrag: 'React.Fragment' },
});
const OUT = join(here, 'boundary.mjs');
writeFileSync(OUT, compiled.code, 'utf8');
const { default: ErrorBoundary } = await import(pathToFileURL(OUT).href);

// Render the boundary the way React does: try the child, and on a throw derive
// state from the error and render again.
function renderWith(error, locale) {
  const boundary = new ErrorBoundary({ locale, children: 'THE-CHILD' });
  if (error) boundary.state = ErrorBoundary.getDerivedStateFromError(error);
  const tree = boundary.render();
  return JSON.stringify(tree);
}

const results = {
  healthy: renderWith(null, 'he'),
  stale_he: renderWith(new Error('Failed to fetch dynamically imported module: /assets/x.js'), 'he'),
  stale_en: renderWith(new Error('error loading dynamically imported module'), 'en'),
  other_he: renderWith(new TypeError('x is not a function'), 'he'),
};
writeFileSync(outPath, JSON.stringify(results), 'utf8');
"""


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("boundary")
    harness = tmp / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    out = tmp / "out.json"
    result = subprocess.run(
        [_node(), str(harness), str(BOUNDARY), str(DASHBOARD), str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_healthy_child_is_passed_through_untouched(rendered):
    assert rendered["healthy"] == '"THE-CHILD"'


def test_a_stale_build_is_named_and_promises_the_data_is_untouched(rendered):
    he = rendered["stale_he"]
    assert "שוחררה גרסה חדשה יותר" in he
    assert "לא נשמר ולא אבד" in he, "the operator must be told the data survived"
    assert "רענון" in he, "and given the one action that fixes it"
    assert "error loading" not in he, "the raw exception text is never shown"
    en = rendered["stale_en"]
    assert "A newer version was released" in en


def test_any_other_failure_is_reported_rather_than_blanked(rendered):
    other = rendered["other_he"]
    assert "לא ניתן היה להציג את המסך הזה" in other
    assert "לא נגעו בכשל" in other
    assert "is not a function" not in other, "an exception message is not operator copy"


def test_the_boundary_is_an_ancestor_of_the_workspace_not_its_sibling():
    """The defect this fix was first written with: renderWorkspace was CALLED
    inline, so its throw happened during the parent's render and escaped. It
    must be rendered as an element inside the boundary's subtree."""
    source = SHELL.read_text(encoding="utf-8")
    assert "<ErrorBoundary" in source
    assert "<WorkspaceRoute" in source, "the workspace must render as a component"
    boundary_at = source.index("<ErrorBoundary")
    route_at = source.index("<WorkspaceRoute")
    closed_at = source.index("</ErrorBoundary>")
    assert boundary_at < route_at < closed_at, "the workspace must sit INSIDE the boundary"
    # And the inline call form must not come back.
    assert "{renderWorkspace({" not in source
