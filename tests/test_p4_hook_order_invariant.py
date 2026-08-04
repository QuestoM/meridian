"""P4: no hook in the clients tree runs on only some of its renders.

``test_p4_first_paint_hook_order.py`` executes one component through the exact
sequence that crashed. This file states the property that component broke, over
every source file in the destination, so the next one to break it is named at the
line rather than found in a browser.

The property is React's own rule: a component calls the same hooks, in the same
order, on every render. Two shapes break it. A hook below a return that a
condition can take runs only once the data arrives, which is the defect that
shipped in ``cdc3935f`` and unmounted the application. A hook inside a condition
or a loop runs only when that branch is taken, which fails the same way.

The check is a parse, not a search: the dashboard's own Babel parser reads each
file, and each function body is walked in statement order. It is scoped to
``src/clients``, which is P4's tree. Other destinations are other builders' files
and this file does not assert against them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
TREE = DASHBOARD / "src" / "clients"
BROKEN_COMMIT = "cdc3935f"

CHECKER = """
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const [dashboardDir, ...files] = process.argv.slice(2);
const req = createRequire(pathToFileURL(join(dashboardDir, 'resolve-from-here.js')));
const parser = req('@babel/parser');

const HOOK = /^use[A-Z]/;
const FUNCTIONS = new Set([
  'FunctionDeclaration',
  'FunctionExpression',
  'ArrowFunctionExpression',
  'ObjectMethod',
  'ClassMethod',
]);
const BRANCHES = new Set([
  'IfStatement',
  'ConditionalExpression',
  'LogicalExpression',
  'SwitchStatement',
  'ForStatement',
  'ForOfStatement',
  'ForInStatement',
  'WhileStatement',
  'DoWhileStatement',
  'TryStatement',
]);

function children(node) {
  const out = [];
  for (const key of Object.keys(node)) {
    if (key === 'loc' || key.endsWith('Comments')) continue;
    const value = node[key];
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item && typeof item.type === 'string') out.push(item);
      });
    } else if (value && typeof value.type === 'string') {
      out.push(value);
    }
  }
  return out;
}

function callee(node) {
  if (node.callee && node.callee.type === 'Identifier') return node.callee.name;
  if (node.callee && node.callee.type === 'MemberExpression' && node.callee.property.type === 'Identifier') {
    return node.callee.property.name;
  }
  return '';
}

// Every hook call inside one statement. A nested function is its own scope and
// is reached by the outer walk instead.
function hooksIn(node, branched, found) {
  if (FUNCTIONS.has(node.type)) return found;
  if (node.type === 'CallExpression' && HOOK.test(callee(node))) {
    found.push({ name: callee(node), line: node.loc.start.line, branched });
  }
  const deeper = branched || BRANCHES.has(node.type);
  children(node).forEach((child) => hooksIn(child, deeper, found));
  return found;
}

function returnsIn(node, found) {
  if (FUNCTIONS.has(node.type)) return found;
  if (node.type === 'ReturnStatement') found.push(node.loc.start.line);
  children(node).forEach((child) => returnsIn(child, found));
  return found;
}

const violations = [];
let inspected = 0;

function scan(node, file) {
  if (FUNCTIONS.has(node.type) && node.body && node.body.type === 'BlockStatement') {
    const statements = node.body.body;
    let escaped = 0;
    statements.forEach((statement, index) => {
      hooksIn(statement, false, []).forEach((hook) => {
        inspected += 1;
        if (escaped) {
          violations.push({ file, ...hook, reason: `after the return at line ${escaped}` });
        } else if (hook.branched) {
          violations.push({ file, ...hook, reason: 'inside a condition or a loop' });
        }
      });
      if (!escaped && index < statements.length - 1) {
        const returns = returnsIn(statement, []);
        if (returns.length) [escaped] = returns;
      }
    });
  }
  children(node).forEach((child) => scan(child, file));
}

for (const file of files) {
  const ast = parser.parse(readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  scan(ast.program, file);
}
process.stdout.write(JSON.stringify({ violations, inspected, files: files.length }));
"""

# A component shaped like the one that shipped, used to prove the check bites
# even when this clone has no history.
SAMPLE = """
import React, { useMemo, useState } from 'react';

export default function Panel({ payload }) {
  const [query, setQuery] = useState('');
  if (!payload) {
    return <p>loading</p>;
  }
  const total = useMemo(() => payload.rows.length, [payload]);
  return <p>{query}{total}</p>;
}
"""

BRANCHED_SAMPLE = """
import React, { useMemo } from 'react';

export default function Panel({ payload }) {
  if (payload) {
    const total = useMemo(() => payload.rows.length, [payload]);
    return <p>{total}</p>;
  }
  return <p>loading</p>;
}
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the sources cannot be parsed here")
    if not (DASHBOARD / "node_modules" / "@babel" / "parser").is_dir():
        pytest.skip("the dashboard has no installed Babel parser, so the sources cannot be parsed here")
    return found


def _check(tmp_path: Path, files: list[Path]) -> dict:
    checker = tmp_path / "hook-order.mjs"
    checker.write_text(CHECKER, encoding="utf-8")
    result = subprocess.run(
        [_node(), str(checker), str(DASHBOARD), *[str(path) for path in files]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def sources() -> list[Path]:
    found = sorted(TREE.glob("*.jsx")) + sorted(TREE.glob("*.js"))
    assert found, "the clients tree has no sources, so this file is measuring nothing"
    return found


def test_every_hook_in_the_clients_tree_runs_on_every_render(tmp_path, sources):
    """The property the crash broke, over the whole destination."""
    report = _check(tmp_path, sources)
    assert report["files"] == len(sources)
    assert report["inspected"] >= 40, "the parser found almost no hooks, so this pass would be vacuous"
    named = [
        f"{Path(item['file']).name}:{item['line']} {item['name']} {item['reason']}"
        for item in report["violations"]
    ]
    assert named == []


def test_the_check_names_the_defect_that_shipped(tmp_path):
    """Proof the check bites, run against the file that actually shipped."""
    read = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BROKEN_COMMIT}:tv-break-dashboard/src/clients/ClientTree.jsx"],
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        pytest.skip(f"{BROKEN_COMMIT} is not in this clone, so the shipped defect cannot be replayed")
    broken = tmp_path / "ClientTree.jsx"
    broken.write_text(read.stdout, encoding="utf-8")
    report = _check(tmp_path, [broken])
    assert len(report["violations"]) == 1
    found = report["violations"][0]
    assert found["name"] == "useMemo"
    assert found["line"] == 213
    assert found["reason"] == "after the return at line 196"


def test_the_check_bites_on_both_shapes_without_needing_history(tmp_path):
    """A hook below a conditional return, and a hook inside the condition itself."""
    below = tmp_path / "Below.jsx"
    below.write_text(SAMPLE, encoding="utf-8")
    inside = tmp_path / "Inside.jsx"
    inside.write_text(BRANCHED_SAMPLE, encoding="utf-8")
    report = _check(tmp_path, [below, inside])
    reasons = sorted(item["reason"] for item in report["violations"])
    assert len(reasons) == 2
    assert reasons[0].startswith("after the return at line")
    assert reasons[1] == "inside a condition or a loop"
