"""Diff the published API surface between two trees, route by route.

The app cannot be imported twice in one process, so each side is dumped in its
own subprocess with its own tree on the import path. Both sides resolve their
data from `Path(__file__).parents[...]`, so a subprocess pointed at a copied
tree reads that copy's data and nothing from the shared one.

The comparison is deliberately not a count. Counts agreeing while a route moved
from one module to another with a different response model is exactly the defect
a refactor produces, so paths, methods and response schemas are compared by
identity and every difference is named.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from result import Result
from materialise import isolated_env

WRITE_METHODS = {"post", "put", "patch", "delete"}

# A wave is not always a pure refactor. Section 8.2 lets a piece add a path if it
# published and froze it under section 8.8 first, and spec.md:1454 states the bar
# in those terms: the reference paths unchanged, plus exactly what was published.
# So a declared addition is reported and allowed, while an undeclared one still
# fails. Removals and in-place changes fail either way, which is the real risk a
# late mount carries.
DECLARED_APPENDS = {
    ("put", "/api/auth/job"): "W0-4, contracts/W0-4.md:25",
    ("get", "/api/advertisers/identity"): "W0-3, the identity read",
}

PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from kairos_api.server import app
schema = app.openapi()
out = {"paths": {}, "components": sorted((schema.get("components") or {}).get("schemas", {}))}
for path, item in (schema.get("paths") or {}).items():
    for method, op in item.items():
        if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
            continue
        responses = op.get("responses") or {}
        models = {}
        for code, body in responses.items():
            content = (body or {}).get("content") or {}
            js = content.get("application/json") or {}
            models[code] = json.dumps((js.get("schema") or {}), sort_keys=True)
        out["paths"].setdefault(path, {})[method.lower()] = {
            "operation_id": op.get("operationId"),
            "responses": models,
            "params": sorted((p.get("name", ""), p.get("in", "")) for p in (op.get("parameters") or [])),
        }
sys.stdout.write("<<<SCHEMA>>>" + json.dumps(out))
"""


def _dump(python: str, tree: Path, scratch: Path) -> tuple[dict | None, str]:
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / "probe_openapi.py"
    probe.write_text(PROBE, encoding="utf-8")
    proc = subprocess.run([python, str(probe), str(tree)], cwd=str(tree),
                          env=isolated_env(scratch), capture_output=True, text=True, timeout=600)
    if "<<<SCHEMA>>>" not in proc.stdout:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, "\n".join(tail[-6:]) if tail else "no output"
    return json.loads(proc.stdout.split("<<<SCHEMA>>>", 1)[1]), ""


def _ops(schema: dict) -> set[tuple[str, str]]:
    return {(m, p) for p, methods in schema["paths"].items() for m in methods}


def check_api_surface(python: str, ref: Path, work: Path, scratch: Path,
                      declared: dict[str, int]) -> Result:
    r = Result("api", "API surface, route by route")
    started = time.time()

    ref_schema, ref_err = _dump(python, ref, scratch / "api-ref")
    if ref_schema is None:
        return r.cannot_check("the reference tree's app would not import: %s" % ref_err)
    work_schema, work_err = _dump(python, work, scratch / "api-work")
    if work_schema is None:
        return r.cannot_check("the working tree's app would not import: %s" % work_err)
    r.seconds = time.time() - started

    ref_ops, work_ops = _ops(ref_schema), _ops(work_schema)
    added, removed = sorted(work_ops - ref_ops), sorted(ref_ops - work_ops)

    changed: list[str] = []
    for method, path in sorted(ref_ops & work_ops):
        a = ref_schema["paths"][path][method]
        b = work_schema["paths"][path][method]
        if a["responses"] != b["responses"]:
            codes = sorted(set(a["responses"]) | set(b["responses"]))
            moved = [c for c in codes if a["responses"].get(c) != b["responses"].get(c)]
            changed.append("%s %s: response model changed on %s" % (method.upper(), path, ", ".join(moved)))
        elif a["params"] != b["params"]:
            changed.append("%s %s: parameters changed, %s to %s" % (method.upper(), path, a["params"], b["params"]))

    counts = {
        "reference_paths": len(ref_schema["paths"]),
        "reference_operations": len(ref_ops),
        "working_paths": len(work_schema["paths"]),
        "working_operations": len(work_ops),
        "working_writes": sum(1 for m, _ in work_ops if m in WRITE_METHODS),
        "reference_writes": sum(1 for m, _ in ref_ops if m in WRITE_METHODS),
    }

    undeclared = [op for op in added if op not in DECLARED_APPENDS]
    for method, path in added:
        who = DECLARED_APPENDS.get((method, path))
        r.note("added: %s %s (%s)" % (method.upper(), path,
                                      "declared by %s" % who if who else "NOT DECLARED"))
    for method, path in removed:
        r.note("removed: %s %s" % (method.upper(), path))
    for line in changed:
        r.note(line)

    baseline_note = "declared baseline %d paths, %d operations, %d writes" % (
        declared.get("paths", 0), declared.get("operations", 0), declared.get("writes", 0))
    matches_declared = (counts["reference_paths"] == declared.get("paths")
                        and counts["reference_operations"] == declared.get("operations")
                        and counts["reference_writes"] == declared.get("writes"))
    r.note("reference measures %d paths, %d operations, %d writes; %s; reference %s the declared baseline"
           % (counts["reference_paths"], counts["reference_operations"], counts["reference_writes"],
              baseline_note, "matches" if matches_declared else "DOES NOT match"))

    if undeclared or removed or changed:
        return r.failed("%d undeclared addition(s), %d removed, %d changed in place"
                        % (len(undeclared), len(removed), len(changed)), **counts)
    if added:
        return r.passed("every reference route intact, plus %d declared addition(s): "
                        "%d paths, %d operations, %d writes"
                        % (len(added), counts["working_paths"], counts["working_operations"],
                           counts["working_writes"]), **counts)
    return r.passed("identical: %d paths, %d operations, %d writes on both sides"
                    % (counts["working_paths"], counts["working_operations"], counts["working_writes"]), **counts)
