"""Compare the actual response bodies, which is the bar a route split is held to.

The schema agreeing is necessary and not sufficient. A route can keep its path,
its method and its response model and still return different bytes, and that is
precisely what a module split can break. So every GET that needs no arguments is
called on both trees and its body compared.

Some fields move on their own between two runs a minute apart: a computed-at
stamp, an elapsed time, a file modification time, a freshness fingerprint. Those
are not normalised away, because silently ignoring a field is how a real
difference hides. They are reported, flagged as time-like, and counted separately
so a reader can tell a clock apart from a defect.

One caveat worth stating, because it decides how a failure here should be read.
This compares the two trees as they actually are, code and data together. A body
can therefore differ because a data file moved rather than because any code did,
and the `moved` check is what tells the two apart. Read a failure here next to
that one before calling it a regression.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from result import Result
from materialise import isolated_env

TIME_LIKE = ("_at", "timestamp", "elapsed", "duration", "_ms", "generated", "computed",
             "fingerprint", "uptime", "now", "modified", "mtime")

# Routes that need arguments before they will terminate, with exactly what is asked
# of each one and why, so a reader can see the question rather than guess at it.
#
# Both preview routes run the day optimizer twice over whatever segments the query
# selects. Unscoped that is every channel and every day, which 06-baseline.md
# measured at 55.60s scoped to no channel and running past thirty minutes with no
# arguments at all. Scoped to one channel-day it is bounded work, measured there at
# 16.55s. The per-route alarm cannot rescue the unscoped call, because SIGALRM is
# delivered between bytecodes and these spend their time inside numpy, so bounding
# the question is the only honest way to ask it.
#
# The two values are not arbitrary. The channel is the operator's own, read from
# data/kairos_settings.json where operator_channel is רשת 13, and the day is inside
# the saved plan's window: Programmes.csv carries 30 days for that channel across
# November 2024, and 2024-11-23 is one of them, the same day 06-baseline.md timed.
# Both are inputs rather than outputs, so they are stable across trees.
ROUTE_ARGS = {
    "/api/constraints/effect": {
        "params": {"channel": "רשת 13", "day": "2024-11-23"},
        "why": "unbounded without a scope; one channel-day is the bounded form of the same question",
    },
    "/api/overrides/effect": {
        "params": {"channel": "רשת 13", "day": "2024-11-23"},
        "why": "unbounded without a scope; one channel-day is the bounded form of the same question",
    },
}

# Routes that genuinely cannot be compared, with the reason. Empty today: the two
# that used to sit here were given bounded arguments above rather than excluded.
# Anything added here is a hole in the proof and reads as unproven, never as passing.
UNCOMPARABLE: dict[str, str] = {}

PROBE = r"""
import hashlib, json, signal, sys
sys.path.insert(0, sys.argv[1])
DEADLINE = float(sys.argv[2])
OUT_PATH = sys.argv[3]
ROUTE_ARGS = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}
UNCOMPARABLE = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
from fastapi.testclient import TestClient
from kairos_api.server import app


class _Deadline(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Deadline()


# One route with no bound must not swallow the whole check. GET /api/constraints/effect
# with no parameters is the known case: the baseline measured it running past thirty
# minutes. A route that trips this is reported as untimed, never as agreeing.
signal.signal(signal.SIGALRM, _on_alarm)

client = TestClient(app)
schema = app.openapi()
targets = []
for path, item in (schema.get("paths") or {}).items():
    if "{" in path:
        continue
    if "get" not in {k.lower() for k in item}:
        continue
    op = item.get("get") or {}
    required = [p for p in (op.get("parameters") or []) if p.get("required")]
    if required:
        continue
    targets.append(path)

out = {}


def flush():
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, default=str)

for path in sorted(targets):
    if path in UNCOMPARABLE:
        out[path] = {"status": None, "timed_out": True,
                     "error": "not comparable: " + UNCOMPARABLE[path]}
        flush()
        continue
    spec = ROUTE_ARGS.get(path) or {}
    params = spec.get("params") or None
    signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    try:
        resp = client.get(path, params=params) if params else client.get(path)
        body = resp.content
        out[path] = {
            "status": resp.status_code,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "params": params,
            "json": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None,
        }
    except _Deadline:
        out[path] = {"status": None, "timed_out": True,
                     "error": "no response within %gs" % DEADLINE}
    except Exception as exc:
        out[path] = {"status": None, "error": "%s: %s" % (type(exc).__name__, exc)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    flush()
sys.stdout.write("<<<BODIES>>>" + json.dumps(out, default=str))
"""


def _dump(python: str, tree: Path, scratch: Path, timeout: int, deadline: float) -> tuple[dict | None, str]:
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / "probe_bodies.py"
    probe.write_text(PROBE, encoding="utf-8")
    partial = scratch / "partial.json"
    args = [python, str(probe), str(tree), str(deadline), str(partial),
            json.dumps(ROUTE_ARGS, ensure_ascii=False), json.dumps(UNCOMPARABLE, ensure_ascii=False)]
    try:
        proc = subprocess.run(args, cwd=str(tree), env=isolated_env(scratch),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hang still leaves everything measured before it, and those routes are
        # real evidence. The ones never reached come back as unproven, not as agreeing.
        if partial.is_file():
            return json.loads(partial.read_text()), ""
        return None, "timed out after %ds with nothing written" % timeout
    if "<<<BODIES>>>" not in proc.stdout:
        if partial.is_file():
            return json.loads(partial.read_text()), ""
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, "\n".join(tail[-6:]) if tail else "no output"
    return json.loads(proc.stdout.split("<<<BODIES>>>", 1)[1]), ""


def _differing_keys(a, b, prefix: str = "") -> list[str]:
    """Where two payloads part company, named by key path rather than by diff noise."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        out: list[str] = []
        for k in keys:
            if k not in a:
                out.append("%s%s (only on the working tree)" % (prefix, k))
            elif k not in b:
                out.append("%s%s (only on the reference)" % (prefix, k))
            else:
                out.extend(_differing_keys(a[k], b[k], "%s%s." % (prefix, k)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return ["%s[] length %d against %d" % (prefix, len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                out.extend(_differing_keys(x, y, "%s[%d]." % (prefix, i)))
        return out[:6]
    if a != b:
        return [prefix.rstrip(".")]
    return []


def _looks_time_like(key_path: str) -> bool:
    low = key_path.lower()
    return any(token in low for token in TIME_LIKE)


def check_response_bodies(python: str, ref: Path, work: Path, scratch: Path, timeout: int,
                          deadline: float = 30.0) -> Result:
    r = Result("bodies", "Response bodies of every GET the harness can ask")
    started = time.time()

    ref_out, err = _dump(python, ref, scratch / "bodies-ref", timeout, deadline)
    if ref_out is None:
        return r.cannot_check("reference side: %s" % err)
    work_out, err = _dump(python, work, scratch / "bodies-work", timeout, deadline)
    if work_out is None:
        return r.cannot_check("working side: %s" % err)
    r.seconds = time.time() - started

    shared = sorted(set(ref_out) & set(work_out))
    identical, time_only, real, untimed = [], [], [], []
    for path in shared:
        a, b = ref_out[path], work_out[path]
        if a.get("timed_out") or b.get("timed_out"):
            untimed.append(path)
            why = a.get("error") or b.get("error") or "no response inside the per-route deadline"
            r.note("%s: unproven, %s" % (path, why))
            continue
        if a.get("error") or b.get("error"):
            untimed.append(path)
            r.note("%s: could not be called (%s)" % (path, a.get("error") or b.get("error")))
            continue
        if a["status"] != b["status"]:
            real.append(path)
            r.note("%s: status %s against %s" % (path, a["status"], b["status"]))
            continue
        if a["sha256"] == b["sha256"]:
            identical.append(path)
            continue
        keys = _differing_keys(a.get("json"), b.get("json")) or ["(body differs, not json)"]
        if keys and all(_looks_time_like(k) for k in keys):
            time_only.append(path)
            r.note("%s: differs only in time-like fields: %s" % (path, ", ".join(keys[:4])))
        else:
            real.append(path)
            r.note("%s: %s" % (path, ", ".join(keys[:6]) + (" ..." if len(keys) > 6 else "")))

    measurements = {"compared": len(shared), "identical": len(identical),
                    "time_like_only": len(time_only), "differing": len(real),
                    "unproven": len(untimed)}
    r.note("%d of %d GET routes returned byte-identical bodies" % (len(identical), len(shared)))
    for path, spec in sorted(ROUTE_ARGS.items()):
        state = ("identical" if path in identical else
                 "differs" if path in real else
                 "time-like only" if path in time_only else
                 "unproven" if path in untimed else "not reached")
        r.note("asked with arguments: %s %s, because %s; result: %s"
               % (path, spec["params"], spec["why"], state))
    for path, why in sorted(UNCOMPARABLE.items()):
        r.note("declared not comparable: %s, because %s" % (path, why))

    if real:
        return r.failed("%d route(s) return different bodies" % len(real), **measurements)
    if untimed:
        return r.cannot_check(
            "%d of %d routes could not be timed or called, so they are unproven; the other %d matched"
            % (len(untimed), len(shared), len(identical) + len(time_only)))
    if time_only:
        return r.passed("all %d routes match once time-like fields are accounted for, and those %d are named above"
                        % (len(shared), len(time_only)), **measurements)
    return r.passed("all %d routes byte-identical" % len(shared), **measurements)
