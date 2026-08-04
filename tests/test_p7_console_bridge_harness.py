"""The plumbing behind the console's real-browser measurements.

Extracted from ``test_p7_console_bridge_session.py`` when that file reached the
450-line law, and nothing changed on the way out: the same stand-in server, the
same page, the same bundler and the same headless Chrome. It defines no test of
its own. The scenario and every assertion stayed in the file that owns them.

Two things were added later, both because a training run measurement needed
something a static server cannot express, and both inert for the scenarios that
do not ask for them:

- **A route may answer a sequence.** Hand a list instead of one body and each
  read takes the next, with the last one repeating for ever. A training run ends
  on the server with no click behind it, so the only way to reproduce that here
  is a route whose answer changes between two reads that nobody triggered.
- **A write may be answered.** ``do_POST`` answers the paths a scenario hands in
  ``writes`` and still refuses every other one, so a scenario that presses a
  control the product would POST gets the product's own answer rather than a
  404 the panel would have to interpret.

The stand-in reproduces the two things the bridge depends on and nothing else:
the application mounts into ``#root``, and the shell returns the sign-in card
INSTEAD of the workspace, so an authentication swaps ``#root``'s only child.
Both of those live in frozen files and are asserted separately from source, so a
shell that stops behaving that way fails a test rather than quietly invalidating
this page.

``/api/model/console`` is answered here with an honest header payload: this
harness has no artifacts on disk, so the version reads unavailable with its
reason, exactly as the product's own console would render on a tree with no
trained model. Any other route is answered only if the caller hands it a body,
and a scenario that hands none leaves every section route unanswered, so each
panel shows its own unreachable state, which is what a console with no server
behind it is supposed to look like. The bodies a scenario does hand over are
built by the product's own payload code against the real artifacts, never typed,
so a measurement made through this page is made on the figures the product
serves.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "tv-break-dashboard"
BRIDGE = FRONTEND / "src" / "model" / "console-bridge.jsx"
SHELL_NAV = FRONTEND / "src" / "shell" / "nav.js"
VITE = FRONTEND / "node_modules" / ".bin" / "vite"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# How long the browser gets to finish a whole scenario.
SCENARIO_BUDGET_S = 120

SESSIONS = {
    "out": None,
    "channel": {
        "username": "chan1", "display_name": "chan1", "role": "operator",
        "affiliation": "channel", "job": "planner", "must_change_password": False,
    },
    "company": {
        "username": "steward1", "display_name": "steward1", "role": "operator",
        "affiliation": "company", "job": "model_steward", "must_change_password": False,
    },
}

# The console header on a tree with no trained artifact: unavailable, with the
# reason, which is the state the console itself renders rather than a stand-in
# for one that would be there.
CONSOLE_PAYLOAD = {
    "model_version": {
        "available": False,
        "reason_he": "אין בדיסק מודל מאומן.",
        "reason_en": "No trained model on disk.",
    },
    "gate_counts": {},
    "gate_states": [],
    "activation": {"available": False},
}

HARNESS_HTML = """<!doctype html>
<html lang="en">
  <head><meta charset="UTF-8" /><title>bridge harness</title></head>
  <body>
    <div id="root"><div class="login-screen" dir="rtl" lang="he"></div></div>
    <script type="module" src="/harness.js"></script>
  </body>
</html>
"""

VITE_CONFIG = """
export default {
  root: '%(root)s',
  logLevel: 'error',
  build: { outDir: '%(out)s', emptyOutDir: true },
};
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Harness(ThreadingHTTPServer):
    """The stand-in server: one session flag, one result slot, and the bundle."""

    daemon_threads = True

    def __init__(self, address, directory: Path, payloads: "dict | None" = None,
                 writes: "dict | None" = None):
        self.mode = "out"
        self.result: "dict | None" = None
        self.payloads = {"/api/model/console": CONSOLE_PAYLOAD, **(payloads or {})}
        self.writes = dict(writes or {})
        self.reads: "dict[str, int]" = {}
        self.lock = threading.Lock()
        super().__init__(address, partial(_HarnessHandler, directory=str(directory)))

    def count(self, key: str) -> int:
        """One more call on this key. Returns how many there were before it."""
        with self.lock:
            seen = self.reads.get(key, 0)
            self.reads[key] = seen + 1
        return seen

    def next_body(self, prefix: str, body):
        """One body, or the next of a sequence with the last one repeating.

        Counted on the server, so a scenario has a count of what was really
        asked for beside whatever the page reports about itself.
        """
        seen = self.count(prefix)
        if not isinstance(body, list):
            return body
        if not body:
            return {}
        return body[min(seen, len(body) - 1)]


class _HarnessHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: D102 - the browser is noisy
        return

    def _send(self, status: int, body: "dict") -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - the base class spells it this way
        if self.path.startswith("/api/auth/me"):
            session = SESSIONS[self.server.mode]
            if session is None:
                self._send(401, {"detail": "not signed in"})
            else:
                self._send(200, session)
            return
        for prefix, body in self.server.payloads.items():
            if self.path.startswith(prefix):
                self._send(200, self.server.next_body(prefix, body))
                return
        if self.path.startswith("/testctl/session"):
            mode = self.path.split("as=")[-1]
            self.server.mode = mode if mode in SESSIONS else "out"
            self._send(200, {"mode": self.server.mode})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - the base class spells it this way
        if self.path.startswith("/testctl/result"):
            length = int(self.headers.get("Content-Length") or 0)
            self.server.result = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, {"stored": True})
            return
        for prefix, body in self.server.writes.items():
            if self.path.startswith(prefix):
                # Counted under its own key. The browser's resource timeline
                # carries no method, so a scenario comparing its own count with
                # the server's has to be able to add the writes back.
                self.server.count(f"POST {prefix}")
                self._send(200, body)
                return
        self._send(404, {"detail": "no such path"})


def skip_unless_a_real_browser_is_available() -> None:
    """Three preconditions, each skipped with its own reason rather than failed."""
    if not (FRONTEND / "node_modules").exists() or not VITE.exists():
        pytest.skip("no node_modules on this tree, and installing one is not this test's job")
    if not CHROME.exists():
        pytest.skip("no Chrome on this machine, and this measurement needs a real browser")
    if shutil.which("node") is None:
        pytest.skip("no node on this machine")


def build_harness(work: Path, harness_js: str) -> Path:
    """Bundle the real bridge with the product's own bundler. Returns the dist.

    The work directory is resolved first because the bundler normalises paths to
    their real form, and on this platform the temporary directory is reached
    through a symlink: an import written relative to the unresolved path lands
    somewhere else once the bundler has resolved the root.
    """
    work = work.resolve()
    source = work / "src"
    source.mkdir(parents=True, exist_ok=True)
    (source / "index.html").write_text(HARNESS_HTML, encoding="utf-8")
    (source / "harness.js").write_text(harness_js, encoding="utf-8")
    dist = work / "dist"
    config = work / "vite.harness.config.mjs"
    config.write_text(VITE_CONFIG % {"root": source.as_posix(), "out": dist.as_posix()},
                      encoding="utf-8")
    built = subprocess.run(
        [str(VITE), "build", "--config", str(config)],
        cwd=str(FRONTEND), capture_output=True, text=True, timeout=300,
    )
    if built.returncode != 0 or not (dist / "index.html").is_file():
        tail = (built.stderr or built.stdout).strip().splitlines()[-12:]
        pytest.fail("the harness bundle did not build: " + " / ".join(tail))
    return dist


def run_scenario(dist: Path, work: Path, payloads: "dict | None" = None,
                 writes: "dict | None" = None) -> "dict":
    """Serve the bundle, run one headless Chrome against it, return its report."""
    server = _Harness(("127.0.0.1", _free_port()), dist, payloads, writes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    chrome = subprocess.Popen(
        [
            str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
            f"--user-data-dir={work / 'chrome'}", url,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + SCENARIO_BUDGET_S
        while time.time() < deadline and server.result is None:
            time.sleep(0.1)
        result = server.result
        if isinstance(result, dict):
            # What the server was actually asked for, counted by the server. A
            # page reporting its own request count is a page marking its own
            # homework, so both counts are taken and a scenario may compare them.
            result["server_reads"] = dict(server.reads)
    finally:
        chrome.kill()
        chrome.wait(timeout=30)
        server.shutdown()
        server.server_close()
    if result is None:
        pytest.fail(f"the browser never reported a result within {SCENARIO_BUDGET_S}s")
    return result


# The shell's own address resolver, driven rather than read.
#
# The console addresses a page in the frozen shell, so the console's control is
# only a destination because that resolver says so. It is a function in a file
# this piece may not write, and it reads `window.location.hash`, so the honest
# way to depend on it is to bundle it with the product's own bundler and ask it,
# in a browser, what each address resolves to. Reading the line that does the
# resolving is what a previous round did, and wave one replaced that line with
# an equivalent one, which turned a working behaviour into a red test.
NAV_PROBE_JS = """
import { viewFromLocation } from '%(nav)s';

const HASHES = %(hashes)s;
const resolved = {};

HASHES.forEach((hash) => {
  window.location.hash = hash;
  resolved[hash] = viewFromLocation();
});

fetch('/testctl/result', {
  method: 'POST',
  body: JSON.stringify({ resolved, href: window.location.href }),
});
"""


def resolve_shell_views(work: Path, hashes: "list[str]") -> "dict":
    """What the frozen shell resolves each address to, measured in a browser."""
    skip_unless_a_real_browser_is_available()
    probe = NAV_PROBE_JS % {
        "nav": os.path.relpath(SHELL_NAV, work.resolve() / "src"),
        "hashes": json.dumps(hashes),
    }
    return run_scenario(build_harness(work, probe), work)
