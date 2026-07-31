"""Build both frontends and compare the rendered text of every route.

The comparison is run against one API, not two. Whether the API moved is
already check one's job, so serving both builds from the same backend isolates
the frontend as the only variable. A difference here is then a frontend
difference and nothing else.

Chrome is launched privately, on its own port and its own profile, because the
shared browser in this environment belongs to other agents and has twice been
navigated out from under a measurement.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from result import Result
from materialise import isolated_env

ROUTES = ["Overview", "Optimizer", "Schedule", "Inventory", "Break Library", "Campaigns",
          "Forecasts", "Calendar", "Reports", "Data", "Advertisers", "Agencies", "Pricing",
          "Overrides", "Assistant", "Versions", "Settings"]

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build(tree: Path, timeout: int) -> tuple[Path | None, str]:
    front = tree / "tv-break-dashboard"
    if not (front / "node_modules").exists():
        return None, "no node_modules on this side, and installing one is not this harness's job"
    try:
        proc = subprocess.run(["npm", "run", "build"], cwd=str(front), capture_output=True,
                              text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return None, "build did not run: %s" % exc
    dist = front / "dist"
    if proc.returncode != 0 or not (dist / "index.html").is_file():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return None, "build failed: %s" % " / ".join(tail)
    return dist, ""


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler(directory: Path, api_port: int | None):
    """Serve the build, and forward /api to one backend.

    Both builds are pointed at the same API on purpose. Whether the API moved is
    check one's question; holding it fixed here makes the frontend the only
    variable, so a difference in rendered text is a frontend difference and
    cannot be the backend disagreeing with itself.
    """

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):  # keep the harness output readable
            return

        def do_GET(self):
            if api_port and self.path.startswith("/api"):
                url = "http://127.0.0.1:%d%s" % (api_port, self.path)
                try:
                    with urllib.request.urlopen(url, timeout=60) as up:
                        body, status = up.read(), up.status
                        ctype = up.headers.get("content-type", "application/json")
                except Exception as exc:
                    body, status, ctype = str(exc).encode(), 502, "text/plain"
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    return Handler


def _serve(directory: Path, api_port: int | None = None) -> tuple[_Server, int]:
    port = _free_port()
    httpd = _Server(("127.0.0.1", port), _handler(directory, api_port))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _start_api(python: str, tree: Path, scratch: Path, env_builder) -> tuple[subprocess.Popen | None, int]:
    """One backend, started from the reference tree, on a port of its own."""
    port = _free_port()
    proc = subprocess.Popen(
        [python, "-m", "uvicorn", "kairos_api.server:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(tree), env=env_builder(scratch / "api"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/api/settings" % port, timeout=3):
                return proc, port
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    return None, 0


def _warm(api_port: int, paths: tuple[str, ...] = ("/api/overview", "/api/schedule", "/api/settings",
                                                  "/api/impact", "/api/parameters")) -> None:
    """Pay the cold cost once, before either side is measured.

    Both builds share one backend, so whichever is measured first would otherwise
    absorb every cold read and the second would look faster purely for being
    second. That is a property of the measurement, not of the frontends, and it
    is exactly the kind of artifact that reads as a real difference.
    """
    for path in paths:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d%s" % (api_port, path), timeout=120):
                pass
        except Exception:
            pass


def _cdp(port: int, url: str, routes: list[str], settle: float) -> dict[str, str] | str:
    """Drive one private Chrome and read the rendered text of each route."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return "the websockets package is not importable, so Chrome cannot be driven"
    import asyncio
    import websockets as ws

    async def run() -> dict[str, str] | str:
        for _ in range(80):
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/json/version" % port, timeout=2):
                    break
            except Exception:
                await asyncio.sleep(0.25)
        else:
            return "Chrome never opened its debugging port"
        with urllib.request.urlopen("http://127.0.0.1:%d/json/list" % port, timeout=5) as fh:
            targets = json.loads(fh.read())
        pages = [t for t in targets if t.get("type") == "page"]
        if not pages:
            return "Chrome exposed no page target"
        out: dict[str, str] = {}
        async with ws.connect(pages[0]["webSocketDebuggerUrl"], max_size=200 * 1024 * 1024) as sock:
            n = 0

            async def send(method: str, **params):
                nonlocal n
                n += 1
                await sock.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await sock.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            await send("Page.enable")
            await send("Runtime.enable")
            for route in routes:
                await send("Page.navigate", url="%s#%s" % (url, route.replace(" ", "%20")))
                await asyncio.sleep(settle)
                res = await send("Runtime.evaluate",
                                 expression="document.body ? document.body.innerText : ''",
                                 returnByValue=True)
                out[route] = (res.get("result") or {}).get("value") or ""
        return out

    return asyncio.run(run())


def _launch_chrome(profile: Path) -> tuple[subprocess.Popen | None, int]:
    if not Path(CHROME).exists():
        return None, 0
    port = _free_port()
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
         "--remote-debugging-port=%d" % port, "--user-data-dir=%s" % profile,
         "--window-size=1600,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, port


def check_frontend_text(python: str, ref: Path, work: Path, scratch: Path, build_timeout: int,
                        settle: float) -> Result:
    r = Result("frontend", "Rendered text of all seventeen routes")
    started = time.time()

    if shutil.which("npm") is None:
        return r.cannot_check("npm is not on PATH")
    ref_dist, err = _build(ref, build_timeout)
    if ref_dist is None:
        return r.cannot_check("reference build: %s" % err)
    work_dist, err = _build(work, build_timeout)
    if work_dist is None:
        return r.cannot_check("working build: %s" % err)

    api_proc, api_port = _start_api(python, ref, scratch, isolated_env)
    if api_proc is None:
        r.note("no backend came up, so both sides render their offline state; "
               "that still compares the shell but not the data-bearing pages")
    ref_srv, ref_port = _serve(ref_dist, api_port or None)
    work_srv, work_port = _serve(work_dist, api_port or None)
    chrome, cdp_port = _launch_chrome(scratch / "chrome-profile")
    if chrome is None:
        ref_srv.shutdown(); work_srv.shutdown()
        return r.cannot_check("Chrome is not installed at %s" % CHROME)
    try:
        if api_port:
            _warm(api_port)
        # Measured in both orders. If a route only disagrees in one order, the
        # disagreement is the measurement warming up, not the frontends differing.
        ref_text = _cdp(cdp_port, "http://127.0.0.1:%d/" % ref_port, ROUTES, settle)
        work_text = _cdp(cdp_port, "http://127.0.0.1:%d/" % work_port, ROUTES, settle)
        ref_again = _cdp(cdp_port, "http://127.0.0.1:%d/" % ref_port, ROUTES, settle)
    finally:
        chrome.terminate()
        ref_srv.shutdown()
        work_srv.shutdown()
        if api_proc is not None:
            api_proc.terminate()
    r.seconds = time.time() - started

    if isinstance(ref_text, str):
        return r.cannot_check("reference render: %s" % ref_text)
    if isinstance(work_text, str):
        return r.cannot_check("working render: %s" % work_text)
    if isinstance(ref_again, str):
        ref_again = {}

    differing, unstable = [], []
    for route in ROUTES:
        a, b = ref_text.get(route, ""), work_text.get(route, "")
        if a != b and ref_again and ref_again.get(route, "") != a:
            # The reference did not even agree with itself between two passes, so
            # this route is not steady enough to compare. Reported, not counted.
            unstable.append(route)
            r.note("%s: the reference rendered %d chars then %d, so it is not steady; not compared"
                   % (route, len(a), len(ref_again.get(route, ""))))
            continue
        if a != b:
            differing.append(route)
            r.note("%s: %d chars on the reference, %d on the working tree" % (route, len(a), len(b)))
            only_ref = sorted(set(a.split("\n")) - set(b.split("\n")))[:3]
            only_work = sorted(set(b.split("\n")) - set(a.split("\n")))[:3]
            for line in only_ref:
                r.note("    only on reference: %s" % line[:90])
            for line in only_work:
                r.note("    only on working:   %s" % line[:90])

    measurements = {"routes": len(ROUTES), "differing": len(differing), "unstable": len(unstable),
                    "empty_on_both": sum(1 for x in ROUTES if not ref_text.get(x) and not work_text.get(x))}
    if measurements["empty_on_both"] == len(ROUTES):
        return r.cannot_check("every route rendered empty on both sides, so the comparison proves nothing")
    if differing:
        return r.failed("%d of %d routes render different text" % (len(differing), len(ROUTES)), **measurements)
    if unstable:
        return r.cannot_check(
            "%d of %d routes never settled on the reference, so they are unproven; the other %d matched"
            % (len(unstable), len(ROUTES), len(ROUTES) - len(unstable)))
    return r.passed("all %d routes render identical text" % len(ROUTES), **measurements)
