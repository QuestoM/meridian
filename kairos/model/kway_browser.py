"""Driving the application's own login, because only it can do the handshake.

Split from :mod:`kairos.model.kway_session` so the session logic — storage,
verification, the decision to renew — stays readable without a browser protocol
in the middle of it, and so the browser half can be replaced without touching
the half that decides.

Why a browser at all, when everything else here speaks plain HTTPS: the login is
Google OAuth with PKCE. The application generates a code verifier, keeps it, and
exchanges it. ``/api/auth/google?env=prod`` answers "Code challenge is missing."
to anyone who tries to start the flow without one, so a hand-rolled version
would be a second implementation of somebody else's protocol, wrong in a
different way every time they change it. Clicking the application's own control
lets the code that owns the verifier use it.

The cookie that matters is HttpOnly and no page script can read it. The
debugging protocol can, which is the only reason this is possible without
scraping a browser's cookie database.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

API_USER = "https://api.kway.co.il/api/user"
APP_LOGIN = "https://app.kway.co.il/auth/login"
LOGIN_SELECTOR = ".continue-with-google"

# Google's own pages that mean a person has to act. Matched on the path so a
# query string carrying a redirect URL cannot produce a false alarm.
HUMAN_NEEDED = ("/signin/identifier", "/signin/challenge", "/signin/v2", "/pwd")

# The account chooser is NOT one of those, and mistaking it for one cost a day.
# It is Google asking which of the profile's signed-in accounts to continue
# with — no password, no code, nothing only a person knows. It appears on some
# runs and not others, which is why a renewal that worked once can stop working
# with no change on either side. Left unanswered it simply waits, so the failure
# arrives as a timeout: the least informative shape a blocked flow can take.
GOOGLE_CHOOSER = "/signin/accountchooser"

# Consent, by contrast, IS a person's decision. Granting an application access
# to an account is not a click to automate on somebody's behalf, even when the
# somebody has said yes before, so it is reported rather than answered.
GOOGLE_CONSENT = ("/signin/oauth/consent", "/o/oauth2/auth")

WANTED_COOKIES = ("sfp_access", "XSRF-TOKEN")

# Clicking the same tile forever would turn a changed page into an infinite
# loop that still reports a timeout at the end. Three is generous for a step
# that works on the first click.
MAX_CHOOSER_CLICKS = 3


def _rpc(sock: Any, method: str, params: Optional[dict] = None,
         session_id: Optional[str] = None, timeout: float = 30.0) -> dict:
    _rpc.counter = getattr(_rpc, "counter", 0) + 1
    message = {"id": _rpc.counter, "method": method, "params": params or {}}
    if session_id:
        message["sessionId"] = session_id
    sock.send(json.dumps(message))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = sock.recv(timeout=max(0.1, deadline - time.monotonic()))
        if raw is None:
            continue
        data = json.loads(raw)
        if data.get("id") == message["id"]:
            if "error" in data:
                raise RuntimeError(f"{method}: {data['error'].get('message')}")
            return data.get("result", {})
    raise TimeoutError(f"{method} did not answer within {timeout}s")


def _debugger_url(port: int, tries: int = 40) -> str:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                return json.loads(r.read())["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 - the browser is still coming up
            time.sleep(0.5)
    raise RuntimeError(f"the browser never opened a debugging port on {port}")


# Answering the chooser. The tile carries the address it belongs to, which is
# what makes this safe to automate: the click goes to a NAMED account, never to
# whichever tile happens to be first. A profile that later holds a second Google
# account must not silently start signing in as the wrong person.
#
# TWO PAGES WEAR THIS NAME. Measured, at the same URL, in the same profile,
# minutes apart: with a window, the accounts are ``[data-identifier]`` tiles
# driven by script; without one, Google serves its no-JavaScript variant — the
# form carries ``bgresponse=js_disabled`` — and each account is a submit BUTTON
# named ``chooser[select]`` whose address sits on a ``[data-email]`` child.
#
# Finding the account by its TEXT is what fails there, and it fails in a way
# worth stating precisely, because it is not obvious and it cost two days. In
# that variant the button is wrapped in an ``li`` that contains nothing else, so
# the wrapper and the button carry the IDENTICAL text. Ranking candidates by
# text length therefore ties, the sort is stable, and the winner is whichever
# comes first in the document — the wrapper. Clicking it does nothing, silently,
# until the budget expires and the caller reports a timeout.
#
# So the address is looked for under either ATTRIBUTE, never by text, and the
# click climbs to the nearest genuinely clickable ancestor: from inside the
# button, that is the button, not the wrapper two levels above it.
CLICKABLE = "button, a, li, div[role=link], div[role=button]"

CHOOSE_ACCOUNT_JS = """(()=>{
  const want = %s, CLICKABLE = %s;
  const pick = n => n.closest(CLICKABLE) || n;
  const owners = [...document.querySelectorAll('[data-identifier],[data-email]')]
    .map(n => [(n.getAttribute('data-identifier') || n.getAttribute('data-email') || '').trim(), n])
    .filter(([id]) => id.includes('@'));
  if (want) {
    const hit = owners.find(([id]) => id.toLowerCase() === want.toLowerCase());
    if (hit) { pick(hit[1]).click(); return 'chose ' + hit[0]; }
    const leaf = [...document.querySelectorAll('*')]
      .filter(n => !n.children.length && (n.textContent || '').includes(want))
      .map(pick).find(n => n.matches(CLICKABLE));
    if (leaf) { leaf.click(); return 'chose by text'; }
    return 'absent:' + owners.map(([id]) => id).join(',');
  }
  if (owners.length === 1) { pick(owners[0][1]).click(); return 'chose the only account: ' + owners[0][0]; }
  return 'ambiguous:' + owners.map(([id]) => id).join(',');
})()"""


def _wait_for_profile(profile: Path, seconds: float = 12.0) -> None:
    """Wait until no other browser holds this profile.

    Chrome keeps a lock while it runs and clears it on the way out, but the way
    out is not instant. A second launch against a still-locked profile does not
    fail loudly — it hands its arguments to the dying instance and exits, and
    what the caller sees is a sign-in that never completes. That is a minute and
    a half of nothing, blamed on the wrong thing.
    """
    lock = profile / "SingletonLock"
    deadline = time.monotonic() + seconds
    while lock.exists() and time.monotonic() < deadline:
        time.sleep(0.4)


def renew_via_profile(
    *,
    chrome: str,
    profile: Path,
    headless: bool = True,
    budget_seconds: float = 90.0,
    port: int = 9444,
    account_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Open the profile, click the app's login, and take the cookies it earns.

    Returns a dict that always explains itself: ``renewed`` with the cookies, or
    ``needs_human`` with the single step to take, or a stated failure. It never
    returns cookies it did not see the server accept.
    """
    try:
        from websockets.sync.client import connect
    except Exception as exc:  # noqa: BLE001
        return {"renewed": False, "reason": f"the websocket client is unavailable ({exc})"}

    if not Path(chrome).exists():
        return {"renewed": False,
                "reason": f"no browser at {chrome}; set KAIROS_CHROME to its path"}
    if not profile.exists():
        return {
            "renewed": False,
            "needs_human": True,
            "reason": f"there is no signed-in browser profile at {profile}",
            "do_this": (
                f"Run once, sign in with the Kway account, then close the window:\n"
                f"  '{chrome}' --user-data-dir='{profile}' https://app.kway.co.il/dashboard"
            ),
        }

    started = time.monotonic()
    _wait_for_profile(profile)
    args = [
        chrome, f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
    ]
    if headless:
        args.append("--headless=new")
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = _debugger_url(port)
        with connect(url, max_size=None) as sock:
            target = _rpc(sock, "Target.createTarget", {"url": "about:blank"})["targetId"]
            session_id = _rpc(sock, "Target.attachToTarget",
                              {"targetId": target, "flatten": True})["sessionId"]
            _rpc(sock, "Page.enable", {}, session_id)
            _rpc(sock, "Runtime.enable", {}, session_id)

            def evaluate(expression: str, timeout: float = 25.0) -> Any:
                result = _rpc(sock, "Runtime.evaluate", {
                    "expression": expression, "returnByValue": True, "awaitPromise": True,
                }, session_id, timeout=timeout)
                return result.get("result", {}).get("value")

            _rpc(sock, "Page.navigate", {"url": APP_LOGIN}, session_id)
            # Wait for the control the application renders, rather than a fixed
            # sleep: an app that is slow today must not read as an app that is
            # broken.
            deadline = time.monotonic() + min(30.0, budget_seconds)
            while time.monotonic() < deadline:
                if evaluate(f"!!document.querySelector({LOGIN_SELECTOR!r})"):
                    break
                time.sleep(0.5)
            else:
                # Already signed in? Then there is no login control to find.
                if evaluate(f"fetch({API_USER!r},{{credentials:'include'}}).then(r=>r.status)") == 200:
                    return _harvest(sock, session_id, evaluate, started)
                return {"renewed": False,
                        "reason": "the login page never rendered its sign-in control"}

            evaluate(
                "(()=>{const el=document.querySelector(%r);"
                "if(!el) return false; (el.closest('button,a')||el).click(); return true;})()"
                % LOGIN_SELECTOR
            )

            deadline = time.monotonic() + budget_seconds
            chooser_clicks = 0
            while time.monotonic() < deadline:
                time.sleep(1.5)
                here = str(evaluate("location.href") or "")
                if GOOGLE_CHOOSER in here:
                    if chooser_clicks >= MAX_CHOOSER_CLICKS:
                        return {
                            "renewed": False,
                            "reason": "Google's account chooser did not accept the account",
                            "at": here[:160],
                        }
                    chooser_clicks += 1
                    chose = str(evaluate(CHOOSE_ACCOUNT_JS % (
                        json.dumps(account_hint or ""), json.dumps(CLICKABLE))) or "")
                    if chose.startswith(("absent", "ambiguous")):
                        return {
                            "renewed": False,
                            "needs_human": True,
                            "reason": (
                                f"Google offered accounts this could not choose between ({chose})"
                                if chose.startswith("ambiguous") else
                                f"the account {account_hint} is not signed in to this profile ({chose})"
                            ),
                            "do_this": (
                                f"Run once and sign in with the right account, then close the window:\n"
                                f"  '{chrome}' --user-data-dir='{profile}' https://app.kway.co.il/dashboard"
                            ),
                        }
                    continue
                if any(mark in here for mark in GOOGLE_CONSENT):
                    return {
                        "renewed": False,
                        "needs_human": True,
                        "reason": "Google is asking to grant this application access, which is a decision for a person",
                        "do_this": (
                            f"Run once, approve the access, then close the window:\n"
                            f"  '{chrome}' --user-data-dir='{profile}' https://app.kway.co.il/dashboard"
                        ),
                        "at": here[:160],
                    }
                if any(mark in here for mark in HUMAN_NEEDED):
                    return {
                        "renewed": False,
                        "needs_human": True,
                        "reason": "Google is asking for a sign-in that only a person can complete",
                        "do_this": (
                            f"Run once and sign in, then close the window:\n"
                            f"  '{chrome}' --user-data-dir='{profile}' https://app.kway.co.il/dashboard"
                        ),
                        "at": here[:160],
                    }
                if evaluate(f"fetch({API_USER!r},{{credentials:'include'}}).then(r=>r.status).catch(()=>0)") == 200:
                    return _harvest(sock, session_id, evaluate, started)
            return {"renewed": False,
                    "reason": f"the sign-in did not complete within {budget_seconds:.0f}s"}
    except Exception as exc:  # noqa: BLE001 - a stated failure, never a silent one
        return {"renewed": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _harvest(sock: Any, session_id: str, evaluate: Any, started: float) -> dict[str, Any]:
    """Take the cookies only after the server has accepted them."""
    jar = _rpc(sock, "Network.getAllCookies", {}, session_id).get("cookies", [])
    cookies = {
        c["name"]: c["value"] for c in jar
        if c["name"] in WANTED_COOKIES and "kway.co.il" in c.get("domain", "")
    }
    if "sfp_access" not in cookies:
        return {"renewed": False,
                "reason": "the sign-in reported success but issued no session cookie"}
    account: dict[str, Any] = {}
    try:
        raw = evaluate(
            f"fetch({API_USER!r},{{credentials:'include'}}).then(r=>r.text())")
        body = json.loads(raw) if raw else {}
        data = body.get("data") or body
        account = {
            "email": data.get("email"),
            "account_id": (data.get("account") or {}).get("id"),
            "role": (data.get("role") or {}).get("name"),
        }
    except Exception:  # noqa: BLE001 - identity is a nicety, the session is the point
        account = {}
    return {
        "renewed": True,
        "cookies": cookies,
        "account": account,
        "seconds": round(time.monotonic() - started, 1),
    }
