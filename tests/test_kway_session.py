"""The session that keeps the competitor feed alive, and the day it cost.

The renewal worked once, in nine seconds, and then stopped working — with no
change on either side. Two days of that were spent blaming the wrong things:
first a profile lock, then "Google refuses headless browsers", a sentence that
was written into a docstring before it was ever measured and turned out to be
false. What was actually happening is in :data:`CHOOSER_NO_JS` below.

So these tests are written against the two pages, as captured, rather than
against a description of them. They need a browser but not a network: every
fixture is a static document, no account is real, and nothing here can pass
because somebody happened to be signed in.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest

from kairos.model import kway_browser as kb
from kairos.model import kway_session as ks

# ---------------------------------------------------------------- the pages
#
# Both are Google's account chooser. Which one is served is decided by Google,
# not by us, and the choice flips between runs of the same code on the same
# machine — which is exactly why supporting only the one that happened to
# appear first produced a failure that looked intermittent.

# With a window: the account is a scripted tile carrying data-identifier.
CHOOSER_SCRIPTED = """
<div class="list">
  <li><div data-identifier="{email}" role="link" class="tile">
    <div class="name">Netanel Bezalel</div><div class="mail">{email}</div>
  </div></li>
</div>"""

# Without one: Google's no-JavaScript variant, copied in shape from the real
# page. The account is a submit button wrapped in an li that contains nothing
# else — so the WRAPPER AND THE BUTTON CARRY IDENTICAL TEXT. That tie is the
# whole defect: rank the candidates by text length and the stable sort hands the
# win to the wrapper, which is the one element here that does nothing when
# clicked. The wrapper must stay in this fixture; without it, the broken
# selector passes.
CHOOSER_NO_JS = """
<form method="post" action="/chosen">
  <input type="hidden" name="bgresponse" value="js_disabled">
  <li class="row"><div class="cell">
    <button type="submit" name="chooser[select]" value="1146675877" class="tile">
      <div class="name">Netanel Bezalel</div>
      <div class="mail" data-email="{email}">{email}</div>
    </button>
  </div></li>
</form>"""

WATCHER = """
<script>
  window.__clicked = null;
  document.addEventListener('click', function (e) {
    window.__clicked = e.target.tagName + ':' +
      (e.target.getAttribute('data-email') || e.target.getAttribute('data-identifier') || '');
  }, true);
</script>"""

EMAIL = "planner@example.test"
OTHER = "someone.else@example.test"


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """One throwaway browser, on a profile that has never signed in anywhere."""
    chrome = ks._chrome_binary()
    from pathlib import Path

    if not Path(chrome).exists():
        pytest.skip("no browser on this machine")
    try:
        from websockets.sync.client import connect
    except Exception:  # noqa: BLE001
        pytest.skip("the websocket client is unavailable")

    port = 9497
    profile = tmp_path_factory.mktemp("chooser-profile")
    process = subprocess.Popen(
        [chrome, f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
         "--headless=new", "--no-first-run", "--no-default-browser-check"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with connect(kb._debugger_url(port), max_size=None) as sock:
            target = kb._rpc(sock, "Target.createTarget", {"url": "about:blank"})["targetId"]
            sid = kb._rpc(sock, "Target.attachToTarget",
                          {"targetId": target, "flatten": True})["sessionId"]
            for method in ("Page.enable", "Runtime.enable"):
                kb._rpc(sock, method, {}, sid)

            def run(body: str, hint: str) -> tuple[str, str]:
                """Load a chooser, answer it, and report what got clicked."""
                document = f"<!doctype html><meta charset=utf-8>{WATCHER}{body}"
                kb._rpc(sock, "Page.navigate", {
                    "url": "data:text/html;charset=utf-8," + urllib.parse.quote(document),
                }, sid)
                time.sleep(0.6)

                def js(expression):
                    return kb._rpc(sock, "Runtime.evaluate", {
                        "expression": expression, "returnByValue": True,
                    }, sid).get("result", {}).get("value")

                said = js(kb.CHOOSE_ACCOUNT_JS % (json.dumps(hint), json.dumps(kb.CLICKABLE)))
                return str(said or ""), str(js("window.__clicked") or "")

            yield run
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


# ------------------------------------------------------- answering the page

def test_the_no_javascript_chooser_clicks_the_button_and_not_its_label(page):
    """THE BUG, pinned. The address sits on a div inside the button.

    Clicking the element that holds the text is the intuitive thing and it is
    silently inert: the page does not move, nothing is logged, and ninety
    seconds later the caller reports a timeout that names the network.
    """
    said, clicked = page(CHOOSER_NO_JS.format(email=EMAIL), EMAIL)
    assert said.startswith("chose"), said
    assert clicked.startswith("BUTTON:"), (
        f"the click landed on {clicked} — a label, which does nothing")


def test_the_scripted_chooser_clicks_the_account_tile(page):
    said, clicked = page(CHOOSER_SCRIPTED.format(email=EMAIL), EMAIL)
    assert said == f"chose {EMAIL}", said
    assert clicked.startswith(("DIV:", "LI:")), clicked


def test_a_single_account_needs_no_name_to_be_chosen(page):
    said, clicked = page(CHOOSER_NO_JS.format(email=EMAIL), "")
    assert said.startswith("chose the only account"), said
    assert clicked.startswith("BUTTON:")


def test_the_named_account_is_chosen_and_not_merely_the_first(page):
    """A profile that gains a second Google account must not quietly switch."""
    two = (CHOOSER_SCRIPTED.format(email=OTHER) + CHOOSER_SCRIPTED.format(email=EMAIL))
    said, _ = page(two, EMAIL)
    assert said == f"chose {EMAIL}", said


def test_two_accounts_and_no_name_is_refused_rather_than_picked(page):
    two = (CHOOSER_SCRIPTED.format(email=OTHER) + CHOOSER_SCRIPTED.format(email=EMAIL))
    said, clicked = page(two, "")
    assert said.startswith("ambiguous:"), said
    assert clicked == "", "an account was chosen on somebody's behalf"


def test_an_account_that_is_not_signed_in_is_reported_not_guessed(page):
    said, clicked = page(CHOOSER_SCRIPTED.format(email=OTHER), EMAIL)
    assert said.startswith("absent:"), said
    assert OTHER in said, "the reason does not say which accounts were offered"
    assert clicked == ""


# ------------------------------------------------- telling the states apart

def test_choosing_an_account_is_not_mistaken_for_a_step_only_a_person_can_take():
    """The whole failure in one assertion.

    The chooser asks for no password and no code. Filing it under "a human is
    needed" would hand a solvable step to a person who is asleep.
    """
    chooser = "https://accounts.google.com/v3/signin/accountchooser?client_id=x"
    assert kb.GOOGLE_CHOOSER in chooser
    assert not any(mark in chooser for mark in kb.HUMAN_NEEDED)
    assert not any(mark in chooser for mark in kb.GOOGLE_CONSENT)


@pytest.mark.parametrize("url", [
    "https://accounts.google.com/v3/signin/identifier?flow=x",
    "https://accounts.google.com/v3/signin/challenge/pwd?x=1",
])
def test_a_password_or_a_second_factor_is_left_to_a_person(url):
    assert any(mark in url for mark in kb.HUMAN_NEEDED)
    assert kb.GOOGLE_CHOOSER not in url


def test_granting_an_application_access_is_left_to_a_person():
    """Choosing among accounts already signed in is mechanical. Granting an
    application access to one is not, however often it has been granted."""
    consent = "https://accounts.google.com/signin/oauth/consent?authuser=0"
    assert any(mark in consent for mark in kb.GOOGLE_CONSENT)


# --------------------------------------------------------- the session itself

def test_a_session_sends_the_token_the_server_handed_out():
    session = ks.Session(cookies={"sfp_access": "abc", "XSRF-TOKEN": "tok"},
                         verified_at=datetime.now(timezone.utc))
    head = session.headers()
    assert head["Cookie"] == "sfp_access=abc; XSRF-TOKEN=tok"
    assert head["X-XSRF-TOKEN"] == "tok"


def test_a_session_without_that_token_does_not_invent_one():
    session = ks.Session(cookies={"sfp_access": "abc"},
                         verified_at=datetime.now(timezone.utc))
    assert "X-XSRF-TOKEN" not in session.headers()


def test_age_is_measured_from_when_it_was_last_proven_alive():
    then = datetime.now(timezone.utc) - timedelta(hours=3)
    assert ks.Session(cookies={}, verified_at=then).age_hours() == pytest.approx(3.0, abs=0.1)


def test_the_account_to_renew_is_named_by_the_environment(monkeypatch):
    """Whoever runs this engine is not whoever built it."""
    monkeypatch.setenv(ks.ACCOUNT_ENV, "someone@elsewhere.test")
    assert ks.account_hint() == "someone@elsewhere.test"


def test_with_nothing_named_and_nothing_stored_no_account_is_assumed(monkeypatch):
    monkeypatch.delenv(ks.ACCOUNT_ENV, raising=False)
    monkeypatch.setattr(ks, "load_session", lambda **_: None)
    assert ks.account_hint() == ""


def test_the_stored_account_names_itself_when_the_environment_does_not(monkeypatch):
    monkeypatch.delenv(ks.ACCOUNT_ENV, raising=False)
    monkeypatch.setattr(ks, "load_session", lambda **_: ks.Session(
        cookies={}, verified_at=datetime.now(timezone.utc),
        account={"email": "stored@example.test"}))
    assert ks.account_hint() == "stored@example.test"


def test_a_session_is_never_handed_out_without_being_asked_for(monkeypatch):
    """Expiry on a cookie is a claim. Acceptance by the server is a fact.

    Measured: the cookie still looked valid for hours after the server had
    stopped honouring it.
    """
    monkeypatch.setattr(ks, "load_session", lambda **_: ks.Session(
        cookies={"sfp_access": "stale"}, verified_at=datetime.now(timezone.utc)))
    monkeypatch.setattr(ks, "alive", lambda *_a, **_k: False)
    session, status = ks.current(allow_renew=False)
    assert session is None
    assert status["state"] == "expired"


def test_a_renewal_a_person_must_finish_says_so_and_says_what_to_do(monkeypatch):
    monkeypatch.setattr(ks, "load_session", lambda **_: None)
    monkeypatch.setattr(ks, "renew", lambda **_: {
        "renewed": False, "needs_human": True,
        "reason": "Google is asking for a sign-in that only a person can complete",
        "do_this": "run this once",
    })
    session, status = ks.current()
    assert session is None
    assert status["state"] == "needs_human"
    assert status["do_this"], "a person was told there is a problem but not what to do"


def test_the_profile_is_kept_outside_this_repository():
    """It holds a live login. A repository is the wrong place for one."""
    from pathlib import Path

    here = Path(__file__).resolve().parents[1]
    assert here not in ks.profile_path().resolve().parents
