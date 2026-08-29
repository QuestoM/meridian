"""What the open internet may learn about this deployment before signing in.

The demo is reachable from the public internet and is scanned by it: measured
in one day of real logs, fourteen probes for known exploit paths, every one
answered 401 by the auth wall. The wall is doing its job. This file pins the
quieter half of the posture, which nothing else asserted:

- A crawler is told not to index. The login wall stops a reader; it does not
  stop a search engine from listing the door, and a broadcaster's advertisers,
  agencies and agreed terms have no business in a search index.
- The pre-auth page carries no operator or advertiser name. The shell must be
  data-free until a session exists, or the wall protects the API while the
  HTML gives the game away.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
ROBOTS = DASHBOARD / "public" / "robots.txt"
INDEX = DASHBOARD / "index.html"


def test_crawlers_are_told_to_index_nothing():
    assert ROBOTS.is_file(), "a deployment on the open internet needs a robots.txt"
    body = ROBOTS.read_text(encoding="utf-8")
    lines = [line.strip().lower() for line in body.splitlines() if line.strip()
             and not line.strip().startswith("#")]
    assert "user-agent: *" in lines
    assert "disallow: /" in lines, "the whole surface is disallowed, not a corner of it"


def test_robots_ships_from_the_static_root_rather_than_the_spa_fallback():
    """The server tries the real path before falling back to index.html, so a
    file that exists is served as itself. Without the file, /robots.txt served
    the application's HTML and said nothing to a crawler at all."""
    served = DASHBOARD / "dist" / "robots.txt"
    if not served.exists():  # a clean checkout has no build yet
        return
    assert "Disallow: /" in served.read_text(encoding="utf-8")


def test_the_pre_auth_shell_names_no_operator_and_no_advertiser():
    """Measured against the live deployment: the HTML served before a session
    exists carries no channel, advertiser or agency name. This keeps it that
    way, because the wall protects the API and not the markup."""
    html = INDEX.read_text(encoding="utf-8")
    for leak in ("רשת 13", "קשת", "כאן 11", "עכשיו 14", "OMD", "פריסבי"):
        assert leak not in html, f"the pre-auth shell must not name {leak!r}"
