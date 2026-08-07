"""P9: one invariant over every path into the History address, not one leg.

This replaces test_p9_kai_history_second_use.py, which was named for the leg it
measured, which is exactly how a defect of this class survived a loop built to
catch it: two rounds closed the repeat-use leg of showRestoreVersion, that file
stayed green, and the first-use leg of the same function was broken the whole
time. So the guard is stated once, as a property of every path in:

  The address the operator is standing on names the page they are looking at,
  no click writes an address onto any other history entry, and no click grows
  the Back stack, which is what every other navigation in this product already
  promises: the shell's own setActiveView replaces the current entry.

showRestoreVersion has four paths in (a version id or none, crossed with
standing on the History page already or not) and three of them were broken:

  id, elsewhere    writeAddress is a replaceState, so writing the address before
                   the hash moved stamped ?entry=version:<id> onto the page
                   being LEFT. One Back press landed the operator there under a
                   url naming a History row that page cannot open.
  id, on History   the one leg two rounds fixed.
  none, elsewhere  no address write at all, so the destination opened at
                   whatever ?entry the url already carried, which after the
                   first defect above is a row the click never named.
  none, on History a same-value hash assignment, which fires no event at all:
                   the chip was inert, measured on a real restore point.

Measured here in real Chrome and load-bearing for all of it: a Back press fires
hashchange only when the two entries differ in their fragment ALONE. A traversal
that also changes the query fires popstate and no hashchange, and the shell
listens for hashchange only, so an implementation that pushes the whole
destination url in one navigation leaves the address bar naming one page while
the screen shows another, which is the same defect one gesture over. The fix
therefore pushes nothing at all, and this file measures that on every path.

The functions under test are extracted verbatim from the shipped source. The
stand-in for the shell keeps one view in a variable and ignores a hashchange
resolving to the view it already holds, exactly as React bails out of a setState
to the value already held; the older harness mounted on every hashchange, which
would have blessed a replaceState plus a synthetic event that was measured NOT
to remount the live product. Three controls, each the exact shape of one shipped
round, each still reproducing the defect a critic measured on the running
product, so no pass here can be vacuous. The last test carries the invariant to
the only other function on this surface that writes the address.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
HISTORY = DASHBOARD / "src" / "history"
PANEL_STATE = (KAI / "assistant-panel-state.js").read_text(encoding="utf-8")
SHORTCUTS = (KAI / "kai-shortcuts.js").read_text(encoding="utf-8")
LABELS = (HISTORY / "history-labels.js").read_text(encoding="utf-8")
ADDRESS = (HISTORY / "history-address.js").read_text(encoding="utf-8")
PROBE = Path(__file__).with_name("test_p9_paint_probe.mjs")
STALE = "version%3Astale999stale"
PATHS = ("firstUse", "firstUseNoId", "repeatWithId", "repeatNoId", "repeatWithIdAgain")


def _read_template(source: str, start: int) -> int:
    """Index just past the template literal opening at ``start``, honouring
    arbitrarily nested ``${...}`` and the backticks that can appear inside
    them (history-labels.js's writeAddress nests one)."""
    index = start + 1
    depth = 0
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if depth == 0 and char == "`":
            return index + 1
        if char == "$" and source[index + 1 : index + 2] == "{":
            depth += 1
            index += 2
            continue
        if depth and char == "{":
            depth += 1
        elif depth and char == "}":
            depth -= 1
        elif depth and char == "`":
            index = _read_template(source, index)
            continue
        index += 1
    return index


def _extract(source: str, signature: str) -> str:
    """The exact shipped text of one top-level declaration, found by its
    signature and closed by its own balanced brace, quote-aware so a `}` inside
    a string or a nested template never closes it early."""
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    index = brace
    while index < len(source):
        char = source[index]
        if char in "'\"":
            index += 1
            while index < len(source) and source[index] != char:
                index += 2 if source[index] == "\\" else 1
            index += 1
            continue
        if char == "`":
            index = _read_template(source, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces extracting {signature!r}")


def _constant(source: str, name: str) -> str:
    return (_extract(source, f"export const {name}").split(";")[0] + ";").replace("export const", "const", 1)


def _function(source: str, name: str) -> str:
    return _extract(source, f"export function {name}").replace("export function", "function", 1)


ADDRESS_PARAM = _constant(LABELS, "ADDRESS_PARAM")
POINT_PREFIX = _constant(ADDRESS, "POINT_PREFIX")
OPEN_HASH = _constant(SHORTCUTS, "OPEN_HASH")
WRITE_ADDRESS = _function(LABELS, "writeAddress")
POINT_ADDRESS = _function(ADDRESS, "pointAddress")
SHOW_RESTORE_VERSION = _function(PANEL_STATE, "showRestoreVersion")
OPEN_DOCK = _function(SHORTCUTS, "openDock")

# The three shapes that shipped before this one, kept as literals rather than
# re-derived from git history so this file reads as one self-contained
# measurement. Each reproduces a defect a critic measured on the running
# product, and every assertion about the real function is paired with a control
# that proves this harness can still see that defect.
#
# Round 7, the original: the repeat use is a same-value hash assignment, which
# fires no hashchange, so the address moves and the screen does not.
CONTROL_NO_BOUNCE = """function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  window.location.hash = 'Versions';
}"""

# Round 8: the bounce that closed that dead end, written with two plain hash
# assignments. Each is a navigation, so every repeat click pushed two entries.
CONTROL_HASH_BOUNCE = """function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  const onVersions = decodeURIComponent(window.location.hash.replace(/^#/, '')) === 'Versions';
  if (versionId && onVersions) {
    window.location.hash = '';
    window.setTimeout(() => { window.location.hash = 'Versions'; }, 0);
    return;
  }
  window.location.hash = 'Versions';
}"""

# Round 9: the replace-based bounce, whose repeat leg a critic proved flat and
# correct across four clicks. Its first-use leg is byte-identical to round 7's,
# which is the leg no round looked at.
CONTROL_REPLACE_BOUNCE = """function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  const onVersions = decodeURIComponent(window.location.hash.replace(/^#/, '')) === 'Versions';
  if (versionId && onVersions) {
    const base = window.location.pathname + window.location.search;
    window.location.replace(`${base}#`);
    window.setTimeout(() => { window.location.replace(`${base}#Versions`); }, 0);
    return;
  }
  window.location.hash = 'Versions';
}"""


def _walk(show_restore_version_source: str) -> str:
    """Every path into the address, walked once in one document: a use with an
    id from another page, a Back press off it if it created anything to go back
    to, a use with no id over a url already carrying somebody else's address,
    and three repeat uses on the destination itself. The stand-in router holds
    one view and remounts the destination only on a real transition into it,
    which is what the shell's setActiveViewState does; the destination reads its
    address once, on mount, which is what HistoryPage's initialisers do."""
    return f"""
{ADDRESS_PARAM}
{POINT_PREFIX}
{WRITE_ADDRESS}
{POINT_ADDRESS}
{show_restore_version_source}

window.__probe = (async () => {{
  const KNOWN = new Set(['Overview', 'Versions']);
  const settle = () => new Promise((resolve) => setTimeout(resolve, 90));
  const path = window.location.pathname;
  const viewFromLocation = () => {{
    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
    return KNOWN.has(hash) ? hash : 'Overview';
  }};
  let view = viewFromLocation();
  const mounted = [];
  window.addEventListener('hashchange', () => {{
    const next = viewFromLocation();
    if (next === view) return;
    view = next;
    if (view === 'Versions') {{
      mounted.push(new URLSearchParams(window.location.search).get(ADDRESS_PARAM) || '');
    }}
  }});

  const marks = {{}};
  const mark = (name) => {{
    marks[name] = {{
      search: window.location.search,
      hash: window.location.hash,
      view,
      mounts: mounted.length,
      lastMounted: mounted.length ? mounted[mounted.length - 1] : '',
      len: window.history.length,
    }};
  }};

  // An unrelated query parameter the real product carries, which no path may drop.
  window.history.replaceState(null, '', `${{path}}?keep=1`);
  mark('start');

  showRestoreVersion('aaa111aaa111');
  await settle();
  mark('firstUse');

  // A Back press is only possible off an entry the click created. The shipped
  // function creates none, which is the point of it; the shapes that shipped
  // before it create one, and this is where what they wrote onto the entry
  // behind them becomes visible.
  if (marks.firstUse.len > marks.start.len) {{
    window.history.back();
    await settle();
    mark('back');
  }}

  // The operator walks off to another page while an address is still in the
  // url, which is the state a shared History link arrives in and the state the
  // stamping defect leaves behind. Moved the way the shell's own setActiveView
  // moves: the entry is replaced and the view is set directly, firing nothing.
  window.history.replaceState(null, '', `${{path}}?keep=1&${{ADDRESS_PARAM}}={STALE}#Overview`);
  view = 'Overview';
  showRestoreVersion(null);
  await settle();
  mark('firstUseNoId');

  showRestoreVersion('bbb222bbb222');
  await settle();
  mark('repeatWithId');

  showRestoreVersion(undefined);
  await settle();
  mark('repeatNoId');

  showRestoreVersion('ccc333ccc333');
  await settle();
  mark('repeatWithIdAgain');

  return marks;
}})();
window.__probe;
"""


def _dock_walk() -> str:
    """The only other function on this surface that writes the address. It opens
    the dock through the shell's published #Assistant route and puts the address
    bar back, so the same invariant applies to it."""
    return f"""
{OPEN_HASH}
{OPEN_DOCK}

window.__probe = (async () => {{
  const settle = () => new Promise((resolve) => setTimeout(resolve, 90));
  let dockOpened = false;
  let view = 'Versions';
  window.addEventListener('hashchange', () => {{
    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
    if (hash === OPEN_HASH) {{ dockOpened = true; return; }}
    view = hash;
  }});
  const here = () => window.location.pathname + window.location.search + window.location.hash;
  window.history.replaceState(null, '', `${{window.location.pathname}}?keep=1&entry={STALE}#Versions`);
  const before = {{ url: here(), len: window.history.length }};
  openDock();
  await settle();
  return {{ before, after: {{ url: here(), len: window.history.length, dockOpened, view }} }};
}})();
window.__probe;
"""


def _run(expression: str, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the address cannot be measured")
    tmp_path.mkdir(parents=True, exist_ok=True)
    document = tmp_path / "doc.html"
    document.write_text("<!doctype html><meta charset=\"utf-8\"><body></body>", encoding="utf-8")
    script = tmp_path / "expr.js"
    script.write_text(expression, encoding="utf-8")
    done = subprocess.run([node, str(PROBE), str(document), str(script)],
                          capture_output=True, text=True, timeout=90)
    if done.returncode == 2 and "no chrome" in done.stderr:
        pytest.skip("no Chrome on this machine, so the address cannot be measured")
    assert done.returncode == 0, f"the probe failed: {done.stderr[-800:]}"
    return json.loads(done.stdout)


def _entry(search: str) -> str:
    return (parse_qs(search.lstrip("?")).get("entry") or [""])[0]


def _kept(search: str) -> str:
    return (parse_qs(search.lstrip("?")).get("keep") or [""])[0]


@pytest.fixture(scope="module")
def shipped(tmp_path_factory) -> dict:
    return _run(_walk(SHOW_RESTORE_VERSION), tmp_path_factory.mktemp("shipped"))


def test_the_address_names_the_screen_on_every_path_in(shipped) -> None:
    """The address written and the address the destination actually mounted at
    are the same one, on all four paths in: with an id and without, arriving
    from another page and standing on the destination already. The two paths
    with no id must clear the address rather than leave the destination opening
    at a row the click never named."""
    wanted_by_path = dict(zip(PATHS, ("version:aaa111aaa111", "", "version:bbb222bbb222",
                                      "", "version:ccc333ccc333")))
    for name, wanted in wanted_by_path.items():
        step = shipped[name]
        assert step["view"] == "Versions", f"{name} must land on the destination"
        assert _entry(step["search"]) == wanted, f"{name} must address exactly what it named"
        assert step["lastMounted"] == wanted, (
            f"{name}: the destination must mount at the address in the url, not at whichever "
            f"row it had already selected and not at one left over from an earlier click")
        assert _kept(step["search"]) == "1", f"{name} must not drop an unrelated query parameter"

    assert shipped["repeatNoId"]["mounts"] > shipped["repeatWithId"]["mounts"], (
        "the path with no id must still open the destination while standing on it: a chip that "
        "was handed no version id was measured inert on a real restore point, no event, no "
        "address change, nothing on screen")


def test_no_click_grows_the_back_stack_or_writes_on_another_entry(shipped) -> None:
    """The whole point of the shape. Every path in replaces the entry the
    operator is standing on, exactly as the shell's own setActiveView does, so
    there is no second entry for an address to be written onto and no Back press
    that has to route a screen back. Six clicks, no entry created by any of
    them."""
    flat = shipped["start"]["len"]
    for name in PATHS:
        assert shipped[name]["len"] == flat, (
            f"{name} must add nothing to the Back stack: a pushed entry either carries the "
            f"address of the page being left, or is a Back press the shell cannot route "
            f"because the query changed across it and no hashchange fires")

    back = shipped.get("back")
    assert back is None, (
        "no path in may create an entry to go back to at all, so the harness never pressed "
        "Back: if a later round pushes one again, this is where it has to prove the entry "
        "behind it is clean")


def test_the_control_that_shipped_writes_its_address_on_the_page_it_left(tmp_path) -> None:
    """The round-9 shipped function, whose repeat leg a critic proved flat and
    correct and whose first-use leg is byte-identical to round 7's. It
    reproduces here exactly what was measured on the running product: one click,
    one Back press, and the operator is on the page they came from under the
    address of a History row that page cannot open."""
    control = _run(_walk(CONTROL_REPLACE_BOUNCE), tmp_path / "replace-bounce")

    assert control["firstUse"]["lastMounted"] == "version:aaa111aaa111", (
        "the control's first use does land on the right row, which is why this survived four "
        "rounds: what it costs is only visible one gesture later")
    assert "back" in control, "the control pushes an entry, which is what makes the stamp reachable"
    assert _entry(control["back"]["search"]) == "version:aaa111aaa111", (
        "the control must still show the defect: one Back press lands on the page it left, "
        "carrying the address of a History row that page cannot open")
    assert control["firstUseNoId"]["lastMounted"] == "version:stale999stale", (
        "and its path with no id must still open the destination at an address the click never "
        "named, because it writes no address at all when it is handed none")
    assert control["repeatNoId"]["mounts"] == control["repeatWithId"]["mounts"], (
        "and its repeat with no id must still be inert: a same-value hash assignment fires "
        "nothing, which is what a critic measured on a real restore point")


def test_the_earlier_controls_reproduce_the_dead_end_and_the_history_climb(tmp_path) -> None:
    """The two rounds before that, each still visibly broken here, so a pass on
    the real function is a pass against every shape this has had."""
    no_bounce = _run(_walk(CONTROL_NO_BOUNCE), tmp_path / "no-bounce")
    assert _entry(no_bounce["repeatWithId"]["search"]) == "version:bbb222bbb222", (
        "the original still writes the new address, which is what made the defect silent")
    assert no_bounce["repeatWithId"]["lastMounted"] == no_bounce["firstUseNoId"]["lastMounted"], (
        "and the destination must not follow it: a same-value hash assignment fires no "
        "hashchange, so the screen keeps the row it mounted with while the url names another")

    hash_bounce = _run(_walk(CONTROL_HASH_BOUNCE), tmp_path / "hash-bounce")
    assert hash_bounce["repeatWithId"]["lastMounted"] == "version:bbb222bbb222", (
        "the hash bounce did remount, which is why it shipped")
    assert hash_bounce["repeatWithId"]["len"] > hash_bounce["firstUseNoId"]["len"], (
        "and it must still show what that cost: two plain hash assignments are two "
        "navigations, and each one pushes a browser-history entry")
    assert hash_bounce["repeatWithIdAgain"]["len"] > hash_bounce["repeatWithId"]["len"], (
        "on every later click too, matching the climb measured across four of them")


def test_the_other_address_writer_on_this_surface_obeys_the_same_invariant(tmp_path) -> None:
    """kai-shortcuts.js's openDock is the only other function on this surface
    that writes the address. It already obeys the invariant, and this is here so
    that stays measured rather than assumed: this class survived four rounds
    because each one guarded a site instead of the property."""
    result = _run(_dock_walk(), tmp_path / "dock")
    before, after = result["before"], result["after"]

    assert after["dockOpened"] is True, "the dock must actually open through the published route"
    assert after["url"] == before["url"], (
        "the operator must be left standing on the address they were standing on, ?entry and "
        "all: a shortcut that leaves #Assistant in the bar reloads elsewhere")
    assert after["len"] == before["len"], "opening the dock must add nothing to the Back stack"
    assert after["view"] == "Versions", "and it must not move the page underneath it"
