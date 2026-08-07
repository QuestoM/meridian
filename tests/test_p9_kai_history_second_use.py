"""P9: the second click on the same control, not just the first.

A blind critic proved the first use of "see it in the history" real: the chip
writes ?entry=version:<id> and HistoryPage selects that exact row. What was
still a dead end is the second use of the same control, in exactly the state
the first use leaves the operator in. showRestoreVersion writes the address
and then sets window.location.hash = 'Versions'. A successful first use lands
on the History page with the dock still open and more chips on screen, so
that hash is already 'Versions', the assignment is a same-value no-op, no
hashchange event fires, the shell never remounts the destination, and
HistoryPage reads the address only in its mount-time initialisers. The url
now names one entry while the screen keeps showing whichever row was already
selected: copy the link and a colleague opens something nobody is looking at.

That was closed with a bounce: when the hash is already 'Versions', set it
off a value the router does not recognise, one tick apart, so the shell's own
hashchange listener actually processes a transition away before the
transition back. A later critic proved that bounce itself has a cost when it
was written as two plain `window.location.hash = ...` assignments: every hash
assignment is a navigation, and every navigation of that kind pushes a
browser-history entry even when it lands back on the address the page was
already at. Four chip clicks made while already on the History page pushed
eight entries between them, and one Back press from the correctly addressed
History page then landed on Overview at the just-abandoned address, a url
naming a History entry on a page that cannot open it. The fix keeps the same
bounce shape but performs it with `window.location.replace`, which fires the
same hashchange without pushing a history entry, passing the pathname and
search explicitly because a bare `location.replace('#')` drops the ?entry
search the bounce needs to preserve.

This measures the actual shipped function in a real browser, not a
description of it. The three functions this depends on (showRestoreVersion,
and the writeAddress/pointAddress it calls, owned by history-labels.js and
history-address.js) are extracted verbatim from the shipped source and run
under a minimal stand-in for the shell's own routing: one page shown at a
time, switched only by a real 'hashchange' event exactly as
shell/TVBreakDashboard.jsx's handleHashChange does, with a History-page analog
that reads the address only when it mounts, exactly as HistoryPage.jsx's
useState initialisers do. It carries two controls: the original pre-fix jump
(hash assignment, nothing else, second use dead) and the hash-assignment
bounce that replaced it in the round before this one (second use works, but
pushes two history entries per click). A pass here can never be vacuous
against either defect.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
HISTORY = DASHBOARD / "src" / "history"
PANEL_STATE = (KAI / "assistant-panel-state.js").read_text(encoding="utf-8")
LABELS = (HISTORY / "history-labels.js").read_text(encoding="utf-8")
ADDRESS = (HISTORY / "history-address.js").read_text(encoding="utf-8")
PROBE = Path(__file__).with_name("test_p9_paint_probe.mjs")


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


def _extract_function(source: str, signature: str) -> str:
    """The exact shipped text of one top-level function, found by its
    signature and closed by its own balanced brace, quote-aware so a `}`
    inside a string or a nested template never closes it early."""
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


ADDRESS_PARAM = (_extract_function(LABELS, "export const ADDRESS_PARAM").split(";")[0] + ";").replace(
    "export const", "const", 1)
WRITE_ADDRESS = _extract_function(LABELS, "export function writeAddress").replace(
    "export function", "function", 1)
POINT_PREFIX = (_extract_function(ADDRESS, "export const POINT_PREFIX").split(";")[0] + ";").replace(
    "export const", "const", 1)
POINT_ADDRESS = _extract_function(ADDRESS, "export function pointAddress").replace(
    "export function", "function", 1)
SHOW_RESTORE_VERSION = _extract_function(PANEL_STATE, "export function showRestoreVersion").replace(
    "export function", "function", 1)

# The pre-fix behaviour, the shape both the critic's report and P9's own
# history describe: writeAddress, then an unconditional hash assignment, with
# no bounce for the case where that assignment is a same-value no-op. Kept
# here as the control rather than re-derived from git history, so this file
# reads as one self-contained measurement.
SHOW_RESTORE_VERSION_PRE_FIX = """function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  window.location.hash = 'Versions';
}"""

# The round-before-this-one's fix: it closes the same-value-no-op dead end by
# bouncing the hash off a value the router does not recognise, but performs
# the bounce with two plain hash ASSIGNMENTS. Kept here as a second control,
# again rather than re-derived from git history, because the defect it
# reproduces (each hash assignment pushes a browser-history entry, so every
# second use adds two) is exactly what a critic measured live and exactly
# what this file now has to prove is gone from the shipped function.
SHOW_RESTORE_VERSION_HASH_BOUNCE = """function showRestoreVersion(versionId) {
  if (versionId) writeAddress(pointAddress(versionId));
  const onVersions = decodeURIComponent(window.location.hash.replace(/^#/, '')) === 'Versions';
  if (versionId && onVersions) {
    window.location.hash = '';
    window.setTimeout(() => { window.location.hash = 'Versions'; }, 0);
    return;
  }
  window.location.hash = 'Versions';
}"""


def _harness(show_restore_version_source: str) -> str:
    """A minimal stand-in for the shell's own router and HistoryPage, wired to
    the real (or control) showRestoreVersion. One page shown at a time,
    switched only by a real 'hashchange' event; the Versions page analog
    records the address it was addressed at, once, at the moment it mounts."""
    return f"""
{ADDRESS_PARAM}
{POINT_PREFIX}
{WRITE_ADDRESS}
{POINT_ADDRESS}
{show_restore_version_source}

window.__probe = (async () => {{
  const KNOWN = new Set(['Overview', 'Versions']);
  function viewFromLocation() {{
    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
    return KNOWN.has(hash) ? hash : 'Overview';
  }}
  const mounted = [];
  function mountIfVersions(view) {{
    if (view !== 'Versions') return;
    const address = new URLSearchParams(window.location.search).get(ADDRESS_PARAM) || '';
    mounted.push(address);
  }}
  window.addEventListener('hashchange', () => mountIfVersions(viewFromLocation()));

  // The genuine first use: the operator is elsewhere (hash unset, same as a
  // fresh load) and opens the chip for one restore point. This is a real
  // hashchange either way (the hash actually differs), so it is not the part
  // either version was reported broken on.
  showRestoreVersion('aaa111aaa111');
  await new Promise((resolve) => setTimeout(resolve, 80));
  const afterFirstUse = {{
    search: window.location.search,
    lastMounted: mounted[mounted.length - 1] || '',
    historyLength: window.history.length,
  }};

  // The second use of the same control, still on the History page, for a
  // different restore point. This is the click the critic proved dead, and
  // then proved costly: a colleague measured the fix for it pushing two
  // browser-history entries per click here.
  showRestoreVersion('bbb222bbb222');
  await new Promise((resolve) => setTimeout(resolve, 80));
  const afterSecondUse = {{
    search: window.location.search,
    lastMounted: mounted[mounted.length - 1] || '',
    historyLength: window.history.length,
  }};

  // A third and fourth use, still on the History page each time, addressing
  // a third restore point. The critic measured history.length climbing by
  // two on every one of four clicks; two clicks already distinguish a
  // per-click push from a one-time cost, and a fourth confirms it is not a
  // fluke of the second.
  showRestoreVersion('ccc333ccc333');
  await new Promise((resolve) => setTimeout(resolve, 80));
  showRestoreVersion('ddd444ddd444');
  await new Promise((resolve) => setTimeout(resolve, 80));
  const afterFourthUse = {{
    search: window.location.search,
    lastMounted: mounted[mounted.length - 1] || '',
    historyLength: window.history.length,
  }};

  return {{ afterFirstUse, afterSecondUse, afterFourthUse }};
}})();
window.__probe;
"""


def _run(show_restore_version_source: str, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the second use cannot be measured")
    document = tmp_path / "doc.html"
    document.write_text("<!doctype html><meta charset=\"utf-8\"><body></body>", encoding="utf-8")
    expression = tmp_path / "expr.js"
    expression.write_text(_harness(show_restore_version_source), encoding="utf-8")
    done = subprocess.run([node, str(PROBE), str(document), str(expression)],
                          capture_output=True, text=True, timeout=60)
    if done.returncode == 2 and "no chrome" in done.stderr:
        pytest.skip("no Chrome on this machine, so the second use cannot be measured")
    assert done.returncode == 0, f"the probe failed: {done.stderr[-800:]}"
    return json.loads(done.stdout)


def test_the_second_use_addresses_a_different_row_while_already_on_versions(tmp_path) -> None:
    """The exact sequence the critic measured dead: dock open over History,
    hash already 'Versions', a chip clicked for a point that is not the row
    already selected. The shipped function must remount the destination so
    the newly written address is the one actually read, and it must do so
    without paying for it in browser history: a critic measured a working
    version of this bounce still pushing two entries per click, resurrecting
    the same defect one Back press away."""
    result = _run(SHOW_RESTORE_VERSION, tmp_path)

    first = result["afterFirstUse"]
    assert first["search"] == "?entry=version%3Aaaa111aaa111"
    assert first["lastMounted"] == "version:aaa111aaa111", (
        "the first use, landing fresh on Versions, must address the row it named")

    second = result["afterSecondUse"]
    assert second["search"] == "?entry=version%3Abbb222bbb222", (
        "the second click must still write the new address")
    assert second["lastMounted"] == "version:bbb222bbb222", (
        "the destination must remount on the second use too: a hash already at "
        "'Versions' fires no event on a same-value assignment, so without a bounce "
        "the page underneath keeps showing whatever row was already selected while "
        "the address bar now names a different one")
    assert second["historyLength"] == first["historyLength"], (
        "the bounce must fire a real hashchange without pushing a browser-history "
        "entry: a Back press from the correctly addressed History page must not "
        "land the operator on a different page at an address it cannot open"
    )

    fourth = result["afterFourthUse"]
    assert fourth["lastMounted"] == "version:ddd444ddd444", (
        "a third and fourth use in a row must keep remounting, not just the second")
    assert fourth["historyLength"] == first["historyLength"], (
        "history.length must stay flat across every repeated use on this page, "
        "not just the first two: the critic measured it climbing on every one "
        "of four clicks, 10 -> 12 -> 14 -> 16"
    )


def test_the_control_reproduces_the_defect_the_critic_measured(tmp_path) -> None:
    """Without the bounce, the first use still works (a fresh hash assignment
    away from whatever the operator was on) and the second is exactly the
    dead end reported: the address is written, the destination never
    remounts, and the row on screen does not move."""
    result = _run(SHOW_RESTORE_VERSION_PRE_FIX, tmp_path)

    first = result["afterFirstUse"]
    assert first["lastMounted"] == "version:aaa111aaa111", (
        "the control's first use is not the part that was broken")

    second = result["afterSecondUse"]
    assert second["search"] == "?entry=version%3Abbb222bbb222", (
        "the control still writes the new address, which is exactly what made "
        "the defect a silent one: the url looks right and the screen does not follow"
    )
    assert second["lastMounted"] == "version:aaa111aaa111", (
        "the control must NOT pick up the new address: a same-value hash "
        "assignment fires no hashchange, so the destination never remounts and "
        "keeps showing whatever it mounted with on the first use"
    )


def test_the_hash_bounce_control_reproduces_the_history_push_the_critic_measured(tmp_path) -> None:
    """The bounce that closed the second-use dead end in the round before this
    one still worked (remounted, selected the right row) but paid for it with
    two pushed browser-history entries per click on this page, because a plain
    `window.location.hash = ...` assignment is itself a navigation. This is the
    control that proves the shipped fix's flat history.length is not a fluke
    of the harness: the same harness, given the shape the critic measured live,
    reproduces the climb."""
    result = _run(SHOW_RESTORE_VERSION_HASH_BOUNCE, tmp_path)

    first = result["afterFirstUse"]
    second = result["afterSecondUse"]
    fourth = result["afterFourthUse"]

    assert second["lastMounted"] == "version:bbb222bbb222", (
        "the hash-bounce control does remount on the second use, which is why "
        "it shipped: the defect it carries is not in what the screen shows")
    assert second["historyLength"] > first["historyLength"], (
        "the hash-bounce control must show the push the critic measured: two "
        "plain hash assignments are two navigations, and each pushes an entry"
    )
    assert fourth["historyLength"] > second["historyLength"], (
        "the push must repeat on every later use, not just the first bounce, "
        "matching the critic's measured climb across four clicks"
    )
