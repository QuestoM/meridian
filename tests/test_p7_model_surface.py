"""Every read this surface publishes reaches a person, and writes nothing.

Split out of ``test_p7_model_console.py`` under the 450-line rule. That file
asserts what the console says; this one asserts the shape of the surface it says
it through, which is where the round-one failure lived:

1. **Every GET has a screen.** The routes are enumerated from the application's
   own route table, never typed here. Typed by hand, the list omitted
   ``/api/model/candidates/{candidate_id}`` for a whole round, and the guard
   written to catch a route without a screen could not see the one route that
   had none.
2. **Every reader has a caller.** A reader published by ``console-api.js`` and
   called by no component is the same dead end one level down.
3. **The route carries what the card renders.** A verdict recorded about a
   candidate is readable back from that candidate's own route, which is what
   makes JS-19's "a later reader can see what was tried" true on the screen
   where the question is asked.
4. **A read never writes.** The whole releases store is byte-identical after
   every GET, including the parameterised one.
5. **The competitor boundary holds.** No rival channel name reaches any payload
   on any of them.
6. **Every destination the console names opens.** The header states that the
   activation switch lives on Rules and carries no switch itself, so that
   sentence has to be the way there. Measured on the live DOM in round two:
   ``.mc-header-activation`` held zero controls and the whole header held one,
   so the name reached nothing, which is section 3.6's dead end exactly.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from kairos_api import model_console_artifacts as artifacts
from kairos_api import model_console_candidates as candidates_module
from kairos_api import model_version_store as store
from test_p7_console_bridge_harness import resolve_shell_views

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "kairos_api"
FRONTEND = ROOT / "tv-break-dashboard"

# The backend modules this piece owns that are not named ``model_console*``.
# The console's own helpers are globbed rather than listed, because a helper
# added under the 450-line rule has to inherit the laws with no edit here.
OWNED_BACKEND = ["model_version_store.py", "model_impact_api.py", "model_audience_api.py",
                 "audience_api.py"]
CONSOLE_DIR = FRONTEND / "src" / "model" / "console"
BRIDGE = FRONTEND / "src" / "model" / "console-bridge.jsx"
SHELL_ROOT_VIEW = FRONTEND / "src" / "shell" / "TVBreakDashboard.jsx"

# The word the header uses for the state it mirrors, and the address the bridge
# sends the reader to when they press it.
ACTIVATION_NOTE_KEY = "header.control_on_rules"
RULES_HASH = "Settings"

# An address the shell ships no page for, which is what makes the resolution of
# the one above an answer rather than an echo.
UNKNOWN_HASH = "#NoPageIsFiledUnderThisName"

# The header the shell reads for itself. Every other GET on the surface has to
# be reachable by a person, which is what the guards below measure.
HEADER_ROUTE = "/api/model/console"

# Every field of a recorded verdict that the candidate's own card renders. The
# route is asserted to carry each one and the card to name each one, out of this
# one tuple, so the two halves of the trace cannot drift apart in silence.
VERDICT_FIELDS = ("decision", "actor", "recorded_at", "model_version_name",
                  "reason", "evidence", "adoption")

# The figures inside those two blocks, which are the part a verdict is worth
# reading for: what it was decided on, and what was and was not done about it.
VERDICT_FIGURES = ("evidence.money_state", "evidence.revenue_delta",
                   "evidence.scope", "adoption.state")


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    """The console's own routes on a throwaway store, so no test writes models/."""
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    from kairos_api.model_console_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _get_routes(app) -> "list[str]":
    """Every GET the surface publishes, from the application's own route table."""
    paths = {route.path for route in app.routes
             if isinstance(route, APIRoute)
             and "GET" in route.methods
             and route.path.startswith("/api/model")}
    assert HEADER_ROUTE in paths, "the model surface publishes no console header"
    return sorted(paths)


def _shape(path: str) -> str:
    """A path with its parameter names removed, so a route and a template match."""
    return re.sub(r"\{[^}]*\}", "{}", path)


def _concrete(path: str) -> str:
    """A callable path: a templated one is filled with a real candidate id."""
    if "{" not in path:
        return path
    ids = [candidates_module.candidate_id(p) for p in candidates_module.candidate_paths()]
    assert ids, "no candidate artifact on the shelf, so a detail route cannot be called"
    return path.replace("{candidate_id}", ids[0])


def _unreachable(app) -> "list[str]":
    """Every GET on this application that ``console-api.js`` names no path for."""
    source = (CONSOLE_DIR / "console-api.js").read_text(encoding="utf-8")
    named = {_shape(re.sub(r"\$\{[^}]*\}", "{}", path))
             for path in re.findall(r"['`](/api/model/[^'`]*)['`]", source)}
    return [path for path in _get_routes(app)
            if path != HEADER_ROUTE and _shape(path) not in named]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _store_fingerprint() -> dict[str, str]:
    directory = store.store_dir()
    if not directory.is_dir():
        return {}
    return {path.relative_to(directory).as_posix(): _sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


# ---------------------------------------------------------------------------
# 1 and 2: every read reaches a person, and every reader reaches a screen
# ---------------------------------------------------------------------------


def test_the_surface_publishes_the_reads_the_contract_names(client) -> None:
    """A guard against the two guards below passing because they found nothing."""
    routes = _get_routes(client.app)
    assert len(routes) >= 9, f"the model surface lost reads: {routes}"
    assert "/api/model/candidates/{candidate_id}" in routes


def test_every_get_on_the_surface_has_a_place_on_the_console(client) -> None:
    """No route on this surface is reachable by the server and unreachable by a person.

    The console header is read by the shell itself. Every other GET either owns
    a named section or is a read about one named thing, in which case it is
    published as its own reader and rendered on that thing's card. Both live in
    the frontend, so both are read from there rather than restated here, and the
    routes are enumerated from the running application rather than typed, which
    is the half that was missing when a detail route shipped with no screen.
    """
    unreachable = _unreachable(client.app)
    assert unreachable == [], f"these reads have no place on the console: {unreachable}"
    # The sections are exactly the reads that stand on their own. A read keyed
    # by a name belongs to the section that lists that name, so it is checked
    # above and not counted here.
    source = (CONSOLE_DIR / "console-api.js").read_text(encoding="utf-8")
    standalone = [path for path in _get_routes(client.app)
                  if "{" not in path and path != HEADER_ROUTE]
    sections = re.search(r"SECTIONS = \[(.*?)\]", source, re.S).group(1)
    assert len(re.findall(r"'", sections)) // 2 == len(standalone)


def test_that_guard_reports_a_route_that_has_no_screen(tmp_path, monkeypatch) -> None:
    """The guard above is worth its words only if it bites, so it is made to bite.

    A guard that has never failed is a guard nobody has measured, and this one
    replaces a hand-typed list that passed for a whole round while the route it
    omitted had no screen. The real router is mounted beside one decoy read, and
    the decoy is the only thing reported: the guard sees a route it was never
    told about, which is the whole point of enumerating from the route table.
    """
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    from kairos_api.model_console_api import router

    app = FastAPI()
    app.include_router(router)

    @app.get("/api/model/reads-nobody-renders")
    def _decoy() -> "dict[str, str]":
        return {}

    assert _unreachable(app) == ["/api/model/reads-nobody-renders"]


def test_the_card_names_every_field_of_the_verdict_the_route_carries() -> None:
    """The other half of the trace: the route carries it, and the card says it.

    ``test_the_candidate_detail_route_carries_the_verdict_the_card_renders``
    asserts the payload from the server side and could pass against a card that
    renders none of it, which is the failure it was written for one level down.
    Both read the same tuple, so a field can only leave the card by leaving this
    file. What this cannot prove is pixels; that is measured in a browser and
    recorded in the piece's contract.
    """
    source = (CONSOLE_DIR / "CandidateVerdict.jsx").read_text(encoding="utf-8")
    missing = [field for field in VERDICT_FIELDS if f"record.{field}" not in source]
    assert missing == [], f"the route carries these and no card renders them: {missing}"
    figures = [name for name in VERDICT_FIGURES if name not in source]
    assert figures == [], f"the verdict renders without its basis: {figures}"


def test_every_reader_the_console_publishes_is_called_by_a_screen() -> None:
    """A reader nothing calls is the same dead end as a route nothing reads.

    The detail route was walled, tested and screenless for a round. Publishing
    ``readCandidate`` and leaving it uncalled would be that failure again one
    level down, so every reader the module exports is grepped for in the rest of
    the console tree.
    """
    source = (CONSOLE_DIR / "console-api.js").read_text(encoding="utf-8")
    readers = re.findall(r"export const (\w+) = [^\n]*\bread\(", source)
    assert len(readers) >= 3, f"console-api.js publishes almost no reader: {readers}"
    callers = "\n".join(path.read_text(encoding="utf-8")
                        for path in sorted(CONSOLE_DIR.glob("*.js*"))
                        if path.name != "console-api.js")
    dead = [name for name in readers if name not in callers]
    assert dead == [], f"these readers are published and no screen calls them: {dead}"


# ---------------------------------------------------------------------------
# 3: the route carries what the card renders
# ---------------------------------------------------------------------------


def test_the_candidate_detail_route_carries_the_verdict_the_card_renders(client) -> None:
    """The payload the card reads, asserted end to end on a real candidate.

    Recording a verdict must leave a trace on the candidate it was about, and
    the trace is this block. Without this the screen could render a verdict the
    route never carries, or carry one no screen shows, which is the shape the
    round-one measurement found.
    """
    candidate = candidates_module.candidate_id(candidates_module.candidate_paths()[0])
    before = client.get(f"/api/model/candidates/{candidate}").json()
    assert before["candidate"]["id"] == candidate
    assert before["decision"] is None, "a throwaway store already holds a verdict"

    recorded = client.post("/api/model/decisions", json={
        "decision": "not_shipped", "subject": "candidate", "candidate_id": candidate,
        "reason": "the held-out figures do not move far enough to be worth the restatement",
    })
    assert recorded.status_code == 200, recorded.text

    after = client.get(f"/api/model/candidates/{candidate}").json()["decision"]
    assert after is not None, "the verdict left no trace on the candidate it was about"
    for key in VERDICT_FIELDS:
        assert key in after, f"the card renders {key} and the route does not carry it"
    assert after["decision_id"] == recorded.json()["decision_id"]
    assert after["evidence"]["money_state"] in ("measured", "stale", "not_measured")


def test_a_verdict_about_one_candidate_reaches_no_other_candidate(client) -> None:
    """The trace lands on its own card and on no one else's.

    A verdict smeared across the shelf would read as five decisions where one
    was taken, which is worse than the blank card it replaced.
    """
    paths = candidates_module.candidate_paths()
    assert len(paths) >= 2, "one candidate on the shelf proves nothing about scoping"
    subject = candidates_module.candidate_id(paths[0])
    others = [candidates_module.candidate_id(path) for path in paths[1:]]
    client.post("/api/model/decisions", json={
        "decision": "not_shipped", "subject": "candidate", "candidate_id": subject,
        "reason": "recorded to check that it lands on one card only",
    })
    for other in others:
        assert client.get(f"/api/model/candidates/{other}").json()["decision"] is None, other


# ---------------------------------------------------------------------------
# 4 and 5: a read never writes, and no rival name is on any of them
# ---------------------------------------------------------------------------


def test_no_get_on_the_surface_writes_the_store(client) -> None:
    before = _store_fingerprint()
    for route in _get_routes(client.app):
        assert client.get(_concrete(route)).status_code == 200, route
    assert _store_fingerprint() == before, "a GET mutated the releases store"


def test_no_rival_channel_name_reaches_any_console_payload(client) -> None:
    """The rival names are inside the artifact this surface reads, so this is real.

    The audience base carries a per-channel map for every channel in the
    training data, and the series factor's cell keys are prefixed with the
    channel name. The rivals below are taken from those keys rather than from
    settings, so the test names exactly the strings that are one careless
    serialization away from a payload.
    """
    payload = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    base = payload.get("base") or {}
    owned = str(base.get("owned_channel") or "")
    rivals = sorted(name for name in (base.get("hist_channel") or {}) if name != owned)
    assert owned, "the artifact records no owned channel, so this test would prove nothing"
    assert rivals, "the artifact holds no rival channels, so this test would prove nothing"
    for route in _get_routes(client.app):
        body = json.dumps(client.get(_concrete(route)).json(), ensure_ascii=False)
        for rival in rivals:
            assert rival not in body, f"{route} names the rival channel {rival}"


# ---------------------------------------------------------------------------
# 6: the one destination the console names in words, and the way to it
# ---------------------------------------------------------------------------


def test_the_sentence_naming_rules_is_the_control_that_goes_there() -> None:
    """The activation mirror's own words open the page they name.

    The console mirrors whether runs consume the audience model and owns no
    switch, because throwing it changes a run and its home is Rules. That makes
    the sentence a promise. Round two rendered it as a plain note: measured on
    the live DOM, ``.mc-header-activation`` contained no anchor, button, role or
    tab stop at all, so the reader who was told where the switch lives could not
    get there from the sentence telling him.
    """
    source = (CONSOLE_DIR / "ModelConsole.jsx").read_text(encoding="utf-8")
    control = re.search(r"onClick=\{onOpenRules\}>(.{0,200}?)</(?:button|Button|Pressable)>", source, re.S)
    assert control is not None, "the console renders no control that opens Rules"
    assert ACTIVATION_NOTE_KEY in control.group(1), (
        f"the control that opens Rules does not carry {ACTIVATION_NOTE_KEY}, so some other string is the one that moves and the sentence is still a dead end"
    )


def test_the_bridge_hands_the_console_both_ways_out_and_a_real_address() -> None:
    """The console falls back to a plain note with no handler, so the handler must ship.

    Both controls come from the same host and this asserts they arrive together,
    which is what makes the fallback branch unreachable in the product rather
    than merely unlikely.
    """
    bridge = BRIDGE.read_text(encoding="utf-8")
    mount = re.search(r"<ModelConsole([^/]*)/>", bridge)
    assert mount is not None, "the bridge no longer mounts the console"
    for prop in ("onBack={back}", "onOpenRules={toRules}"):
        assert prop in mount.group(1), f"the bridge stopped passing {prop}"
    assert re.search(rf"RULES_HASH = '{RULES_HASH}'", bridge), (
        "the bridge's address for Rules moved; re-measure it against the shell's route table"
    )
    assert "window.location.hash = RULES_HASH" in bridge, (
        "the control no longer addresses anything, so pressing it goes nowhere"
    )


def test_the_frozen_shell_still_resolves_that_address_to_a_page(tmp_path) -> None:
    """The premise the control rests on, driven rather than read.

    A hash is only a destination because the shell turns it into one, and the
    resolver that turns it lives in a file this piece may not write. Round three
    pinned the line of source that did the resolving, wave one replaced that
    line with an equivalent one, and a test went red while the behaviour it
    guards kept working. So the resolver is now run: the frozen module is
    bundled by the product's own bundler and asked, in a browser, what the
    console's address for Rules resolves to.

    The unknown address is asked with it, because a resolver that echoed
    whatever it was handed would satisfy the first assertion and prove nothing.

    The shell's re-read on a hash change is the one half that cannot be driven
    here: it lives inside the whole workspace component. It is asserted as the
    two facts it is, the listener and the re-read, rather than as the formatting
    of the lines that carry them, which is the mistake this test is correcting.
    """
    views = resolve_shell_views(tmp_path, [f"#{RULES_HASH}", UNKNOWN_HASH, ""])
    assert "failed" not in views, views.get("failed")
    resolved = views["resolved"]
    assert resolved[f"#{RULES_HASH}"] == "Governance", (
        f"the shell resolves the console's address for Rules to {resolved[f'#{RULES_HASH}']}"
    )
    assert resolved[UNKNOWN_HASH] != UNKNOWN_HASH.lstrip("#"), (
        "the resolver returns any address it is handed, so the assertion above proves nothing"
    )
    assert resolved[UNKNOWN_HASH] == resolved[""], (
        f"an address the shell does not know resolves to {resolved[UNKNOWN_HASH]}"
        f" and no address at all resolves to {resolved['']}"
    )
    shell = SHELL_ROOT_VIEW.read_text(encoding="utf-8")
    assert re.search(r"addEventListener\(\s*'hashchange'", shell), (
        "the shell no longer listens for the hash change the control makes"
    )
    assert "routeFromLocation({" in shell and "setActiveViewState(route.view);" in shell, (
        "the shell no longer re-reads and normalises the address into a canonical view"
    )


# ---------------------------------------------------------------------------
# 7: the display strings this surface sends, one to a source line
# ---------------------------------------------------------------------------


def _wrapped_strings(path: Path) -> list:
    """Every string constant over 40 characters that spans more than one line.

    Docstrings are excluded because they are addressed to a reader of the file,
    not to a reader of the screen. Everything else on this surface is a sentence
    somebody reads rendered, so a line break in the source is a line break the
    writer chose on behalf of a layout he cannot see.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documentation = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = (node.body or [None])[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            documentation.add(id(first.value))
    return [
        f"{path.name}:{node.lineno}-{node.end_lineno} {node.value[:60]}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in documentation
        and len(node.value) > 40 and node.end_lineno > node.lineno
    ]


def test_no_display_string_on_this_surface_is_wrapped_across_source_lines() -> None:
    """One display string per source line, and the UI does the wrapping.

    Round two shipped fourteen of them, in Hebrew and in English, on the modules
    below. A sentence broken by hand in the source is a sentence broken at a
    width nobody measured, and it is the reason the same paragraph reads
    differently in the two languages of this product.
    """
    modules = sorted(BACKEND.glob("model_console*.py"))
    modules += [BACKEND / name for name in OWNED_BACKEND]
    assert len(modules) >= 8, f"the module list went empty: {[p.name for p in modules]}"
    wrapped = [row for path in modules for row in _wrapped_strings(path)]
    assert wrapped == [], "display strings wrapped across source lines: " + "; ".join(wrapped)
