"""W0-2 shell seams: the structure the split established, pinned.

Wave zero broke the 6,236-line `TVBreakDashboard.jsx` into a shell plus one
destination tree per wave-1 piece, and lifted the design tokens into
`src/tokens.css`. These tests pin the invariants that make that split hold, so
a later piece cannot quietly reassemble the monolith, define a second token
source, or drop a file back at the top level.

They also carry, against the new paths, the four contracts
`tests/test_qa2_dashboard_components.py` asserts against the old top-level
ones, so no coverage lapses while that file's path resolution is ruled on.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"

# The trees section 8.2 of docs/ux-gauntlet/spec.md names for wave zero.
DESTINATION_TREES = (
    "shell",
    "today",
    "plan/week",
    "plan/day",
    "plan/break",
    "clients",
    "rules",
    "sources",
    "model",
    "history",
    "kai",
)

# Two paths stay at the top level of src/ and each has a reason.
#   index.jsx  - the Vite entry, named by the frozen tv-break-dashboard/index.html
#   tokens.css - the spec names this exact path for the design tokens
# vocabulary.js and session.js belong to W0-4, not to this piece.
TOP_LEVEL_ALLOWED = {"index.jsx", "tokens.css", "vocabulary.js", "session.js"}

# Over the 450-line law before wave zero and moved, not created, by W0-2. Each
# is now inside its wave-1 owner's tree and is that owner's to split, so this is
# the inherited debt as an UPPER BOUND, not a snapshot: an owner who splits one
# of these is doing the thing the law asks for and must not fail this test.
KNOWN_OVERSIZE = {
    "shell/styles.css",
    "plan/day/ScheduleEditor.jsx",
    "rules/ConstraintBuilder.jsx",
}

LINE_CAP = 450


def _sources(*suffixes: str) -> list[Path]:
    return sorted(path for path in SRC.rglob("*") if path.suffix in suffixes)


def _read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def _object_body(source: str, anchor: str) -> str:
    """The brace-balanced object literal that follows `anchor`, without braces."""
    text = source[source.index(anchor):]
    start = text.index("{", 0 if anchor.startswith("const") else text.index("const labels"))
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    raise AssertionError(f"unbalanced object literal after {anchor!r}")


def _object_keys(source: str, anchor: str) -> set[str]:
    return set(re.findall(r"(?m)^\s*'?([A-Za-z][A-Za-z ]*?)'?\s*:", _object_body(source, anchor)))


def _object_pairs(source: str, anchor: str) -> dict[str, str]:
    pair = r"(?m)^\s*'?([A-Za-z][A-Za-z ]*?)'?\s*:\s*'([^']*)'"
    return dict(re.findall(pair, _object_body(source, anchor)))


def _exported_definitions(source: str) -> dict[str, str]:
    """Every top-level export in a module, mapped to its verbatim source text."""
    found = {}
    for match in re.finditer(r"(?m)^export (?:default )?(?:async )?(?:function|const|let|class) (\w+)", source):
        if source[match.start():].startswith("export function"):
            end = source.index("\n}\n", match.start()) + 3
        else:
            end = source.find("\n", match.start())
        found[match.group(1)] = source[match.start():end]
    return found


def test_every_top_level_source_file_moved_into_a_tree() -> None:
    """Nothing is left loose at src/ except the entry and the token sheet."""
    loose = {
        path.name
        for path in SRC.iterdir()
        if path.is_file() and path.suffix in {".js", ".jsx", ".css"}
    }
    assert loose <= TOP_LEVEL_ALLOWED, f"unexpected top-level src files: {sorted(loose - TOP_LEVEL_ALLOWED)}"


def test_every_source_file_lives_in_a_named_destination_tree() -> None:
    """Each file resolves to exactly one of the trees the ownership table names."""
    for path in _sources(".js", ".jsx", ".css"):
        relative = path.relative_to(SRC).as_posix()
        if relative in TOP_LEVEL_ALLOWED:
            continue
        assert any(
            relative.startswith(f"{tree}/") for tree in DESTINATION_TREES
        ), f"{relative} is in no destination tree"


def test_the_shell_no_longer_carries_the_page_components() -> None:
    """The monolith is gone: each page component has exactly one home."""
    shell = _read("shell/TVBreakDashboard.jsx")
    assert len(shell.splitlines()) < LINE_CAP
    for name in (
        "function OverviewPage",
        "function SchedulePage",
        "function InventoryPage",
        "function BreakLibraryPage",
        "function CampaignsPage",
        "function ForecastsPage",
        "function ReportsPage",
        "function DataPage",
        "function SettingsPanel",
        "function OptimizerWorkspace",
    ):
        assert name not in shell, f"{name} is still inside the shell"
        homes = [
            path.relative_to(SRC).as_posix()
            for path in _sources(".jsx")
            if name in path.read_text(encoding="utf-8")
        ]
        assert len(homes) == 1, f"{name} is defined in {homes}"


def test_no_file_this_piece_created_is_over_the_line_cap() -> None:
    """The 450-line law: no new breach, and the inherited ones only ever shrink.

    The rule this pins is one-directional. A file over the cap that is not one
    of the inherited breaches is a violation whoever wrote it must fix. A file
    that leaves the inherited set has been split by its owner, which is the law
    being obeyed, so it is not a failure to report.
    """
    oversize = {
        path.relative_to(SRC).as_posix()
        for path in _sources(".js", ".jsx", ".css")
        if len(path.read_text(encoding="utf-8").splitlines()) > LINE_CAP
    }
    assert oversize <= KNOWN_OVERSIZE, (
        f"over the {LINE_CAP}-line cap and not an inherited breach: {sorted(oversize - KNOWN_OVERSIZE)}"
    )


def test_tokens_are_defined_in_tokens_css_and_nowhere_else() -> None:
    """One token source. A stylesheet may read a token; it may not define one."""
    definition = re.compile(r"(?m)^\s*(--[a-z0-9-]+)\s*:")
    tokens = definition.findall(_read("tokens.css"))
    assert len(tokens) >= 40, f"tokens.css defines only {len(tokens)} variables"
    for path in _sources(".css"):
        if path.name == "tokens.css":
            continue
        found = definition.findall(path.read_text(encoding="utf-8"))
        assert found == [], f"{path.relative_to(SRC)} defines tokens {found}"


def test_tokens_css_is_loaded_before_the_shell_stylesheet() -> None:
    """The entry imports the tokens first, so no sheet renders before them."""
    entry = _read("index.jsx")
    assert entry.index("./tokens.css") < entry.index("./shell/styles.css")


def test_all_seventeen_navigation_entries_survive_the_split() -> None:
    """The route list is the shell's, and it still has every entry."""
    nav = _read("shell/nav.js")
    for label in (
        "Overview",
        "Optimizer",
        "Schedule",
        "Inventory",
        "Break Library",
        "Campaigns",
        "Forecasts",
        "Calendar",
        "Reports",
        "Data",
        "Advertisers",
        "Agencies",
        "Pricing",
        "Overrides",
        "Assistant",
        "Versions",
        "Settings",
    ):
        assert f"'{label}'" in nav, f"navigation entry {label} is missing"
    assert nav.count("],\n") == 17, "the navigation list no longer holds 17 entries"


def test_assistant_hash_still_opens_the_dock_over_the_current_page() -> None:
    """#Assistant must never become a page: it opens the dock where you are."""
    shell = _read("shell/TVBreakDashboard.jsx")
    assert "viewFromLocation() === 'Assistant'" in shell
    assert "setAssistantOpen(true)" in shell
    assert "if (next === 'Assistant')" in shell


def test_the_repointed_page_text_helper_is_the_same_function() -> None:
    """Eight modules now read pageText from the shell rather than from clients.

    That is only safe because the two implementations are the same three lines,
    so this asserts the equality the repoint relied on.
    """
    body = re.compile(r"export function pageText\(locale, en, he\) \{\n(.*?)\n\}", re.S)
    shell_body = body.search(_read("shell/surface-helpers.js"))
    clients_body = body.search(_read("clients/advertisers-helpers.js"))
    assert shell_body and clients_body
    assert shell_body.group(1) == clients_body.group(1)


# --- carried over from tests/test_qa2_dashboard_components.py, new paths ---


def test_staleness_banner_covers_every_backend_group_label() -> None:
    """Every label schedule_freshness can put into `changed` maps in the banner."""
    from kairos.export.schedule_freshness import GROUP_LABELS

    source = _read("shell/ScheduleStalenessBanner.jsx")
    for key, label in GROUP_LABELS.items():
        assert f"'{label}'" in source or f"{key}:" in source, (
            f"ScheduleStalenessBanner.jsx has no label mapping for backend group {key!r}"
        )


def test_staleness_banner_keeps_its_agreement_free_frame() -> None:
    """Unknown entries render verbatim and the double-verb fallback stays gone."""
    source = _read("shell/ScheduleStalenessBanner.jsx")
    assert "changedLabels[key] || String(key" in source
    assert "הקלט השתנה" not in source
    assert "${changedPhrase} השתנו" not in source
    assert "חל שינוי ב${changedPhrase}" in source
    assert "חל שינוי בקלט הלוח" in source


def test_daypart_label_helper_covers_engine_keys() -> None:
    """surface-helpers daypartLabel covers every key daypart_for_hour can emit."""
    from kairos.data.dayparts import daypart_for_hour

    engine_keys = {daypart_for_hour(hour) for hour in range(24)}
    engine_keys.discard(None)
    assert engine_keys, "engine produced no daypart keys at all"
    source = _read("shell/surface-helpers.js")
    for key in sorted(engine_keys) + ["unclassified"]:
        assert f"{key}:" in source, f"daypartLabel is missing engine key {key!r}"


def test_the_two_shell_daypart_helpers_serve_disjoint_vocabularies() -> None:
    """Two daypart tables, two key spaces, ruled in contracts/W0-2.md section 3.

    `shell/surface-helpers.js` answers the engine taxonomy that arrives over the
    wire; `shell/labels.js` answers the client-side planning grid built by
    `shell/plan-model.js`. Neither understands the other's keys, so a page that
    imports the wrong one renders raw English inside a Hebrew surface.
    """
    from kairos.data.dayparts import daypart_for_hour

    engine_keys = {daypart_for_hour(hour) for hour in range(24)}
    engine_keys.discard(None)
    engine_table = _object_keys(_read("shell/surface-helpers.js"), "const DAYPART_LABELS")
    client_source = _read("shell/plan-model.js")
    client_list = re.search(r"daypartKeys = \[(.*?)\]", client_source)
    assert client_list, "shell/plan-model.js no longer declares daypartKeys"
    client_keys = set(re.findall(r"'([A-Za-z][A-Za-z ]*)'", client_list.group(1)))
    client_table = _object_keys(_read("shell/labels.js"), "export function daypartLabel")
    assert engine_keys, "the engine produced no daypart keys at all"
    assert engine_keys <= engine_table, f"engine keys missing from the table: {sorted(engine_keys - engine_table)}"
    assert client_keys == client_table, f"the planning grid and its label table disagree: {sorted(client_keys ^ client_table)}"
    assert not engine_table & client_table, f"the two vocabularies now overlap on {sorted(engine_table & client_table)}"


def test_every_duplicated_shell_export_is_identical_or_a_declared_fork() -> None:
    """The complete census of names the shell exports from two modules.

    The duplication is inherited: at 342a2896 nine of these were defined both
    inside `TVBreakDashboard.jsx` and in `surface-helpers.js`, and the split gave
    the monolith copy a module home. Nine bodies are byte identical, which is
    what makes the two import paths interchangeable. Two are deliberate forks
    over different vocabularies and are ruled in contracts/W0-2.md section 3.
    """
    identical = {
        "API_BASE": ["api.js", "surface-helpers.js"],
        "finiteNumber": ["format.jsx", "surface-helpers.js"],
        "formatCurrency": ["format.jsx", "surface-helpers.js"],
        "formatCurrencyAxis": ["format.jsx", "surface-helpers.js"],
        "formatMinutes": ["format.jsx", "surface-helpers.js"],
        "formatNumber": ["format.jsx", "surface-helpers.js"],
        "formatPercent": ["format.jsx", "surface-helpers.js"],
        "normalizeRows": ["plan-model.js", "surface-helpers.js"],
        "pageText": ["format.jsx", "surface-helpers.js"],
    }
    forked = {
        "daypartLabel": ["labels.js", "surface-helpers.js"],
        "programTypeLabel": ["labels.js", "surface-helpers.js"],
    }
    census: dict[str, dict[str, str]] = {}
    for path in sorted((SRC / "shell").iterdir()):
        if path.suffix not in {".js", ".jsx"}:
            continue
        for name, body in _exported_definitions(path.read_text(encoding="utf-8")).items():
            census.setdefault(name, {})[path.name] = body
    duplicated = {name: modules for name, modules in census.items() if len(modules) > 1}
    assert sorted(duplicated) == sorted({**identical, **forked}), f"the duplicate-export census moved: {sorted(duplicated)}"
    for name, modules in identical.items():
        assert sorted(duplicated[name]) == modules, f"{name} now comes from {sorted(duplicated[name])}"
        assert len(set(duplicated[name].values())) == 1, f"{name} has diverged between {modules}"
    for name, modules in forked.items():
        assert sorted(duplicated[name]) == modules, f"{name} now comes from {sorted(duplicated[name])}"
        assert len(set(duplicated[name].values())) == 2, f"{name} was merged without a ruling in contracts/W0-2.md"


def test_the_two_shell_program_type_tables_are_the_same_table() -> None:
    """`programTypeLabel` is a real duplicate, so both copies stay one table."""
    helpers = _object_pairs(_read("shell/surface-helpers.js"), "const PROGRAM_TYPE_LABELS_HE")
    labels = _object_pairs(_read("shell/labels.js"), "export function programTypeLabel")
    assert len(helpers) == 18, f"the classifier vocabulary moved to {len(helpers)} entries"
    assert helpers == labels, f"the two copies disagree on {sorted(set(helpers.items()) ^ set(labels.items()))}"


def test_removed_dead_exports_stay_gone_and_unreferenced() -> None:
    """The dead helpers were deleted and nothing under src/ still references them."""
    removed = [
        "DAYPART_PRESETS",
        "chipOptions",
        "filterAdvertisers",
        "sortAdvertisers",
        "computeSummary",
        "collectDaypartTokens",
        "fetchJsonOrError",
    ]
    sources = {
        path.relative_to(SRC).as_posix(): path.read_text(encoding="utf-8")
        for path in _sources(".js", ".jsx")
    }
    for name in removed:
        hits = [path for path, text in sources.items() if name in text]
        assert hits == [], f"dead export {name!r} is still referenced in {hits}"
