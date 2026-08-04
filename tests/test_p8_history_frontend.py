"""P8 History, the surface: the contracts the rebuilt destination has to keep.

This is Bar 3 and the mechanics. History replaced a restore-point list called
``AssistantVersions.jsx`` with a timeline that contains it, so every capability
that file carried has to be provably present in the tree that replaced it, one
test per capability. The sharpest of them is the viewer write-lock, which the
regression row names by file and line: it is now the session's own ``canWrite``
plus the endpoint's own ``can_edit``, which is stricter than what it replaced,
and the test would fail if either half went missing.

Two siblings hold the rest, because this file reached 491 lines against the
450-line law: ``tests/test_p8_history_laws.py`` for the laws a surface can be
read for, and ``tests/test_p8_history_modules.py`` for the rules that are
executed with node rather than read.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
HISTORY = SRC / "history"


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


# --- the destination exists and the shell still reaches it ----------------------

def test_the_shell_entry_point_still_exists_and_renders_history() -> None:
    """The shell's router is frozen and imports history/VersionsPage, so that
    module has to keep its name and its two props."""
    router = (SRC / "shell" / "workspace-router.jsx").read_text(encoding="utf-8")
    assert "from '../history/VersionsPage'" in router
    assert "<VersionsPage locale={locale} notify={notify} />" in router
    entry = _read("VersionsPage.jsx")
    assert "export default function VersionsPage({ locale, notify })" in entry
    assert "<HistoryPage locale={locale} notify={notify} />" in entry


def test_the_two_components_other_trees_import_still_export_the_same_names() -> None:
    """The shell renders ActivityFeed and a rules surface renders
    ActivityLogPanel. Neither import may break because History moved in, and
    neither component may lose its default export while an importer exists."""
    assert "export default function ActivityFeed(" in _read("ActivityFeed.jsx")
    assert "export default ActivityLogPanel;" in _read("ActivityLogPanel.jsx")
    assert "from '../history/ActivityFeed'" in (SRC / "shell" / "TVBreakDashboard.jsx").read_text(encoding="utf-8")
    importers = [
        path.name for path in SRC.rglob("*.jsx")
        if "history/ActivityLogPanel" in path.read_text(encoding="utf-8")
    ]
    assert importers, "ActivityLogPanel has an importer outside this tree, so it stays"


def test_the_destination_is_named_from_the_frozen_vocabulary() -> None:
    labels = _read("history-labels.js")
    assert "from '../vocabulary.js'" in labels
    assert "word('place.history'" in labels
    assert "historyPlace(locale)" in _read("HistoryPage.jsx")


# --- Bar 3: every capability the replaced component carried -------------------

def test_the_removed_component_is_gone_and_nothing_imports_it() -> None:
    assert not (HISTORY / "AssistantVersions.jsx").exists()
    assert "AssistantVersions" not in "\n".join(
        path.read_text(encoding="utf-8") for path in SRC.rglob("*.jsx"))


def test_the_restore_point_list_survives() -> None:
    page = _read("HistoryPage.jsx")
    assert "fetchTimeline" in page
    assert "restore_point" in _read("history-labels.js")


def test_the_diff_on_a_restore_point_survives() -> None:
    restore = _read("HistoryRestore.jsx")
    assert "fetchVersionDiff" in restore
    assert "HistoryDiff" in restore
    diff = _read("HistoryDiff.jsx")
    for store in ("settings", "advertisers"):
        assert store in diff, f"the {store} diff shape is still rendered"


def test_a_row_the_restore_would_add_or_remove_is_read_as_an_identity() -> None:
    """The measured defect: chipText fell back to JSON.stringify because it
    looked for item.id or item.name and not one of the eight stores uses either.
    What the identity returns is executed in tests/test_p8_history_modules.py;
    what is asserted here is that the surface asks for it and that a chip that
    can hold Hebrew is no longer forced left to right."""
    diff = _read("HistoryDiff.jsx")
    assert "import { rowIdentity } from './history-rows';" in diff
    assert "JSON.stringify" not in diff, "nothing on this surface reaches a person as a dumped record"
    assert '<code dir="ltr"' not in diff, "the chip that can hold Hebrew is not forced left to right"
    assert 'className="hist-diff-chip-name" dir="auto"' in diff
    assert 'dir={part.ltr ? \'ltr\' : \'auto\'}' in diff, (
        "only an id, a date and a clock are isolated left to right")
    words = _read("history-row-words.js")
    labels = _read("history-labels.js")
    assert "from './history-row-words.js';" in labels, (
        "the row vocabulary is re-exported, so this destination still has one import for a word")
    for table in ("ROW_EFFECTS", "ROW_KINDS", "ROW_MODES", "ROW_EVENT_TYPES", "ROW_PARTS"):
        assert f"export const {table}" in words and table in labels


def test_a_changed_field_is_read_as_the_values_inside_it() -> None:
    """The same defect in the one store the chips did not cover. A settings
    change carries the value the store holds and one of them is a whole nested
    object, so the row that decided a rate-card restore printed a record cut at
    seventy-seven characters. What the reader gets is executed in
    tests/test_p8_history_modules.py; what is asserted here is that the surface
    asks for it, that no cell a store writes into is forced left to right, and
    that the words are read from the product's own rather than spelled twice."""
    diff = _read("HistoryDiff.jsx")
    assert "import { changeRows } from './history-fields';" in diff
    assert 'dir="ltr"' not in diff, "no cell a store can write Hebrew into is forced left to right"
    assert "<bdi>{cur}</bdi>" in diff and "<bdi>{ver}</bdi>" in diff, (
        "each value is isolated, so a Hebrew name and a negative number both read as written")
    fields = _read("history-fields.js")
    assert "import { copyByLocale } from '../shell/copy.js';" in fields, (
        "a settings key is named by the word its own surface already uses")
    assert "FORCE_LABELS" in fields, "and by the run report's table where that carries it"
    assert "const CHANNEL_FIELD = 'operator_channel';" in fields, (
        "the one settings field that can carry a channel name is guarded by name")


def test_no_channel_name_can_reach_a_restore_preview() -> None:
    """Two shapes could carry one: a pin's target_id, which is
    `2024-11-01|<channel>|001`, and the legacy flat channel scope on a
    restriction. Neither value is printed, and the predicate summary reads a
    closed set of six fields so a seventh could not open the door either."""
    rows = _read("history-rows.js")
    assert "target_id" not in rows.split("function overrideRow")[1].split("return {")[0], (
        "a pin is identified by its anchor, never by the id that embeds a channel")
    assert "if (type === 'channel') return [rowPhrase(ROW_CHANNEL_SCOPE, locale)];" in rows
    assert "const SUMMARISED = ['programme', 'genre', 'daypart', 'weekday', 'date', 'hour'];" in rows
    assert "if (!SUMMARISED.includes(field)) return '';" in rows


def test_the_file_selection_restore_survives() -> None:
    restore = _read("HistoryRestore.jsx")
    assert "restoreVersion(versionId, chosen)" in restore
    assert 'type="checkbox"' in restore
    assert "hist-restore-files" in restore


def test_the_rename_survives() -> None:
    assert "renameVersion(versionId, label.trim())" in _read("HistoryRestore.jsx")


def test_the_manual_restore_point_survives() -> None:
    page = _read("HistoryPage.jsx")
    assert "saveRestorePoint(pointLabel.trim())" in page


def test_the_viewer_write_lock_survives_and_is_now_two_gates() -> None:
    """Bar 3 names the viewer write-lock by file and line. It is now the
    session's own canWrite through the frozen session.js, plus the payload's own
    can_edit, which is the endpoint speaking. Both halves are asserted."""
    page = _read("HistoryPage.jsx")
    assert "payloadCanEdit(body || {}, session, WALLS.readOnlyRole)" in page
    assert "from '../session.js'" in page
    assert "gate.canEdit ? (" in page, "the write control is conditional on the gate"
    assert "gate.reason" in page, "the refusal is legible before the click"
    restore = _read("HistoryRestore.jsx")
    assert "canEdit ? (" in restore
    assert "canEditReason" in restore


def test_the_safety_point_promise_is_still_on_the_restore_control() -> None:
    restore = _read("HistoryRestore.jsx")
    assert "נשמר קודם כנקודת שחזור" in restore
    assert "restore point first" in restore


def test_a_file_this_account_may_not_put_back_is_locked_with_the_servers_own_words() -> None:
    """The refusal is read before the click, per file, and it is the endpoint's
    string rather than one this tree wrote: a channel account is refused the
    settings document and the calendar, and the other seven stay restorable.
    The control waits for the diff, because that is the read that carries the
    per-file answer and a control offered over permissions it has not read is
    the 403-after-the-click this closed."""
    restore = _read("HistoryRestore.jsx")
    assert "result.data.file_permissions" in restore, "the per-file answer is read from the payload"
    assert ".can_edit !== false" in restore
    assert "can_edit_reason" in restore, "the reason is the server's own string, verbatim"
    assert "diff.state === 'ready' ?" in restore, "the control waits for the per-file answer"
    assert "selected.has(file) && permitted(file)" in restore, "a withheld file never reaches the request"
    assert "disabled={busy || !permitted(file)}" in restore
    assert "hist-file-why" in restore and "<Lock size={11} />" in restore


# --- what the destination adds, and the reference mechanics -------------------

def test_the_keyboard_is_taught_on_the_surface_that_answers_to_it() -> None:
    labels = _read("history-labels.js")
    page = _read("HistoryPage.jsx")
    assert "KEY_HINTS" in labels and "KEY_HINTS" in page
    for key in ("'J'", "'K'", "'Enter'", "'Esc'", "'/'"):
        assert key in labels, f"{key} is taught in the hint row"
    assert "event.key === 'j' || event.key === 'ArrowDown'" in page
    assert "event.key === 'k' || event.key === 'ArrowUp'" in page


def test_the_opened_record_keeps_its_place_in_the_set() -> None:
    detail = _read("HistoryDetail.jsx")
    assert "`${position} / ${total}`" in detail
    assert "onStep(-1)" in detail and "onStep(1)" in detail


def test_the_filters_live_in_the_content_not_in_the_navigation() -> None:
    """The six kinds are tabs inside the page, not six rail entries.

    The rail's own length is not this piece's to freeze: it stood at 17 when this
    was written and at 15 today, because Calendar and Pricing were folded into
    destinations by the pieces that own them, which is the whole point of the
    rebuild. What this piece owes is that it adds no entry and removes none, so
    the assertion is on its own single entry and on the fact that nothing in this
    tree writes the rail at all.
    """
    page = _read("HistoryPage.jsx")
    assert 'role="tablist"' in page
    nav = (SRC / "shell" / "nav.js").read_text(encoding="utf-8")
    entries = re.findall(r"\['([^']+)', \w+\],", nav)
    assert entries.count("Versions") == 1, f"the destination is in the rail exactly once: {entries}"
    assert not any("nav.js" in path.read_text(encoding="utf-8") for path in HISTORY.glob("*.js*")), (
        "and no file in this tree writes a navigation entry of its own")


def test_a_row_is_never_a_dead_end() -> None:
    labels = _read("history-labels.js")
    assert "ACTION_DOORS" in labels
    detail = _read("HistoryDetail.jsx")
    assert "hist-door" in detail
    assert "onOpenVersion(facts.safety_version_id)" in detail
    assert "{facts.version_id ? (" in detail, "an unrecorded id is words, never a button that does nothing"


def test_an_asked_for_entry_is_never_answered_with_a_different_one() -> None:
    """The failure this closes, measured by a blind critic on their own
    instance: with the Restore filter on, following "Undo it with 1337540bd866"
    left the url and the detail exactly as they were, because the auto-select
    effect snapped the selection back to row one. Selecting the newest row when
    a specific entry was asked for answers a question nobody asked, and does it
    silently."""
    page = _read("HistoryPage.jsx")
    assert "if (selected || !entries.length || requested.current) return;" in page, (
        "the default selection stands aside while a specific entry is pending")
    assert "onOpenVersion={openVersion}" in page
    assert "const query = addressQuery(address);" in page
    assert "requested.current = address;" in page
    assert "setKind(query.kind);" in page and "setLimit(query.limit);" in page
    # A row the reader picks drops the pending request, so the note cannot outlive it.
    assert "const choose = useCallback((id) => {" in page
    assert "onSelect={(entry) => choose(entry.id)}" in page
    assert "choose(entries[next].id);" in page
    # And the note cannot fire against a body that is one request behind.
    assert "loaded.kind === kind" in page


def test_an_act_that_saved_nothing_reads_as_one() -> None:
    """A preview is its own kind on the surface as well as in the payload, it is
    drawn differently, and its opened entry says in words that there is nothing
    to put back."""
    labels = _read("history-labels.js")
    assert "preview: ['Preview', 'תצוגה מקדימה']" in labels
    assert "placement_preview" in labels and "price_preview" in labels
    css = _read("history.css")
    assert ".hist-dot.k-preview" in css
    assert '.hist-row[data-kind="preview"]' in css
    detail = _read("HistoryDetail.jsx")
    assert "Nothing was saved." in detail
    assert "לא נשמר דבר." in detail


def test_repeated_previews_fold_into_one_row_that_opens_into_all_of_them() -> None:
    """Measured on the running instance: 200 entries rendered as 106 rows, and
    the largest fold held five identical price tests one second apart. Folding
    hides nothing, so the opened row lists every member."""
    fold = _read("history-fold.js")
    assert "export function foldPreviews" in fold
    assert "a.kind === 'preview'" in fold and "b.kind === 'preview'" in fold
    assert "isoDay(a.ts) === isoDay(b.ts)" in fold
    page = _read("HistoryPage.jsx")
    assert "foldPreviews(" in page
    detail = _read("HistoryDetail.jsx")
    assert "hist-members" in detail
    assert "members.map(" in detail


def test_a_change_is_never_folded_away() -> None:
    """Folding a change would hide the one thing this destination is for."""
    fold = _read("history-fold.js")
    body = fold.split("function sameFold")[1].split("}")[0]
    assert "'preview'" in body
    assert "'change'" not in body


def test_an_entry_has_an_address_that_survives_a_reload() -> None:
    """Kai hands back a restore point and its own control has to land on that
    point. The address is a query parameter because the shell resolves the hash
    by exact match against its seventeen entries, and the search string survives
    a hash change untouched."""
    labels = _read("history-labels.js")
    assert "export const ADDRESS_PARAM = 'entry';" in labels
    assert "window.history.replaceState" in labels
    assert "window.location.hash" in labels, "the hash is preserved, never rewritten"
    page = _read("HistoryPage.jsx")
    assert "useState(readAddress)" in page
    assert "writeAddress(selected ? addressOf(selected) : '')" in page
    assert "addressMissed" in page, "a link into an unloaded range says so"


def test_a_share_of_one_is_not_printed_as_a_share_of_a_hundred() -> None:
    """Measured on the running instance before the fix: the retention floor of
    0.72 rendered as 0.7 percent, because every other percentage in this product
    is stored already scaled and formatPercent does not scale."""
    detail = _read("HistoryDetail.jsx")
    assert "if (unit === 'fraction') return formatPercent(Number(value) * 100, locale);" in detail
    fmt = (SRC / "shell" / "format.jsx").read_text(encoding="utf-8")
    assert "return `${formatNumber(value, locale)}%`;" in fmt, (
        "the shared formatter still does not scale, which is what makes the line above necessary")


def test_no_engine_key_and_no_internal_token_is_rendered_as_a_name() -> None:
    """Two measured leaks: a break act arrives as
    /api/breaks/2024-11-01|<channel>|000~1/placement, and the recorder writes
    auth-disabled where a person's name goes."""
    labels = _read("history-labels.js")
    assert "export function pathStem" in labels
    assert "export const ACTOR_LABELS" in labels
    assert "'auth-disabled': ['No sign-in', 'ללא כניסה']" in labels
    row = _read("HistoryRow.jsx")
    assert "pathStem(facts.path)" in row
    assert "actorLabel(entry.actor, locale)" in row
    assert "{entry.actor}" not in row, "the raw token never reaches a row"


def test_the_provenance_of_every_source_is_printed() -> None:
    page = _read("HistoryPage.jsx")
    assert "hist-provenance" in page
    assert "<HistoryRunsSource" in page
    runs = _read("history-runs.js")
    assert "the run log could not be read" in runs, "an unreadable record says so by name"
    assert "יומן ההרצות" in runs


def test_the_runs_source_is_tri_state_and_the_withheld_state_names_its_remedy() -> None:
    """Real, unavailable, unknown. The third state here is the boundary one: with
    no operator channel the product cannot tell which runs are the operator's,
    so it says so and offers the control that fixes it."""
    runs = _read("history-runs.js")
    for state in ("'available'", "'unreadable'", "'withheld_no_operator_channel'"):
        assert state in runs
    assert "Set the operator channel" in runs and "הגדרת ערוץ המפעיל" in runs
    assert 'href="#Settings"' in _read("HistoryRunsSource.jsx"), "the remedy is a door, not a sentence"
    assert "runsSourceState(sources)" in _read("HistoryPage.jsx")


def test_the_three_places_a_run_count_is_read_all_ask_the_same_source() -> None:
    """The strip, the tab and the empty list under it. The payload has carried
    the state all along; what failed was that two of the three read the tally
    instead of the state, and the tally is zero for a reason that is not zero."""
    since = _read("HistorySince.jsx")
    assert "runsSourceState(body && body.examined)" in since, (
        "the attestation reads the sources it examined, which travel with the verdict")
    lines = since.splitlines()
    printed = [number for number, line in enumerate(lines) if "counts.run" in line]
    assert len(printed) == 1, "the run tally is printed in exactly one branch"
    assert "{changeCount && counted" in lines[printed[0] - 1], (
        "and it is the branch where the product may count the runs")
    assert "runsCountLine(runsState)" in since, "the other branches say why instead"
    assert "<RunsRemedyLink" in since, "and the strip offers the same door the footer does"

    page = _read("HistoryPage.jsx")
    assert "const counted = !body || runsCounted(runsState);" in page, (
        "a record that has not arrived is not a record that failed")
    assert "{name === 'run' && runsHint ? (" in page, "the run tab prints unknown rather than zero"
    assert "aria-label={runsHint}" in page, "and the reason is on the control itself"
    assert "const runsBlocked = kind === 'run' && !counted;" in page
    assert page.count("<HistoryRunsSource") == 2, (
        "the footer and the empty list say one sentence, from one place")
    # The sentence under an empty list is chosen in one of three branches and the
    # withheld source answers first. The assertion is on the guard rather than on
    # one branch's words, because the branch it used to name is the one that sent
    # a reader looking for a colleague away with "nothing matches those filters".
    empty = page.split("{state === 'ready' && !shown ? (")[1].split("{state === 'ready' && shown ? (")[0]
    assert empty.count("{runsBlocked ? (") == 1, "the withheld run log answers in its own sentence"
    assert empty.count("{!runsBlocked && ") == 2, (
        "and every other branch is guarded on it, so no sentence blames the filters while the "
        "source is the cause")
    assert "{!runsBlocked && windowed ? (" in empty and "{!runsBlocked && !windowed ? (" in empty, (
        "a day window and a page without one each decide in their own module")


# The laws a surface can be read for live in tests/test_p8_history_laws.py,
# which this file was split into when it reached 491 lines against the cap.
