"""P8 History: the provenance footer says whose figure each figure is.

The defect this file pins, measured by a blind critic on the running instance and
reproduced here before the fix. Signed in as an operator, the footer read "The
request recorder holds 1 line, which become changes, previews and sign-ins"
directly beside "The record starts on 2026-07-26; the request recorder keeps the
newest 5,000 lines and the version store keeps the newest 200 restore points".
As a viewer it read 4. As the admin it read 5,010. The store on disk held 5,005
lines at that moment.

A store that keeps 5,000 and holds 1 is a self-contradiction printed in one
paragraph, and the figure that was wrong was the one whose whole job is to say
how large the evidence is. The cause was one line: the sources block took
``records`` from the caller's scoped slice while the ``starts`` beside it was
read from the store itself, which is the doctrine this destination had already
written down and applied to one of the two figures.

Reproduced against this repository's own recorder on 2026-08-01 over a copy of
the store, at 5,261 lines: operator 1, viewer 4, admin 5,261.

The second half is the same error class one field over. The rename control this
destination itself ships is ``PATCH /api/versions/{id}``, and it fell through to
the version family's catch-all, so renaming a point at 23:24 put "Restore
applied | PATCH /api/versions/a3bd7ff7f743 | 200" on the Change tab. A
compliance owner reading that tab counted a restore that never happened.

Two rules are enforced here and nothing else is:

- how large a record is, and how far back it goes, are facts about the store and
  never about the reader, so no account may be told a smaller store than exists,
- and an act is recorded as the act it was, so naming a restore point and
  putting one back are never the same word.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.version_store as vs
from kairos_api import activity_log

from test_p8_history import (  # noqa: F401 - fixtures are used by name
    _as,
    auth_env,
    history_env,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "tv-break-dashboard" / "src" / "history"

# Over the retention floor by five lines, which is the state the defect was
# measured in and the only state in which the contradiction is visible: a store
# below its own floor has never pruned and every figure agrees.
OVER_THE_FLOOR = activity_log.MAX_KEPT_ENTRIES + 5


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


def _line(user: str, role: str, index: int) -> str:
    """One recorder line, shaped exactly as activity_log._entry writes it."""
    return json.dumps({
        "ts": f"2026-07-30T{8 + index % 12:02d}:00:00.000+00:00",
        "user": user,
        "role": role,
        "event": "request",
        "method": "POST",
        "path": "/api/break-decisions",
        "status": 200,
        "duration_ms": 4.0,
        "via": "dashboard",
    }, ensure_ascii=False)


def _fill_recorder() -> int:
    """A store larger than its own retention floor, with a named slice per account.

    Returns the number of lines on disk, counted by this test rather than by the
    module under test, because the whole question is whether the module reports
    the store's own size.
    """
    rows = [_line("someone-else", "operator", index) for index in range(OVER_THE_FLOOR - 3)]
    rows += [_line("operator1", "operator", 1), _line("operator1", "operator", 2)]
    rows += [_line("viewer1", "viewer", 3)]
    path = activity_log.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    activity_log.reset_runtime_state()
    return sum(1 for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip())


# --- how large the record is, and whose figure that is ---------------------------

def test_a_non_admin_read_reports_the_stores_own_size_and_never_its_own_slice(
        history_env, auth_env) -> None:
    """The named gap, closed and pinned.

    Three accounts read the same store and are told the same size, because how
    much evidence exists does not change with who is looking. What does change is
    how much of it each may read, which now travels under its own key with the
    rule that produced it, so the surface can print both and neither can be
    mistaken for the other.
    """
    on_disk = _fill_recorder()
    assert on_disk == OVER_THE_FLOOR == 5005

    expected = {"admin": ("all", on_disk), "operator1": ("self", 2), "viewer1": ("self", 1)}
    reported = {}
    for username, role in (("admin", "admin"), ("operator1", "operator"), ("viewer1", "viewer")):
        client = _as(history_env, auth_env, username, role)
        changes = client.get("/api/history", params={"limit": 1}).json()["sources"]["changes"]
        scope, slice_size = expected[username]

        assert changes["records"] == on_disk, (
            f"{username} was told the recorder holds {changes['records']} of {on_disk} lines")
        assert changes["in_scope"] == slice_size, f"{username} reads its own slice"
        assert changes["scope"] == scope, "and the rule that produced the slice rides with it"
        reported[username] = changes["records"]

        # The invariant the two adjacent sentences broke: the figure printed as
        # the store's size may never be smaller than what the retention clause
        # beside it claims the store keeps, once the store is over that floor.
        keeps = changes["retention"]["keeps"]
        assert changes["records"] >= min(on_disk, keeps), (
            f"{username} reads a store of {changes['records']} lines beside a clause keeping {keeps}")
        assert changes["records"] >= keeps == activity_log.MAX_KEPT_ENTRIES

    assert len(set(reported.values())) == 1, "every account is told the same store size"
    assert expected["operator1"][1] < on_disk, (
        "and the slice really is narrower, so the two keys are not one figure twice")


def test_the_store_size_is_the_stores_and_the_slice_is_the_readers(history_env, auth_env) -> None:
    """The unit under the route. ``entry_count`` reads the file and takes no
    caller, which is what makes it unable to answer differently to two readers."""
    assert activity_log.entry_count() == 0, "an empty store holds nothing, and says so"
    on_disk = _fill_recorder()
    assert activity_log.entry_count() == on_disk == len(activity_log._read_entries())

    # A line that does not parse still occupies one of the kept lines, so it is
    # counted as held and is not counted as read. A blank line is neither.
    path = activity_log.log_path()
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n\n", encoding="utf-8")
    activity_log.reset_runtime_state()
    assert activity_log.entry_count() == on_disk + 1
    assert len(activity_log._read_entries()) == on_disk

    admin = _as(history_env, auth_env, "admin", "admin")
    changes = admin.get("/api/history", params={"limit": 1}).json()["sources"]["changes"]
    assert changes["records"] == on_disk + 1 and changes["in_scope"] == on_disk, (
        "so even for an account that may read everything, the two figures are told apart")


def test_the_attestation_reads_the_same_two_figures(history_env, auth_env) -> None:
    """The compliance half reads the same block under ``examined``, so a count
    attested since a day cannot be weighed against a different store size."""
    on_disk = _fill_recorder()
    operator = _as(history_env, auth_env, "operator1", "operator")
    body = operator.get("/api/history/since", params={"day": "2026-07-01"}).json()
    assert body["examined"]["changes"]["records"] == on_disk
    assert body["examined"]["changes"]["in_scope"] == 2
    assert body["examined"]["changes"]["scope"] == "self" == body["scope"]


def test_the_other_two_sources_are_unchanged_by_this(history_env, auth_env) -> None:
    """Only the request recorder is scoped per account, so only it carries the
    pair. Restore points are the shared operating record, and the runs figure is
    the operator's own channel by the competitor boundary, which is stated in
    words by ``run_scope`` rather than by a count of what was excluded."""
    vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    operator = _as(history_env, auth_env, "operator1", "operator")
    sources = operator.get("/api/history", params={"limit": 1}).json()["sources"]
    assert set(sources["restore_points"]) == {"records", "available", "starts", "retention"}
    assert set(sources["runs"]) == {"records", "available", "state", "starts", "retention"}
    assert sources["restore_points"]["records"] == 1
    assert sources["runs"]["records"] == 2, "the operator's own two runs, and no rival's"


# --- an act is recorded as the act it was ----------------------------------------

def test_a_rename_is_recorded_as_a_rename_and_never_as_a_restore(history_env, auth_env) -> None:
    """Measured live before the fix: renaming a restore point put "Restore
    applied | PATCH /api/versions/... | 200" on the Change tab, because the
    rename fell through to the version family's catch-all. Nothing was put back."""
    admin = _as(history_env, auth_env, "admin", "admin")
    version_id = vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    renamed = admin.patch(f"/api/versions/{version_id}", json={"label": "before the audit"})
    assert renamed.status_code == 200 and renamed.json()["label"] == "before the audit"

    rows = admin.get("/api/history", params={"kind": "change", "limit": 500}).json()["entries"]
    acts = {(row["facts"].get("method"), row["facts"].get("action")) for row in rows
            if str(row["facts"].get("path") or "").startswith("/api/versions")}
    assert ("PATCH", "restore_point_renamed") in acts, f"the rename is recorded as itself: {acts}"
    assert ("PATCH", "restore") not in acts, "and never as an act that put something back"

    # The other two acts in the same family are untouched, which is the half that
    # was right before this row was added above them.
    assert admin.post(f"/api/versions/{version_id}/restore", json={}).status_code == 200
    assert admin.post("/api/versions/snapshot", json={"label": "after the audit"}).status_code == 200
    after = admin.get("/api/history", params={"kind": "change", "limit": 500}).json()["entries"]
    later = {(row["facts"].get("method"), row["facts"].get("action")) for row in after
             if str(row["facts"].get("path") or "").startswith("/api/versions")}
    assert ("POST", "restore") in later, "putting a version back is still a restore"
    assert ("POST", "restore_point_saved") in later, "and taking a point by hand is still its own act"


def test_the_rename_carries_its_own_word_in_both_languages(history_env, auth_env) -> None:
    labels = _read("history-labels.js")
    assert "restore_point_renamed: ['Restore point renamed', 'שם נקודת שחזור שונה']," in labels
    assert "restore_point_renamed: 'Versions'," in labels, "and the row is not a dead end"
    assert "restore: ['Restore applied', 'שחזור בוצע']," in labels, (
        "the restore keeps its own words, which were never the problem")


# --- the sentence, executed rather than read -------------------------------------

# The payloads measured on this repository's own recorder on 2026-08-01, over a
# copy of the store holding 5,261 lines, read by three live sessions.
SOURCE_PROBE = """
import { changesSourceLine, recordStartLine } from './tv-break-dashboard/src/history/history-reach.js';
const retention = {pruned: true, keeps: 5000, prune_at: 6000, unit: 'lines'};
const changes = (in_scope, scope) => ({records: 5261, in_scope, scope, starts: '2026-08-01', retention});
const record = (source) => ({record_starts: '2026-07-26', sources: {changes: source,
  restore_points: {records: 200, starts: '2026-08-01',
    retention: {pruned: true, keeps: 200, prune_at: 200, unit: 'restore_points'}}}});
const operator = changes(1, 'self');
console.log(JSON.stringify({
  operator: changesSourceLine(operator),
  viewer: changesSourceLine(changes(4, 'self')),
  admin: changesSourceLine(changes(5261, 'all')),
  torn: changesSourceLine(changes(5260, 'all')),
  unread: changesSourceLine(undefined),
  older: changesSourceLine({records: 5261}),
  paragraph: [changesSourceLine(operator)[0], recordStartLine(record(operator))[0]],
  paragraphHe: [changesSourceLine(operator)[1], recordStartLine(record(operator))[1]],
}));
"""


def _digits(text: str) -> list[str]:
    return re.findall(r"\d+", text.replace(",", ""))


def test_the_footer_says_which_figure_is_the_stores_and_which_is_the_readers() -> None:
    """Executed rather than grepped, because the defect was two true figures
    printed as one, and the only way to catch that is to read the sentence.

    The paragraph assertion is the defect itself: the two sentences that
    contradicted each other are built here from one payload and compared, so a
    store size smaller than the retention floor beside it fails rather than
    ships."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    result = subprocess.run([node, "--input-type=module", "-e", SOURCE_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    measured = json.loads(result.stdout.strip().splitlines()[-1])

    assert measured["operator"][0] == (
        "The request recorder holds 5,261 lines, of which 1 is yours; each line is a change, a preview or a sign-in.")
    assert measured["viewer"][0].startswith("The request recorder holds 5,261 lines, of which 4 are yours;")
    assert measured["operator"][1].startswith("רישום הבקשות מחזיק 5,261 שורות, ומתוכן 1 שלכם;")
    assert measured["viewer"][1].startswith("רישום הבקשות מחזיק 5,261 שורות, ומתוכן 4 שלכם;")

    # An account that may read all of it reads the sentence it read before, which
    # is the one state where a single figure was never a lie.
    assert measured["admin"][0] == (
        "The request recorder holds 5,261 lines, which become changes, previews and sign-ins.")
    assert measured["unread"][0] == measured["older"][0].replace("5,261", "0"), (
        "a source that says nothing is counted as nothing, and a payload without the new key "
        "reads exactly as it read before")

    # Some lines could not be parsed, which is the only other way the two figures
    # part company, and it is a difference between the store and the page too.
    assert measured["torn"][0] == (
        "The request recorder holds 5,261 lines, of which this read could use 5,260; each line is a change, a preview or a sign-in.")

    # Every figure in every sentence is the payload's own, in both languages.
    for name in ("operator", "viewer", "admin", "torn"):
        for language in (0, 1):
            assert set(_digits(measured[name][language])) <= {"5261", "1", "4", "5260"}, name

    # And the paragraph the critic read: the store's own size, printed beside the
    # clause that says what the store keeps, in both languages.
    for paragraph in (measured["paragraph"], measured["paragraphHe"]):
        held = int(_digits(paragraph[0])[0])
        keeps = int(_digits(paragraph[1])[3])
        assert held == 5261 and keeps == 5000
        assert held >= keeps, f"the footer says the recorder holds {held} and keeps {keeps}"


def test_the_surface_still_hands_that_sentence_the_whole_source() -> None:
    """One argument, so a surface cannot pass half the pair and get the old
    sentence back by accident."""
    page = _read("HistoryPage.jsx")
    assert "{pageText(locale, ...changesSourceLine(sources.changes))}" in page
    reach = _read("history-reach.js")
    assert "export function changesSourceLine(changes) {" in reach
    assert "const mine = Number(source.in_scope || 0);" in reach
