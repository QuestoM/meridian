"""P8 History: no sentence built for a self-scoped reader may name an exclusion
set narrower than the set the same store proves is filtered.

The class, rather than one member of it. Four rounds in a row fixed one sentence
and the next round found the next one, because the invariant each round pinned
named one kind: round 15's test forbade denying `restore` and nothing else, so it
stayed green while the sentence beside it was wrong about `preview` and `sign_in`.
Measured by a blind critic on 2026-08-07 and reproduced here on a real store: a
self-scoped reader was withheld 2,533 changes, 636 previews and 2,586 sign-ins,
and not one restore, restore point or run, while the strip printed "Only changes
made by other accounts are withheld here". A compliance owner told that only
changes are withheld reads the sign-in count beside it as the sign-in record and
attests to it.

So what is pinned here is measured, not listed. The store is seeded so that every
kind the payload names carries an entry made by an account other than the reader,
the same store is read twice in the same test, once as an admin and once as a
self-scoped operator, and the difference between those two payloads is what the
sentences are held to: every kind that difference proves is filtered must be named
by whichever clause denies something, and no kind it proves is not filtered may be
named there. Nothing in this file enumerates the kinds itself.

`assert_the_denial_names_every_filtered_kind` is the assertion in one function so
it can be run against an earlier revision's sentences from a scratch copy, which is
how this test was shown to fail on the behaviour it was written to catch.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.activity_log as activity_log
import kairos_api.version_store as vs

from test_p8_history import (  # noqa: F401 - fixtures are used by name
    _as,
    auth_env,
    history_env,
)

ROOT = Path(__file__).resolve().parents[1]

# The plural word this destination uses for each kind, in English and in both the
# plain and the definite Hebrew. Written out here rather than imported from the
# module under test, so renaming a word in the module fails this file instead of
# moving both sides of the comparison at once.
KIND_WORDS = {
    "change": ["changes", "שינויים", "השינויים"],
    "preview": ["previews", "תצוגות מקדימות", "התצוגות המקדימות"],
    "run": ["runs", "הרצות", "ההרצות"],
    "restore_point": ["restore points", "נקודות שחזור", "נקודות השחזור"],
    "restore": ["restores", "שחזורים", "השחזורים"],
    "sign_in": ["sign-ins", "כניסות", "הכניסות"],
}

# What a clause that keeps something from the reader can say, in either language.
DENIALS = ("withheld", "not shown", "excluded", "אינם מוצגים", "אינן מוצגות",
           "לא מוצגים", "מוסתר", "אינו מוסתר")

# Every sentence this destination builds for a self-scoped reader, from the two
# payloads the same store answered with, so one probe covers the strip and the
# page footer and neither is read from a hardcoded figure.
SENTENCE_PROBE = """
import fs from 'fs';
import { sinceCountLine, sinceEmptyLine } from './tv-break-dashboard/src/history/history-since.js';
import { coveredSets, pageCoveredLine } from './tv-break-dashboard/src/history/history-scope.js';
const paid = JSON.parse(fs.readFileSync(process.argv[1], 'utf-8'));
const since = paid.since;
const timeline = paid.timeline;
console.log(JSON.stringify({
  sentences: {
    empty: sinceEmptyLine('self', since),
    count: sinceCountLine(since.changed || 1, since.counts.run, 'self', since),
    footer: pageCoveredLine(timeline),
  },
  sets: {
    attestation: coveredSets(since.attested_kinds, since.scope_kinds),
    page: coveredSets(timeline.kinds, (timeline.attestation || {}).scope_kinds),
  },
}));
"""


def _clauses(sentence: str) -> list[str]:
    """One rendered string as the separate claims a reader reads in it."""
    return [part.strip() for part in sentence.split(". ") if part.strip()]


def _names(clause: str, kind: str) -> bool:
    """Whether this clause names that kind, in either language and either form.

    Folded to lower case because a kind word can open a sentence, and a sentence
    that names a kind names it whether or not it is the first word."""
    return any(word in clause.lower() for word in KIND_WORDS[kind])


def assert_the_denial_names_every_filtered_kind(label: str, sentence: str,
                                                filtered: set[str], shown: set[str]) -> None:
    """The invariant, in one place so an earlier revision can be held to it too.

    ``filtered`` is what the store itself proves this reader is not shown, taken as
    the difference between two live reads of one store. ``shown`` is what the same
    two reads prove reaches the reader whole. A clause that denies something must
    name all of the first and none of the second; a sentence that denies nothing is
    not examined, because naming no exclusion is not the same as naming a short one.
    """
    for clause in _clauses(sentence):
        if not any(phrase in clause for phrase in DENIALS):
            continue
        missing = sorted(kind for kind in filtered if not _names(clause, kind))
        assert not missing, (
            f"{label} denies less than this store proves is filtered. The clause "
            f"{clause!r} names no {missing}, and the same store answered a self-scoped "
            f"reader with fewer of each of them than it answered an admin with.")
        overclaimed = sorted(kind for kind in shown if _names(clause, kind))
        assert not overclaimed, (
            f"{label} denies {overclaimed}, which this store proves reach a self-scoped "
            f"reader whole: the clause is {clause!r}")


def _seed_every_kind(client, auth_store) -> None:
    """One store holding an entry of every kind, all by an account other than the
    reader the sentences are built for.

    The restore writes the restore and the safety restore point and is recorded as
    a change; the scenario request is a preview whatever it answers, because the act
    is derived from the method and the path and this app mounts no scenario route;
    the auth event is the recorder's own sign-in line. The run log is the fixture's.
    """
    admin = _as(client, auth_store, "admin", "admin")
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    assert admin.post(f"/api/versions/{version_id}/restore", json={}).status_code == 200
    admin.post("/api/scenario", json={})
    activity_log.record_auth_event("login", "admin", "admin")


def _read_both_ways(client, auth_store) -> dict:
    """The same store read twice in the same second: once by an account that may
    read all of it, once by an account that may read only its own slice."""
    admin = _as(client, auth_store, "admin", "admin")
    admin_since = admin.get("/api/history/since").json()
    admin_timeline = admin.get("/api/history?limit=1").json()
    operator = _as(client, auth_store, "operator1", "operator")
    return {
        "admin_since": admin_since,
        "admin_timeline": admin_timeline,
        "since": operator.get("/api/history/since").json(),
        "timeline": operator.get("/api/history?limit=1").json(),
    }


def _run_probe(tmp_path: Path, read: dict) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine, so the module cannot be executed here")
    payload = tmp_path / "history-payloads.json"
    payload.write_text(json.dumps({"since": read["since"], "timeline": read["timeline"]}),
                       encoding="utf-8")
    result = subprocess.run([node, "--input-type=module", "-e", SENTENCE_PROBE, str(payload)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _difference(wider: dict, narrower: dict, kinds: list[str]) -> tuple[set[str], set[str]]:
    """What the account filter took, and what it left, per kind, from two reads."""
    filtered = {kind for kind in kinds if wider["counts"][kind] > narrower["counts"][kind]}
    shown = {kind for kind in kinds if wider["counts"][kind] == narrower["counts"][kind]}
    return filtered, shown


def test_no_self_scoped_sentence_names_an_exclusion_narrower_than_the_store_proves(
        history_env, auth_env, tmp_path) -> None:
    """The class defect, held to a live store rather than to a list of kinds.

    The strip's sentences are checked against the window they are printed over and
    the footer's against the whole record, because those are the two sets the two
    payloads carry, and both are measured by differencing the admin's read against
    the operator's read of the same store in the same test."""
    _seed_every_kind(history_env, auth_env)
    read = _read_both_ways(history_env, auth_env)
    kinds = read["timeline"]["kinds"]

    # Not a vacuous pass: the store must actually carry every kind the payload
    # names, and the account filter must actually be narrowing something.
    empty = sorted(kind for kind in kinds if not read["admin_timeline"]["counts"][kind])
    assert not empty, f"this store carries no {empty}, so it could prove nothing about them"
    assert read["since"]["scope"] == "self" and read["admin_since"]["scope"] == "all"

    window_filtered, window_shown = _difference(read["admin_since"], read["since"], kinds)
    record_filtered, record_shown = _difference(read["admin_timeline"], read["timeline"], kinds)
    assert window_filtered, "the operator's own window is narrowed, or this proves nothing"

    probe = _run_probe(tmp_path, read)
    for label in ("empty", "count"):
        for sentence in probe["sentences"][label]:
            assert_the_denial_names_every_filtered_kind(
                f"the strip's {label} sentence", sentence, window_filtered, window_shown)
    for sentence in probe["sentences"]["footer"]:
        assert_the_denial_names_every_filtered_kind(
            "the page footer", sentence, record_filtered, record_shown)

    # And the other half of the same class: a phrase saying a kind reaches this
    # reader whole may not name one the store proves is filtered.
    for name, sets in probe["sets"].items():
        overclaimed = sorted(set(sets["shared"]) & record_filtered)
        assert not overclaimed, (
            f"the {name} phrase says this reader sees every {overclaimed} on record, "
            f"and this store answered them fewer of each than it answered an admin")
        assert not set(sets["own"]) - record_filtered - record_shown


def test_the_empty_and_footer_sentences_never_deny_a_kind_the_payload_contains(
        history_env, auth_env, tmp_path) -> None:
    """The round-15 member of the class, kept as its own regression.

    A restore and its safety restore point carry no per-account scope: both are
    merged outside the filter that narrows the recorder's lines, so a self-scoped
    read contains every one of them by any account. The payload says so in
    `attested_kinds` and in counts that do not move with the reader, and no sentence
    built for that reader may deny showing them."""
    _seed_every_kind(history_env, auth_env)
    read = _read_both_ways(history_env, auth_env)
    since = read["since"]

    assert "restore" in since["attested_kinds"] and "restore_point" in since["attested_kinds"]
    assert since["counts"]["restore"] == read["admin_since"]["counts"]["restore"] >= 1
    assert since["counts"]["restore_point"] == read["admin_since"]["counts"]["restore_point"] >= 1

    probe = _run_probe(tmp_path, read)
    for label, pair in probe["sentences"].items():
        for sentence in pair:
            for clause in _clauses(sentence):
                if not any(phrase in clause for phrase in DENIALS):
                    continue
                for kind in ("restore", "restore_point"):
                    assert not _names(clause, kind), (
                        f"{label} denies showing {kind}, which this store's own payload "
                        f"proves a self-scoped reader is shown: {clause!r}")


def test_the_attestation_phrase_and_the_page_phrase_are_not_one_string(
        history_env, auth_env, tmp_path) -> None:
    """The round-15 fix's own defect: one phrase shared by two scopes.

    The attestation covers three kinds and the page renders six, so a single
    constant cannot be true in both places. Measured before this split: the footer
    read "each line is a change, a preview or a sign-in" two spans above "You see
    every restore point and restore on record, plus your own changes"."""
    _seed_every_kind(history_env, auth_env)
    read = _read_both_ways(history_env, auth_env)
    probe = _run_probe(tmp_path, read)

    attestation, page = probe["sets"]["attestation"], probe["sets"]["page"]
    assert set(attestation["own"]) | set(attestation["shared"]) == set(read["since"]["attested_kinds"])
    assert set(page["own"]) | set(page["shared"]) < set(read["timeline"]["kinds"]), (
        "the page phrase covers fewer kinds than the page renders, because the runs "
        "answer to the competitor boundary and are disclosed by run_scope instead")
    assert set(page["own"]) > set(attestation["own"]), (
        "the page names every kind the recorder scopes per account, the attestation "
        "only the one of them it attests over")

    footer = probe["sentences"]["footer"]
    for language, sentence in enumerate(probe["sentences"]["empty"]):
        assert footer[language] not in sentence and sentence not in footer[language]
