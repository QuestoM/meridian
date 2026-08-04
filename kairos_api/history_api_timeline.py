"""The merge behind History: one attributed timeline over four real records.

Discovery found three separate answers to "what happened" and none of them
covered the plan: the version timeline, the settings activity log and an
in-memory bell feed. A fourth record, ``output/run_log.jsonl``, has held every
run's provenance since the engine was built and no endpoint ever served it.

This module merges them into one ordered list of entries and nothing else. It
computes no number of its own: every figure an entry carries was written by the
engine, the version store or the request recorder, and a field that cannot be
resolved is absent rather than guessed. How far back each of those records still
reaches, and what prunes the rest, is decided beside this in
:mod:`kairos_api.history_api_reach`.

Three rules are enforced here rather than at the surface, because a surface
that has to remember them will one day forget.

- **The artifact root decides visibility.** An act whose output lands under
  ``models/`` is training (specification section 4.1), and a channel-affiliated
  account never sees one. The filter is applied on the read, so there is no
  rendered trace that the other side exists.
- **The competitor boundary applies to runs.** Measured on this deployment's
  own log: 523 records, of which 444 belong to three channels the operator does
  not own. They are dropped by :mod:`kairos_api.channel_scope` before anything
  is serialised, and the disclosure note travels with the payload.
- **Entries carry no prose.** Every word a person reads is chosen by the
  surface from the two-language vocabulary, so the payload passes the lexicon
  test of specification section 4.2 by construction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from kairos_api import channel_scope, version_store
from kairos_api.history_api_actions import (  # noqa: F401 - the classifier's public surface
    ARTIFACT_ROOTS,
    KINDS,
    OUTCOMES,
    PREVIEW_ACTIONS,
    action_for,
    artifact_root,
    kind_for,
    outcome_for,
)

# The sign-in events the recorder writes beside the requests. They are the only
# entries in that file that are not a request, so they are named here and every
# other event name is left alone rather than guessed at.
_SIGN_IN_EVENTS = frozenset({"login", "login_failed", "logout"})


def _entry(
    entry_id: str,
    kind: str,
    ts: str,
    actor: str,
    via: str,
    root: str,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """One timeline entry. Every entry has the same seven fields and its own
    facts, so a surface renders one row shape rather than five."""
    return {
        "id": entry_id,
        "kind": kind,
        "ts": str(ts or ""),
        "actor": str(actor or ""),
        "via": via,
        "artifact_root": root,
        "facts": facts,
    }


def restore_block(manifest: dict[str, Any], live: Optional[dict[str, str]] = None) -> Optional[str]:
    """Why this version cannot be restored into this deployment, or None.

    A manifest records the absolute path each file was captured from. Restoring
    copies those bytes onto whatever path the same logical name resolves to
    now, so a version captured against a different store would write foreign
    content into the operator's live files. Measured on this deployment: 196 of
    the 200 manifests on disk were captured by pytest against temporary paths,
    and every one of them offers a restore button today.

    ``foreign_store`` means the paths do not match. ``missing_snapshot`` means
    the manifest says the file existed and its bytes are gone. None means the
    version is restorable, which is the only state that offers the control.
    """
    paths = live_paths() if live is None else live
    covered = [
        item for item in manifest.get("files", [])
        if item.get("logical") in version_store._LOGICAL_ORDER
    ]
    if not covered:
        return "no_files"
    version_id = str(manifest.get("version_id") or "")
    for item in covered:
        logical = str(item.get("logical"))
        if str(item.get("path") or "") != paths.get(logical, ""):
            return "foreign_store"
        if item.get("existed") and version_id:
            snapshot = version_store._versions_root() / version_id / str(item.get("name"))
            if not snapshot.exists():
                return "missing_snapshot"
    return None


def live_paths() -> dict[str, str]:
    """The path each logical file resolves to in this process, resolved once."""
    resolved: dict[str, str] = {}
    for logical in version_store._LOGICAL_ORDER:
        try:
            resolved[logical] = str(version_store._logical_path(logical))
        except Exception:  # pragma: no cover - a store that cannot resolve is unknown
            continue
    return resolved


def version_entries(manifests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore points, one entry per recorded version."""
    paths = live_paths()
    entries: list[dict[str, Any]] = []
    for manifest in manifests:
        version_id = str(manifest.get("version_id") or "")
        if not version_id:
            continue
        source = str(manifest.get("source") or "")
        block = restore_block(manifest, paths)
        entries.append(_entry(
            f"version:{version_id}",
            "restore_point",
            str(manifest.get("created_at") or ""),
            str(manifest.get("actor") or ""),
            "assistant" if source == "assistant_apply" else "dashboard",
            "data",
            {
                "version_id": version_id,
                "source": source,
                "label": manifest.get("label"),
                "batch_id": manifest.get("batch_id"),
                "files": [f.get("logical") for f in manifest.get("files", [])],
                "restorable": block is None,
                "restore_block": block,
            },
        ))
    return entries


def restore_entries() -> list[dict[str, Any]]:
    """Restores that actually happened, from the version store's own audit log.

    This is the other half of "how to put it back": the safety version id on
    each row is the point that undoes the restore, so a reversal is itself an
    addressable object rather than a claim in a sentence.
    """
    path = version_store._versions_root() / "audit.jsonl"
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("event") != "restore":
            continue
        version_id = str(record.get("version_id") or "")
        entries.append(_entry(
            f"restore:{version_id}:{record.get('ts')}",
            "restore",
            str(record.get("ts") or ""),
            str(record.get("actor") or ""),
            "dashboard",
            "data",
            {
                "version_id": version_id,
                "restored": record.get("restored") or [],
                "safety_version_id": record.get("safety_version_id"),
            },
        ))
    return entries


def activity_entries(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Changes and account events, from the append-only request recorder.

    The recorder stores metadata only and this reads it as stored: no body, no
    query string, no header ever entered that file and none can leave here.
    """
    entries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        event = str(record.get("event") or "")
        ts = str(record.get("ts") or "")
        actor = str(record.get("user") or "")
        if event in _SIGN_IN_EVENTS:
            entries.append(_entry(
                f"account:{ts}:{index}",
                "sign_in",
                ts,
                actor,
                "dashboard",
                "data",
                {"event": event, "role": record.get("role") or ""},
            ))
            continue
        if event != "request":
            continue
        path = record.get("path")
        action = action_for(record.get("method"), path)
        entries.append(_entry(
            f"change:{ts}:{index}",
            kind_for(action),
            ts,
            actor,
            "assistant" if record.get("via") == "assistant" else "dashboard",
            artifact_root(path),
            {
                "action": action,
                "method": record.get("method"),
                "path": path,
                "status": record.get("status"),
                # What that status means, decided once here rather than by every
                # surface that reads a number. Without it each reader has to know
                # that a 403 on a save is not a save, and none of them did.
                "outcome": outcome_for(record.get("status")),
                "duration_ms": record.get("duration_ms"),
                "role": record.get("role") or "",
            },
        ))
    return entries


_RUN_SUMMARY_FIELDS = (
    "projected_revenue",
    "total_breaks",
    "total_ad_seconds",
    "average_retention",
    "objective",
    "compliant",
)


def run_facts(record: dict[str, Any]) -> dict[str, Any]:
    """The run's own recorded outcome, plus the scope every figure is on.

    The scope travels with the figures because a revenue number without the
    channel, the day and the engine version that produced it is not an answer.
    """
    summary = record.get("summary") or {}
    facts: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "channel": record.get("channel"),
        "day": record.get("day"),
        "engine_version": record.get("engine_version"),
        "segment_count": record.get("segment_count"),
    }
    for field in _RUN_SUMMARY_FIELDS:
        if field in summary:
            facts[field] = summary[field]
    return facts


def run_entries(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plan runs, newest last in file order. Already channel-scoped by the caller."""
    return [
        _entry(
            f"run:{record.get('run_id')}",
            "run",
            str(record.get("created_at") or ""),
            "engine",
            "engine",
            "output",
            run_facts(record),
        )
        for record in records
        if record.get("run_id")
    ]


def scoped_runs(records: list[dict[str, Any]], settings: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The operator's own runs, and the disclosure of what the scope removed."""
    return channel_scope.scope_records(records, key="channel", settings=settings)


def visible(entries: Iterable[dict[str, Any]], is_company: bool) -> list[dict[str, Any]]:
    """Drop training entries for an account that is not on the company side.

    A filter on the read, never a hidden section: a channel account's timeline
    contains no row, no count and no marker that a models entry was removed.
    """
    if is_company:
        return list(entries)
    return [entry for entry in entries if entry.get("artifact_root") != "models"]


def order(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Newest first, with a stable tie-break on the entry id."""
    return sorted(entries, key=lambda item: (str(item.get("ts") or ""), str(item.get("id"))), reverse=True)


def counts(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    """How many entries of each kind, so a filter control can print its own count."""
    tally = {kind: 0 for kind in KINDS}
    for entry in entries:
        kind = str(entry.get("kind"))
        if kind in tally:
            tally[kind] += 1
    return tally


def outcome_counts(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    """How many of the recorded changes landed, and how many were refused.

    Taken over the ``change`` kind alone, which is exactly the set the Change
    filter shows and the set the attestation counts. A refused write stays on
    the timeline because somebody attempted it and that is worth reading; this
    tally is what keeps the figure beside the filter from being read as a count
    of things that changed.
    """
    tally = {outcome: 0 for outcome in OUTCOMES}
    for entry in entries:
        if entry.get("kind") != "change":
            continue
        outcome = str((entry.get("facts") or {}).get("outcome") or "")
        if outcome in tally:
            tally[outcome] += 1
    return tally


# A broadcast day is an Israeli day. Every record in this product stamps UTC, so
# the day an entry is filed under is its timestamp read in this zone, and the
# surface reads it in the same one. Without this a change made at half past one
# in the morning in Tel Aviv files itself under the previous day.
BROADCAST_ZONE = ZoneInfo("Asia/Jerusalem")


def broadcast_day(ts: Any) -> str:
    """The calendar day, in the broadcast zone, that a timestamp falls on.

    An unparseable stamp falls back to its own first ten characters rather than
    to today, because a record with a broken timestamp must not silently become
    a record of now.
    """
    text = str(ts or "")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text[:10]
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(BROADCAST_ZONE).date().isoformat()


def since_day(entries: Iterable[dict[str, Any]], day: str) -> list[dict[str, Any]]:
    """Entries recorded on or after ``day``, an ISO calendar date in the
    broadcast zone."""
    marker = str(day or "")
    return [entry for entry in entries if broadcast_day(entry.get("ts")) >= marker]


def until_day(entries: Iterable[dict[str, Any]], day: str) -> list[dict[str, Any]]:
    """Entries recorded on or before ``day``, in the broadcast zone.

    The other half of ``since_day``, and the half that reaches backwards. A page
    is always the newest of whatever matched, so without this the record before
    the current page is unreachable: measured on the running instance, a change
    filter matched 2,027 entries and served the newest 500, which spanned one
    afternoon, and no control could ask for the day before.
    """
    marker = str(day or "")
    return [entry for entry in entries if broadcast_day(entry.get("ts")) <= marker]


# The cursor a page hands back so the next one can start exactly where it ended.
# A calendar day cannot do this on its own: a single busy day can hold more
# entries than a page, and a day-granular step would then serve the same rows
# again forever. The cursor is the sort key itself, so it always advances.
CURSOR_SEPARATOR = "|"


def sort_key(entry: dict[str, Any]) -> tuple[str, str]:
    """The key ``order`` sorts on, which is the order a page is cut from."""
    return (str(entry.get("ts") or ""), str(entry.get("id") or ""))


def cursor_of(entry: dict[str, Any]) -> str:
    """The cursor that resumes immediately after this entry."""
    stamp, entry_id = sort_key(entry)
    return f"{stamp}{CURSOR_SEPARATOR}{entry_id}"


def older_than(entries: Iterable[dict[str, Any]], cursor: str) -> list[dict[str, Any]]:
    """The entries strictly older than a cursor, in the list's own total order."""
    stamp, _, entry_id = str(cursor or "").partition(CURSOR_SEPARATOR)
    marker = (stamp, entry_id)
    return [entry for entry in entries if sort_key(entry) < marker]

