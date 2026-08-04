"""History: one attributed timeline, what a restore would change, and the restore.

Two routers live here and they answer two halves of one question.

``/api/versions`` is the operation-state version store's HTTP layer, moved
verbatim from ``version_store.py`` in the wave-zero router split. Its behaviour
is unchanged: the same role gates (a signed-in session to read, an operator or
admin role to snapshot, restore or rename), the same nine logical files, and
the same pre-restore safety point, so a restore is always itself undoable.

``/api/history`` is new. It merges the four records the product already keeps
and never showed together: the version timeline, the request recorder, the
version store's own restore audit, and ``output/run_log.jsonl``, which has held
every run's provenance since the engine was built without a single endpoint
reading it.

Three things this layer adds and the reason each one is honest rather than
cosmetic.

**A restore point says whether it can be restored.** A manifest records the
absolute path each file was captured from, and a restore copies those bytes
onto whatever the same logical name resolves to now. Measured on this
deployment: 196 of the 200 manifests were captured by pytest against temporary
paths, and every one of them offers a restore today, which would write foreign
bytes into the operator's live settings. They are now marked unrestorable, the
control is withheld, and the refusal is legible before the click.

**A payload says whether the reader may act on it.** ``can_edit`` and its
reason ride the read, so a viewer sees the timeline and the diffs with the
restore control rendered as state rather than failing after a click. The same
answer rides it per logical file: measured before that gate, a channel operator
refused ``PUT /api/rules/model-activation`` restored the settings file here with
a 200 and flipped that switch, moved a regulatory limit and wrote a rival
channel. :mod:`kairos_api.history_api_files` states that rule once.

**A timeline entry carries no prose and no rival.** Every word is chosen by the
surface from the two-language vocabulary, and runs are scoped to the operator's
own channel before serialisation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kairos_api import (
    activity_log,
    channel_scope,
    history_api_attestation,
    history_api_files,
    history_api_reach,
    history_api_runs,
    history_api_timeline,
    version_store,
)
from kairos_api.affiliation_wall import is_company

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/versions")

# The second router carries every new path and is mounted by its own stanza in
# the append-only region of server.py. The versions router above is already
# mounted through the version_store stanza, so mounting this module once more
# would define all five version paths twice.
timeline_router = APIRouter(prefix="/api/history", tags=["history"])

# Reading history and applying a restore are separately permissioned, which is
# the rule Figma's version history already follows: viewing is available to
# viewers, restoring needs edit access. Affiliation gates two of the nine files
# a restore writes rather than the surface itself, so the timeline and the diffs
# stay readable and the write answers to the wall the file's own routes use.
HISTORY_WALL = history_api_files.RESTORE_WALL

# Version ids are uuid4().hex[:12]; accept 8-32 lowercase hex so nothing else
# (a traversal path, a stray label) ever reaches the manifest reader.
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")

_RUN_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A cursor is a timestamp and an entry id joined by a pipe, which is the sort key
# the list is already ordered on. Nothing else reaches the comparison.
_CURSOR_RE = re.compile(r"^[0-9T:.+\-]{4,40}\|[0-9A-Za-z:.+\-]{0,100}$")

MAX_TIMELINE_ENTRIES = 500


def _require_day(value: str, name: str) -> str:
    """400 on anything that is not a day the calendar has.

    A shape regex is not a calendar, and a day is compared downstream as a string
    against a broadcast day, so an impossible one is answered rather than refused.
    Measured before this guard: ``until=2026-13-99`` answered 200 over the whole
    record, ``until=2026-02-31`` answered 200 over nothing, ``until=2026-07-32``
    served 74 entries, and ``since?day=2026-13-99`` answered 500 from a parse
    three modules later. Every day on these routes comes through here.
    """
    text = str(value or "")
    if not _DAY_RE.fullmatch(text):
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO calendar day, YYYY-MM-DD.")
    try:
        date.fromisoformat(text)
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"{name} must be a day the calendar has; {text} is not one.") from None
    return text


def _require_version_id(version_id: str) -> str:
    """404 on anything that is not a well-formed version id (hex, 8-32 chars)."""
    cleaned = str(version_id or "")
    if not _VERSION_ID_RE.fullmatch(cleaned):
        raise HTTPException(status_code=404, detail=f"no version {version_id!r}")
    return cleaned


# Auth seam: mirror the assistant action plane so roles behave identically.
def _require_session(request: Request, writer: bool = False) -> str:
    """401 without a signed-in session; 403 when ``writer`` and the role is read-only.
    With auth disabled every call is allowed and acts as 'auth-disabled'."""
    from kairos_api import auth
    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="A signed-in session is required.")
    if writer and session["role"] not in auth.WRITE_ROLES:
        raise HTTPException(status_code=403, detail=(
            "The operator or admin role is required to snapshot, restore or rename versions."))
    return str(session["username"])


def _require_writer(request: Request) -> str:
    return _require_session(request, writer=True)


_SCOPE_NOTE = ("Versions snapshot the operation-state files the operator edits: settings "
               "(with pricing overrides), placement constraints, manual overrides, "
               "advertiser rules, scoped advertiser conditions and calendar events. "
               "History is append-only; a restore first records the current state, "
               "so it is always undoable.")


def _public_entry(manifest: dict[str, Any], live: Optional[dict[str, str]] = None,
                  withheld: Iterable[str] = ()) -> dict[str, Any]:
    """One version as a surface reads it, with what this reader may put back."""
    block = history_api_timeline.restore_block(manifest, live)
    return history_api_files.public_entry(manifest, block, withheld)


@router.get("", tags=["versions"])
def list_versions(request: Request, limit: int = 50) -> dict[str, Any]:
    """Recorded versions, newest first."""
    _require_session(request)
    limit = max(1, min(int(limit), version_store.MAX_VERSIONS))
    live = history_api_timeline.live_paths()
    blocked = history_api_files.withheld(request)
    entries = [_public_entry(m, live, blocked) for m in version_store._all_manifests()[:limit]]
    body = {
        "entries": entries,
        "note": _SCOPE_NOTE,
        "restorable_count": sum(1 for entry in entries if entry["restorable"]),
        "file_permissions": history_api_files.permissions(request),
    }
    return HISTORY_WALL.stamp(body, request)


@router.get("/{version_id}/diff", tags=["versions"])
def version_diff(version_id: str, request: Request) -> dict[str, Any]:
    """Per logical file: what restoring this version would change from now."""
    _require_session(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    diff = {entry["logical"]: version_store._diff_logical(version_id, entry["logical"])
            for entry in manifest.get("files", []) if entry.get("logical") in version_store._LOGICAL_ORDER}
    # The exact per-file answer: this route knows which fields would move.
    settings_changed = (diff.get("settings") or {}).get("changed")
    body = {"version_id": version_id, "created_at": manifest.get("created_at"),
            "source": manifest.get("source"), "diff": diff,
            "file_permissions": history_api_files.permissions(request, settings_changed),
            "entry": _public_entry(manifest, None, history_api_files.withheld(request))}
    return HISTORY_WALL.stamp(body, request)


class RestoreRequest(BaseModel):
    files: Optional[list[str]] = None


@router.post("/{version_id}/restore", tags=["versions"])
def restore_version(version_id: str, request: Request,
                    body: RestoreRequest | None = None) -> dict[str, Any]:
    """Put the selected files back. Snapshots the current state first (undoable)."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    block = history_api_timeline.restore_block(manifest)
    if block == "foreign_store":
        raise HTTPException(status_code=409, detail=(
            "This version was recorded against a different store location, so restoring it "
            "would write files this deployment never produced. It stays readable as history."))
    if block == "missing_snapshot":
        raise HTTPException(status_code=409, detail=(
            "This version is missing the snapshot bytes for a file it covers, so it cannot "
            "be put back. It stays readable as history."))
    covered = [f["logical"] for f in manifest.get("files", []) if f.get("logical") in version_store._LOGICAL_ORDER]
    requested = body.files if body and body.files else covered
    selected = [name for name in version_store._LOGICAL_ORDER if name in set(requested) and name in covered]
    if not selected:
        raise HTTPException(status_code=400,
                            detail=f"no restorable files selected; this version covers {covered}")
    # Before the safety point, so a refused restore leaves no version, no audit
    # line and nothing put back. A restore writes files, and two of the nine
    # answer to a wall of their own on the way back exactly as on the way in.
    history_api_files.require_restore(version_id, selected, request)
    safety = version_store.snapshot(source="pre_restore", actor=actor, files=selected, force=True)
    restored = [version_store._restore_logical(version_id, logical) for logical in selected]
    version_store._audit("restore", actor, version_id=version_id, restored=restored, safety_version_id=safety)
    return {"restored": restored, "safety_version_id": safety}


class LabelRequest(BaseModel):
    label: Optional[str] = None


@router.post("/snapshot", tags=["versions"])
def create_snapshot(request: Request, body: LabelRequest | None = None) -> dict[str, Any]:
    """A named manual snapshot of the full operation state."""
    actor = _require_writer(request)
    label = body.label if body else None
    version_id = version_store.snapshot(source="manual_snapshot", actor=actor,
                                        files=list(version_store._LOGICAL_ORDER), label=label, force=True)
    version_store._audit("snapshot", actor, version_id=version_id, label=label)
    return _public_entry(version_store._read_manifest(str(version_id)), None,
                         history_api_files.withheld(request))


@router.patch("/{version_id}", tags=["versions"])
def rename_version(version_id: str, body: LabelRequest, request: Request) -> dict[str, Any]:
    """Rename (relabel) a version. Writer roles only."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    manifest["label"] = body.label
    version_store._atomic_write(version_store._manifest_path(version_id),
                                json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    version_store._audit("rename", actor, version_id=version_id, label=body.label)
    return _public_entry(manifest, None, history_api_files.withheld(request))


# ---------------------------------------------------------------------------
# The timeline: four records, one order, one filter
# ---------------------------------------------------------------------------

# The three states the run log can be in, so a surface can tell them apart. An
# unreadable log and a log the product may not serve are different news, and neither of them is "no runs".
RUNS_AVAILABLE = "available"
RUNS_UNREADABLE = "unreadable"
RUNS_WITHHELD = "withheld_no_operator_channel"


def _run_records() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """The operator's own runs, oldest first, with the scope disclosure.

    **Withheld is not a defensive nicety, it closes a measured breach.** The run
    log holds every channel's runs and the boundary is the filter to
    ``settings.operator_channel``. With that setting empty the scope helper passes
    every record through and says so, which is right for a helper and wrong here:
    measured the moment another surface blanked the field, a ``kind=run`` read
    returned 217, 124 and 76 rows belonging to three channels the operator does
    not own, with their names and their revenue on them. A product that cannot
    tell which runs are its own may not guess and may not serve all four, so it
    withholds them and names the missing input and where to supply it.
    """
    from kairos.observability.run_log import DEFAULT_RUN_LOG_PATH, read_run_log

    empty = channel_scope.scope_note("", 0, 0, 0, scoped=False)
    if not DEFAULT_RUN_LOG_PATH.exists():
        return [], empty, RUNS_UNREADABLE
    try:
        records = read_run_log()
    except (OSError, ValueError):
        logger.exception("the run log could not be read; history serves the rest")
        return [], empty, RUNS_UNREADABLE
    scoped, note = history_api_timeline.scoped_runs(records)
    if not note.get("scoped"):
        withheld = dict(note)
        withheld["rows_out"] = 0
        return [], withheld, RUNS_WITHHELD
    return scoped, note, RUNS_AVAILABLE


def _assemble(request: Request) -> dict[str, Any]:
    """Every source, merged, ordered and filtered for this caller.

    Two filters apply and they answer two different questions. The activity
    scope is the rule the settings activity log already enforces: an admin sees
    every account's changes and sign-ins, anybody else sees only their own, and
    History may not widen that by merging. The affiliation filter is the
    training test: an entry whose output lands under ``models/`` is company-only.
    Restore points and runs are the operator's shared operating record and carry
    no per-account scope, which is what ``/api/versions`` already does today.
    """
    scope, self_user = activity_log.visibility_scope(request)
    manifests = version_store._all_manifests()
    activity = activity_log._read_entries()
    if scope == "self":
        activity = [record for record in activity if record.get("user") == self_user]
    runs, run_note, runs_state = _run_records()
    entries = (
        history_api_timeline.version_entries(manifests)
        + history_api_timeline.restore_entries()
        + history_api_timeline.activity_entries(activity)
        + history_api_timeline.run_entries(runs)
    )
    visible = history_api_timeline.order(history_api_timeline.visible(entries, is_company(request)))
    # Two of these four records are pruned and one is scoped, so a dropped day, a
    # quiet day and a day outside this reader's slice are three different answers.
    sources, starts = history_api_reach.assemble(
        manifests, activity, runs, runs_state, runs_state == RUNS_AVAILABLE, visible, scope)
    return {
        "entries": visible,
        "runs": runs,
        "run_scope": run_note,
        "scope": scope,
        "sources": sources,
        "record_starts": starts,
    }


@timeline_router.get("")
def history_timeline(request: Request, limit: int = 120, kind: str | None = None,
                     actor: str | None = None, since: str | None = None,
                     until: str | None = None, before: str | None = None) -> dict[str, Any]:
    """The merged timeline, newest first, and how far back this page reaches.

    ``kind`` narrows to one of the six kinds, ``actor`` to one account, ``since``
    to an ISO calendar day and ``until`` to the day the list ends on. The counts
    are taken inside the day window and before the kind and the actor, so a
    filter control prints a real count of what clicking it would reveal.

    **A page is a window, and the window used to be nailed to the newest end.**
    A page carries at most ``MAX_TIMELINE_ENTRIES``, cut from the newest of
    whatever matched, so the record before it was unreachable and the payload did
    not say so: measured, ``kind=change`` matched 2,027 entries and served 500
    spanning one afternoon. ``until`` moves the window by calendar day, ``before``
    steps it by exactly one page, and ``newer``, ``served`` and ``older`` say
    where the window sits so the surface can print a true sentence.
    """
    _require_session(request)
    assembled = _assemble(request)
    entries = assembled["entries"]
    total = len(entries)
    if since:
        entries = history_api_timeline.since_day(entries, _require_day(since, "since"))
    if until:
        entries = history_api_timeline.until_day(entries, _require_day(until, "until"))
    # Inside the day window: a tab that says 2,027 over a list of 17 counts a set the reader is not looking at.
    # Beside it, how many of those changes the server refused, so no figure here reads as a count of what happened.
    tally = history_api_timeline.counts(entries)
    outcomes = history_api_timeline.outcome_counts(entries)
    window_total = len(entries)
    if kind:
        if kind not in history_api_timeline.KINDS:
            raise HTTPException(status_code=400, detail=f"kind must be one of {list(history_api_timeline.KINDS)}.")
        entries = [entry for entry in entries if entry["kind"] == kind]
    if actor:
        wanted = str(actor).strip()
        entries = [entry for entry in entries if entry["actor"] == wanted]
    # matched is the whole result set here and never moves as the reader pages: it is the footer's denominator.
    matched = len(entries)
    if before:
        if not _CURSOR_RE.fullmatch(str(before)):
            raise HTTPException(status_code=400,
                                detail="before must be a cursor taken from next_before.")
        entries = history_api_timeline.older_than(entries, str(before))
    capped = max(1, min(int(limit), MAX_TIMELINE_ENTRIES))
    page = entries[:capped]
    older = len(entries) - len(page)
    today = history_api_timeline.broadcast_day(datetime.now(timezone.utc).isoformat())
    body = {
        "entries": page,
        "matched": matched,
        "served": len(page),
        "newer": matched - len(entries),
        "older": older,
        "next_before": history_api_timeline.cursor_of(page[-1]) if older and page else None,
        "page_max": MAX_TIMELINE_ENTRIES,
        "window": {"since": since or None, "until": until or None, "before": before or None},
        "total": total,
        "window_total": window_total,
        "record_starts": assembled["record_starts"],
        "counts": tally,
        "outcomes": outcomes,
        "today": today,
        "attestation": history_api_attestation.since_body(assembled, today),
        "kinds": list(history_api_timeline.KINDS),
        "actors": sorted({entry["actor"] for entry in assembled["entries"] if entry["actor"]}),
        "scope": assembled["scope"],
        "sources": assembled["sources"],
        "run_scope": assembled["run_scope"],
        "note": _SCOPE_NOTE,
    }
    return HISTORY_WALL.stamp(body, request)


@timeline_router.get("/runs/{run_id}")
def history_run(run_id: str, request: Request) -> dict[str, Any]:
    """One run: what it read, what was in force, what it produced, and the delta
    against the previous run over the same channel and the same day."""
    _require_session(request)
    if not _RUN_ID_RE.fullmatch(str(run_id or "")):
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    runs, note, state = _run_records()
    if state == RUNS_WITHHELD:
        raise HTTPException(status_code=409, detail=(
            "No operator channel is set, so this product cannot tell which runs belong to the "
            "operator. Runs stay withheld until the channel is set on Rules."))
    record = history_api_runs.find_run(runs, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
    body = history_api_runs.detail(runs, record)
    body["run_scope"] = note
    body["run_log_available"] = state == RUNS_AVAILABLE
    body["run_log_state"] = state
    return body


@timeline_router.get("/since")
def history_since(request: Request, day: str | None = None) -> dict[str, Any]:
    """What changed since a calendar day, and the evidence when nothing did.

    This is the attestation half of the compliance job: the counts per kind
    since the day, and the regulatory guardrail store's own change record, which
    is the only place a limit's movement is recorded with a date and an actor.

    The landing answer for today rides the timeline read, so opening History
    costs one request and this route serves the other days. No day means that
    same today, named on the answer; a supplied day is still held to the calendar.
    """
    _require_session(request)
    day = day or history_api_timeline.broadcast_day(datetime.now(timezone.utc).isoformat())
    return history_api_attestation.since_body(_assemble(request), _require_day(day, "day"))
