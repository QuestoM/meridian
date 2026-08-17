"""Day proposals: several people proposing competing versions of one day.

The week has a freeze (:mod:`kairos_api.plan_version_store`) and the browser has
private day drafts (``src/plan/week/local-plan-variants.js``). Between them sat
the thing the operation actually does: the daily plan gets built, people change
it by hand, and two schedulers who disagree about one Tuesday had nowhere to put
two answers. A browser-local draft cannot be read by the person you are arguing
with, and a week freeze is the wrong grain: it moves seven days to settle one.

A day proposal is that missing object. It is a named, authored, server-side
version of ONE operator channel-day, it lives beside its rivals, and exactly one
of them is ever adopted. This module is a SIBLING of the week freeze, built on
the same idioms and sharing its arithmetic rather than repeating it: the same
``clean_name``, the same ``_totals``, the same ``_summarize`` through
:mod:`kairos_api.channel_scope`, so a day's money and a week's money can never
be two different sums.

Four rules hold it.

- **The frozen bytes are the proposal.** ``plan.csv`` holds the proposed day's
  rows verbatim with their sha256, so adopting one publishes exactly what its
  author saw, and a comparison is computed from the frozen file rather than from
  a summary somebody wrote down.
- **The money is the operator's, and says so.** Day rows reach here already
  scoped to one channel, and the scope note still travels with every figure,
  because a figure without its scope is not a figure.
- **Staleness is first class.** A proposal records the baseline it was authored
  against - the day's own row bytes, the engine's run stamp and the operator
  settings in force. When any of those move the proposal is stale, the reason
  names which one moved, and adoption is refused until somebody explicitly
  re-bases it. The identity and the comparison live in
  :mod:`kairos_api.day_proposal_identity` and are re-exported here, because a
  caller asking "may this be adopted" should not have to know the answer is
  assembled from two modules.
- **Adoption is terminal and singular.** One proposal per day may be adopted.
  A second attempt is refused by name, and the rivals are marked rejected with
  the lineage that says which proposal superseded them. Nothing is deleted:
  a rejected alternative stays readable, which is the whole point of having
  argued in public.

The store lives under ``data/day_proposals`` by default and wherever
``KAIROS_DAY_PROPOSALS_DIR`` points when it is set, which is how a test
relocates it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# What a day IS and whether it moved. Re-exported here because the store is the
# module every caller names, and a caller asking "may this be adopted" should not
# have to know that the answer is assembled from two modules.
from kairos_api.day_proposal_identity import (  # noqa: F401  (re-exported)
    ProposalRefused,
    baseline_ref,
    canonical_bytes,
    channel_slug,
    clean_date as _clean_date,
    now_iso as _now_iso,
    sha256_of as _sha256,
    staleness,
)

# The week freeze's own arithmetic, imported rather than repeated. A day total
# and the week total that contains it are then the same function of the same
# rows, so they cannot drift into two answers.
from kairos_api.plan_version_store import (  # noqa: F401  (clean_name re-exported)
    _settings_basis,
    _summarize,
    _totals,
    clean_name,
)

ROOT = Path(__file__).resolve().parents[1]
PROPOSALS_DIR_ENV = "KAIROS_DAY_PROPOSALS_DIR"
MAX_PROPOSALS_PER_DAY = 50
PLAN_FILENAME = "plan.csv"
MANIFEST_FILENAME = "manifest.json"
EDITS_FILENAME = "edits.json"

PROPOSED = "proposed"
ADOPTED = "adopted"
REJECTED = "rejected"
WITHDRAWN = "withdrawn"
STATUSES = (PROPOSED, ADOPTED, REJECTED, WITHDRAWN)
# Every terminal state is terminal for the same reason: a decision that can be
# taken back is not a decision, and history that can be edited is not history.
TERMINAL = (ADOPTED, REJECTED, WITHDRAWN)

_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def proposals_root() -> Path:
    raw = os.environ.get(PROPOSALS_DIR_ENV, "").strip()
    return Path(raw) if raw else ROOT / "data" / "day_proposals"


def day_root(channel: str, date: str) -> Path:
    return proposals_root() / channel_slug(channel) / _clean_date(date)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))


# ------------------------------------------------------------------- read side

def list_for_day(channel: str, date: str) -> list[dict[str, Any]]:
    """Every proposal for one channel-day, newest first. Broken directories skip."""
    root = day_root(channel, date)
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    wanted = str(channel or "").strip()
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        manifest = _read_json(directory / MANIFEST_FILENAME)
        if not isinstance(manifest, dict):
            continue
        if str(manifest.get("channel") or "").strip() != wanted:
            continue
        found.append(manifest)
    found.sort(key=lambda item: int(item.get("seq", 0)), reverse=True)
    return found


def get(channel: str, date: str, proposal_id: str) -> Optional[dict[str, Any]]:
    manifest = _read_json(day_root(channel, date) / str(proposal_id) / MANIFEST_FILENAME)
    return manifest if isinstance(manifest, dict) else None


def rows_for(channel: str, date: str, proposal_id: str) -> Optional[pd.DataFrame]:
    """The proposal's frozen day rows, or None when the file is missing."""
    path = day_root(channel, date) / str(proposal_id) / PLAN_FILENAME
    if not path.exists():
        return None
    return pd.read_csv(path)


def edits_for(channel: str, date: str, proposal_id: str) -> dict[str, Any]:
    payload = _read_json(day_root(channel, date) / str(proposal_id) / EDITS_FILENAME)
    return payload if isinstance(payload, dict) else {}


def adopted_for_day(channel: str, date: str) -> Optional[dict[str, Any]]:
    return next((item for item in list_for_day(channel, date)
                 if item.get("status") == ADOPTED), None)


# ------------------------------------------------------------------ write side

def _next_seq(manifests: list[dict[str, Any]]) -> int:
    return max((int(item.get("seq", 0)) for item in manifests), default=0) + 1


def create_proposal(
    *,
    channel: str,
    date: str,
    name: str,
    author: str,
    rows: pd.DataFrame,
    baseline_ref: dict[str, Any],
    edits: Optional[dict[str, Any]] = None,
    note: str = "",
    rows_source: str = "",
    engine: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Freeze one person's version of this day and return its manifest.

    Refuses an empty name for the reason the week freeze refuses one: a version
    nobody can name is a version nobody can find again, and this surface exists
    so people can argue about named things.
    """
    clean = clean_name(name)
    if not clean:
        raise ProposalRefused("a day proposal needs a name",
                              "להצעה ליום צריך שם", code="no_name")
    if rows is None or not len(rows):
        raise ProposalRefused("a day proposal needs the day's rows",
                              "להצעה ליום צריכות להיות שורות היום", code="no_rows")
    day = _clean_date(date)
    manifests = list_for_day(channel, day)
    proposal_id = uuid.uuid4().hex[:12]
    directory = day_root(channel, day) / proposal_id
    payload = canonical_bytes(rows)
    _atomic_write(directory / PLAN_FILENAME, payload)
    edit_map = dict(edits or {})
    _write_json(directory / EDITS_FILENAME, edit_map)
    manifest = {
        "proposal_id": proposal_id,
        "channel": str(channel).strip(),
        "date": day,
        "name": clean,
        "note": clean_name(note) if note else "",
        "author": str(author or "").strip(),
        "created_at": _now_iso(),
        "seq": _next_seq(manifests),
        "status": PROPOSED,
        "baseline_ref": dict(baseline_ref or {}),
        "rows_source": str(rows_source or "").strip(),
        "rows_sha256": _sha256(payload),
        "rows_bytes": len(payload),
        "edit_count": len(edit_map),
        # The engine's own verdict on this arrangement AT AUTHORING TIME: the
        # guardrail run and totals the optimizer produced for these very rows.
        # Kept because a CSV of rows cannot be re-run through the guardrail
        # engine later (spacing and the retention floor need placements, which
        # the plan schema does not carry), so a comparison that only had the
        # rows would have to report compliance as unknown forever. None when the
        # author had no engine verdict, which reads downstream as exactly that.
        "engine": dict(engine) if isinstance(engine, dict) else None,
        "summary": _summarize(rows),
        "settings_basis": _settings_basis(),
        "decision": None,
        "lineage": {"superseded_by": None, "rebased_from": None, "rebased_at": None},
    }
    _write_json(directory / MANIFEST_FILENAME, manifest)
    prune(channel, day)
    return manifest


def _require(channel: str, date: str, proposal_id: str) -> tuple[dict[str, Any], Path]:
    if not _ID_RE.fullmatch(str(proposal_id or "")):
        raise ProposalRefused(f"no such proposal {proposal_id!r}",
                              "אין הצעה כזו", code="unknown_proposal")
    manifest = get(channel, date, proposal_id)
    if manifest is None:
        raise ProposalRefused(f"no proposal {proposal_id} for {channel} on {date}",
                              "אין הצעה כזו ליום הזה", code="unknown_proposal")
    return manifest, day_root(channel, date) / str(proposal_id)


def check_adoptable(
    channel: str,
    date: str,
    proposal_id: str,
    *,
    current_ref: Optional[dict[str, Any]] = None,
    allow_stale: bool = False,
) -> dict[str, Any]:
    """Every reason this proposal may not be adopted, raised before anything moves.

    Separate from :func:`update_status` so a caller that publishes rows into the
    live plan can run the whole refusal set FIRST and never end up having written
    a day it then cannot record as adopted.
    """
    manifest, _directory = _require(channel, date, proposal_id)
    # The day-level fact comes FIRST, before this proposal's own state. Adopting
    # a rival of the winner is the common mistake, and "this day already adopted
    # X" is what the person needs to hear; "your proposal is rejected" answers a
    # question they did not ask and leaves them looking for the reason why.
    held = adopted_for_day(channel, date)
    if held is not None:
        raise ProposalRefused(
            f"this day already adopted {held.get('name')!r} ({held.get('proposal_id')}); "
            "only one proposal per day may be adopted",
            f"ליום הזה כבר אומצה ההצעה {held.get('name')!r}; אפשר לאמץ הצעה אחת בלבד ליום",
            code="already_adopted",
        )
    current = str(manifest.get("status") or PROPOSED)
    if current in TERMINAL:
        raise ProposalRefused(
            f"proposal {manifest['name']!r} is already {current} and that is final",
            f"ההצעה {manifest['name']!r} כבר במצב סופי ולא ניתן לשנותה",
            code="already_decided",
        )
    state = staleness(manifest, current_ref)
    if (state["stale"] or not state["known"]) and not allow_stale:
        fields = ", ".join(item["field"] for item in state["moved"]) or "unknown"
        raise ProposalRefused(
            f"proposal {manifest['name']!r} was authored against a baseline that has since "
            f"moved ({fields}); re-base it explicitly before adopting",
            f"ההצעה {manifest['name']!r} נכתבה על בסיס שהשתנה מאז; יש לרענן אותה במפורש לפני אימוץ",
            code="stale",
        )
    return manifest


def update_status(
    channel: str,
    date: str,
    proposal_id: str,
    status: str,
    *,
    actor: str,
    note: str = "",
    current_ref: Optional[dict[str, Any]] = None,
    superseded_by: Optional[str] = None,
    allow_stale: bool = False,
) -> dict[str, Any]:
    """Move one proposal's status, refusing every move the machine does not allow.

    ``proposed`` is the only state anything leaves. Adoption additionally needs
    a fresh baseline and an unclaimed day: a stale proposal is refused naming
    what moved, and a second adoption is refused naming the proposal that
    already holds the day.
    """
    if status not in STATUSES:
        raise ProposalRefused(f"status must be one of {STATUSES}, got {status!r}",
                              "מצב ההצעה אינו מוכר", code="bad_status")
    if status == PROPOSED:
        raise ProposalRefused("a proposal is already proposed when it is created",
                              "הצעה נוצרת כבר במצב מוצע", code="bad_status")
    if status == ADOPTED:
        manifest = check_adoptable(channel, date, proposal_id,
                                   current_ref=current_ref, allow_stale=allow_stale)
        directory = day_root(channel, date) / str(proposal_id)
    else:
        manifest, directory = _require(channel, date, proposal_id)
        current = str(manifest.get("status") or PROPOSED)
        if current in TERMINAL:
            raise ProposalRefused(
                f"proposal {manifest['name']!r} is already {current} and that is final",
                f"ההצעה {manifest['name']!r} כבר במצב סופי ולא ניתן לשנותה",
                code="already_decided",
            )
    manifest["status"] = status
    manifest["decision"] = {
        "verdict": status,
        "by": str(actor or "").strip(),
        "at": _now_iso(),
        "note": str(note or "").strip(),
    }
    lineage = dict(manifest.get("lineage") or {})
    lineage["superseded_by"] = superseded_by
    manifest["lineage"] = lineage
    _write_json(directory / MANIFEST_FILENAME, manifest)
    return manifest


def withdraw(channel: str, date: str, proposal_id: str, *, actor: str,
             note: str = "") -> dict[str, Any]:
    """The author taking their own version off the table. Still readable after."""
    return update_status(channel, date, proposal_id, WITHDRAWN, actor=actor, note=note)


def rebase(channel: str, date: str, proposal_id: str, *, actor: str,
           new_ref: dict[str, Any], note: str = "") -> dict[str, Any]:
    """Point a stale proposal at the day as it stands now, on the record.

    The re-base does not touch the frozen rows: the author's version is what it
    was. It records that a person looked at what moved and said this version
    still stands, and it keeps the baseline it used to hold so the move itself
    stays visible in history.
    """
    manifest, directory = _require(channel, date, proposal_id)
    if str(manifest.get("status")) in TERMINAL:
        raise ProposalRefused(
            f"proposal {manifest['name']!r} is {manifest['status']} and cannot be re-based",
            f"ההצעה {manifest['name']!r} במצב סופי ולא ניתן לרענן אותה",
            code="already_decided",
        )
    lineage = dict(manifest.get("lineage") or {})
    lineage["rebased_from"] = manifest.get("baseline_ref")
    lineage["rebased_at"] = _now_iso()
    lineage["rebased_by"] = str(actor or "").strip()
    lineage["rebase_note"] = str(note or "").strip()
    manifest["lineage"] = lineage
    manifest["baseline_ref"] = dict(new_ref or {})
    _write_json(directory / MANIFEST_FILENAME, manifest)
    return manifest


def reject_rivals(channel: str, date: str, adopted_id: str, *, actor: str,
                  note: str) -> list[dict[str, Any]]:
    """Close every still-open rival of the adopted proposal, with the lineage.

    Rejection here is not deletion and not a verdict on quality: it is the
    record that this day was settled by another version, and it names which.
    """
    closed: list[dict[str, Any]] = []
    for item in list_for_day(channel, date):
        if item.get("proposal_id") == adopted_id or item.get("status") != PROPOSED:
            continue
        closed.append(update_status(
            channel, date, str(item["proposal_id"]), REJECTED,
            actor=actor, note=note, superseded_by=adopted_id,
        ))
    return closed


def prune(channel: str, date: str) -> list[str]:
    """Drop the oldest proposals past the per-day cap. Decided ones are kept first."""
    manifests = list_for_day(channel, date)
    if len(manifests) <= MAX_PROPOSALS_PER_DAY:
        return []
    # Newest first already; keep decided history over open drafts at the margin.
    ranked = sorted(manifests, key=lambda item: (
        0 if item.get("status") in TERMINAL else 1, -int(item.get("seq", 0)),
    ))
    pruned: list[str] = []
    for manifest in ranked[MAX_PROPOSALS_PER_DAY:]:
        proposal_id = str(manifest.get("proposal_id", ""))
        directory = day_root(channel, date) / proposal_id
        if proposal_id and directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            pruned.append(proposal_id)
    return pruned
