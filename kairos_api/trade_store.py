"""The agreement store: signed trade agreements as first-class, versioned objects.

Layout, one directory per agreement under ``data/agreements`` (relocatable via
``KAIROS_AGREEMENTS_DIR`` for tests, the same pattern as the plan versions):

    <agreements root>/<agreement_id>/
        agreement.json              the mutable head: status, parties, window
        documents/<doc_id>.pdf      immutable source bytes (sha256 in the head)
        extractions/<doc_id>.json   the pipeline's proposal for one document
        review/<doc_id>.json        reviewer state per proposed instance
        versions/<version_id>/      immutable approved snapshots
            manifest.json           who approved, when, coverage, counts
            termset.json            the approved term instances, post-review

Three rules hold this store honest:

- **Nothing is deleted.** An agreement leaves service by status
  (``superseded``/``expired``/``withdrawn``), never by removal; a source
  document, once attached, is immutable and its hash is pinned.
- **The approval gate is server truth, not a disabled button.** ``approve()``
  re-derives the gate from stored state and refuses with the exact reasons.
- **A version is bytes plus provenance.** The approved termset embeds every
  instance's citations and review history, so "why does this rule exist" is
  answerable from the version alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
AGREEMENTS_DIR_ENV = "KAIROS_AGREEMENTS_DIR"

# Status machine. Draft holds uploads and extraction runs; review begins when a
# proposal exists; approval freezes a version and compiles; the terminal states
# record why an agreement stopped governing without erasing that it did.
DRAFT = "draft"
IN_REVIEW = "in_review"
APPROVED = "approved"
SUPERSEDED = "superseded"
EXPIRED = "expired"
WITHDRAWN = "withdrawn"
STATUSES = (DRAFT, IN_REVIEW, APPROVED, SUPERSEDED, EXPIRED, WITHDRAWN)
TRANSITIONS: dict[str, frozenset[str]] = {
    DRAFT: frozenset({IN_REVIEW, WITHDRAWN}),
    IN_REVIEW: frozenset({DRAFT, APPROVED, WITHDRAWN}),
    APPROVED: frozenset({IN_REVIEW, SUPERSEDED, EXPIRED, WITHDRAWN}),
    SUPERSEDED: frozenset(),
    EXPIRED: frozenset(),
    WITHDRAWN: frozenset(),
}

LEVELS = ("agency_framework", "advertiser", "campaign")

# EVERY AGREEMENT HAS AN END DATE. Owner rule, and it is a modelling rule
# rather than a formality: an obligation with no closing date has no
# measurement window, so its pace, its projection and its alarm are all
# undefined — the commitment silently stops being tracked, which is the exact
# failure this engine exists to prevent. An agreement the parties intend to run
# until somebody cancels it is therefore recorded with the FOREVER date below,
# and every surface prints that as open-ended rather than as a real 2099
# deadline.
FOREVER = "2099-12-31"
FOREVER_LABEL_HE = "ללא מועד סיום (נרשם עד 2099)"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Reviewer verdicts on one proposed instance. ``edited`` keeps BOTH the
# extraction's params and the reviewer's; the diff is part of the record.
PROPOSED = "proposed"
CONFIRMED = "confirmed"
EDITED = "edited"
REJECTED = "rejected"
REVIEW_STATES = (PROPOSED, CONFIRMED, EDITED, REJECTED)

_STORE_LOCK = threading.Lock()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


def lock() -> threading.Lock:
    return _STORE_LOCK


def now_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def agreements_root() -> Path:
    raw = os.environ.get(AGREEMENTS_DIR_ENV, "").strip()
    return Path(raw) if raw else ROOT / "data" / "agreements"


def _dir_for(agreement_id: str) -> Path:
    if not _ID_RE.fullmatch(agreement_id):
        raise ValueError(f"invalid agreement id {agreement_id!r}")
    return agreements_root() / agreement_id


def _head_path(agreement_id: str) -> Path:
    return _dir_for(agreement_id) / "agreement.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalise_window(window: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Every agreement window carries a start and an end, or this raises.

    ``ends_on`` may be given as ``None``/``""``/``"indefinite"``/``"forever"``
    to mean "until one side cancels"; that is stored as ``FOREVER`` with
    ``open_ended: true`` so the intent survives and the measurement window
    still exists. A start date is never invented: an agreement whose own start
    nobody knows is a document that has not been read yet.
    """
    window = dict(window or {})
    start = str(window.get("starts_on") or "").strip()
    if not _DATE_RE.match(start):
        raise ValueError(
            "an agreement needs a start date in YYYY-MM-DD form; "
            f"got {window.get('starts_on')!r}"
        )
    raw_end = str(window.get("ends_on") or "").strip().lower()
    open_ended = raw_end in ("", "none", "indefinite", "forever", "ללא", "פתוח")
    end = FOREVER if open_ended else str(window.get("ends_on")).strip()
    if not _DATE_RE.match(end):
        raise ValueError(
            "an agreement needs an end date in YYYY-MM-DD form, or an "
            "explicitly open-ended marker; got "
            f"{window.get('ends_on')!r}"
        )
    if end < start:
        raise ValueError(f"the agreement ends ({end}) before it starts ({start})")
    out = {**window, "starts_on": start, "ends_on": end, "open_ended": open_ended}
    if open_ended:
        out["open_ended_label_he"] = FOREVER_LABEL_HE
    return out


# ---------------------------------------------------------------- agreement head

def new_agreement_id() -> str:
    return "agr-" + uuid.uuid4().hex[:10]


def create(
    *,
    title: str,
    level: str,
    actor: str,
    counterparty: Optional[dict[str, Any]] = None,
    window: Optional[dict[str, Any]] = None,
    parent_agreement_id: Optional[str] = None,
    note: str = "",
) -> dict[str, Any]:
    """Create a draft agreement head. Raises on a bad level or missing title."""
    clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
    if not clean_title:
        raise ValueError("an agreement needs a title")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if parent_agreement_id is not None:
        load_head(parent_agreement_id)  # raises when the parent does not exist
    window = normalise_window(window)
    agreement_id = new_agreement_id()
    head = {
        "agreement_id": agreement_id,
        "title": clean_title,
        "level": level,
        "status": DRAFT,
        "counterparty": counterparty or {},
        "window": window,
        "parent_agreement_id": parent_agreement_id,
        "note": str(note or ""),
        "created_at": now_stamp(),
        "created_by": actor,
        "updated_at": now_stamp(),
        "updated_by": actor,
        "documents": [],
        "current_version_id": None,
        "status_history": [
            {"status": DRAFT, "at": now_stamp(), "by": actor, "note": "created"}
        ],
    }
    _write_json(_head_path(agreement_id), head)
    return head


def load_head(agreement_id: str) -> dict[str, Any]:
    path = _head_path(agreement_id)
    if not path.exists():
        raise KeyError(f"no agreement {agreement_id!r}")
    return _read_json(path)


def save_head(head: dict[str, Any], actor: str) -> dict[str, Any]:
    head = dict(head)
    head["updated_at"] = now_stamp()
    head["updated_by"] = actor
    _write_json(_head_path(head["agreement_id"]), head)
    return head


def list_agreements() -> list[dict[str, Any]]:
    """Every agreement head, newest first, resilient to a broken directory."""
    root = agreements_root()
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for directory in root.iterdir():
        head_file = directory / "agreement.json"
        if not directory.is_dir() or not head_file.exists():
            continue
        try:
            found.append(_read_json(head_file))
        except (json.JSONDecodeError, OSError):
            continue
    found.sort(key=lambda h: str(h.get("created_at", "")), reverse=True)
    return found


def set_status(agreement_id: str, target: str, actor: str, note: str = "") -> dict[str, Any]:
    """Move the status machine, refusing transitions the machine does not allow."""
    if target not in STATUSES:
        raise ValueError(f"unknown status {target!r}")
    head = load_head(agreement_id)
    current = head["status"]
    if target not in TRANSITIONS.get(current, frozenset()):
        raise ValueError(
            f"agreement {agreement_id} is {current!r} and cannot become "
            f"{target!r}; allowed: {sorted(TRANSITIONS.get(current, frozenset()))}"
        )
    head["status"] = target
    head.setdefault("status_history", []).append(
        {"status": target, "at": now_stamp(), "by": actor, "note": str(note or "")}
    )
    return save_head(head, actor)


# ---------------------------------------------------------------- documents

def attach_document(
    agreement_id: str,
    *,
    filename: str,
    payload: bytes,
    actor: str,
    ingest_route: str = "digital",
) -> dict[str, Any]:
    """Attach an immutable source document to an agreement.

    Draft and in-review agreements accept documents directly. An APPROVED
    agreement accepts one too - that is how an amendment or an appendix
    arrives in this market - and the arrival sends the agreement back to
    review: the new document's clauses must pass the completeness gate like
    any others, while the approved version KEEPS GOVERNING until a new
    approval supersedes it (binding moves only on approve/supersede, never
    on attach).
    """
    head = load_head(agreement_id)
    if head["status"] not in (DRAFT, IN_REVIEW, APPROVED):
        raise ValueError(
            "הסכם במצב "
            f"{head['status']!r} "
            "אינו מקבל מסמכים; מסמך מצטרף לטיוטה, לסקירה, או כתיקון להסכם מאושר"
        )
    if not payload:
        raise ValueError("קובץ ריק אינו מסמך")
    # The reading pipeline consumes PDFs (digital or scanned). Anything else -
    # a spreadsheet, a Word file, an image - would be stored happily and fail
    # confusingly at extraction, so the refusal happens here, at the boundary,
    # by content rather than by filename: a renamed .xlsx is still an xlsx.
    if not bytes(payload[:5]) == b"%PDF-":
        raise ValueError(
            "הקובץ אינו PDF. המערכת קוראת הסכמים חתומים כקובצי PDF בלבד - "
            "גם סרוקים; גיליון אלקטרוני או מסמך וורד יש לייצא ל־PDF תחילה"
        )
    doc_id = "doc-" + uuid.uuid4().hex[:8]
    suffix = Path(str(filename or "")).suffix.lower() or ".pdf"
    target = _dir_for(agreement_id) / "documents" / f"{doc_id}{suffix}"
    _atomic_write(target, payload)
    entry = {
        "document_id": doc_id,
        "filename": str(filename or f"{doc_id}{suffix}"),
        "stored_as": target.name,
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "ingest_route": ingest_route,
        "attached_at": now_stamp(),
        "attached_by": actor,
    }
    head.setdefault("documents", []).append(entry)
    save_head(head, actor)
    if head["status"] == APPROVED:
        set_status(
            agreement_id, IN_REVIEW, actor,
            note=(
                "מסמך נוסף הועלה (נספח או תיקון) וההסכם חזר לסקירה. "
                "הגרסה המאושרת ממשיכה לחול עד שאישור חדש יחליף אותה."
            ),
        )
    return entry


def document_path(agreement_id: str, document_id: str) -> Path:
    head = load_head(agreement_id)
    entry = next(
        (d for d in head.get("documents", []) if d.get("document_id") == document_id),
        None,
    )
    if entry is None:
        raise KeyError(f"agreement {agreement_id} has no document {document_id!r}")
    path = _dir_for(agreement_id) / "documents" / str(entry["stored_as"])
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


# ---------------------------------------------------------------- extraction

def save_extraction(agreement_id: str, document_id: str, payload: dict[str, Any],
                    actor: str) -> None:
    """Store the pipeline's proposal for one document and open review state.

    The payload is a DocumentExtraction.to_payload() dict; it is validated by
    round-tripping through the shapes so a malformed proposal cannot enter the
    review flow. Saving a NEW extraction for a document resets its review
    state: the old review file is archived beside it, never overwritten in place.
    """
    from kairos.trade.documents import extraction_from_payload

    extraction_from_payload(payload)  # raises on structural dishonesty
    document_path(agreement_id, document_id)  # raises when the doc is unknown
    base = _dir_for(agreement_id)
    _write_json(base / "extractions" / f"{document_id}.json", payload)
    review_file = base / "review" / f"{document_id}.json"
    if review_file.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        review_file.rename(review_file.with_name(f"{document_id}.{stamp}.superseded.json"))
    fresh = {
        "document_id": document_id,
        "opened_at": now_stamp(),
        "opened_by": actor,
        "instances": {
            inst["instance_id"]: {"state": PROPOSED}
            for inst in payload.get("instances", [])
        },
        "clauses_seen": {},
        "reviewer_added": [],
        "conflicts": {},
    }
    _write_json(review_file, fresh)


def load_extraction(agreement_id: str, document_id: str) -> dict[str, Any]:
    path = _dir_for(agreement_id) / "extractions" / f"{document_id}.json"
    if not path.exists():
        raise KeyError(f"agreement {agreement_id} has no extraction for {document_id!r}")
    return _read_json(path)


def load_review(agreement_id: str, document_id: str) -> dict[str, Any]:
    path = _dir_for(agreement_id) / "review" / f"{document_id}.json"
    if not path.exists():
        raise KeyError(f"agreement {agreement_id} has no review state for {document_id!r}")
    return _read_json(path)


def save_review(agreement_id: str, document_id: str, review: dict[str, Any]) -> None:
    _write_json(_dir_for(agreement_id) / "review" / f"{document_id}.json", review)


# ---------------------------------------------------------------- versions

def versions_dir(agreement_id: str) -> Path:
    return _dir_for(agreement_id) / "versions"


def list_versions(agreement_id: str) -> list[dict[str, Any]]:
    root = versions_dir(agreement_id)
    if not root.exists():
        return []
    found = []
    for directory in root.iterdir():
        manifest = directory / "manifest.json"
        if not directory.is_dir() or not manifest.exists():
            continue
        try:
            found.append(_read_json(manifest))
        except (json.JSONDecodeError, OSError):
            continue
    found.sort(key=lambda m: (str(m.get("created_at", "")), int(m.get("seq", 0))), reverse=True)
    return found


def load_termset(agreement_id: str, version_id: str) -> dict[str, Any]:
    path = versions_dir(agreement_id) / version_id / "termset.json"
    if not path.exists():
        raise KeyError(f"agreement {agreement_id} version {version_id} has no termset")
    return _read_json(path)


def write_version(agreement_id: str, manifest: dict[str, Any],
                  termset: dict[str, Any]) -> None:
    directory = versions_dir(agreement_id) / str(manifest["version_id"])
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "termset.json", termset)
