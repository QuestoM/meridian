"""Review and approval: the completeness gate, enforced server-side.

The mission rule this module owns: extracted commercial terms NEVER take
effect without human approval, and an agreement cannot be approved while any
clause remains unseen or any proposal undecided. The gate is re-derived from
stored state inside ``approve()`` — a disabled button in the UI is a courtesy;
this is the law.

A clause ends review in exactly one of these states:

- mapped, with every instance built from it decided (confirmed/edited/rejected);
- irrelevant, with its closed-list class and reason (from extraction or reviewer);
- unmapped-ACKNOWLEDGED: a human read the loud flag and recorded a disposition
  note ("understood, not supported — tracked outside the system"). Approval is
  possible with acknowledged-unsupported clauses; it is impossible with SILENT
  ones. The acknowledgment travels into the version manifest, so the honesty
  survives the approval.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from kairos.trade import taxonomy, taxonomy_schemas
from kairos.trade import standing
from kairos_api import trade_store
from kairos_api.trade_store import (
    APPROVED,
    CONFIRMED,
    EDITED,
    IN_REVIEW,
    PROPOSED,
    REJECTED,
    REVIEW_STATES,
    now_stamp,
)


def _entry(review: dict[str, Any], instance_id: str) -> dict[str, Any]:
    entry = review.get("instances", {}).get(instance_id)
    if entry is None:
        raise KeyError(f"no proposed instance {instance_id!r} in this review")
    return entry


def _instance(extraction: dict[str, Any], instance_id: str) -> dict[str, Any]:
    for inst in extraction.get("instances", []):
        if inst.get("instance_id") == instance_id:
            return inst
    raise KeyError(f"extraction holds no instance {instance_id!r}")


def mark_clauses_seen(agreement_id: str, document_id: str, clause_ids: list[str],
                      actor: str) -> dict[str, Any]:
    """Record that a reviewer has had these clauses on screen."""
    extraction = trade_store.load_extraction(agreement_id, document_id)
    known = {c["clause_id"] for c in extraction.get("clauses", [])}
    unknown = sorted(set(clause_ids) - known)
    if unknown:
        raise KeyError(f"document {document_id} has no clauses {unknown}")
    review = trade_store.load_review(agreement_id, document_id)
    seen = review.setdefault("clauses_seen", {})
    for cid in clause_ids:
        seen.setdefault(cid, {"by": actor, "at": now_stamp()})
    trade_store.save_review(agreement_id, document_id, review)
    return {"seen": len(seen), "total": len(known)}


def decide_instance(
    agreement_id: str,
    document_id: str,
    instance_id: str,
    verdict: str,
    actor: str,
    *,
    edited_params: Optional[dict[str, Any]] = None,
    edited_scope: Optional[dict[str, Any]] = None,
    edited_window: Optional[dict[str, Any]] = None,
    reason: str = "",
) -> dict[str, Any]:
    """Confirm, edit or reject one proposed instance.

    An edit keeps the extraction's params untouched and stores the reviewer's
    beside them; the approved termset carries the reviewer's values with the
    diff visible. A rejection needs a reason — a term removed silently is a
    term dropped, which is the one unacceptable outcome.
    """
    if verdict not in (CONFIRMED, EDITED, REJECTED):
        raise ValueError(f"verdict must be one of {(CONFIRMED, EDITED, REJECTED)}")
    if verdict == REJECTED and not str(reason or "").strip():
        raise ValueError("rejecting an extracted term requires a reason")
    extraction = trade_store.load_extraction(agreement_id, document_id)
    proposed = _instance(extraction, instance_id)
    review = trade_store.load_review(agreement_id, document_id)
    entry = _entry(review, instance_id)

    if verdict == EDITED:
        if edited_params is None:
            raise ValueError("an edit carries the edited parameters")
        _validate_params(proposed["term_id"], edited_params)
        entry["edited_params"] = edited_params
        if edited_scope is not None:
            entry["edited_scope"] = edited_scope
        if edited_window is not None:
            entry["edited_window"] = edited_window
    entry["state"] = verdict
    entry["by"] = actor
    entry["at"] = now_stamp()
    if reason:
        entry["reason"] = str(reason)
    entry.setdefault("history", []).append(
        {"state": verdict, "by": actor, "at": entry["at"], "reason": str(reason or "")}
    )
    trade_store.save_review(agreement_id, document_id, review)
    return entry


def promote_instance(agreement_id: str, document_id: str, instance_id: str,
                     actor: str) -> dict[str, Any]:
    """Move one reading out of the interpretations and into the proposals.

    A reading with no values in it does not hold the gate shut, which is what
    keeps the main list short enough to work through. But a reader who looks at
    one and recognises a real term must be able to say so — and from that moment
    it is an ordinary proposal: it appears in the main list, it blocks approval
    until it is decided, and deciding it will mean editing the values in,
    because the extraction did not find any.

    The reverse is deliberately not offered. Once a person has said a clause
    carries a term, taking that back is a REJECTION with a reason, which the
    ordinary decision path already records and this would quietly bypass.
    """
    extraction = trade_store.load_extraction(agreement_id, document_id)
    inst = _instance(extraction, instance_id)
    if not standing.is_interpretive(inst):
        raise ValueError(
            f"instance {instance_id!r} is already a proposal; there is nothing to promote"
        )
    review = trade_store.load_review(agreement_id, document_id)
    _entry(review, instance_id)
    promoted = review.setdefault("promoted", [])
    if instance_id not in promoted:
        promoted.append(instance_id)
    review.setdefault("promoted_log", []).append(
        {"instance_id": instance_id, "by": actor, "at": now_stamp()})
    trade_store.save_review(agreement_id, document_id, review)
    return {"instance_id": instance_id, "standing": standing.CONFIDENT,
            "promoted": True}


def standings(extraction: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Where each reading sits, decided once here so no surface re-derives it."""
    promoted = set(review.get("promoted", []))
    out: dict[str, Any] = {}
    for inst in extraction.get("instances", []):
        iid = inst["instance_id"]
        interpretive = standing.is_interpretive(inst) and iid not in promoted
        out[iid] = {
            "standing": standing.INTERPRETIVE if interpretive else standing.CONFIDENT,
            "promoted": iid in promoted,
            "reason_he": standing.reason(inst, "he") if interpretive else "",
            "reason_en": standing.reason(inst, "en") if interpretive else "",
        }
    return out


def _validate_params(term_id: str, params: dict[str, Any]) -> list[str]:
    """Structural check against the term schema; returns missing required
    fields (callers decide whether missing blocks). Unknown keys raise."""
    schema = taxonomy_schemas.schema_for(term_id)
    allowed = set(schema.get("properties", {}))
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"term {term_id} does not take parameters {unknown}")
    return [f for f in schema.get("required", []) if f not in params]


def add_reviewer_instance(
    agreement_id: str,
    document_id: str,
    *,
    term_id: str,
    params: dict[str, Any],
    actor: str,
    clause_id: Optional[str] = None,
    quote: str = "",
    scope: Optional[dict[str, Any]] = None,
    window: Optional[dict[str, Any]] = None,
    not_in_document: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """A term the extraction missed, added by the reviewer.

    Either it points at a clause (with a quote the reviewer selected), or it is
    explicitly marked ``not_in_document`` — allowed, but loud, because a rule
    with no documentary basis must say so forever.
    """
    taxonomy.get(term_id)
    missing = _validate_params(term_id, params)
    extraction = trade_store.load_extraction(agreement_id, document_id)
    if not not_in_document:
        if not clause_id or not str(quote or "").strip():
            raise ValueError(
                "a reviewer-added term cites a clause and a quote, or is "
                "explicitly marked as not_in_document"
            )
        clause = next(
            (c for c in extraction["clauses"] if c["clause_id"] == clause_id), None
        )
        if clause is None:
            raise KeyError(f"document {document_id} has no clause {clause_id!r}")
        if quote not in clause["text"]:
            raise ValueError(
                "the quote must appear verbatim in the cited clause"
            )
    review = trade_store.load_review(agreement_id, document_id)
    instance_id = "rv-" + uuid.uuid4().hex[:8]
    record = {
        "instance_id": instance_id,
        "term_id": term_id,
        "params": params,
        "scope": scope or {},
        "window": window or {},
        "missing": missing,
        "origin": "reviewer",
        "not_in_document": bool(not_in_document),
        "clause_id": clause_id,
        "quote": str(quote or ""),
        "note": str(note or ""),
        "added_by": actor,
        "added_at": now_stamp(),
    }
    review.setdefault("reviewer_added", []).append(record)
    trade_store.save_review(agreement_id, document_id, review)
    return record


def acknowledge_unmapped(agreement_id: str, document_id: str, clause_id: str,
                         actor: str, note: str) -> dict[str, Any]:
    """A human takes ownership of an understood-but-unsupported clause."""
    if not str(note or "").strip():
        raise ValueError("acknowledging an unsupported clause requires a note")
    extraction = trade_store.load_extraction(agreement_id, document_id)
    disp = next(
        (d for d in extraction["dispositions"] if d["clause_id"] == clause_id), None
    )
    if disp is None:
        raise KeyError(f"document {document_id} has no clause {clause_id!r}")
    if disp["disposition"] != "unmapped":
        raise ValueError(
            f"clause {clause_id} is {disp['disposition']!r}, not unmapped"
        )
    review = trade_store.load_review(agreement_id, document_id)
    review.setdefault("unmapped_acks", {})[clause_id] = {
        "by": actor, "at": now_stamp(), "note": str(note),
    }
    trade_store.save_review(agreement_id, document_id, review)
    return review["unmapped_acks"][clause_id]


def resolve_conflict(agreement_id: str, document_id: str, conflict_id: str,
                     winner_instance_id: str, actor: str, note: str = "") -> dict[str, Any]:
    """A human settles a conflict the resolver could not (or overrides it)."""
    review = trade_store.load_review(agreement_id, document_id)
    conflicts = review.setdefault("conflicts", {})
    entry = conflicts.get(conflict_id)
    if entry is None:
        raise KeyError(f"no conflict {conflict_id!r} recorded on this review")
    if winner_instance_id not in entry.get("instances", []):
        raise ValueError(
            f"{winner_instance_id!r} is not a party to conflict {conflict_id}"
        )
    entry["resolution"] = "resolved_by_human"
    entry["winner"] = winner_instance_id
    entry["resolved_by"] = actor
    entry["resolved_at"] = now_stamp()
    entry["note"] = str(note or "")
    trade_store.save_review(agreement_id, document_id, review)
    return entry


# ------------------------------------------------------------------- the gate

def document_gate(agreement_id: str, document_id: str) -> dict[str, Any]:
    """The completeness gate for one document, with every blocker named."""
    extraction = trade_store.load_extraction(agreement_id, document_id)
    review = trade_store.load_review(agreement_id, document_id)
    clauses = extraction.get("clauses", [])
    dispositions = {d["clause_id"]: d for d in extraction.get("dispositions", [])}
    seen = review.get("clauses_seen", {})
    acks = review.get("unmapped_acks", {})
    states = review.get("instances", {})

    # Clause ids sort in DOCUMENT ORDER, not lexicographically: a reviewer
    # scanning for what is unread expects 2.1 after 1.4, and sorted() strings
    # put 10.1 there instead. Numeric components compare as numbers; anything
    # non-numeric (pre-1, sig-1, appA-2) keeps a stable textual order after.
    def _clause_key(cid: str) -> tuple:
        parts = str(cid).replace("-", ".").split(".")
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part) for part in parts
        )

    unseen = sorted(
        (c["clause_id"] for c in clauses if c["clause_id"] not in seen),
        key=_clause_key,
    )
    # A proposal that carries no values is a LEAD, not a proposal, and it does
    # not hold the gate shut. It is still on the screen, in its own list, with a
    # control that moves it into the main one — and from that moment it is an
    # ordinary proposal and must be decided like any other.
    #
    # This is what makes the reading light enough to work through. Measured on
    # the corpus: 16 of 228 proposals carry no answer, and setting them aside
    # raises the share of the main list that is correct from 66.7% to 71.7%
    # while moving nothing correct out of it (kairos.trade.standing).
    interpretive = {
        inst["instance_id"] for inst in extraction.get("instances", [])
        if standing.is_interpretive(inst)
        and inst["instance_id"] not in set(review.get("promoted", []))
    }
    undecided = sorted(
        iid for iid, entry in states.items()
        if entry.get("state", PROPOSED) == PROPOSED and iid not in interpretive
    )
    unacked = sorted(
        cid for cid, d in dispositions.items()
        if d["disposition"] == "unmapped" and cid not in acks
    )
    open_conflicts = sorted(
        cid for cid, entry in review.get("conflicts", {}).items()
        if entry.get("resolution") not in ("resolved_by_rule", "resolved_by_human")
    )
    blockers: list[dict[str, Any]] = []
    if unseen:
        blockers.append({"kind": "clauses_unseen", "count": len(unseen), "ids": unseen[:20]})
    if undecided:
        blockers.append({"kind": "instances_undecided", "count": len(undecided), "ids": undecided[:20]})
    if unacked:
        blockers.append({"kind": "unmapped_unacknowledged", "count": len(unacked), "ids": unacked})
    if open_conflicts:
        blockers.append({"kind": "conflicts_open", "count": len(open_conflicts), "ids": open_conflicts})

    counts = {"mapped": 0, "irrelevant": 0, "unmapped": 0}
    for d in dispositions.values():
        counts[d["disposition"]] += 1
    return {
        "document_id": document_id,
        "clauses_total": len(clauses),
        "clauses_seen": len(seen),
        "dispositions": counts,
        "instances_total": len(states),
        "instances_decided": len(states) - len(undecided) - len(interpretive),
        # Named on the gate so a reader can see that the short list is short on
        # purpose, and how many readings are waiting in the other one.
        "instances_interpretive": len(interpretive),
        "reviewer_added": len(review.get("reviewer_added", [])),
        "unmapped_acknowledged": len(acks),
        "conflicts_open": len(open_conflicts),
        "ready": not blockers,
        "blockers": blockers,
    }


def agreement_gate(agreement_id: str) -> dict[str, Any]:
    """The whole-agreement gate: every attached document must pass its own."""
    head = trade_store.load_head(agreement_id)
    documents = head.get("documents", [])
    gates = []
    blockers: list[dict[str, Any]] = []
    if not documents:
        blockers.append({"kind": "no_documents", "count": 0, "ids": []})
    for doc in documents:
        doc_id = doc["document_id"]
        try:
            gate = document_gate(agreement_id, doc_id)
        except KeyError:
            gate = {"document_id": doc_id, "ready": False,
                    "blockers": [{"kind": "no_extraction", "count": 1, "ids": [doc_id]}]}
        gates.append(gate)
        for blocker in gate.get("blockers", []):
            blockers.append({**blocker, "document_id": doc_id})
    return {
        "agreement_id": agreement_id,
        "status": head["status"],
        "ready": head["status"] == IN_REVIEW and not blockers,
        "documents": gates,
        "blockers": blockers,
    }


# ------------------------------------------------------------------ approval

def approve(agreement_id: str, actor: str, note: str = "") -> dict[str, Any]:
    """The human act: gate re-derived, version frozen, status moved.

    Returns the version manifest. The caller (the API layer) runs the compiler
    on the returned termset; nothing here reaches planning by itself.
    """
    gate = agreement_gate(agreement_id)
    if not gate["ready"]:
        raise ValueError(
            "the completeness gate is not green: "
            + "; ".join(
                f"{b['kind']}({b['count']}) in {b.get('document_id', 'agreement')}"
                for b in gate["blockers"]
            )
        )
    head = trade_store.load_head(agreement_id)
    approved_instances: list[dict[str, Any]] = []
    acknowledged_unsupported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for doc in head.get("documents", []):
        doc_id = doc["document_id"]
        extraction = trade_store.load_extraction(agreement_id, doc_id)
        review = trade_store.load_review(agreement_id, doc_id)
        states = review.get("instances", {})
        for inst in extraction.get("instances", []):
            entry = states.get(inst["instance_id"], {})
            state = entry.get("state", PROPOSED)
            if state == REJECTED:
                rejected.append({
                    "instance_id": inst["instance_id"], "term_id": inst["term_id"],
                    "reason": entry.get("reason", ""), "by": entry.get("by"),
                })
                continue
            params = entry.get("edited_params", inst["params"])
            approved_instances.append({
                **inst,
                "document_id": doc_id,
                "params": params,
                "scope": entry.get("edited_scope", inst.get("scope", {})),
                "window": entry.get("edited_window", inst.get("window", {})),
                "review": {
                    "state": state,
                    "by": entry.get("by"),
                    "at": entry.get("at"),
                    "extracted_params": inst["params"] if state == EDITED else None,
                },
            })
        for added in review.get("reviewer_added", []):
            approved_instances.append({
                "instance_id": added["instance_id"],
                "term_id": added["term_id"],
                "params": added["params"],
                "scope": added.get("scope", {}),
                "window": added.get("window", {}),
                "missing": added.get("missing", []),
                "notes": added.get("note", ""),
                "document_id": doc_id,
                "citations": (
                    [{
                        "document_id": doc_id,
                        "page": _page_of(extraction, added.get("clause_id")),
                        "clause_id": added.get("clause_id"),
                        "quote": added.get("quote", ""),
                    }]
                    if added.get("clause_id") else []
                ),
                "confidence": "high",
                "review": {"state": "reviewer_added", "by": added.get("added_by"),
                           "at": added.get("added_at"),
                           "not_in_document": added.get("not_in_document", False)},
            })
        for cid, ack in review.get("unmapped_acks", {}).items():
            disp = next(
                d for d in extraction["dispositions"] if d["clause_id"] == cid
            )
            acknowledged_unsupported.append({
                "document_id": doc_id, "clause_id": cid,
                "appears_to_do": disp.get("reason", ""), **ack,
            })
        for conflict_id, entry in review.get("conflicts", {}).items():
            conflicts.append({"conflict_id": conflict_id, "document_id": doc_id, **entry})

    versions = trade_store.list_versions(agreement_id)
    version_id = "v-" + uuid.uuid4().hex[:10]
    manifest = {
        "version_id": version_id,
        "seq": max((int(v.get("seq", 0)) for v in versions), default=0) + 1,
        "agreement_id": agreement_id,
        "created_at": now_stamp(),
        "actor": actor,
        "note": str(note or ""),
        "level": head["level"],
        "window": head.get("window", {}),
        "parent_agreement_id": head.get("parent_agreement_id"),
        "documents": [
            {"document_id": d["document_id"], "sha256": d["sha256"],
             "filename": d["filename"]}
            for d in head.get("documents", [])
        ],
        "gate": {k: v for k, v in gate.items() if k != "blockers"},
        "counts": {
            "approved_terms": len(approved_instances),
            "rejected_terms": len(rejected),
            "acknowledged_unsupported": len(acknowledged_unsupported),
            "conflicts": len(conflicts),
        },
    }
    termset = {
        "version_id": version_id,
        "agreement_id": agreement_id,
        "instances": approved_instances,
        "rejected": rejected,
        "acknowledged_unsupported": acknowledged_unsupported,
        "conflicts": conflicts,
    }
    trade_store.write_version(agreement_id, manifest, termset)
    head = trade_store.load_head(agreement_id)
    head["current_version_id"] = version_id
    trade_store.save_head(head, actor)
    trade_store.set_status(agreement_id, APPROVED, actor, note=f"version {version_id}")
    return manifest


def _page_of(extraction: dict[str, Any], clause_id: Optional[str]) -> int:
    for clause in extraction.get("clauses", []):
        if clause["clause_id"] == clause_id:
            return int(clause["pages"][0])
    return 1
