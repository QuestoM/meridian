"""HTTP routes for the trade-agreement engine.

Thin routes over the machinery: trade_store (lifecycle), trade_review (the
completeness gate), the extraction runner (as a background job), the compiler
+ trade_bind (approval makes rules bind), and the obligations engine (live
standings). Auth rides the global middleware: reads need a session, writes
need the operator/admin role, exactly like every other store.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from kairos_api import trade_review, trade_store

router = APIRouter(prefix="/api/trade", tags=["trade"])


def _actor(request: Request) -> str:
    from kairos_api import auth

    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request)
    return str(session["username"]) if session else "anonymous"


def _head_or_404(agreement_id: str) -> dict[str, Any]:
    try:
        return trade_store.load_head(agreement_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


# ------------------------------------------------------------------ agreements

class AgreementCreate(BaseModel):
    title: str
    level: str
    counterparty: dict[str, Any] | None = None
    window: dict[str, Any] | None = None
    parent_agreement_id: str | None = None
    note: str = ""


@router.get("/agreements")
def list_agreements() -> dict[str, Any]:
    heads = trade_store.list_agreements()
    out = []
    for head in heads:
        entry = {k: head.get(k) for k in (
            "agreement_id", "title", "level", "status", "counterparty",
            "window", "created_at", "created_by", "current_version_id",
            "parent_agreement_id",
        )}
        entry["documents"] = len(head.get("documents", []))
        try:
            gate = trade_review.agreement_gate(head["agreement_id"])
            entry["gate_ready"] = gate["ready"]
            entry["gate_blockers"] = len(gate["blockers"])
        except Exception:  # noqa: BLE001 - a broken gate is a fact to show
            entry["gate_ready"] = False
            entry["gate_blockers"] = None
        out.append(entry)
    return {"agreements": out}


@router.post("/agreements")
def create_agreement(body: AgreementCreate, request: Request) -> dict[str, Any]:
    try:
        return trade_store.create(
            title=body.title, level=body.level, actor=_actor(request),
            counterparty=body.counterparty, window=body.window,
            parent_agreement_id=body.parent_agreement_id, note=body.note,
        )
    except (ValueError, KeyError) as exc:
        raise _fail(exc) from exc


@router.get("/agreements/{agreement_id}")
def agreement_detail(agreement_id: str) -> dict[str, Any]:
    head = _head_or_404(agreement_id)
    versions = trade_store.list_versions(agreement_id)
    try:
        gate = trade_review.agreement_gate(agreement_id)
    except Exception:  # noqa: BLE001
        gate = {"ready": False, "blockers": [], "documents": []}
    from kairos_api import trade_bind

    return {
        "agreement": head,
        "versions": versions,
        "gate": gate,
        "bound_rules": trade_bind.bound_rules(agreement_id),
    }


class StatusChange(BaseModel):
    target: str
    note: str = ""


@router.post("/agreements/{agreement_id}/status")
def change_status(agreement_id: str, body: StatusChange, request: Request) -> dict[str, Any]:
    _head_or_404(agreement_id)
    actor = _actor(request)
    try:
        head = trade_store.set_status(agreement_id, body.target, actor, note=body.note)
    except (ValueError, KeyError) as exc:
        raise _fail(exc) from exc
    # Leaving service takes the live rules with it, visibly.
    if body.target in (trade_store.SUPERSEDED, trade_store.EXPIRED, trade_store.WITHDRAWN):
        from kairos_api import trade_bind

        unbound = trade_bind.unbind(agreement_id, actor)
        return {"agreement": head, "unbound": unbound}
    return {"agreement": head}


# ------------------------------------------------------------------- documents

@router.post("/agreements/{agreement_id}/documents")
async def upload_document(agreement_id: str, request: Request,
                          file: UploadFile) -> dict[str, Any]:
    _head_or_404(agreement_id)
    payload = await file.read()
    try:
        return trade_store.attach_document(
            agreement_id, filename=file.filename or "document.pdf",
            payload=payload, actor=_actor(request),
        )
    except ValueError as exc:
        raise _fail(exc) from exc


@router.get("/agreements/{agreement_id}/documents/{document_id}/file")
def document_file(agreement_id: str, document_id: str) -> FileResponse:
    try:
        path = trade_store.document_path(agreement_id, document_id)
    except (KeyError, FileNotFoundError) as exc:
        raise _fail(exc) from exc
    return FileResponse(path, media_type="application/pdf")


@router.post("/agreements/{agreement_id}/documents/{document_id}/extract")
def start_extraction(agreement_id: str, document_id: str, request: Request) -> dict[str, Any]:
    _head_or_404(agreement_id)
    try:
        trade_store.document_path(agreement_id, document_id)
    except (KeyError, FileNotFoundError) as exc:
        raise _fail(exc) from exc
    from kairos_api import jobs

    existing = jobs.running_job(f"trade_extract:{agreement_id}:{document_id}")
    if existing is not None:
        return {"job_id": existing, "already_running": True}
    actor = _actor(request)
    job_id = jobs.submit(
        f"trade_extract:{agreement_id}:{document_id}",
        _run_extraction_job, agreement_id, document_id, actor,
    )
    return {"job_id": job_id, "already_running": False}


def _run_extraction_job(agreement_id: str, document_id: str, actor: str) -> dict[str, Any]:
    """The background worker: ingest → extract → store proposal → open review."""
    from kairos.trade import extract_provider, extract_run

    path = trade_store.document_path(agreement_id, document_id)
    client = extract_provider.build_client()  # raises ProviderUnavailable honestly
    caller = extract_provider.StageCaller(client=client, stats=extract_provider.RunStats())
    extraction = extract_run.run_pdf(
        path, caller, document_id=document_id, agreement_id=agreement_id,
    )
    payload = extraction.to_payload()
    trade_store.save_extraction(agreement_id, document_id, payload, actor)
    _seed_conflicts(agreement_id, document_id, payload)
    coverage = payload["coverage"]
    return {
        "document_id": document_id,
        "clauses": coverage["total_clauses"],
        "mapped": coverage["mapped"],
        "unmapped": coverage["unmapped"],
        "instances": len(payload.get("instances", [])),
        "conflicts": len(payload.get("stats", {}).get("conflicts", [])),
        "provider": payload.get("stats", {}).get("provider", {}),
    }


def _seed_conflicts(agreement_id: str, document_id: str, payload: dict[str, Any]) -> None:
    """Conflicts the assembler detected become review state: auto-resolved ones
    carry their rule and explanation; open ones block the gate until a human
    settles them."""
    conflicts = payload.get("stats", {}).get("conflicts", [])
    if not conflicts:
        return
    review = trade_store.load_review(agreement_id, document_id)
    for conflict in conflicts:
        review.setdefault("conflicts", {})[conflict["conflict_id"]] = {
            "instances": conflict.get("instances", []),
            "contested": conflict.get("contested", ""),
            "resolution": conflict.get("resolution"),
            "winner": conflict.get("winner"),
            "rule": conflict.get("rule"),
            "explanation_he": conflict.get("explanation_he", ""),
        }
    trade_store.save_review(agreement_id, document_id, review)


@router.get("/agreements/{agreement_id}/documents/{document_id}/proposal")
def document_proposal(agreement_id: str, document_id: str) -> dict[str, Any]:
    """The proposal a reviewer works through, with what each term WILL DO.

    The effect sentences come from the engine's own compiler verdict
    (kairos.trade.explain), so the review screen never has to re-derive an
    effect the backend already decided — and a term the compiler cannot bind
    reads as "will not act automatically" with the reason, before approval
    rather than after.
    """
    head = _head_or_404(agreement_id)
    try:
        extraction = trade_store.load_extraction(agreement_id, document_id)
        review = trade_store.load_review(agreement_id, document_id)
        gate = trade_review.document_gate(agreement_id, document_id)
    except KeyError as exc:
        raise _fail(exc) from exc
    from kairos.trade import explain

    # Explain the CURRENT reviewed state: a reviewer's edit changes what the
    # rule will do, so the sentence must follow the edit, not the extraction.
    states = review.get("instances", {})
    effective = []
    for inst in extraction.get("instances", []):
        entry = states.get(inst["instance_id"], {})
        if entry.get("state") == trade_store.REJECTED:
            continue
        effective.append({
            **inst,
            "params": entry.get("edited_params", inst.get("params", {})),
            "scope": entry.get("edited_scope", inst.get("scope", {})),
        })
    effective.extend(review.get("reviewer_added", []))
    explained = explain.explain_termset(
        {"version_id": "draft", "agreement_id": agreement_id,
         "instances": effective},
        head,
    )
    return {"extraction": extraction, "review": review, "gate": gate,
            "effects": explained}


# --------------------------------------------------------------- review actions

class SeenBody(BaseModel):
    clause_ids: list[str]


@router.post("/agreements/{agreement_id}/documents/{document_id}/seen")
def mark_seen(agreement_id: str, document_id: str, body: SeenBody,
              request: Request) -> dict[str, Any]:
    try:
        return trade_review.mark_clauses_seen(
            agreement_id, document_id, body.clause_ids, _actor(request))
    except (KeyError, ValueError) as exc:
        raise _fail(exc) from exc


class DecideBody(BaseModel):
    verdict: str
    edited_params: dict[str, Any] | None = None
    edited_scope: dict[str, Any] | None = None
    edited_window: dict[str, Any] | None = None
    reason: str = ""


@router.post("/agreements/{agreement_id}/documents/{document_id}/instances/{instance_id}/decide")
def decide(agreement_id: str, document_id: str, instance_id: str,
           body: DecideBody, request: Request) -> dict[str, Any]:
    try:
        return trade_review.decide_instance(
            agreement_id, document_id, instance_id, body.verdict, _actor(request),
            edited_params=body.edited_params, edited_scope=body.edited_scope,
            edited_window=body.edited_window, reason=body.reason,
        )
    except (KeyError, ValueError) as exc:
        raise _fail(exc) from exc


class AddInstanceBody(BaseModel):
    term_id: str
    params: dict[str, Any]
    clause_id: str | None = None
    quote: str = ""
    scope: dict[str, Any] | None = None
    window: dict[str, Any] | None = None
    not_in_document: bool = False
    note: str = ""


@router.post("/agreements/{agreement_id}/documents/{document_id}/instances")
def add_instance(agreement_id: str, document_id: str, body: AddInstanceBody,
                 request: Request) -> dict[str, Any]:
    try:
        return trade_review.add_reviewer_instance(
            agreement_id, document_id, term_id=body.term_id, params=body.params,
            actor=_actor(request), clause_id=body.clause_id, quote=body.quote,
            scope=body.scope, window=body.window,
            not_in_document=body.not_in_document, note=body.note,
        )
    except (KeyError, ValueError) as exc:
        raise _fail(exc) from exc


class AckBody(BaseModel):
    note: str


@router.post("/agreements/{agreement_id}/documents/{document_id}/clauses/{clause_id}/acknowledge")
def acknowledge(agreement_id: str, document_id: str, clause_id: str,
                body: AckBody, request: Request) -> dict[str, Any]:
    try:
        return trade_review.acknowledge_unmapped(
            agreement_id, document_id, clause_id, _actor(request), body.note)
    except (KeyError, ValueError) as exc:
        raise _fail(exc) from exc


class ResolveBody(BaseModel):
    winner_instance_id: str
    note: str = ""


@router.post("/agreements/{agreement_id}/documents/{document_id}/conflicts/{conflict_id}/resolve")
def resolve(agreement_id: str, document_id: str, conflict_id: str,
            body: ResolveBody, request: Request) -> dict[str, Any]:
    try:
        return trade_review.resolve_conflict(
            agreement_id, document_id, conflict_id, body.winner_instance_id,
            _actor(request), note=body.note)
    except (KeyError, ValueError) as exc:
        raise _fail(exc) from exc


@router.get("/agreements/{agreement_id}/gate")
def gate(agreement_id: str) -> dict[str, Any]:
    _head_or_404(agreement_id)
    return trade_review.agreement_gate(agreement_id)


# -------------------------------------------------------------------- approval

class ApproveBody(BaseModel):
    note: str = ""


@router.post("/agreements/{agreement_id}/approve")
def approve(agreement_id: str, body: ApproveBody, request: Request) -> dict[str, Any]:
    """The human act: gate → immutable version → compile → BIND.

    The response says exactly what now binds, what the compiler skipped by
    name, and what the stores refused — the honesty travels with the act.
    """
    _head_or_404(agreement_id)
    actor = _actor(request)
    try:
        manifest = trade_review.approve(agreement_id, actor, note=body.note)
    except (ValueError, KeyError) as exc:
        raise _fail(exc) from exc
    from kairos.trade.compile import compile_termset
    from kairos_api import trade_bind

    head = trade_store.load_head(agreement_id)
    termset = trade_store.load_termset(agreement_id, manifest["version_id"])
    artifacts = compile_termset(termset, head)
    bound = trade_bind.bind(artifacts, actor)
    compiled = {
        "summary": artifacts.summary(),
        "skipped": artifacts.skipped,
        "settlement_terms": len(artifacts.settlement.get("terms", [])),
    }
    # Persist what this version compiled to, beside the version itself.
    directory = trade_store.versions_dir(agreement_id) / manifest["version_id"]
    import json as _json

    (directory / "compiled.json").write_text(
        _json.dumps({
            "conditions": artifacts.conditions,
            "frequency_rules": artifacts.frequency_rules,
            "settlement": artifacts.settlement,
            "skipped": artifacts.skipped,
            "bound": bound,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return {"version": manifest, "compiled": compiled, "bound": bound}


# ----------------------------------------------------------------- obligations

@router.get("/agreements/{agreement_id}/obligations")
def obligations(agreement_id: str) -> dict[str, Any]:
    head = _head_or_404(agreement_id)
    version_id = head.get("current_version_id")
    if not version_id:
        return {"available": False,
                "reason": "אין גרסה מאושרת; התחייבויות נמדדות רק מהסכם מאושר"}
    termset = trade_store.load_termset(agreement_id, version_id)
    import pandas as pd

    from kairos.trade import obligations as ob
    from kairos_api import agency_conditions, campaigns_api_store, campaigns_delivery

    links_path = Path(agency_conditions.LINKS_PATH)
    links = (
        pd.read_csv(links_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if links_path.exists() else pd.DataFrame(columns=["agency_id", "advertiser"])
    )
    inputs = ob.Inputs(
        delivery=campaigns_delivery.load_frame(),
        campaigns=campaigns_api_store.load_frame(),
        agency_links=links,
        today=date.today(),
        preferred_rate=None,
    )
    snapshots = ob.evaluate_all(termset, head, inputs)
    alarms: dict[str, int] = {}
    for snap in snapshots:
        alarms[snap.get("alarm", "unknown")] = alarms.get(snap.get("alarm", "unknown"), 0) + 1
    return {"available": True, "obligations": snapshots, "alarm_counts": alarms,
            "evaluated_at": inputs.today.isoformat()}


# ----------------------------------------------------------------- simulation

class SimulateBody(BaseModel):
    """Simulate a proposed or approved version against real activity."""

    version_id: str | None = None
    window: dict[str, str] | None = None


def _obligation_inputs(today: date):
    import pandas as pd

    from kairos_api import agency_conditions, campaigns_api_store, campaigns_delivery

    links_path = Path(agency_conditions.LINKS_PATH)
    links = (
        pd.read_csv(links_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if links_path.exists() else pd.DataFrame(columns=["agency_id", "advertiser"])
    )
    return campaigns_delivery.load_frame(), campaigns_api_store.load_frame(), links


@router.post("/agreements/{agreement_id}/simulate")
def simulate_agreement(agreement_id: str, body: SimulateBody) -> dict[str, Any]:
    """What this agreement WOULD do to real activity. Writes nothing.

    Without a version_id the newest version is used; a draft agreement with no
    version yet says so rather than simulating an empty termset.
    """
    head = _head_or_404(agreement_id)
    version_id = body.version_id or head.get("current_version_id")
    if not version_id:
        versions = trade_store.list_versions(agreement_id)
        version_id = versions[0]["version_id"] if versions else None
    if not version_id:
        return {"available": False,
                "reason": "אין עדיין גרסה לסימולציה; יש להשלים סקירה ואישור, "
                          "או לשמור גרסת טיוטה"}
    try:
        termset = trade_store.load_termset(agreement_id, version_id)
    except KeyError as exc:
        raise _fail(exc) from exc
    from kairos.trade import simulate as trade_simulate

    delivery, campaigns, links = _obligation_inputs(date.today())
    inputs = trade_simulate.SimulationInputs(
        delivery=delivery, campaigns=campaigns, agency_links=links,
        today=date.today(), window=body.window,
    )
    result = trade_simulate.simulate(termset, head, inputs)
    return {"available": True, "version_id": version_id, **result}


# ----------------------------------------------------------------- attribution

@router.get("/attribution/{rule_id:path}")
def attribution(rule_id: str) -> dict[str, Any]:
    """From a live rule id back to the clause that put it there."""
    from kairos.trade.compile import parse_rule_id

    parsed = parse_rule_id(rule_id)
    if parsed is None:
        return {"trade_rule": False}
    try:
        head = trade_store.load_head(parsed["agreement_id"])
        termset = trade_store.load_termset(
            parsed["agreement_id"], parsed["version_id"])
    except KeyError:
        return {"trade_rule": True, **parsed, "resolved": False}
    instance = next(
        (i for i in termset.get("instances", [])
         if i.get("instance_id") == parsed["instance_id"]),
        None,
    )
    from kairos.trade import taxonomy

    return {
        "trade_rule": True,
        "resolved": instance is not None,
        **parsed,
        "agreement_title": head.get("title"),
        "term": (
            {
                "term_id": instance["term_id"],
                "name_he": taxonomy.get(instance["term_id"]).name_he,
                "params": instance.get("params", {}),
                "citations": instance.get("citations", []),
            }
            if instance else None
        ),
    }
