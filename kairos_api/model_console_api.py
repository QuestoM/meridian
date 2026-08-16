"""The model console: the company side of the line, and the only side it has.

Eleven paths, one wall. Every one of them requires ``affiliation = company`` on
the read as well as the write, because the whole surface is training content
and section 4.5 of the rebuild specification puts training content behind
affiliation, not behind role. Role still decides the write: a company viewer
reads every verdict and records none.

The surface answers the six questions section 4.4 gives the model dashboard,
and each answer is a route:

- ``GET /api/model/console``, the header: which version, what it decided, what is open.
- ``GET /api/model/gates``, what each gate decided, on what basis.
- ``GET /api/model/coverage``, how much contrast the data carries, and what is blocked.
- ``GET /api/model/drift``, what drifted, and why there is no series across versions yet.
- ``GET /api/model/candidates`` and ``.../{id}``, what a train would change.
- ``POST /api/model/candidates/{id}/measure``, the money it would move, measured.
- ``GET /api/model/provenance``, fingerprints, seeds, method, and the flags nobody records.
- ``GET`` and ``POST /api/model/training``, the runs, and starting one.
- ``GET`` and ``POST /api/model/versions``, the versions recorded, and recording one.
- ``POST /api/model/decisions``, a ship or no-ship verdict against a named version.

Two routes that predate this module are walled by the same wall, from their own
files: ``GET /api/impact`` and ``GET /api/model/audience``. They are two of the
four open reads section 4.5 names, they are P7's two, and until this wave every
account on the product fetched the full coefficient metadata on every page load.

The console renders the audience-model activation switch as **state only**. The
switch is a run-side configuration act whose surface is Rules, so throwing it
from here would be exactly the blur the training-versus-runs rule exists to
prevent.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos.optimize.inventory import InventoryInputError, load_inventory
from kairos_api import model_console_artifacts as artifacts
from kairos_api import model_console_candidates as candidates
from kairos_api import model_console_api_payloads as payloads
from kairos_api import model_console_coverage as coverage
from kairos_api import model_console_gates as gates
from kairos_api import model_version_store as store
from kairos_api.affiliation_wall import Wall, session_for

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model", tags=["model"])

# The model surface's wall, declared once. The refusal is the shipped
# company-surface constant rather than a new string, so the words a person
# reads before the click and the words the server sends cannot drift, and the
# frontend's own copy of them stays the one that is already pinned by a test.
MODEL_WALL = Wall(company_only=True)

# The measurements in flight, so a second request does not start a second run of
# a hundred-second job. Single process by the same assumption three other
# modules in this package already state about themselves.
_RUNNING: dict[str, dict[str, Any]] = {}
_RUNNING_LOCK = threading.Lock()


def _actor(request: Optional[Request]) -> str:
    session = session_for(request)
    return str(session["username"]) if session else ""


def _in_flight() -> dict[str, dict[str, Any]]:
    """The register, taken BEFORE the store is read and never after it.

    The order is the whole point. A measurement ends by writing its record and
    then clearing its own entry, so a reader that takes the store first and the
    register second can be overtaken between the two and answer with the record
    that was there before the write and with nothing in flight. That answer is a
    superseded figure with nothing on it to say a newer one exists, and it is
    the answer that makes the screen stop watching. Taken in this order the same
    race answers measuring once more, which costs one further read a second and
    a half later and can never publish a superseded figure as a settled one.

    Measured on the running instance on 2026-08-04: a real measurement ended at
    15:22:24.511, the shelf carried its money 0.21 s later with no press behind
    it, and the record on the screen at that moment was the one from 15:09:32.
    """
    with _RUNNING_LOCK:
        return {key: dict(value) for key, value in _RUNNING.items()}


@router.get("/console")
@MODEL_WALL.guard()
def console(request: Request) -> dict[str, Any]:
    """The console header: the version in force, the gate counts, what is open.

    Stamped with ``can_edit``, so the header answers "who may run a rebuild"
    before anyone reaches the training panel: the same wall that would refuse
    ``POST /api/model/training`` writes its verdict here first.
    """
    version = payloads.current_version()
    ledger = gates.ledger()
    decision = (store.latest_decision(str(version.get("id") or ""))
                if version.get("available") else None)
    window = coverage.training_window()
    measurements = store.measurements()
    result = {
        "model_version": version,
        "gate_counts": ledger["counts"],
        "gate_states": ledger["states"],
        "gate_total": ledger["total"],
        "layers": len(ledger["layers"]),
        "drift": payloads.drift_block(),
        "window": window,
        "activation": payloads.activation_mirror(),
        "candidates": {
            "count": len(candidates.candidate_paths()),
            "measured": sum(1 for path in candidates.candidate_paths()
                            if (measurements.get(candidates.candidate_id(path)) or {}).get("fingerprint")
                            == candidates.measurement_fingerprint(path)),
        },
        "latest_decision": decision,
        "decisions_recorded": len(store.decisions()),
    }
    return MODEL_WALL.stamp(result, request)


@router.get("/gates")
@MODEL_WALL.guard()
def gate_ledger() -> dict[str, Any]:
    """Every gate with its state, its basis and the artifact's own reason."""
    return {"model_version": payloads.current_version(), **gates.ledger()}


@router.get("/coverage")
@MODEL_WALL.guard()
def contrast() -> dict[str, Any]:
    """How much contrast the window carries, and the register of what is blocked."""
    return {"model_version": payloads.current_version(), **coverage.coverage(gates.ledger()["gates"])}


@router.get("/drift")
@MODEL_WALL.guard()
def drift() -> dict[str, Any]:
    """The level-drift measurement, and the honest state of the cross-version series."""
    recorded = [record for record in store.versions() if isinstance(record.get("drift"), dict)]
    series = [
        {"model_version_id": record.get("id"), "name": record.get("name"),
         "first_seen_at": record.get("first_seen_at"),
         "drift_per_week": (record.get("drift") or {}).get("drift_per_week"),
         "binding": (record.get("drift") or {}).get("binding")}
        for record in recorded
        if isinstance((record.get("drift") or {}).get("drift_per_week"), (int, float))
    ]
    return {
        "model_version": payloads.current_version(),
        "current": payloads.drift_block(),
        "series": series,
        "series_state": "available" if len(series) > 1 else "one_point",
        "series_reason_en": "A drift series needs at least two recorded model versions. Training overwrote the artifact in place before this store existed, so no earlier version was kept; every version seen from now on is recorded with its drift.",
        "series_reason_he": "סדרת סחיפה דורשת לפחות שתי גרסאות מודל רשומות. האימון דרס את הקובץ במקום לפני שהמאגר הזה היה קיים, ולכן לא נשמרה גרסה קודמת; כל גרסה שתיראה מכאן ואילך נרשמת עם הסחיפה שלה.",
    }


@router.get("/candidates")
@MODEL_WALL.guard()
def candidate_list() -> dict[str, Any]:
    """Every candidate on the shelf, with its gate deltas and its money state."""
    running = _in_flight()
    metadata = artifacts.retention_metadata()
    measurements = store.measurements()
    rows = [candidates.summary_row(path, metadata, measurements.get(candidates.candidate_id(path)))
            for path in candidates.candidate_paths()]
    for row in rows:
        if row["id"] in running:
            row["money"] = {"state": "measuring", **running[row["id"]]}
    return {
        "model_version": payloads.current_version(),
        "candidates": rows,
        "directory": candidates.CANDIDATE_DIR.relative_to(candidates.ROOT).as_posix(),
        "measurement_cost_en": "Measuring one candidate computes the whole weekly plan twice.",
        "measurement_cost_he": "מדידת מועמד אחד מחשבת את התוכנית השבועית פעמיים.",
    }


@router.get("/candidates/{candidate_id}")
@MODEL_WALL.guard()
def candidate_detail(candidate_id: str) -> dict[str, Any]:
    """One candidate in full, with the decision already taken about it, if any."""
    path = candidates.candidate_path(candidate_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no candidate artifact called {candidate_id}")
    running = _in_flight()
    version = payloads.current_version()
    row = candidates.summary_row(path, artifacts.retention_metadata(),
                                 store.measurement(candidate_id))
    if candidate_id in running:
        row["money"] = {"state": "measuring", **running[candidate_id]}
    return {
        "model_version": version,
        "candidate": row,
        "decision": store.latest_decision(str(version.get("id") or ""), "candidate", candidate_id),
    }


@router.post("/candidates/{candidate_id}/measure")
@MODEL_WALL.guard()
def measure_candidate(candidate_id: str) -> dict[str, Any]:
    """Start the money measurement for one candidate, in the background.

    Nothing is written to any artifact: the plan is computed in memory twice and
    only the resulting figures are stored.
    """
    path = candidates.candidate_path(candidate_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no candidate artifact called {candidate_id}")
    try:
        # Refuse synchronously, before the in-flight register or background
        # thread exists. The operator gets an actionable response and no failed
        # job can later look like a measurement that merely disappeared.
        load_inventory(require_usable=True)
    except InventoryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _RUNNING_LOCK:
        if candidate_id in _RUNNING:
            return {"state": "measuring", **_RUNNING[candidate_id]}
        _RUNNING[candidate_id] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "past_durations_seconds": payloads.past_durations(),
        }
        started = dict(_RUNNING[candidate_id])

    def _run() -> None:
        try:
            record = candidates.measure_money_movement(candidate_id)
            store.save_measurement(record)
        except Exception:
            logger.exception("candidate measurement failed for %s", candidate_id)
        finally:
            with _RUNNING_LOCK:
                _RUNNING.pop(candidate_id, None)

    threading.Thread(target=_run, name=f"kairos-candidate-{candidate_id}", daemon=True).start()
    return {"state": "measuring", **started}


@router.get("/provenance")
@MODEL_WALL.guard()
def provenance() -> dict[str, Any]:
    """What produced these artifacts, and the one thing nothing records."""
    metadata = artifacts.retention_metadata()
    audience = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    return {
        "model_version": payloads.current_version(),
        "seeds": {
            "interval_seed": metadata.get("interval_seed"),
            "bootstrap_B": metadata.get("bootstrap_B"),
            "placebo_seed": (metadata.get("placebo_correction") or {}).get("seed")
            if isinstance(metadata.get("placebo_correction"), dict) else None,
        },
        "method": {
            "pooling_method": metadata.get("pooling_method"),
            "interval_method": metadata.get("interval_method"),
            "detrend_baseline_mode": metadata.get("detrend_baseline_mode"),
            "before_after_window_minutes": metadata.get("before_after_window_minutes"),
            "audience_base_kind": (audience.get("base") or {}).get("kind")
            if isinstance(audience.get("base"), dict) else None,
        },
        "gate_override_flags": artifacts.gate_override_flags(),
        "override_gap_en": "None of these flags is written into the artifact, so a forced gate and a gate that activated on its own measurement are indistinguishable after the fact.",
        "override_gap_he": "אף אחד מהדגלים האלה אינו נכתב לקובץ, ולכן שער שנכפה ושער שהופעל מהמדידה שלו עצמו אינם ניתנים להבחנה בדיעבד.",
        "actor_recorded": False,
        "actor_gap_en": "No artifact records who ran the training. Decisions recorded here do carry their actor.",
        "actor_gap_he": "אף קובץ אינו רושם מי הריץ את האימון. ההכרעות הנרשמות כאן כן נושאות את מי שרשם אותן.",
        "training_commands": payloads.shipped_training_commands(),
    }


@router.get("/versions")
@MODEL_WALL.guard()
def version_log() -> dict[str, Any]:
    """The versions this console has recorded, and every decision taken about them."""
    version = payloads.current_version()
    return {
        "model_version": version,
        "observed": store.versions(),
        "decisions": store.decisions(),
        "store_dir": store.store_dir().as_posix(),
    }


@router.post("/versions")
@MODEL_WALL.guard()
def record_version() -> dict[str, Any]:
    """Record the version now on disk, with the drift it measured.

    Idempotent: a version already recorded is returned unchanged. This is the
    act that builds the cross-version drift series, and it is an act rather
    than a side effect of a read so that opening the console changes nothing.
    """
    version = artifacts.model_version()
    if not version.get("available"):
        raise HTTPException(status_code=400,
                            detail="אין בדיסק גרסת מודל לרישום")
    record = store.observe({**version, "drift": payloads.drift_block()})
    return {"recorded": record, "model_version": payloads.current_version()}


@router.get("/training")
@MODEL_WALL.guard()
def training(request: Request) -> dict[str, Any]:
    """What can be trained, what is training, and every run this console started.

    Stamped with ``can_edit``: the Train button reads its own permission
    before the click rather than after the 403, the same contract
    ``Wall.stamp`` already gives the pricing and rules surfaces.
    """
    from kairos_api import model_console_training

    result = {"model_version": payloads.current_version(), **model_console_training.payload()}
    return MODEL_WALL.stamp(result, request)


class TrainingRequest(BaseModel):
    """One training run, into the releases store and never over the shipped file."""

    artifact: str = Field(description="retention or audience")
    flags: dict[str, str] = Field(default_factory=dict,
                                  description="gate overrides, each force-on or force-off")


@router.post("/training")
@MODEL_WALL.guard()
def start_training(payload: TrainingRequest, request: Request) -> dict[str, Any]:
    """Start a training run. It cannot write the shipped artifact, by construction."""
    from kairos_api import model_console_training

    try:
        return model_console_training.start(payload.artifact, actor=_actor(request),
                                            flags=payload.flags)
    except model_console_training.TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class DecisionRequest(BaseModel):
    """A ship or no-ship verdict against the model version now on disk."""

    decision: str = Field(description="shipped or not_shipped")
    subject: str = Field(default="current", description="current or candidate")
    candidate_id: Optional[str] = None
    reason: str = ""
    release_note_he: str = ""
    release_note_en: str = ""
    money_direction: str = "unknown"


@router.post("/decisions")
@MODEL_WALL.guard()
def record_decision(payload: DecisionRequest, request: Request) -> dict[str, Any]:
    """Record a verdict. Never copies an artifact, by design and by test.

    The route declares its own ``Request`` so the verdict carries the account
    that took it. The wall finds that parameter by type and passes it straight
    through, so declaring it costs the route nothing.
    """
    version = artifacts.model_version()
    evidence = payloads.complete_decision_evidence(payload.subject, payload.candidate_id)
    try:
        record = store.record_decision(
            model_version=version,
            decision=payload.decision,
            subject=payload.subject,
            candidate_id=payload.candidate_id,
            reason=payload.reason,
            release_note_he=payload.release_note_he,
            release_note_en=payload.release_note_en,
            money_direction=payload.money_direction,
            actor=_actor(request),
            evidence=evidence,
        )
    except store.ModelVersionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record
