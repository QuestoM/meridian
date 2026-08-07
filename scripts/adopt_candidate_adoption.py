"""Adopting a candidate: the checks, the escalation, the write and the undo.

Adoption is the one act in this product whose output lands under ``models/``,
which is the whole definition of training in section 4.1 of the specification.
It is therefore company staff only, it lives at a terminal in this repository,
and nothing on any operator surface offers it, links to it or names it. A test
holds that line rather than a comment.

**The rule that decides whether an adoption may land.** A shipped figure may not
move without a proof, an owner decision and a measurement, so:

- An adoption whose measured revenue movement is exactly zero may land. Nothing
  a broadcaster reads changes, so there is nothing to approve.
- An adoption whose measured revenue movement is not zero **stops and
  escalates**. No command-line flag releases it, because a keystroke is not an
  owner decision. What releases it is an approval artifact under
  ``models/releases/owner_approvals/`` that names the exact movement in shekels
  the owner approved. If the measured movement later differs from the approved
  one by so much as a cent, the approval no longer matches and the adoption
  escalates again by itself.

**Every adoption is reversible and the undo is byte-exact.** Before anything is
written, the artifact being replaced is copied whole into the adoption's own
directory. Reverting restores those exact bytes and refuses if the shipped
artifact is not the one this adoption left behind, because restoring over
somebody else's change would silently destroy it.

**The verdict is recorded in the artifact itself.** The adopted artifact carries
an ``adoption`` block in its metadata naming who adopted it, from which
candidate, against which model version, on which measured movement and which
re-score verdict, so a reader holding only the file can see what was decided.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scripts import adopt_candidate_ownership as ownership
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_state import (
    gate_evidence,
    live_version,
    money_state,
    recorded_decision,
    ship_decision,
)
from scripts.adopt_candidate_surface import artifact_surface, dropped_field
from scripts.adopt_candidate_rescore import (
    Paths,
    candidate_files,
    candidate_id,
    load_rescore,
    read_artifact,
    rescore_state,
    sha256_file,
)

ADOPTIONS_FILE = "adoptions.jsonl"
ADOPTIONS_DIR = "adoptions"
APPROVALS_DIR = "owner_approvals"

PREVIOUS_NAME = "previous.json"
ADOPTED_NAME = "adopted.json"
MANIFEST_NAME = "manifest.json"

# The sentence a money escalation prints, from the one table that holds both
# halves of every authored string this piece emits.
ESCALATION = words.ESCALATION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, text: str, paths: Optional[Paths] = None) -> None:
    """One write, and the only place this act touches the filesystem.

    Guarded rather than trusted: the ownership row is checked here, at the line
    that writes, so a later caller cannot reach a path outside it by forgetting
    a check somewhere else.
    """
    if paths is not None:
        ownership.guard(paths.root, path, paths.releases_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def approvals_dir(paths: Paths) -> Path:
    return paths.releases_dir / APPROVALS_DIR


def adoptions_dir(paths: Paths) -> Path:
    return paths.releases_dir / ADOPTIONS_DIR


def adoptions_log(paths: Paths) -> Path:
    return paths.releases_dir / ADOPTIONS_FILE


def adoptions(paths: Optional[Paths] = None) -> list[dict[str, Any]]:
    """Every adoption and revert ever performed, newest first."""
    paths = paths or Paths()
    path = adoptions_log(paths)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return list(reversed(records))


def _append(paths: Paths, record: dict[str, Any]) -> dict[str, Any]:
    path = adoptions_log(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def owner_approval(identifier: str, paths: Optional[Paths] = None) -> Optional[dict[str, Any]]:
    """The owner's recorded approval for one candidate, if there is one."""
    paths = paths or Paths()
    path = approvals_dir(paths) / f"{identifier}.json"
    payload = read_artifact(path) if path.is_file() else {}
    return payload or None


def _check(identifier: str, passed: bool, en: str, he: str, how: str = "",
           **fields: Any) -> dict[str, Any]:
    """One condition answered in both languages, including what would clear it.

    ``how`` is a key into the words table rather than a sentence, because the
    sentence is two sentences and neither belongs in the middle of a condition.
    """
    return {"id": identifier, "passed": bool(passed), "reason_en": en, "reason_he": he,
            **words.pair(words.HOW, how, "how", **fields)}


def preconditions(identifier: str, paths: Optional[Paths] = None,
                  approved_by: str = "", reason: str = "") -> dict[str, Any]:
    """Every condition an adoption must clear, each answered yes or no with why.

    Returned whole and in order rather than raised one at a time, so a steward
    reads the entire distance to a landing in one look instead of discovering it
    one refusal per command.
    """
    paths = paths or Paths()
    known = {candidate_id(path): path for path in candidate_files(paths)}
    path = known.get(identifier)
    version = live_version()
    version_id = str(version.get("id") or "")

    checks = [_check(
        "candidate_exists", path is not None,
        f"The candidate {identifier} is in models/candidates/." if path else
        f"There is no candidate called {identifier}. Known: {', '.join(sorted(known)) or 'none'}.",
        "המועמד נמצא בתיקיית המועמדים." if path else "אין מועמד בשם הזה.")]

    state = rescore_state(paths)
    checks.append(_check(
        "rescore_current", state["state"] == "current",
        "The held-out re-score is current." if state["state"] == "current" else str(state.get("reason_en") or state["state"]),
        "המדידה החוזרת עדכנית." if state["state"] == "current" else str(state.get("reason_he") or ""),
        "rescore"))

    verdict = _stored_verdict(identifier, paths)
    checks.append(_check(
        "not_measured_worse", verdict != "worse",
        "The re-score does not call this candidate worse than the shipped model." if verdict != "worse" else
        "The re-score calls this candidate worse than the shipped model, so it may not be adopted.",
        "המדידה אינה קובעת שהמועמד גרוע מהמודל המשודר." if verdict != "worse" else
        "המדידה קובעת שהמועמד גרוע מהמודל המשודר, ולכן אין להטמיע אותו."))

    money = money_state(identifier)
    checks.append(_check(
        "money_current", money["state"] == "measured",
        "The money this would move is measured and current." if money["state"] == "measured" else str(money.get("reason_en") or ""),
        "הכסף שזה יזיז נמדד והוא עדכני." if money["state"] == "measured" else str(money.get("reason_he") or ""),
        str(money.get("how") or ""), id=identifier))

    decision = ship_decision(identifier, version_id)
    taken = None if decision else recorded_decision(identifier)
    checks.append(_check(
        "ship_decision_recorded", decision is not None,
        f"A ship verdict is recorded for this candidate: {decision['decision_id']}." if decision else
        f"The verdict on record for this candidate is no ship, taken {words.when((taken or {}).get('recorded_at'))}. Adopting it would contradict a decision somebody already took." if taken else
        "No verdict of any kind is recorded for this candidate against the model version on disk.",
        "נרשמה הכרעה להשיק את המועמד הזה." if decision else
        "ההכרעה הרשומה למועמד הזה היא לא להשיק, והטמעה תסתור הכרעה שכבר התקבלה." if taken else
        "לא נרשמה שום הכרעה למועמד הזה מול גרסת המודל שעל הדיסק.",
        "record_verdict", id=identifier))

    checks.append(_check(
        "steward_named", bool(str(approved_by).strip()),
        "The steward taking this decision is named." if approved_by else "Nobody is named as taking this decision.",
        "מי שמכריע נקוב בשם." if approved_by else "לא נקוב מי מכריע.",
        "adopted_by"))
    checks.append(_check(
        "reason_given", bool(str(reason).strip()),
        "The adoption carries its reason." if reason else "The adoption carries no reason, and a verdict with no reason is not a record.",
        "להטמעה יש סיבה." if reason else "להטמעה אין סיבה, והכרעה בלי סיבה אינה רישום.",
        "reason"))

    shipped_payload = read_artifact(paths.shipped)
    candidate_payload = read_artifact(path) if path else {}
    surface = artifact_surface(shipped_payload, candidate_payload)
    dropped = surface["engine_inputs_dropped"]
    checks.append(_check(
        "no_engine_input_dropped", not dropped,
        "The candidate carries every field the engine reads out of the shipped artifact." if not dropped else
        "The candidate drops fields the engine reads: " + ", ".join(dropped_field(item) for item in dropped) + ".",
        "המועמד נושא כל שדה שהמנוע קורא מהקובץ המשודר." if not dropped else
        "המועמד משמיט שדות שהמנוע קורא: " + ", ".join(dropped_field(item) for item in dropped) + ".",
        "rebuild_candidate"))

    money_moves = bool(money.get("moved_fields")) if money["state"] == "measured" else False
    approval = owner_approval(identifier, paths)
    approved_delta = (approval or {}).get("approved_revenue_delta")
    matches = (isinstance(approved_delta, (int, float)) and money["state"] == "measured"
               and round(float(approved_delta), 2) == round(float(money["revenue_delta"]), 2))
    if money_moves:
        checks.append(_check(
            "owner_approval_matches_movement", bool(matches),
            "The owner's recorded approval names this exact movement." if matches else ESCALATION["en"],
            "אישור הבעלים הרשום נוקב בדיוק בתנועה הזו." if matches else ESCALATION["he"],
            "owner_approval", id=identifier))
    else:
        checks.append(_check(
            "no_shipped_figure_moves", money["state"] == "measured",
            "The measured movement is zero, so no shipped figure moves and no owner approval is needed." if money["state"] == "measured" else
            "Whether a shipped figure moves is unknown until the money is measured.",
            "התנועה הנמדדת היא אפס, ולכן שום מספר משודר אינו זז." if money["state"] == "measured" else
            "לא ידוע אם מספר משודר זז עד שהכסף נמדד."))

    # The last write of this act lands on a path this piece's ownership row does
    # not carry. It is a condition rather than a comment, because a rule nobody
    # is stopped by is not a rule, and it is the last check because it is the
    # last thing that happens.
    owned = ownership.state(paths.root, paths.releases_dir)
    checks.append(_check(
        "write_target_is_owned", owned["ruled"],
        f"The ruling on record puts {owned['path']} on this piece's ownership row." if owned["ruled"] else
        f"Adopting writes {owned['path']}, which is absent from {owned['spec_row']}, so it is frozen by absence. Everything up to this point is on the row and has run.",
        "הפסיקה הרשומה מציבה את הקובץ המשודר בשורת הבעלות של החלק הזה." if owned["ruled"] else
        "ההטמעה כותבת אל הקובץ המשודר, שאינו נמצא בשורת הבעלות של החלק הזה, ולכן הוא קפוא בהיעדרו.",
        "ownership_ruling", file=owned["ruling_file"], path=owned["path"]))

    return {
        "candidate_id": identifier,
        "checks": checks,
        "ownership": owned,
        "passed": all(check["passed"] for check in checks),
        "blocked_on": [check["id"] for check in checks if not check["passed"]],
        "money": money,
        "money_moves": money_moves,
        "escalated": bool(money_moves and not matches),
        "artifact_surface": surface,
        # JS-19's second sentence is "read what its gates decided differently",
        # and its target asks for each with its held-out figure. Both are here
        # rather than one screen away, because the verdict is taken here.
        "gate_evidence": gate_evidence(shipped_payload, candidate_payload),
        # The third of the three things JS-19's done condition names, beside the
        # gate deltas and the money. Read from the stored re-score rather than
        # recomputed, because attributing a cell needs the per-break errors and
        # those cost the re-score's ten seconds of data loading.
        "cell_deltas": _stored_cell_deltas(identifier, paths),
        "rescore_verdict": verdict,
        "model_version": version,
        "ship_decision": decision,
    }


def _stored_row(identifier: str, paths: Paths) -> dict[str, Any]:
    stored = load_rescore(paths) or {}
    for row in stored.get("candidates") or []:
        if row.get("id") == identifier:
            return row
    return {}


def _stored_verdict(identifier: str, paths: Paths) -> str:
    return str((_stored_row(identifier, paths).get("verdict") or {}).get("state") or "unknown")


def _stored_cell_deltas(identifier: str, paths: Paths) -> dict[str, Any]:
    """The coefficient delta as the re-score measured it, or nothing at all.

    An empty answer is a real state: a re-score taken before this measurement
    existed carries no cell rows, and the render says so rather than showing an
    empty table that reads as "nothing moved".
    """
    return _stored_row(identifier, paths).get("cell_deltas") or {}


def adopt(identifier: str, *, adopted_by: str, reason: str, release_note_he: str = "",
          paths: Optional[Paths] = None, perform: bool = False) -> dict[str, Any]:
    """Plan an adoption, and perform it only when every check has passed.

    Planning is the default and it writes nothing at all. Performing is opt in,
    and it still refuses on the first failed check, so the safe path is the one
    that happens when a flag is forgotten.
    """
    paths = paths or Paths()
    state = preconditions(identifier, paths, approved_by=adopted_by, reason=reason)
    plan = {
        "adoption_id": None,
        "candidate_id": identifier,
        "performed": False,
        "planned_at": _now(),
        **state,
    }
    if not state["passed"]:
        plan["outcome"] = "escalated" if state["escalated"] else "refused"
        return plan
    if not perform:
        plan["outcome"] = "ready"
        return plan

    source = paths.candidates_dir / f"tv_break_coefficients_{identifier}.json"
    adoption_id = f"ad-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    directory = adoptions_dir(paths) / adoption_id
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.shipped, directory / PREVIOUS_NAME)

    payload = read_artifact(source)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    stamp = {
        "adoption_id": adoption_id,
        "adopted_at": _now(),
        "adopted_by": str(adopted_by).strip(),
        "from_candidate": identifier,
        "from_file": source.relative_to(paths.root).as_posix(),
        "superseded_version_id": str((state["model_version"] or {}).get("id") or ""),
        "superseded_sha256": sha256_file(paths.shipped),
        "ship_decision_id": (state["ship_decision"] or {}).get("decision_id"),
        "reason": str(reason).strip(),
        "release_note_he": str(release_note_he).strip(),
        "measured_revenue_delta": state["money"].get("revenue_delta"),
        "measured_revenue_scope": state["money"].get("scope"),
        "rescore_verdict": state["rescore_verdict"],
        # JS-19's done condition, verbatim: the gate deltas, the coefficient
        # deltas and the measured money movement, recorded against a new model
        # version. All three are in this stamp, so a reader holding only the
        # adopted file can see what was decided and on what.
        "gate_deltas": (state["gate_evidence"] or {}).get("verdicts") or [],
        "coefficient_deltas": (state["cell_deltas"] or {}).get("summary") or {},
        "revert_with": f"python scripts/adopt_candidate.py revert {adoption_id}",
    }
    metadata["adoption"] = stamp
    payload["metadata"] = metadata
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    _write_atomic(directory / ADOPTED_NAME, text, paths)
    _write_atomic(paths.shipped, text, paths)

    record = {
        "adoption_id": adoption_id,
        "action": "adopted",
        "recorded_at": _now(),
        "candidate_id": identifier,
        "adopted_by": stamp["adopted_by"],
        "reason": stamp["reason"],
        "release_note_he": stamp["release_note_he"],
        "superseded_version_id": stamp["superseded_version_id"],
        "superseded_sha256": stamp["superseded_sha256"],
        "adopted_sha256": sha256_file(paths.shipped),
        "measured_revenue_delta": stamp["measured_revenue_delta"],
        "rescore_verdict": state["rescore_verdict"],
        "coefficient_deltas": stamp["coefficient_deltas"],
        "ship_decision_id": stamp["ship_decision_id"],
        "directory": directory.relative_to(paths.root).as_posix(),
    }
    _write_atomic(directory / MANIFEST_NAME, json.dumps(record, ensure_ascii=False, indent=1) + "\n", paths)
    _append(paths, record)
    plan.update({"adoption_id": adoption_id, "performed": True, "outcome": "adopted",
                 "record": record})
    return plan


def revert(adoption_id: str, *, reverted_by: str, reason: str,
           paths: Optional[Paths] = None, perform: bool = False) -> dict[str, Any]:
    """Put back the exact bytes an adoption replaced, or refuse and say why."""
    paths = paths or Paths()
    record = next((item for item in adoptions(paths)
                   if item.get("adoption_id") == adoption_id and item.get("action") == "adopted"), None)
    if record is None:
        return {"outcome": "refused", "adoption_id": adoption_id, "performed": False,
                "reason_en": f"There is no adoption called {adoption_id}.",
                "reason_he": "אין הטמעה בשם הזה."}
    if any(item.get("adoption_id") == adoption_id and item.get("action") == "reverted"
           for item in adoptions(paths)):
        return {"outcome": "refused", "adoption_id": adoption_id, "performed": False,
                "reason_en": "That adoption has already been reverted.",
                "reason_he": "ההטמעה הזו כבר בוטלה."}
    previous = paths.root / str(record.get("directory") or "") / PREVIOUS_NAME
    if not previous.is_file():
        return {"outcome": "refused", "adoption_id": adoption_id, "performed": False,
                "reason_en": f"The artifact this adoption replaced is not on disk at {previous}.",
                "reason_he": "הקובץ שההטמעה הזו החליפה אינו על הדיסק."}
    current = sha256_file(paths.shipped)
    if current != record.get("adopted_sha256"):
        return {"outcome": "refused", "adoption_id": adoption_id, "performed": False,
                "reason_en": "The shipped artifact is not the one this adoption left behind, so reverting would destroy a later change. Nothing was written.",
                "reason_he": "הקובץ המשודר אינו זה שההטמעה הזו הותירה, ולכן ביטול היה מוחק שינוי מאוחר יותר. דבר לא נכתב."}
    if not perform:
        return {"outcome": "ready", "adoption_id": adoption_id, "performed": False,
                "restores_sha256": record.get("superseded_sha256"),
                "reason_en": "Ready to revert. Nothing has been written.",
                "reason_he": "מוכן לביטול. דבר לא נכתב."}
    _write_atomic(paths.shipped, previous.read_text(encoding="utf-8"), paths)
    undo = {
        "adoption_id": adoption_id,
        "action": "reverted",
        "recorded_at": _now(),
        "candidate_id": record.get("candidate_id"),
        "reverted_by": str(reverted_by).strip() or "unknown (login is not set up)",
        "reason": str(reason).strip(),
        "restored_sha256": sha256_file(paths.shipped),
        "expected_sha256": record.get("superseded_sha256"),
    }
    undo["restored_exactly"] = undo["restored_sha256"] == undo["expected_sha256"]
    _append(paths, undo)
    return {"outcome": "reverted", "adoption_id": adoption_id, "performed": True, "record": undo}
