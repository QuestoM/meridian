"""The registry view: every artifact on one page, with what is known about it.

A model steward asking "is this candidate genuinely better than what is live"
needs four things joined, and before this module they lived in four places that
never met: the artifact on disk, the held-out re-score, the money the plan would
move, and the verdict somebody already recorded about it.

This joins them and ranks them. Each row carries its own state rather than a
score with the doubt filed off: a re-score that is stale says so, a money figure
that has never been measured says so, and a candidate that predicts exactly what
another candidate predicts is named as a duplicate rather than counted twice.

Nothing here computes a figure. It reads what was measured, states how old it
is, and says what the next act would be. That separation is deliberate: the
expensive measurements each have their own command, so opening the registry
never quietly starts a hundred seconds of optimizer.

Every line this join is rendered as lives in ``adopt_candidate_render.py``,
split out under the naming rule of section 8.2 when this file reached the
450-line cap. ``render`` and ``render_checks`` are re-exported here because they
are the names three callers and the tests already use, and moving a file should
not move a public surface.
"""

from __future__ import annotations

from typing import Any, Optional

from scripts import adopt_candidate_gates as gates
from scripts import adopt_candidate_history as history
from scripts import adopt_candidate_note as note
from scripts import adopt_candidate_origin as origin
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_adoption import adoptions, owner_approval, preconditions
from scripts.adopt_candidate_baselines import standing_finding
from scripts.adopt_candidate_render import render, render_checks  # noqa: F401
from scripts.adopt_candidate_state import gate_evidence, money_state
from scripts.adopt_candidate_rescore import (
    Paths,
    candidate_files,
    candidate_id,
    load_rescore,
    read_artifact,
    rescore_state,
)


def _live_version() -> dict[str, Any]:
    from kairos_api import model_console_artifacts as artifacts

    return artifacts.model_version()


def registry(paths: Optional[Paths] = None) -> dict[str, Any]:
    """Every candidate joined to its measurements, its verdict and its next act."""
    paths = paths or Paths()
    stored = load_rescore(paths) or {}
    scores = {row.get("id"): row for row in stored.get("candidates") or []}
    version = _live_version()
    # One reader for the decision log. This join used to walk it itself for the
    # newest verdict per candidate while the history reading walked it again,
    # and two readers of one append-only file is how two surfaces of one piece
    # come to disagree about what was decided. It also dropped every record that
    # was not about a candidate, which on this tree is a verdict on the live
    # model, and every record but the newest, which on this tree is a second
    # refusal of one candidate for a different stated reason.
    log = history.decision_log(
        [candidate_id(path) for path in candidate_files(paths)],
        version_id=str(version.get("id") or ""),
        version_name=str(version.get("name") or ""))
    performed = {record.get("candidate_id"): record for record in adoptions(paths)
                 if record.get("action") == "adopted"}
    reverted = {record.get("adoption_id") for record in adoptions(paths)
                if record.get("action") == "reverted"}
    basis_rows = {row.get("id"): row
                  for row in ((stored.get("fit_basis") or {}).get("rows") or [])}
    # Read once, outside the loop, because every candidate's gate reading is a
    # comparison against this same file.
    shipped_payload = read_artifact(paths.shipped)
    shipped_metadata = shipped_payload.get("metadata") if isinstance(
        shipped_payload.get("metadata"), dict) else {}

    rows = []
    for path in candidate_files(paths):
        identifier = candidate_id(path)
        payload = read_artifact(path)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        score = scores.get(identifier) or {}
        # The same function the adoption act and the verdict act read, rather
        # than a second one that answered the stale case differently.
        money = money_state(identifier)
        taken = history.history_for(log, identifier)
        adoption = performed.get(identifier)
        rows.append({
            "id": identifier,
            "file": path.relative_to(paths.root).as_posix(),
            "bytes": path.stat().st_size,
            # The bytes this row was measured on. A reader holding this payload
            # away from the tree cannot otherwise tell whether it describes the
            # artifact the server is serving today or one that has been rebuilt
            # since, and that is the difference between a figure and a guess.
            "sha256": score.get("sha256"),
            "computed_at": metadata.get("computed_at"),
            # What it was built for and what data it read. Read from the
            # artifact on disk rather than from the stored score, so it is
            # about the file that is there now and can never be the stale half
            # of a row whose other half is current.
            "origin": origin.origin_row(identifier, metadata, root=paths.root,
                                        shipped_metadata=shipped_metadata),
            "breaks_fitted_on": metadata.get("total_breaks_measured"),
            "rmse": score.get("rmse"),
            "rmse_delta": (score.get("paired") or {}).get("rmse_delta"),
            "paired_statistic": (score.get("paired") or {}).get("paired_statistic"),
            # The bar the statistic is read against. It was measured on every
            # row and dropped by this join, so the table printed a statistic
            # with nothing to compare it with.
            "paired_bar": (score.get("paired") or {}).get("paired_bar"),
            "fold_dispersion": (score.get("paired") or {}).get("fold_dispersion"),
            "verdict": (score.get("verdict") or {}).get("state", "unknown"),
            "verdict_en": (score.get("verdict") or {}).get("en"),
            "verdict_he": (score.get("verdict") or {}).get("he"),
            # The rule that produced the verdict word, with the measurement that
            # ran through it. It was on the row in the re-score and dropped by
            # this join, so a reader of the published payload got a verdict and a
            # statistic with nothing on the row saying what decided them.
            "rule_en": (score.get("verdict") or {}).get("rule_en"),
            "rule_he": (score.get("verdict") or {}).get("rule_he"),
            "duplicate_of": score.get("duplicate_of") or [],
            # What this artifact's own producer recorded about adopting it. Not
            # a figure this table ranks: it is the artifact's own split under
            # its own fit. On this tree the row the table ranks first is a row
            # whose producer advised against adopting it, and the join dropped
            # that entirely.
            "self_reported": score.get("self_reported"),
            # This row's own share of the evaluation, so a screen can mark the
            # row rather than only print a paragraph naming it.
            "fit_basis": basis_rows.get(identifier),
            # The coefficient delta JS-19 names beside the gate deltas and the
            # money. The summary rather than the 36 rows, because this row is a
            # line in a comparison across five candidates; the rows are the
            # diff command.
            "cell_delta": (score.get("cell_deltas") or {}).get("summary"),
            # The first of the three things JS-19's done condition names, and
            # the one this join dropped: what its gates decided differently,
            # with the amount each side decided that on. It lived only inside
            # the adoption checks, which is the last command a steward runs,
            # while the sequence reads the gates before it reads the money.
            "gates": gates.gate_summary(gate_evidence(shipped_payload, payload)),
            "money": money,
            "decisions": taken["count"],
            "latest_decision": taken["rows"][0] if taken["rows"] else None,
            # Every verdict ever recorded on this artifact rather than the newest
            # one, which is the second half of JS-19's done condition: the
            # verdict is stored and a later reader can see what was tried.
            "history": taken,
            # Whether the verdict on record was taken on the common-basis
            # comparison or on the artifacts' own self-reported figures. On this
            # tree every recorded verdict predates the comparison, and a screen
            # that showed only "no ship" would hide what the no ship rests on.
            "decision_on_rescore": bool(taken["rows"] and taken["rows"][0]["on_rescore"]),
            "owner_approval": owner_approval(identifier, paths) is not None,
            "adopted": bool(adoption) and adoption.get("adoption_id") not in reverted,
            "adoption_id": (adoption or {}).get("adoption_id"),
            "next_act": _next_act(identifier, score, money, taken["rows"]),
        })
    rows.sort(key=lambda row: (row["rmse"] is None, row["rmse"] if row["rmse"] is not None else 0.0))
    return {
        "live_version": version,
        # What the other side of the wall reads about that version. The release
        # note is the one training-authored sentence section 4.6 lets across,
        # and whether an operator reads one is a measurement about this act
        # rather than a property of the store.
        "operator_reads": note.operator_reads(),
        # The whole log, grouped by what each record is about, so a reader can
        # account for every record rather than for the five a shelf shows.
        "decision_log": log,
        "rescore_state": rescore_state(paths, stored or None),
        "evaluation": stored.get("evaluation"),
        "limit": stored.get("limit"),
        # Which rows the limit sentence is true of, measured. The sentence is
        # selected from this, so carrying one without the other would leave a
        # reader with a claim about named rows and no way to see the rows.
        "fit_basis": stored.get("fit_basis"),
        "baselines": stored.get("baselines") or [],
        # The live artifact is a row in this comparison like any other, and its
        # origin is read from disk beside every candidate's rather than out of
        # the stored score, for the same reason.
        "shipped": dict(stored.get("shipped") or {},
                        origin=origin.origin_row("shipped", shipped_metadata,
                                                 root=paths.root)),
        "cell_structure": stored.get("cell_structure"),
        # The one finding on this surface that no candidate answers, sized
        # against the choice the candidates offer, so it is readable as the
        # larger question rather than as a footnote under the baselines.
        "structure_finding": standing_finding(stored.get("cell_structure") or {}, rows),
        "duplicate_groups": stored.get("duplicate_groups") or [],
        "candidates": rows,
        "adoptions": adoptions(paths),
    }


def _next_act(identifier: str, score: dict[str, Any], money: dict[str, Any],
              decisions: list[dict[str, Any]]) -> dict[str, str]:
    """One sentence naming what would move this candidate forward, and the command.

    Never a suggestion to adopt. Adoption runs its own checks and the registry
    would be guessing at their outcome, so the furthest this goes is pointing at
    the command that reports them.
    """
    if not score:
        return words.next_act("rescore")
    if money.get("state") != "measured":
        return words.next_act("measure", id=identifier)
    if not decisions:
        return words.next_act("decide", id=identifier)
    # Read off the row rather than re-derived from a raw record. These are the
    # history rows now, and the evidence a raw record carries is not on them.
    if not decisions[0].get("on_rescore"):
        return words.next_act("redecide", id=identifier)
    return words.next_act("checks", id=identifier)


def checks_for(identifier: str, paths: Optional[Paths] = None, *, adopted_by: str = "",
               reason: str = "") -> dict[str, Any]:
    """The precondition report for one candidate, without planning a write."""
    state = preconditions(identifier, paths or Paths(), approved_by=adopted_by, reason=reason)
    state["outcome"] = ("ready" if state["passed"] else
                        "escalated" if state["escalated"] else "refused")
    return with_origin(state, paths)


def with_origin(state: dict[str, Any], paths: Optional[Paths] = None) -> dict[str, Any]:
    """Join what a candidate was built for onto a checks or an adoption payload.

    Joined here rather than computed inside the act, which is this module's whole
    role: the act answers conditions and this one joins what is already known
    about an artifact onto them. Both callers pass through it, so the checks a
    steward reads and the plan an adoption prints carry the same line rather than
    one of them carrying it.
    """
    paths = paths or Paths()
    identifier = str(state.get("candidate_id") or "")
    known = {candidate_id(path): path for path in candidate_files(paths)}
    path = known.get(identifier)
    metadata = (read_artifact(path).get("metadata") if path else None) or {}
    shipped = read_artifact(paths.shipped).get("metadata") or {}
    return dict(state, origin=origin.origin_row(identifier, metadata, root=paths.root,
                                                shipped_metadata=shipped))
