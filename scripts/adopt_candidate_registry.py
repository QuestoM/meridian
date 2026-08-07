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
"""

from __future__ import annotations

from typing import Any, Optional

from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_adoption import adoptions, owner_approval, preconditions
from scripts.adopt_candidate_state import decision_rests_on_rescore, moved_inputs
from scripts.adopt_candidate_rescore import (
    Paths,
    candidate_files,
    candidate_id,
    load_rescore,
    read_artifact,
    rescore_state,
)

# What each verdict looks like at a terminal. Words, never a colour and never a
# mark, because a steward reading this over a session log gets the same reading.
VERDICT_TAGS = {
    "identical": "identical",
    "better": "better",
    "worse": "worse",
    "not_distinguishable": "no difference",
    "unknown": "not re-scored",
}

MONEY_TAGS = {
    "measured": "measured",
    "stale": "stale",
    "not_measured": "not measured",
}

# The verdict somebody already recorded, which is JS-19's whole done condition.
# A bare count of decisions cannot say which way one went, and four of the five
# candidates on this tree carry a no-ship that a count renders as "1".
DECISION_TAGS = {
    "shipped": "ship",
    "not_shipped": "no ship",
}


def _live_version() -> dict[str, Any]:
    from kairos_api import model_console_artifacts as artifacts

    return artifacts.model_version()


def _decisions_by_candidate() -> dict[str, list[dict[str, Any]]]:
    from kairos_api import model_version_store as store

    out: dict[str, list[dict[str, Any]]] = {}
    for record in store.decisions():
        if record.get("subject") != "candidate":
            continue
        out.setdefault(str(record.get("candidate_id") or ""), []).append(record)
    return out


def _money_state(identifier: str) -> dict[str, Any]:
    from kairos_api import model_console_candidates as console
    from kairos_api import model_version_store as store

    stored = store.measurement(identifier)
    path = console.candidate_path(identifier)
    if stored is None or path is None:
        return {"state": "not_measured", "revenue_delta": None}
    current = str(stored.get("fingerprint") or "") == console.measurement_fingerprint(path)
    delta = (stored.get("operator_channel_delta") or {}).get("revenue_delta")
    return {
        "state": "measured" if current else "stale",
        "revenue_delta": delta,
        "measured_at": stored.get("measured_at"),
        "changed": [] if current else console.changed_inputs(path, stored),
        "scope_rows": ((stored.get("scope") or {}).get("operator_channel") or {}).get("rows"),
    }


def registry(paths: Optional[Paths] = None) -> dict[str, Any]:
    """Every candidate joined to its measurements, its verdict and its next act."""
    paths = paths or Paths()
    stored = load_rescore(paths) or {}
    scores = {row.get("id"): row for row in stored.get("candidates") or []}
    decisions = _decisions_by_candidate()
    performed = {record.get("candidate_id"): record for record in adoptions(paths)
                 if record.get("action") == "adopted"}
    reverted = {record.get("adoption_id") for record in adoptions(paths)
                if record.get("action") == "reverted"}

    rows = []
    for path in candidate_files(paths):
        identifier = candidate_id(path)
        payload = read_artifact(path)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        score = scores.get(identifier) or {}
        money = _money_state(identifier)
        taken = decisions.get(identifier) or []
        adoption = performed.get(identifier)
        rows.append({
            "id": identifier,
            "file": path.relative_to(paths.root).as_posix(),
            "bytes": path.stat().st_size,
            "computed_at": metadata.get("computed_at"),
            "breaks_fitted_on": metadata.get("total_breaks_measured"),
            "rmse": score.get("rmse"),
            "rmse_delta": (score.get("paired") or {}).get("rmse_delta"),
            "paired_statistic": (score.get("paired") or {}).get("paired_statistic"),
            "fold_dispersion": (score.get("paired") or {}).get("fold_dispersion"),
            "verdict": (score.get("verdict") or {}).get("state", "unknown"),
            "verdict_en": (score.get("verdict") or {}).get("en"),
            "verdict_he": (score.get("verdict") or {}).get("he"),
            "duplicate_of": score.get("duplicate_of") or [],
            "money": money,
            "decisions": len(taken),
            "latest_decision": taken[0] if taken else None,
            # Whether the verdict on record was taken on the common-basis
            # comparison or on the artifacts' own self-reported figures. On this
            # tree every recorded verdict predates the comparison, and a screen
            # that showed only "no ship" would hide what the no ship rests on.
            "decision_on_rescore": decision_rests_on_rescore(taken[0] if taken else None),
            "owner_approval": owner_approval(identifier, paths) is not None,
            "adopted": bool(adoption) and adoption.get("adoption_id") not in reverted,
            "adoption_id": (adoption or {}).get("adoption_id"),
            "next_act": _next_act(identifier, score, money, taken),
        })
    rows.sort(key=lambda row: (row["rmse"] is None, row["rmse"] if row["rmse"] is not None else 0.0))
    return {
        "live_version": _live_version(),
        "rescore_state": rescore_state(paths, stored or None),
        "evaluation": stored.get("evaluation"),
        "limit": stored.get("limit"),
        "baselines": stored.get("baselines") or [],
        "shipped": stored.get("shipped"),
        "cell_structure": stored.get("cell_structure"),
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
    if not decision_rests_on_rescore(decisions[0]):
        return words.next_act("redecide", id=identifier)
    return words.next_act("checks", id=identifier)


def _money_cell(money: dict[str, Any]) -> str:
    delta = money.get("revenue_delta")
    tag = MONEY_TAGS.get(str(money.get("state")), str(money.get("state")))
    if money.get("state") == "measured" and isinstance(delta, (int, float)):
        return f"{delta:+,.2f}"
    if money.get("state") == "stale" and isinstance(delta, (int, float)):
        return f"stale ({delta:+,.0f})"
    return tag


def _number(value: Any, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "not measured"


def _decision_cell(row: dict[str, Any]) -> str:
    """The verdict on record, and whether it rests on this comparison.

    A bare "no ship" cannot say what the no ship was decided on, and on this
    tree that is the whole question: every recorded verdict was taken by reading
    two artifacts' own held-out figures, on different test sets. The asterisk is
    explained in the note under the table.
    """
    latest = row.get("latest_decision") or {}
    state = str(latest.get("decision") or "")
    if not state:
        return "none"
    tag = DECISION_TAGS.get(state, state)
    if row["decisions"] > 1:
        tag = f"{tag} ({row['decisions']})"
    return tag if row.get("decision_on_rescore") else f"{tag} *"


def render(payload: dict[str, Any]) -> list[str]:
    """The registry as a terminal reads it, one list of lines and no side effect."""
    lines: list[str] = []
    version = payload.get("live_version") or {}
    lines.append("Live model version")
    lines.append(f"  {version.get('name') or 'none on disk'}  short {version.get('short') or 'none'}")
    for kind, block in (version.get("artifacts") or {}).items():
        if isinstance(block, dict) and block.get("present"):
            lines.append(f"  {kind:10s} {block.get('path')}  {str(block.get('sha256') or '')[:12]}  trained {block.get('computed_at')}")
    lines.append("")

    state = payload.get("rescore_state") or {}
    evaluation = payload.get("evaluation") or {}
    lines.append(f"Held-out re-score: {state.get('state')}")
    if state.get("state") == "current":
        lines.append(f"  measured {words.when(state.get('measured_at'))}")
    elif state.get("reason_en"):
        lines.append(f"  {state['reason_en']}")
    if evaluation:
        lines.append(f"  {evaluation.get('breaks')} breaks, {evaluation.get('cells')} cells, {evaluation.get('window')}, {evaluation.get('folds')} temporal folds")
        lines.append(f"  metric: {evaluation.get('metric_en')}")
        # The single most honest line on this surface. Every rmse below sits
        # against it, and one that is not clearly under it is a model that has
        # not beaten the mean of the thing it is predicting.
        if isinstance(evaluation.get("target_sd"), (int, float)):
            lines.append(f"  target spread: {evaluation['target_sd']:.6f} standard deviation. {evaluation.get('target_sd_en')}")
    limit = payload.get("limit") or {}
    if limit:
        lines.append(f"  limit: {limit.get('en')}")
        lines.append(f"  lifted by: {limit.get('unblocked_by_en')}")
    lines.append("")

    lines.extend(_render_table(payload))
    lines.extend(_render_baselines(payload))
    lines.extend(_render_notes(payload))
    return lines


def _render_table(payload: dict[str, Any]) -> list[str]:
    shipped = payload.get("shipped") or {}
    header = f"  {'artifact':20s} {'rmse':>10s} {'vs shipped':>11s} {'stat':>6s} {'re-score':>14s} {'money on own channel':>21s}  {'verdict on record':>17s}"
    lines = ["Artifacts, closest to the measured effects first", header]
    lines.append(f"  {'shipped (live)':20s} {_number(shipped.get('rmse'), 6):>10s} {'':>11s} {'':>6s} {'':>14s} {'':>21s}  {'':>17s}")
    for row in payload.get("candidates") or []:
        verdict = VERDICT_TAGS.get(row["verdict"], row["verdict"])
        delta = row.get("rmse_delta")
        statistic = row.get("paired_statistic")
        lines.append(
            f"  {row['id']:20s} {_number(row.get('rmse'), 6):>10s} "
            f"{(f'{delta:+.6f}' if isinstance(delta, (int, float)) else ''):>11s} "
            f"{(f'{statistic:+.2f}' if isinstance(statistic, (int, float)) else ''):>6s} "
            f"{verdict:>14s} {_money_cell(row['money']):>21s}  {_decision_cell(row):>17s}")
    lines.append("")
    lines.append("  money is the revenue movement on the operator's own channel, from the model console's own measurement. A cell marked stale names a figure whose inputs have since moved.")
    lines.append("  a verdict marked * was taken before this comparison existed, so it rests on each artifact's own held-out figures, which come from different splits and are not comparable.")
    lines.extend(_render_money_notes(payload))
    lines.append("")
    return lines


def _render_money_notes(payload: dict[str, Any]) -> list[str]:
    """The rows behind each money figure, and the named reason a stale one is stale.

    Both were computed per candidate and dropped by the render, so the count of
    rows a figure was summed over and the input that moved under it were only
    reachable through --json. A figure with no denominator is not a measurement.
    """
    lines: list[str] = []
    for row in payload.get("candidates") or []:
        money = row.get("money") or {}
        rows = money.get("scope_rows")
        if money.get("state") == "measured" and isinstance(rows, int):
            lines.append(f"  {row['id']:20s} summed over {rows:,} plan rows, measured {words.when(money.get('measured_at'))}")
        elif money.get("state") == "stale":
            lines.append(f"  {row['id']:20s} stale since it was measured. What moved: {moved_inputs(money.get('changed'))}")
    return lines


def _render_baselines(payload: dict[str, Any]) -> list[str]:
    lines = ["Baselines, out of sample, no artifact involved"]
    for row in payload.get("baselines") or []:
        lines.append(f"  {row['id']:20s} {_number(row.get('rmse'), 6):>10s}  {row.get('basis_en')}")
    structure = payload.get("cell_structure") or {}
    if structure:
        lines.append(f"  {structure.get('reading_en')}")
        lines.append(f"  cell split minus one constant: {_number(structure.get('rmse_delta'), 6)} rmse")
    lines.append("")
    return lines


def _render_notes(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for group in payload.get("duplicate_groups") or []:
        lines.append(f"Duplicate predictors: {', '.join(group)} predict the same value for every break.")
    if lines:
        lines.append("")
    adopted = [row for row in payload.get("candidates") or [] if row.get("adopted")]
    lines.append("Adopted and live")
    if adopted:
        for row in adopted:
            lines.append(f"  {row['id']} as {row['adoption_id']}")
    elif payload.get("adoptions"):
        lines.append(f"  nothing is live from a candidate. {len(payload['adoptions'])} adoption records exist and every one was reverted")
    else:
        lines.append("  nothing has ever been adopted on this tree, so the live artifact is the one the training script wrote")
    lines.append("")
    lines.append("Next act, per candidate")
    for row in payload.get("candidates") or []:
        lines.append(f"  {row['id']:20s} {row['next_act']['en']}")
        lines.append(f"  {'':20s} {row['next_act']['command']}")
    lines.append("")
    # The path was in the payload and not on the screen, so a steward who wanted
    # to open one had to guess the filename or read the json.
    lines.append("Artifact files")
    for row in payload.get("candidates") or []:
        lines.append(f"  {row['id']:20s} {row['file']}  {row['bytes']:,} bytes")
    return lines


def _render_surface(surface: dict[str, Any]) -> list[str]:
    """What else the adopted file would change, beyond the money and the score.

    Printed on every check run, passing or failing, because a steward who reads
    only the verdict lines would never learn that the file about to replace the
    live one is a narrower file than the one it replaces.
    """
    if not surface:
        return []
    lines = ["What else the adopted artifact would change"]
    intervals = surface.get("intervals") or {}
    if intervals.get("bounds_moved"):
        lines.append(f"  credible bounds moved on {intervals['bounds_moved']} of the {intervals.get('bounds_compared')} bounds compared, across {intervals.get('cells_compared')} cells")
        # ``read_by`` is already a sentence about the line that reads the bound,
        # so introducing it with "read by" made "read by <path> prices the ...".
        lines.append(f"  largest move {intervals['max_abs_move']} at {intervals['max_abs_move_at']}. {intervals['read_by']}")
    for key, label in (("metadata_dropped", "metadata keys dropped"),
                       ("metadata_added", "metadata keys added"),
                       ("detail_fields_dropped", "per-cell fields dropped"),
                       ("detail_fields_added", "per-cell fields added"),
                       ("cells_dropped", "cells dropped")):
        values = surface.get(key) or []
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    if len(lines) == 1:
        lines.append("  nothing beyond the coefficients themselves")
    lines.append("")
    return lines


def _render_gates(evidence: dict[str, Any]) -> list[str]:
    """What its gates decide differently, and what each side decided that on.

    JS-19's sequence says "read what its gates decided differently" and its
    target adds "with its held-out figure". Both were one surface away, on the
    model console, while the verdict is taken here. The two held-out sizes sit
    on one line on purpose: 2,532 against 506 is the argument for the re-score.
    """
    if not evidence:
        return []
    cell = words.gate_cell
    lines = ["What its gates decide differently"]
    for row in evidence.get("verdicts") or []:
        lines.append(f"  {row['key']:34s} shipped {cell(row['shipped'], row['shipped_absent']):22s} candidate {cell(row['candidate'], row['candidate_absent'])}")
    if len(lines) == 1:
        lines.append("  no gate decides differently from the shipped artifact")
    if evidence.get("held_out"):
        lines.append("  how much each gate was decided on, as each artifact reports it about itself")
        for row in evidence["held_out"]:
            lines.append(f"  {row['block']:34s} shipped {words.size_cell(row['shipped_size'], row['shipped_unit'], row['shipped_absent']):22s} candidate {words.size_cell(row['candidate_size'], row['candidate_unit'], row['candidate_absent'])}")
        lines.append("  two sides measured on different amounts are not comparable, which is why show scores every artifact again on one common set.")
    lines.append("")
    return lines


def _render_money(money: dict[str, Any]) -> list[str]:
    """The figure itself, in shekels and with its scope, not just its state.

    JS-19's target asks for the money movement in shekels with its scope, and a
    check that says only "measured and current" has answered a different
    question. When it is not measured the state is printed instead, because an
    absent measurement is a state and never a zero.
    """
    delta = money.get("revenue_delta")
    scope = money.get("scope") or {}
    if money.get("state") != "measured" or not isinstance(delta, (int, float)):
        # A state, never a figure. A stale one still carries the date it was
        # taken and what has moved since, which is what makes it actionable.
        lines = [f"Money if adopted: {MONEY_TAGS.get(str(money.get('state')), 'unknown')}"]
        if money.get("state") == "stale":
            lines.append(f"  last measured {words.when(money.get('measured_at'))}. What moved since: {moved_inputs(money.get('changed'))}")
        return lines + [""]
    whole = money.get("whole_plan_delta")
    rows = scope.get("rows")
    counted = f"{rows:,}" if isinstance(rows, int) else "an unrecorded number of"
    lines = [f"Money if adopted: {delta:+,.2f} on the operator's own channel over {counted} rows"]
    if isinstance(whole, (int, float)):
        lines.append(f"  whole plan, every channel the optimizer schedules: {whole:+,.2f}")
    lines.append(f"  basis: {scope.get('basis')}")
    lines.append(f"  measured {words.when(money.get('measured_at'))}")
    if money.get("moved_fields"):
        lines.append(f"  shipped figures this would move: {', '.join(money['moved_fields'])}")
    lines.append("")
    return lines


def render_checks(state: dict[str, Any]) -> list[str]:
    """The adoption checks as a terminal reads them, passed and failed alike.

    A name that is not a candidate stops after the first check. The payload
    still answers every condition, but rendering the rest would state as fact
    that a file which does not exist drops the engine inputs the shipped one has.
    """
    lines = [f"Adoption checks for {state.get('candidate_id')}"]
    for check in state.get("checks") or []:
        mark = "pass" if check["passed"] else "STOP"
        lines.append(f"  [{mark}] {check['id']:32s} {check['reason_en']}")
        if not check["passed"] and check.get("how_en"):
            lines.append(f"         {'':32s} {check['how_en']}")
        if check["id"] == "candidate_exists" and not check["passed"]:
            lines.append("")
            lines.append("Nothing below can be answered about a candidate that is not on disk.")
            lines.append(f"outcome: {state.get('outcome')}")
            return lines
    lines.append("")
    lines.extend(_render_money(state.get("money") or {}))
    lines.extend(_render_gates(state.get("gate_evidence") or {}))
    lines.extend(_render_surface(state.get("artifact_surface") or {}))
    if state.get("escalated"):
        lines.append("This adoption is escalated and will not land.")
    lines.append(f"outcome: {state.get('outcome')}")
    return lines


def checks_for(identifier: str, paths: Optional[Paths] = None, *, adopted_by: str = "",
               reason: str = "") -> dict[str, Any]:
    """The precondition report for one candidate, without planning a write."""
    state = preconditions(identifier, paths or Paths(), approved_by=adopted_by, reason=reason)
    state["outcome"] = ("ready" if state["passed"] else
                        "escalated" if state["escalated"] else "refused")
    return state
