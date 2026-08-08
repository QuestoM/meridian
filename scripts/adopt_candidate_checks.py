"""The adoption checks for one candidate, as a terminal reads them.

Split out of ``adopt_candidate_render.py`` under the naming rule of section 8.2
when that file reached 447 lines of a 450 cap, and the split falls on the seam
that file's own docstring names. Everything left there is a comparison ACROSS
artifacts, which is what a steward choosing between five candidates reads.
Everything here is one candidate answered against the conditions an adoption
would have to clear, which is what the same steward reads once the choice is
made. The two are read at different moments and they were one file only because
they print to the same terminal.

Nothing here computes a figure either. The conditions are answered in
``adopt_candidate_adoption.py`` and the measurements are taken elsewhere again,
so a change to a sentence in this file can never move a number.
"""

from __future__ import annotations

from typing import Any

from scripts import adopt_candidate_cells as cells
from scripts import adopt_candidate_gates as gates
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_state import moved_inputs

MONEY_TAGS = {
    "measured": "measured",
    "stale": "stale",
    "not_measured": "not measured",
}


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
    cell = gates.gate_cell
    summary = gates.gate_summary(evidence)
    lines = ["What its gates decide differently"]
    # The sentence the rows below amount to, first. Ten rows each reading "does
    # not carry it" is an artifact with no gate decisions on it, and a reader
    # counting the rows reads ten gates decided the other way.
    lines.append(f"  {summary['reading_en']}")
    for row in evidence.get("verdicts") or []:
        lines.append(f"  {row['key']:34s} shipped {cell(row['shipped'], row['shipped_absent']):22s} candidate {cell(row['candidate'], row['candidate_absent'])}")
    if len(lines) == 2:
        lines.append("  no gate decides differently from the shipped artifact")
    if evidence.get("held_out"):
        lines.append("  how much each gate was decided on, as each artifact reports it about itself")
        for row in evidence["held_out"]:
            lines.append(f"  {row['block']:34s} shipped {gates.size_cell(row['shipped_size'], row['shipped_unit'], row['shipped_absent']):22s} candidate {gates.size_cell(row['candidate_size'], row['candidate_unit'], row['candidate_absent'])}")
        # The rule, then whether it bit on this pair. The rule alone used to
        # assert that the amounts disagree, and on calibrated every one of them
        # agrees, so the surface was stating a confound that pair does not carry.
        lines.append(f"  {gates.HELD_OUT_RULE['en']}")
        lines.append(f"  {summary['held_out_basis_en']}")
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
            # The magnitude of the figure being refused, named rather than
            # withheld. "Stale" alone cannot say whether what this check will
            # not use is a rounding error or a million shekels, and the table
            # three commands earlier already prints it.
            last = money.get("last_known_revenue_delta")
            if isinstance(last, (int, float)):
                lines.append(f"  the figure it last measured was {last:+,.2f} on the operator's own channel, and this check will not carry it into a record or an artifact stamp because its inputs have since moved")
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
    # What it was built for, on the surface where it would be adopted. Its own
    # producer's sentence, or the absence, and never an inference from below.
    built_for = (state.get("origin") or {}).get("purpose")
    lines.append(f"  built for: {built_for or 'no purpose recorded in the artifact'}")
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
    # The points before the intervals, because the point is what the engine
    # reads as the retention cost and the interval is what it reads when the
    # operator prices risk. Reporting the interval and not the point read as
    # "the bounds moved and the numbers did not", and on this tree every number
    # moved on every candidate but one.
    lines.extend(cells.render_summary(state.get("cell_deltas") or {}))
    lines.extend(_render_surface(state.get("artifact_surface") or {}))
    if state.get("escalated"):
        lines.append("This adoption is escalated and will not land.")
    lines.append(f"outcome: {state.get('outcome')}")
    return lines
