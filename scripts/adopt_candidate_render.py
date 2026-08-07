"""The registry as a terminal reads it: every line, and no figure computed here.

Split out of ``adopt_candidate_registry.py`` under the naming rule of section
8.2. That file was at 448 lines of a 450 cap, and the split falls on the seam
its own docstring already names: the registry joins what was measured, and
nothing in it computes a figure. This renders the join and computes nothing at
all, so a change to a sentence can never move a number.

Every table here is a comparison across artifacts rather than a page per
artifact, because JS-19's steward is choosing between five candidates and not
reading one. The four blocks are the score, the coefficient delta, the baselines
and the standing state, in that order, because that is the order the question is
asked in: is it better, what did it change, better than what, and what happens
next.
"""

from __future__ import annotations

from typing import Any

from scripts import adopt_candidate_basis as basis
from scripts import adopt_candidate_cells as cells
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_state import moved_inputs
from scripts.adopt_candidate_surface import dropped_field  # noqa: F401

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


def _money_cell(money: dict[str, Any]) -> str:
    """The money on a table row: the figure when it is current, the state when not.

    A stale row prints the magnitude it last measured in brackets, from
    ``last_known_revenue_delta`` rather than from ``revenue_delta``, which stays
    None so that no stale figure can reach a decision record or an artifact
    stamp. The two key names are the whole point: one is for reading, one is for
    recording, and they are not the same figure.
    """
    tag = MONEY_TAGS.get(str(money.get("state")), str(money.get("state")))
    delta = money.get("revenue_delta")
    if money.get("state") == "measured" and isinstance(delta, (int, float)):
        return f"{delta:+,.2f}"
    last = money.get("last_known_revenue_delta")
    if money.get("state") == "stale" and isinstance(last, (int, float)):
        return f"stale ({last:+,.0f})"
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
        # Which rows the limit is naming, and by how much. The sentence above is
        # selected by a measurement now, so when it says "not every artifact"
        # the rows it means are printed rather than left to the reader.
        lines.extend(basis.render_fit_basis(payload))
        lines.append(f"  lifted by: {limit.get('unblocked_by_en')}")
    lines.append("")

    lines.extend(_render_table(payload))
    lines.extend(_render_cells(payload))
    lines.extend(basis.render_self_tests(payload))
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
    # The bar is read off a measured row rather than typed here, so the legend
    # cannot say one thing while the verdict was decided on another.
    bar = next((row.get("paired_bar") for row in payload.get("candidates") or []
                if isinstance(row.get("paired_bar"), (int, float))), None)
    if bar is not None:
        lines.append(f"  {words.PAIRED_LEGEND['en'].format(bar=f'{bar:.1f}')}")
    lines.append("  money is the revenue movement on the operator's own channel, from the model console's own measurement. A cell marked stale names a figure whose inputs have since moved.")
    lines.append("  a verdict marked * was taken before this comparison existed, so it rests on each artifact's own held-out figures, which come from different splits and are not comparable.")
    lines.extend(_render_money_notes(payload))
    lines.append("")
    return lines


def _render_cells(payload: dict[str, Any]) -> list[str]:
    """What each candidate changes in the numbers themselves, side by side.

    The score table says how much closer a candidate is. It cannot say whether
    that came from one cell or from thirty-six that cancel, and those are two
    different artifacts. Measured on this tree, every candidate but one moves
    every cell it carries, and almost all of that movement cancels.
    """
    rows = [row for row in payload.get("candidates") or [] if row.get("cell_delta")]
    if not rows:
        return []
    header = (f"  {'artifact':20s} {'cells moved':>13s} {'largest move':>13s} "
              f"{'where':30s} {'cancels':>8s}")
    lines = ["Coefficients, what each candidate changes against the shipped model", header]
    for row in rows:
        summary = row["cell_delta"]
        largest = summary.get("max_abs_delta")
        moved = f"{summary.get('cells_moved')} of {summary.get('cells_compared')}"
        # A candidate that moves nothing has nothing to cancel, and printing
        # 0.0 percent there states that its movement cancelled rather than that
        # there was none. Three states, and the third is a word. Driven by the
        # figure rather than by the count of moved cells, because a cell that
        # moved on no break scored here also has no cancellation to report.
        share = summary.get("cancelled_share")
        cancels = f"{share * 100:.1f}%" if isinstance(share, (int, float)) else "no move"
        lines.append(
            f"  {row['id']:20s} {moved:>13s} "
            f"{(f'{largest:.9f}' if isinstance(largest, (int, float)) else 'none'):>13s} "
            f"{str(summary.get('max_abs_delta_at') or ''):30s} "
            f"{cancels:>8s}")
    lines.append("")
    lines.append("  cancels is the share of the squared error each candidate moves that its own cells undo between them, so a high figure is a re-fit that landed elsewhere rather than a model that improved.")
    lines.append(f"  {words.CELL_KEY_SHAPE['en']}")
    lines.append("  the whole per-cell table: python scripts/adopt_candidate.py diff <candidate>")
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
        rows = (money.get("scope") or {}).get("rows")
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
    lines.extend(_render_structure_finding(payload.get("structure_finding") or {}))
    lines.append("")
    return lines


def _render_structure_finding(finding: dict[str, Any]) -> list[str]:
    """The standing finding, sized against the candidates, with whose decision it is.

    Printed as a standing finding and never as a next act. Nothing this terminal
    can run changes the cell structure, so an act line here would be an act
    nobody can take.
    """
    if not finding or finding.get("earns_its_place"):
        return []
    times = finding.get("times_the_largest_candidate_move")
    largest = finding.get("largest_candidate_move_rmse")
    lines = ["", "  Standing finding, and no candidate on this shelf answers it"]
    if isinstance(times, (int, float)) and isinstance(largest, (int, float)):
        lines.append(f"  the cell split costs {finding['structure_cost_rmse']:.6f} rmse out of sample, which is {times:.1f} times the largest movement any candidate makes, {largest:.6f}")
    addressing = finding.get("candidates_addressing_it") or []
    lines.append(f"  {len(addressing)} of the {finding.get('candidates_compared')} candidates change the set of cells at all, so all of them are choices made inside a structure that does not pay for itself")
    lines.append(f"  whose decision this is: {finding.get('decision_owner_en')}. This terminal has no command that changes it.")
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
