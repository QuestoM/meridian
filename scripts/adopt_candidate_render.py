"""The registry as a terminal reads it: every line, and no figure computed here.

Split out of ``adopt_candidate_registry.py`` under the naming rule of section
8.2. That file was at 448 lines of a 450 cap, and the split falls on the seam
its own docstring already names: the registry joins what was measured, and
nothing in it computes a figure. This renders the join and computes nothing at
all, so a change to a sentence can never move a number.

Every table here is a comparison across artifacts rather than a page per
artifact, because JS-19's steward is choosing between five candidates and not
reading one. The blocks are the score, the gates, the coefficient delta, the
self-tests, the verdicts already recorded, the baselines and the standing state,
in that order, because that is the order the question is asked in: is it better,
what did it decide differently, what did it change, what did its own producer
say, what has already been decided about it, better than what, and what happens
next.

**The checks for one candidate left this file in round 11**, when the verdict
history took it to 447 of the 450-line cap. They are in
``adopt_candidate_checks.py`` under the same naming rule, on the seam the two
halves already had: this file is read while a steward is choosing between five
artifacts, and that one is read once the choice is made. ``render_checks`` is
re-exported here because three callers already use that name, and moving a file
should not move a public surface.
"""

from __future__ import annotations

from typing import Any

from scripts import adopt_candidate_baselines as baselines
from scripts import adopt_candidate_basis as basis
from scripts import adopt_candidate_gates as gates
from scripts import adopt_candidate_history as history
from scripts import adopt_candidate_note as note
from scripts import adopt_candidate_origin as origin
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_checks import MONEY_TAGS, render_checks  # noqa: F401
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
    # What is on record about that version itself. Every read this piece made of
    # the decision log filtered to the candidate rows, so a verdict whose subject
    # is the live model reached no surface here at all.
    lines.extend(history.render_live_model(payload.get("decision_log") or {}))
    # And what the operator side reads about that version, which is the one
    # sentence this act sends across the wall. Measured on this tree: none, and
    # nothing on any surface of this piece said so.
    lines.extend(note.render_operator_reads(payload.get("operator_reads") or {}))
    lines.append("")

    state = payload.get("rescore_state") or {}
    evaluation = payload.get("evaluation") or {}
    lines.append(f"Held-out re-score: {state.get('state')}")
    if state.get("state") == "current":
        lines.append(f"  measured {words.when(state.get('measured_at'))}")
    elif state.get("reason_en"):
        lines.append(f"  {state['reason_en']}")
    if evaluation:
        lines.append(f"  {evaluation.get('breaks')} breaks, {evaluation.get('cells')} cells, {words.window_line(evaluation)}, {evaluation.get('folds')} temporal folds")
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

    # What each artifact was for, before anything is ranked. A reader who does
    # not know what an artifact was trying to do cannot read a table of five.
    lines.extend(origin.render_purposes(payload))
    lines.extend(_render_table(payload))
    lines.extend(_render_gate_readings(payload))
    lines.extend(_render_cells(payload))
    lines.extend(basis.render_self_tests(payload))
    # Every verdict, in the order it was taken, on the surface where the next one
    # is taken. The table above prints one word and a count in brackets, which
    # cannot say when, by whom, on what, or that two of them are the same word
    # for two different stated reasons.
    lines.extend(history.render_history(payload))
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


def _render_gate_readings(payload: dict[str, Any]) -> list[str]:
    """What each artifact's gates decided, across all five, before any drill.

    JS-19 reads the gates before it reads the money, and until this block the
    only place they appeared was the last command a steward runs. It is a
    sentence per artifact rather than a count, because the count the console's
    comparison returns includes every key the candidate does not carry, and on
    three of the five candidates on this tree that is the whole of it.
    """
    rows = [row for row in payload.get("candidates") or [] if row.get("gates")]
    if not rows:
        return []
    lines = ["Gates, what each artifact decided against the shipped model"]
    for row in rows:
        lines.append(f"  {row['id']:20s} {row['gates']['reading_en']}")
        lines.extend(gates.render_summary(row["gates"])[1:])
    lines.append("")
    lines.append(f"  {gates.HELD_OUT_RULE['en']}")
    lines.append("  the held-out figures themselves, per gate: python scripts/adopt_candidate.py checks <candidate>")
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
    lines.extend(baselines.render_standing_finding(payload.get("structure_finding") or {}))
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
    # With the day each artifact was produced, which this payload has carried on
    # every row and no surface printed. The live block above prints it for the
    # shipped model and the candidate rows did not, so a steward could read a
    # whole shelf without learning that all five of these artifacts predate the
    # model they are being compared against.
    for row in payload.get("candidates") or []:
        lines.append(f"  {row['id']:20s} {row['file']}  {row['bytes']:,} bytes  produced {words.when(row.get('computed_at'))}")
    # And what data each of those files read, checked against the files on disk.
    lines.extend(origin.render_provenance(payload))
    return lines
