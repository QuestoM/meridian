"""The coefficient delta: which numbers move, and which of them buy the metric.

JS-19's done condition names three things that must be recorded against a new
model version: the gate deltas, the **coefficient deltas**, and the measured
money movement. Two of the three were built. The third was not, and its absence
was worse than a hole, because the surface printed something in its place.

Measured on this tree before this module existed: the shipped artifact and the
``afterwindow`` candidate differ in **all 36 of 36** coefficients, the largest
point move being 0.014066885 at ``PrimeShow2_first_short``. The adoption checks
reported none of it. What they did report was "credible bounds moved on 72 of
the 72 bounds", which invites exactly the wrong reading: that the intervals
moved and the points did not. Every point moved.

**What this module adds that a metric delta cannot say.** A candidate that
scores 0.000114 better in RMSE has told a steward that it is better by a
thousandth. It has not said whether that came from one cell it fixed or from
thirty-six cells that moved a lot and almost entirely cancelled. Those are two
different artifacts and only one of them is a finding. So every figure here is
attributed: each cell carries the breaks it was measured on, the squared error
it moved, and its share of the total movement, and the summary states how much
of the movement the cells make against each other cancels.

**A cell key is not a channel.** The keys are the artifacts' own composite keys,
of the form ``PrimeShow2_first_short``: a programme class, a break position and
a break length. Measured on this tree, the four classes are News, Other,
PrimeShow1 and PrimeShow2. No channel name of any kind is in them and none
reaches any payload from here.

**Tri-state, not zero.** A cell one artifact carries and the other does not is
``added`` or ``dropped`` with a delta of ``None``. A missing coefficient is not
a coefficient of zero and it is never rendered as one.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from scripts import adopt_candidate_words as words

# The share of the total absolute movement that the named cells must reach
# before the summary stops naming them. Eight tenths, so the sentence says
# "these carry most of it" about a set that genuinely does.
CARRIES_SHARE = 0.8

# And the size at which naming them stops being a finding. Measured on this
# tree, the competitor candidate needs 16 of its 36 cells to reach four fifths
# of its own movement, and a list of 16 names is not "the cells that carry it",
# it is the opposite finding: the movement is spread and no small set carries
# it. A quarter of the compared cells is the line, and which side a candidate
# falls on is in the payload as ``concentrated`` rather than left to the render.
CONCENTRATED_AT = 0.25

# Below this the two coefficients are the same number. The artifacts are written
# with json float repr, so an exact equality test is the right one and this only
# guards a round trip through a float.
MOVED_AT = 1e-12


def _state(shipped: Optional[float], candidate: Optional[float]) -> str:
    if shipped is None:
        return "added"
    if candidate is None:
        return "dropped"
    return "moved" if abs(float(candidate) - float(shipped)) > MOVED_AT else "unchanged"


def cell_rows(shipped_coefficients: dict[str, Any], candidate_coefficients: dict[str, Any],
              cells: np.ndarray, shipped_errors: np.ndarray,
              candidate_errors: np.ndarray) -> list[dict[str, Any]]:
    """One row per cell either artifact carries, with what it moved and bought.

    ``squared_error_delta`` is the candidate's squared error minus the shipped
    model's, summed over the breaks in that cell. It is signed the way the metric
    is signed, so negative means the candidate is closer to the measured effects
    there, and the sum of it over every cell is the whole movement the metric
    reports.
    """
    difference = np.asarray(candidate_errors, dtype=float) - np.asarray(shipped_errors, dtype=float)
    cells = np.asarray(cells)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(shipped_coefficients) | set(candidate_coefficients)):
        before = shipped_coefficients.get(key)
        after = candidate_coefficients.get(key)
        before = float(before) if isinstance(before, (int, float)) else None
        after = float(after) if isinstance(after, (int, float)) else None
        mask = cells == key
        bought = float(difference[mask].sum()) if difference.size else 0.0
        rows.append({
            "cell": key,
            "shipped": None if before is None else round(before, 9),
            "candidate": None if after is None else round(after, 9),
            # None and never zero. A cell only one side carries has no delta,
            # and printing 0.0 there would state that it did not move.
            "delta": None if before is None or after is None else round(after - before, 9),
            "state": _state(before, after),
            "breaks": int(mask.sum()),
            "squared_error_delta": round(bought, 12),
        })
    total = sum(abs(row["squared_error_delta"]) for row in rows)
    for row in rows:
        row["share_of_absolute"] = round(abs(row["squared_error_delta"]) / total, 6) if total else 0.0
    return rows


def _carries_the_move(rows: list[dict[str, Any]]) -> list[str]:
    """The smallest set of cells that accounts for most of the movement."""
    ordered = sorted(rows, key=lambda row: -abs(row["squared_error_delta"]))
    total = sum(abs(row["squared_error_delta"]) for row in ordered)
    if not total:
        return []
    running, named = 0.0, []
    for row in ordered:
        named.append(row["cell"])
        running += abs(row["squared_error_delta"])
        if running / total >= CARRIES_SHARE:
            break
    return named


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the whole coefficient delta amounts to, stated rather than implied.

    The figure this exists for is the cancellation. A candidate whose 36 cells
    each move and whose net movement is a thousandth of what they move is not a
    model that improved a thousandth; it is a model that was re-fitted and landed
    somewhere else. The metric alone cannot tell those apart and this can.
    """
    moved = [row for row in rows if row["state"] == "moved"]
    compared = [row for row in rows if row["state"] in ("moved", "unchanged")]
    added = [row["cell"] for row in rows if row["state"] == "added"]
    dropped = [row["cell"] for row in rows if row["state"] == "dropped"]
    net = sum(row["squared_error_delta"] for row in rows)
    absolute = sum(abs(row["squared_error_delta"]) for row in rows)
    largest = max(moved, key=lambda row: abs(row["delta"]), default=None)
    weight = sum(row["breaks"] for row in compared)
    weighted = (sum(abs(row["delta"]) * row["breaks"] for row in moved) / weight) if weight else 0.0
    # None and never zero, for the same reason a missing coefficient is not a
    # coefficient of zero. A candidate that moved nothing has no movement to
    # cancel, and 0.0 in this field states that none of its movement cancelled,
    # which is a reading of a measurement that was never taken. The terminal
    # already prints "no move" here; the payload a route would serve did not.
    cancelled = round(1.0 - abs(net) / absolute, 6) if absolute else None
    summary = {
        "cells_compared": len(compared),
        "cells_moved": len(moved),
        "cells_unchanged": len(compared) - len(moved),
        "cells_added": added,
        "cells_dropped": dropped,
        "max_abs_delta": None if largest is None else round(abs(largest["delta"]), 9),
        "max_abs_delta_at": None if largest is None else largest["cell"],
        "breaks_weighted_mean_abs_delta": round(weighted, 9),
        "net_squared_error_delta": round(net, 12),
        "total_abs_squared_error_delta": round(absolute, 12),
        "cancelled_share": cancelled,
        "cells_improved": sum(1 for row in rows if row["squared_error_delta"] < 0),
        "cells_worsened": sum(1 for row in rows if row["squared_error_delta"] > 0),
        "carries_the_move": _carries_the_move(rows),
        "carries_share": CARRIES_SHARE,
        "key_shape_en": words.CELL_KEY_SHAPE["en"],
        "key_shape_he": words.CELL_KEY_SHAPE["he"],
    }
    named = len(summary["carries_the_move"])
    summary["concentrated"] = bool(named and len(compared)
                                   and named <= CONCENTRATED_AT * len(compared))
    summary.update(_reading(summary))
    return summary


def _reading(summary: dict[str, Any]) -> dict[str, str]:
    """The one sentence the summary is for, in both languages."""
    if not summary["cells_moved"] and not summary["cells_added"] and not summary["cells_dropped"]:
        return words.pair(words.CELL_READING, "none", "reading")
    if summary["total_abs_squared_error_delta"] <= 0:
        return words.pair(words.CELL_READING, "no_effect", "reading",
                          moved=summary["cells_moved"], compared=summary["cells_compared"])
    return words.pair(words.CELL_READING,
                      "cancelling" if summary["concentrated"] else "spread", "reading",
                      moved=summary["cells_moved"], compared=summary["cells_compared"],
                      cancelled=f"{summary['cancelled_share'] * 100:.1f}",
                      named=len(summary["carries_the_move"]),
                      share=f"{CARRIES_SHARE * 100:.0f}")


def cell_deltas(shipped_coefficients: dict[str, Any], candidate_coefficients: dict[str, Any],
                cells: np.ndarray, shipped_errors: np.ndarray,
                candidate_errors: np.ndarray) -> dict[str, Any]:
    """Every cell that moved and what it bought, plus the summary over them."""
    rows = cell_rows(shipped_coefficients, candidate_coefficients, cells,
                     shipped_errors, candidate_errors)
    return {"rows": rows, "summary": summarise(rows)}


def _cell_number(value: Any) -> str:
    """A coefficient as a line of text, and a state where there is no number."""
    return f"{value:+.6f}" if isinstance(value, (int, float)) else "not carried"


def render_summary(deltas: dict[str, Any], indent: str = "  ") -> list[str]:
    """The coefficient delta in four lines, for the adoption checks.

    Placed above the credible bounds wherever both are rendered, because the
    point is what the engine reads as the retention cost itself and the interval
    is what it reads when the operator prices risk. Reporting the interval and
    not the point was the defect this closes.
    """
    summary = (deltas or {}).get("summary") or {}
    if not summary:
        return []
    lines = ["What its coefficients change", f"{indent}{summary.get('reading_en')}"]
    if summary.get("max_abs_delta_at"):
        lines.append(f"{indent}largest move {summary['max_abs_delta']:.9f} at {summary['max_abs_delta_at']}. {words.CELL_READ_BY['en']}")
        lines.append(f"{indent}mean absolute move, weighted by the breaks each cell was measured on: {summary['breaks_weighted_mean_abs_delta']:.9f}")
    for key, label in (("cells_added", "cells the candidate adds"),
                       ("cells_dropped", "cells the candidate drops")):
        if summary.get(key):
            lines.append(f"{indent}{label}: {', '.join(summary[key])}")
    # Named only when a small set genuinely carries the movement. When it takes
    # sixteen of thirty-six cells to reach four fifths, the list is not the
    # finding and the sentence above has already said the opposite one.
    if summary.get("concentrated"):
        lines.append(f"{indent}the cells carrying most of it: {', '.join(summary['carries_the_move'])}")
    lines.append(f"{indent}{summary['key_shape_en']}")
    lines.append("")
    return lines


def render_table(identifier: str, deltas: dict[str, Any], limit: int = 0) -> list[str]:
    """Every cell, ranked by how much of the movement it carries.

    This is the view a metric delta cannot replace: it says where the candidate
    differs, on how many breaks, and whether that difference bought anything.
    """
    rows = list((deltas or {}).get("rows") or [])
    summary = (deltas or {}).get("summary") or {}
    if not rows:
        return [f"No coefficient comparison is stored for {identifier}.",
                "Run: python scripts/adopt_candidate.py rescore --force"]
    # Ranking thirty-six identical rows by a contribution that is zero on every
    # one of them is noise dressed as a table. The finding is that nothing
    # moved, and it is said once. ``--all`` still prints every value.
    if limit and not summary.get("cells_moved") and not summary.get("cells_added") \
            and not summary.get("cells_dropped"):
        return [f"Coefficient delta, {identifier} against the shipped model",
                f"  {summary.get('reading_en')}",
                f"  every one of the {summary.get('cells_compared')} cells holds the same number in both artifacts",
                "  the credible bounds and the metadata are a separate comparison, and on this candidate they do move.",
                "  read them with: python scripts/adopt_candidate.py checks " + identifier,
                "  every cell and its value: python scripts/adopt_candidate.py diff " + identifier + " --all",
                ""]
    rows.sort(key=lambda row: -abs(row["squared_error_delta"]))
    shown = rows[:limit] if limit else rows
    header = (f"  {'cell':30s} {'shipped':>11s} {'candidate':>11s} {'delta':>11s} "
              f"{'breaks':>7s} {'squared error moved':>20s} {'share':>7s}")
    lines = [f"Coefficient delta, {identifier} against the shipped model", header]
    for row in shown:
        lines.append(
            f"  {row['cell']:30s} {_cell_number(row['shipped']):>11s} "
            f"{_cell_number(row['candidate']):>11s} {_cell_number(row['delta']):>11s} "
            f"{row['breaks']:>7,d} {row['squared_error_delta']:>+20.9f} "
            f"{row['share_of_absolute'] * 100:>6.2f}%")
    if limit and len(rows) > limit:
        lines.append(f"  {len(rows) - limit} further cells are not shown. Pass --all for every cell.")
    lines.append("")
    lines.append("  squared error moved is the candidate's squared error minus the shipped model's, summed over the breaks in that cell.")
    lines.append("  a negative figure is a cell where the candidate is closer to the measured effects than the shipped model.")
    # The column had no denominator on screen, so a reader had to guess what the
    # percentages were a share of and why a cell that made the metric worse sits
    # at the top of the table.
    lines.append("  share is that cell's part of the total absolute squared error moved, and the rows are ranked by it, so a cell that moved the metric the wrong way still ranks high.")
    lines.append("")
    lines.extend(render_summary(deltas))
    return lines
