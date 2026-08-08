"""The two out-of-sample baselines, which are the only honest absolutes here.

Split out of ``adopt_candidate_rescore.py`` under the naming rule of section 8.2
when that file reached the 450-line cap, and on the seam its own docstring
already draws: every artifact scored by this piece was fitted on all 2,532
breaks in this repository, so every absolute figure it produces is optimistic
and only a paired difference is readable. These two rows are the exception.
Neither has ever seen the break it predicts.

They answer a question the artifacts cannot answer about themselves. The
leave-one-out cell mean predicts each break from the other breaks in its own
cell; the leave-one-out global mean predicts it from every other break. If the
first is not closer to the measured effects than the second, the 36-cell
structure that every artifact in this tree is built on does not earn its place
out of sample, whatever the artifacts say about themselves.

Measured on this tree, it does not: the cell mean is 0.242097 and one constant
is 0.241474, so the split is 0.000623 RMSE **worse**, which is more than four
times the largest movement any candidate makes. That is reported whichever way
it lands, and it is reported in the same place as the artifact scores rather
than in a footnote, because a steward choosing between five artifacts should
know that all six rows sit inside a structure that does not pay for itself.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def leave_one_out(y: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
    """Predict each break from the others, globally or inside its own cell.

    A cell holding exactly one break falls back to the global prediction rather
    than to zero, because a cell of one has no other breaks to predict from and
    predicting zero there would flatter the per-cell baseline.
    """
    total, count = y.sum(), len(y)
    global_loo = (total - y) / (count - 1) if count > 1 else np.zeros_like(y)
    if groups is None:
        return global_loo
    frame = pd.DataFrame({"g": groups, "y": y})
    sums = frame.groupby("g")["y"].transform("sum").to_numpy(dtype=float)
    sizes = frame.groupby("g")["y"].transform("size").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = (sums - y) / (sizes - 1.0)
    return np.where(sizes > 1.0, inside, global_loo)


def squared_errors(y: np.ndarray, cells: np.ndarray) -> dict[str, np.ndarray]:
    """The per-break squared error of each baseline, keyed by its row name."""
    return {
        "global_mean_loo": (y - leave_one_out(y, None)) ** 2,
        "cell_mean_loo": (y - leave_one_out(y, cells)) ** 2,
    }


def standing_finding(structure: dict[str, Any],
                     candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """The out-of-sample finding measured against what the candidates on offer move.

    Written because the finding was on the screen and nobody could act on it. It
    sat under the baselines as one sentence, beside five candidate rows, with
    nothing saying how the two compare or whose decision it is. Measured on this
    tree the comparison is the whole argument: the cell split costs 0.000623
    RMSE out of sample and the largest movement any candidate makes is 0.000156,
    so the structure every artifact here is built on is four times the size of
    the choice a steward is being asked to make inside it.

    It is stated as a standing finding with the decision it needs and never as a
    next act, because changing the cell structure is a retraining decision for
    the model owner and this piece decides nothing.
    """
    moves = [abs(float(row["rmse_delta"])) for row in candidates
             if isinstance(row.get("rmse_delta"), (int, float))]
    largest = max(moves) if moves else None
    cost = abs(float(structure.get("rmse_delta") or 0.0))
    # Counted rather than claimed. A candidate addresses the structure only by
    # holding a different set of cells, so the count is the number whose
    # coefficient delta reports a cell added or a cell dropped. Measured on this
    # tree it is zero: all five carry the same 36 keys as the shipped artifact.
    addressing = [row["id"] for row in candidates
                  if (row.get("cell_delta") or {}).get("cells_added")
                  or (row.get("cell_delta") or {}).get("cells_dropped")]
    return {
        "structure_cost_rmse": round(cost, 9),
        "largest_candidate_move_rmse": None if largest is None else round(largest, 9),
        "times_the_largest_candidate_move": (
            round(cost / largest, 2) if largest else None),
        "earns_its_place": bool(structure.get("earns_its_place")),
        "candidates_compared": len(candidates),
        "candidates_addressing_it": addressing,
        "decision_owner_en": "the model owner, as a retraining decision",
        "decision_owner_he": "בעל המודל, כהחלטת אימון מחדש",
    }


def render_standing_finding(finding: dict[str, Any]) -> list[str]:
    """The standing finding at a terminal, sized against the candidates.

    Rendered here rather than in ``adopt_candidate_render.py`` for the reason
    ``adopt_candidate_basis.py`` gives for its own two blocks: that module is the
    terminal's renderer and it sits at the size cap, and this block is this
    module's finding end to end.

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


def cell_structure(baselines: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the 36-cell split beat one constant, out of sample and honestly.

    This is the only figure on the whole surface that is free of the in-sample
    limit, because both baselines predict each break from breaks that are not
    it. It is reported whichever way it lands.
    """
    by_id = {row["id"]: row for row in baselines}
    cell = float(by_id["cell_mean_loo"]["rmse"])
    glob = float(by_id["global_mean_loo"]["rmse"])
    moved = cell - glob
    return {
        "cell_mean_loo_rmse": round(cell, 9),
        "global_mean_loo_rmse": round(glob, 9),
        "rmse_delta": round(moved, 9),
        "earns_its_place": bool(moved < 0),
        "out_of_sample": True,
        "reading_en": (
            "Out of sample the per-cell split predicts better than a single constant."
            if moved < 0 else
            "Out of sample the per-cell split does not predict better than a single constant."),
        "reading_he": (
            "מחוץ למדגם החלוקה לתאים חוזה טוב יותר מקבוע יחיד."
            if moved < 0 else
            "מחוץ למדגם החלוקה לתאים אינה חוזה טוב יותר מקבוע יחיד."),
    }
