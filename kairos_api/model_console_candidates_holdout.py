"""What each gate verdict was decided on, shipped artifact against candidate.

A candidate can agree with the shipped model on every gate flag and on every
coefficient cell while the figures those flags were decided on have moved a long
way. Measured on the artifacts in this tree,
``models/candidates/tv_break_coefficients_placebo_corrected.json`` is exactly
that case: it moves no cell and flips no flag, and its series gate was decided
on a genre RMSE of 0.24424481 against the shipped 0.24199622, a series RMSE of
0.26463353 against 0.26238666, and 506 held-out breaks against 2,532. A console
that reported that candidate as "no gate decides differently" would tell a model
steward something false, which is what this comparison exists to prevent.

It is not a second reading of the artifacts. It runs the console's own gate
ledger over each artifact's metadata and pairs the rows by gate id, so the
figures here and the figures in the gate table are the same figures by
construction. Every leaf of each gate's own record is compared: a number that
moved is shown as a pair, a number one side does not record is marked absent on
that side, and each artifact's own sentence is carried verbatim on both sides,
never merged and never rewritten.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from kairos_api import model_console_gates as gates

_EMPTY_SIDE: dict[str, Any] = {"reason": "", "basis": {}}


def _leaves(record: Any) -> dict[str, Any]:
    """A gate record's own figure slots, under the artifact's own key names.

    A slot is a key whose value is a number or an explicit null. The null is the
    honest "this gate ran and produced no figure" the event gate records, and it
    is a different fact from a key the artifact does not carry at all, so the two
    are kept apart. Strings are carried as the sentence instead, and a boolean is
    a verdict, which the gate-delta list already reports.
    """
    if not isinstance(record, dict):
        return {}
    return {key: value for key, value in record.items()
            if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))}


def _comparable(value: Any) -> Any:
    return round(value, 9) if isinstance(value, float) else value


def _digits(*values: Any) -> int:
    """Zero for a count the artifact stores as an integer, six for a measurement."""
    numbers = [value for value in values
               if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return 0 if numbers and all(isinstance(value, int) for value in numbers) else 6


def _figures(before: dict[str, Any], after: dict[str, Any],
             skip: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted((set(before) | set(after)) - set(skip)):
        shipped, candidate = before.get(key), after.get(key)
        rows.append({
            "key": key,
            "shipped": shipped,
            "candidate": candidate,
            "shipped_absent": key not in before,
            "candidate_absent": key not in after,
            "moved": ((key in before) != (key in after)
                      or _comparable(shipped) != _comparable(candidate)),
            "digits": _digits(shipped, candidate),
        })
    return rows


def _pair(identity: dict[str, Any], left: dict[str, Any], right: dict[str, Any],
          skip: Iterable[str]) -> dict[str, Any]:
    before = _leaves((left.get("basis") or {}).get("detail"))
    after = _leaves((right.get("basis") or {}).get("detail"))
    figures = _figures(before, after, skip)
    reason_shipped = str(left.get("reason") or "")
    reason_candidate = str(right.get("reason") or "")
    basis = identity.get("basis") or {}
    return {
        "gate_id": identity.get("id"),
        "label_en": identity.get("label_en"),
        "label_he": identity.get("label_he"),
        "statistic_en": basis.get("statistic_en"),
        "statistic_he": basis.get("statistic_he"),
        "figures": figures,
        "reason_shipped": reason_shipped,
        "reason_candidate": reason_candidate,
        "reason_moved": reason_shipped != reason_candidate,
        "shipped_records_nothing": not before and not reason_shipped,
        "candidate_records_nothing": not after and not reason_candidate,
        "moved": (reason_shipped != reason_candidate
                  or any(figure["moved"] for figure in figures)),
    }


def held_out_deltas(shipped_metadata: Optional[dict[str, Any]],
                    candidate_metadata: Optional[dict[str, Any]],
                    skip_keys: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Every gate whose evidence differs, with both sides' figures and sentences.

    ``skip_keys`` are the metadata keys the caller's gate-delta list already
    prints, so no figure is reported twice on one screen as two findings. Gates
    whose evidence is identical are left out for the reason the gate-delta list
    leaves out identical flags: the news is the difference. The order is the gate
    ledger's own, so this block and the gate table read in the same sequence.
    """
    shipped_rows = ({row["id"]: row for row in gates.retention_rows(shipped_metadata)}
                    if shipped_metadata else {})
    candidate_rows = ({row["id"]: row for row in gates.retention_rows(candidate_metadata)}
                      if candidate_metadata else {})
    order = list(shipped_rows) + [key for key in candidate_rows if key not in shipped_rows]
    rows = []
    for gate_id in order:
        left = shipped_rows.get(gate_id)
        right = candidate_rows.get(gate_id)
        row = _pair(left or right or {}, left or _EMPTY_SIDE, right or _EMPTY_SIDE, skip_keys)
        if row["moved"]:
            rows.append(row)
    return rows


def differences(gate_rows: list[dict[str, Any]], held_out: list[dict[str, Any]],
                cells: dict[str, Any]) -> dict[str, Any]:
    """What differs, counted, so no screen can render this candidate as inert.

    Three independent kinds of difference, each countable on its own: a gate flag
    that decides differently, a figure a gate was decided on that moved, and a
    coefficient cell that moved. A candidate that moves none of the three is the
    only one a reader may take as changing nothing.
    """
    return {
        "gate_verdicts": len(gate_rows),
        "held_out_gates": len(held_out),
        "held_out_figures": sum(1 for row in held_out
                                for figure in row["figures"] if figure["moved"]),
        "reasons": sum(1 for row in held_out if row["reason_moved"]),
        "cells_moved": int(cells.get("cells_moved") or 0),
    }
