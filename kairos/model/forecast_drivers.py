"""Why the forecast is the number it is: the multiplicative decomposition.

:meth:`kairos.model.audience_model.AudienceModel.predict_tvr` computes a per-row
log base, adds one delta per activated gate family, exponentiates, and returns a
single float. Every intermediate term is discarded. This module recomputes the
same sum and KEEPS the terms, so the prediction can be shown as what it
structurally is: a starting rating in points, times the channel level, times the
genre (or slot) level, times one multiplier per family that touched the row.

The product of those terms IS the point forecast -- the same exponential of the
same sum -- which the tests assert rather than trust. Two consequences worth
naming:

  * A family that is ON but does not touch this row (no cell for it, or no
    competitor lineup for this date) is not a 1.0 multiplier in the list. It is
    absent from the drivers and present in ``not_applied`` with the row-level
    reason, because "did not apply" and "applied and changed nothing" are
    different facts about a forecast.
  * A family that is OFF is listed with the verdict its own held-out
    measurement returned. Five of the eight are off for ABSENCE OF CONTRAST in a
    one-month window -- no Hanukkah day, one season, every observation on an
    operator-event day -- which is not the same as having been tried and failed,
    and the payload keeps that distinction in the reason text the gate wrote.

Nothing here reads disk or settings; it is pure computation over an already
loaded model and an already scored frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from kairos.model.audience_factors import FAMILIES, cell_deltas_for, family_cells
from kairos.model.audience_frame import attach_pressure
from kairos.model.forecast_basis import FAMILY_LABELS_HE, LEVEL_LABELS_HE


@dataclass(frozen=True)
class FamilyEffects:
    """The per-row family sum, plus everything needed to explain each row.

    ``deltas`` is the total log delta per row and ``applied`` marks the rows an
    activated family actually touched -- exactly the two arrays ``predict_tvr``
    computes internally, so a payload built from these cannot drift from the
    shipped prediction.
    """

    deltas: np.ndarray
    applied: np.ndarray
    contributions: dict[str, dict[str, Any]]


def apply_families(
    model, scored: pd.DataFrame, *, lineup_frame_fn: Optional[Callable] = None
) -> FamilyEffects:
    """Sum every activated family's log delta over ``scored``, keeping the parts.

    Mirrors ``predict_tvr``'s loop term for term: gate off or factor missing is
    skipped, the competitor family rides the continuous pressure (NaN pressure
    means not-applicable for that row), every other family reads a cell table.
    """
    deltas = np.zeros(len(scored))
    applied = np.zeros(len(scored), dtype=bool)
    contributions: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        gate = dict(model.gates.get(family, {}))
        payload = model.factors.get(family)
        if gate.get("verdict") != "on" or payload is None:
            contributions[family] = {"gate": gate, "row_deltas": None}
            continue
        if family == "competitor_lineup":
            pressure, reason = attach_pressure(
                scored, model.owned_channel, lineup_frame_fn=lineup_frame_fn,
            )
            if pressure is None:
                contributions[family] = {
                    "gate": gate, "row_deltas": None, "absent_reason": reason,
                }
                continue
            from kairos.model.audience_factors import pressure_deltas_for

            family_deltas = pressure_deltas_for(pressure, payload)
            deltas += family_deltas
            applied |= np.isfinite(pressure)
            contributions[family] = {
                "gate": gate, "row_deltas": family_deltas,
                "row_applied": np.isfinite(pressure), "pressure": pressure,
            }
            continue
        cells = family_cells(scored, family)
        family_deltas = cell_deltas_for(cells, payload["cells"])
        deltas += family_deltas
        row_applied = np.array([cell is not None for cell in cells])
        applied |= row_applied
        contributions[family] = {
            "gate": gate, "row_deltas": family_deltas, "row_applied": row_applied,
            "cells": cells, "table": payload["cells"],
        }
    return FamilyEffects(deltas=deltas, applied=applied, contributions=contributions)


def series_answered_at(contributions: dict[str, dict[str, Any]], position: int) -> bool:
    """Whether the series factor had a TRAINED cell for this row.

    A series present in the frame but absent from the fitted table contributed
    exactly 1.0, so it did not answer; the level below it did.
    """
    record = contributions.get("series", {})
    cells = record.get("cells")
    if cells is None:
        return False
    cell = cells[position]
    return bool(cell is not None and cell in (record.get("table") or {}))


# ----------------------------------------------------------------- base levels

def base_terms(base, channel: str, genre: str, band: str) -> tuple[str, dict[str, float]]:
    """The base level that answers, and the log terms that build it.

    Mirrors :meth:`AudienceBase.log_base` exactly: genre inside the channel,
    falling to the channel's slot band, falling to the channel, falling to the
    grand mean. The terms are DIFFERENCES, so their sum is the level itself.
    """
    terms = {"global": float(base.global_log), "channel": 0.0, "level": 0.0}
    channel_log = base.channel_log.get(channel)
    if channel_log is not None:
        terms["channel"] = float(channel_log) - float(base.global_log)
    parent = float(channel_log if channel_log is not None else base.global_log)
    by_genre = base.genre_log.get(channel, {})
    if genre in by_genre:
        terms["level"] = float(by_genre[genre]) - parent
        return "genre", terms
    by_slot = base.slot_log.get(channel, {})
    if band in by_slot:
        terms["level"] = float(by_slot[band]) - parent
        return "slot", terms
    return ("channel" if channel_log is not None else "global"), terms


def historical_level(base, channel: str, row: pd.Series) -> str:
    """Which plain-mean table answers when no activated family touches the row."""
    for level, table, key in (
        ("series", base.hist_series, str(row["series_key"])),
        ("genre", base.hist_genre, str(row["genre"])),
        ("slot", base.hist_slot, str(row["slot_band"])),
    ):
        if key and key in table.get(channel, {}):
            return level
    return "channel" if channel in base.hist_channel else "global"


# --------------------------------------------------------------------- drivers

def drivers_for(
    row: pd.Series, position: int, *, terms: dict[str, float], base_level: str,
    contributions: dict[str, dict[str, Any]], model_basis: bool,
    expected: float, resolved: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The multiplicative decomposition, and the levels that were tried.

    The first driver is the starting rating in points; every later driver is a
    multiplier. Their product is the point forecast. On the historical-mean path
    there is nothing to decompose and the single driver says so rather than
    inventing terms.
    """
    fallbacks: list[str] = []
    if not model_basis:
        return [{
            "key": "historical_mean", "kind": "base",
            "label_he": f"ממוצע היסטורי ברמת {LEVEL_LABELS_HE.get(resolved, resolved)}",
            "label_en": f"historical mean at the {resolved} level",
            "value_tvr": round(float(expected), 4),
            "note_he": "אף משפחה פעילה לא נגעה בשורה הזאת, ולכן המספר הוא הממוצע ההיסטורי ולא פירוק מודל",
        }], fallbacks

    drivers: list[dict[str, Any]] = [{
        "key": "global", "kind": "base",
        "label_he": "רמת הבסיס של כל המדידה", "label_en": "grand mean level",
        "value_tvr": round(math.exp(terms["global"]), 4),
        "log_term": round(terms["global"], 6),
    }, {
        "key": "channel", "kind": "multiplier",
        "label_he": "רמת הערוץ", "label_en": "channel level",
        "multiplier": round(math.exp(terms["channel"]), 4),
        "log_term": round(terms["channel"], 6),
    }]
    if base_level in ("genre", "slot"):
        detail = str(row["genre"]) if base_level == "genre" else str(row["slot_band"])
        drivers.append({
            "key": base_level, "kind": "multiplier",
            "label_he": f"{LEVEL_LABELS_HE[base_level]} {detail}",
            "label_en": f"{base_level} {detail}",
            "multiplier": round(math.exp(terms["level"]), 4),
            "log_term": round(terms["level"], 6), "detail": detail,
        })
    else:
        fallbacks.append(
            "no genre or slot level for this programme on this channel; "
            f"the {base_level} level answered"
        )

    for family in FAMILIES:
        record = contributions.get(family, {})
        row_deltas = record.get("row_deltas")
        if row_deltas is None:
            continue
        row_applied = record.get("row_applied")
        if row_applied is not None and not bool(row_applied[position]):
            continue
        delta = float(row_deltas[position])
        entry: dict[str, Any] = {
            "key": family, "kind": "multiplier", "family": family,
            "label_he": FAMILY_LABELS_HE.get(family, family),
            "label_en": family.replace("_", " "),
            "multiplier": round(math.exp(delta), 4), "log_term": round(delta, 6),
            "held_out_delta_pct": record.get("gate", {}).get("held_out_delta_pct"),
        }
        if family == "competitor_lineup":
            # The lineup's own frame carries rival programme titles. Only the
            # scalar pressure crosses into a payload, never a title.
            entry["pressure"] = round(float(record["pressure"][position]), 4)
        else:
            cell = (record.get("cells") or [None] * (position + 1))[position]
            entry["cell"] = cell
            if cell is not None and cell not in (record.get("table") or {}):
                entry["note_he"] = "התא לא נצפה באימון; המקדם הוא 1.0 בדיוק"
                fallbacks.append(
                    f"{family} cell {cell!r} was never observed in training; "
                    "the factor contributed exactly 1.0"
                )
            if family == "series":
                entry["label_he"] = f"סדרה {row['series_key']}"
        drivers.append(entry)
    return drivers, fallbacks


def not_applied_for(
    contributions: dict[str, dict[str, Any]], position: int
) -> list[dict[str, Any]]:
    """Every family that did not move this number, and the measured reason."""
    out: list[dict[str, Any]] = []
    for family in FAMILIES:
        record = contributions.get(family, {})
        gate = record.get("gate", {}) or {}
        row_deltas = record.get("row_deltas")
        row_applied = record.get("row_applied")
        if row_deltas is not None and (row_applied is None or bool(row_applied[position])):
            continue
        entry = {
            "family": family,
            "label_he": FAMILY_LABELS_HE.get(family, family),
            "verdict": gate.get("verdict", "unknown"),
            "reason": gate.get("reason", ""),
            "held_out_delta_pct": gate.get("held_out_delta_pct"),
            "measured_at": gate.get("measured_at"),
        }
        if record.get("absent_reason"):
            entry["row_reason"] = record["absent_reason"]
        elif row_deltas is not None:
            entry["row_reason"] = (
                "the family is active but does not apply to this row "
                "(no cell, or no competitor lineup for this date)"
            )
        out.append(entry)
    return out
