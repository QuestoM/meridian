"""How much contrast the training data carries, and what is blocked on data.

A gate that reads "off" for want of contrast is not a result, it is a question
nobody could ask. This module answers the two things a steward needs next: how
much contrast the window actually holds, and, for each blocked factor, the
condition that would end the block and the first date on which it could.

Every figure here is read or computed, never asserted:

- The training window and the wartime disclosure come from the module that
  already owns them, so the console and the calendar cannot disagree.
- The retention contrast is the cell count and the per-cell observation counts
  from the artifact's own detail block, plus the pooling collapse the artifact
  measures.
- The audience contrast is the base's own observation count and the number of
  levels each factor learned. Only the operator's own channel is ever named;
  the rest are counted, because a competitor may reach a payload only as an
  unnamed aggregate.
- The unblock dates come from the checked-in Israeli calendar table and the
  operator's own event store, so "roughly when" is a date somebody can look up
  rather than a feeling.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from kairos_api import model_console_artifacts as artifacts

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "calendar_events.csv"


def training_window() -> dict[str, Any]:
    """The measured window and its wartime disclosure, from the one owner of both."""
    metadata = artifacts.retention_metadata()
    try:
        from kairos_api.events_api import (
            CEASEFIRE_DATE,
            POST_CEASEFIRE_TAIL_BREAKS,
            TRAINING_WINDOW_END,
            TRAINING_WINDOW_START,
        )
    except Exception as exc:  # pragma: no cover - defensive, a read must not 500
        logger.warning("training window constants unavailable (%s)", exc)
        return {"available": False,
                "reason_en": "The training window constants could not be read in this build.",
                "reason_he": "לא ניתן היה לקרוא את קבועי חלון האימון בגרסה הזו."}
    total = metadata.get("total_breaks_measured")
    total_int = int(total) if isinstance(total, (int, float)) else None
    tail_pct = (round(100.0 * POST_CEASEFIRE_TAIL_BREAKS / total_int, 2)
                if total_int else None)
    return {
        "available": True,
        "start": TRAINING_WINDOW_START.isoformat(),
        "end": TRAINING_WINDOW_END.isoformat(),
        "days": (TRAINING_WINDOW_END - TRAINING_WINDOW_START).days + 1,
        "total_breaks_measured": total_int,
        "ceasefire_date": CEASEFIRE_DATE,
        "post_ceasefire_breaks": POST_CEASEFIRE_TAIL_BREAKS,
        "post_ceasefire_pct": tail_pct,
        "headline_en": f"The whole training window was measured under wartime conditions. The ceasefire took effect on {CEASEFIRE_DATE}, leaving {POST_CEASEFIRE_TAIL_BREAKS} of {total_int} measured breaks after it.",
        "headline_he": f"כל חלון האימון נמדד בתנאי מלחמה. הפסקת האש נכנסה לתוקף בתאריך {CEASEFIRE_DATE}, ונותרו אחריה {POST_CEASEFIRE_TAIL_BREAKS} מתוך {total_int} ברייקים נמדדים.",
    }


def retention_contrast() -> dict[str, Any]:
    """Cells, observations per cell, and how far the cells pool together."""
    payload = artifacts.read_artifact(artifacts.RETENTION_ARTIFACT) or {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
    counts = sorted(int(cell["n"]) for cell in detail.values()
                    if isinstance(cell, dict) and isinstance(cell.get("n"), (int, float)))
    tau2 = metadata.get("between_cell_variance_tau2")
    within = metadata.get("pooled_within_variance")
    ratio = None
    if isinstance(tau2, (int, float)) and isinstance(within, (int, float)) and within:
        ratio = tau2 / within
    return {
        "available": bool(counts),
        "cells": len(detail),
        "observations": sum(counts) if counts else None,
        "per_cell_min": counts[0] if counts else None,
        "per_cell_median": statistics.median(counts) if counts else None,
        "per_cell_max": counts[-1] if counts else None,
        "cells_under_ten": sum(1 for value in counts if value < 10) if counts else None,
        "negative_cells": metadata.get("negative_cells"),
        "between_cell_variance_tau2": tau2,
        "pooled_within_variance": within,
        "contrast_ratio": None if ratio is None else round(ratio, 6),
        "pooling_method": metadata.get("pooling_method"),
        "learned_pseudo_count": metadata.get("learned_pseudo_count"),
        "note_en": "The contrast ratio is the artifact's own between-cell variance over its within-cell variance. Near zero means the cells carry almost no signal the pooled constant does not already carry.",
        "note_he": "יחס הניגודיות הוא השונות בין התאים חלקי השונות בתוך התא, כפי שהקובץ מודד אותן. ערך קרוב לאפס פירושו שהתאים כמעט אינם נושאים אות שהקבוע המאוחד אינו נושא כבר.",
    }


def audience_contrast() -> dict[str, Any]:
    """The audience base's own observation count and how many levels it learned.

    Level counts only. The nested maps are keyed by channel name and would name
    rival channels, so nothing here emits a key: the operator's own channel is
    named because the artifact records it as owned, and every other channel is
    a count.
    """
    payload = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
    if not base:
        return {"available": False,
                "reason_en": "No trained audience model artifact is on disk.",
                "reason_he": "אין בדיסק קובץ מודל קהל מאומן."}
    factors = base.get("factors") if isinstance(base.get("factors"), dict) else {}
    levels: list[dict[str, Any]] = []
    for name in sorted(factors):
        block = factors[name]
        cells = block.get("cells") if isinstance(block, dict) else None
        levels.append({
            "factor": name,
            "levels": len(cells) if isinstance(cells, dict) else None,
            "shape": "cells" if isinstance(cells, dict) else "scalar",
        })
    channel_maps = {key: len(value) for key, value in base.items()
                    if isinstance(value, dict) and key.endswith(("_log", "hist_channel",
                                                                 "hist_genre", "hist_series",
                                                                 "hist_slot"))}
    return {
        "available": True,
        "observations": base.get("n_observations"),
        "kind": base.get("kind"),
        "shrinkage_k": base.get("shrinkage_k"),
        "tvr_floor": base.get("tvr_floor"),
        "operator_channel": base.get("owned_channel"),
        "channels_in_base": max(channel_maps.values()) if channel_maps else None,
        "factor_levels": levels,
        "note_en": "Channels other than the operator's own are counted, never named.",
        "note_he": "ערוצים שאינם של המפעיל נספרים ולעולם אינם מצוינים בשם.",
    }


# ---------------------------------------------------------------------------
# The blocked register: what would end each block, and the first date it could
# ---------------------------------------------------------------------------


def _calendar_rows():
    try:
        from kairos.data.israel_calendar import load_calendar

        return load_calendar()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("israel calendar unavailable (%s)", exc)
        return ()


def _next_range(after: date, predicate) -> "dict[str, Any] | None":
    """The first calendar range starting after ``after`` that matches."""
    candidates = [row for row in _calendar_rows()
                  if row.start_date > after and predicate(row)]
    if not candidates:
        return None
    row = min(candidates, key=lambda item: item.start_date)
    return {"start": row.start_date.isoformat(), "end": row.end_date.isoformat(),
            "name_he": row.name_he, "kind": row.kind}


def _event_days() -> "set[date]":
    """Every day covered by an active operator event, from the event store."""
    if not EVENTS_PATH.is_file():
        return set()
    import csv

    days: set[date] = set()
    try:
        with EVENTS_PATH.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("active", "")).strip().lower() not in ("true", "1", "yes"):
                    continue
                try:
                    start = date.fromisoformat(str(row.get("start_date", "")).strip())
                except ValueError:
                    continue
                end_raw = str(row.get("end_date", "")).strip()
                try:
                    end = date.fromisoformat(end_raw) if end_raw else start
                except ValueError:
                    end = start
                if end < start:
                    end = start
                # An open-ended event covers everything from its start, which is
                # exactly why the window has no ordinary days to contrast against.
                span = min((end - start).days, 3660)
                for offset in range(span + 1):
                    days.add(start + timedelta(days=offset))
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("event store unreadable (%s)", exc)
    return days


def _event_free_days(window: dict[str, Any]) -> dict[str, Any]:
    if not window.get("available"):
        return {"measurable": False}
    start = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    covered = _event_days()
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    free = [day for day in days if day not in covered]
    following = end + timedelta(days=1)
    first_free = None
    for offset in range(0, 3650):
        probe = following + timedelta(days=offset)
        if probe not in covered:
            first_free = probe.isoformat()
            break
    return {"measurable": True, "days_in_window": len(days), "event_free_days": len(free),
            "first_event_free_day_after_window": first_free}


def _season_span(window_end: date) -> "tuple[Optional[int], Optional[str]]":
    """How many seasons the window spans, and the first day of the next one.

    Both come from the engine's own season bands rather than from a month
    arithmetic invented here, so the console cannot disagree with the model
    about what a season is.
    """
    try:
        from kairos.data.israel_calendar import season_of
    except Exception:  # pragma: no cover - defensive
        return None, None
    window = training_window()
    if not window.get("available"):
        return None, None
    start = date.fromisoformat(window["start"])
    days = [start + timedelta(days=offset) for offset in range((window_end - start).days + 1)]
    seasons = {season_of(day) for day in days}
    here = season_of(window_end)
    for offset in range(1, 400):
        probe = window_end + timedelta(days=offset)
        if season_of(probe) != here:
            return len(seasons), probe.isoformat()
    return len(seasons), None


def blocked_register(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per gate that could not run, with the condition and the date."""
    window = training_window()
    end = date.fromisoformat(window["end"]) if window.get("available") else date.today()
    events = _event_free_days(window)
    rows: list[dict[str, Any]] = []
    for gate in gates:
        if gate["state"] != "no_contrast":
            continue
        rows.append({
            "gate_id": gate["id"],
            "label_en": gate["label_en"],
            "label_he": gate["label_he"],
            "reason": gate["reason"],
            **_unblock(gate["family"], end, events),
        })
    return rows


def _unblock(family: str, window_end: date, events: dict[str, Any]) -> dict[str, Any]:
    if family == "calendar_hanukkah":
        found = _next_range(window_end, lambda row: row.kind == "hanukkah")
        return _condition(
            "A training window that contains Hanukkah days as well as ordinary ones.",
            "חלון אימון שמכיל ימי חנוכה לצד ימים רגילים.",
            "kairos/config/israel_calendar.csv", found)
    if family == "calendar_school_and_chol_hamoed":
        found = _next_range(window_end, lambda row: row.is_school_holiday)
        return _condition(
            "A training window that contains school-holiday or Chol HaMoed days as well as ordinary ones.",
            "חלון אימון שמכיל ימי חופשה או חול המועד לצד ימים רגילים.",
            "kairos/config/israel_calendar.csv", found)
    if family == "season":
        seasons, next_season = _season_span(window_end)
        found = ({"start": next_season, "end": next_season, "name_he": "תחילת העונה הבאה",
                  "kind": "season_change"} if next_season else None)
        return _condition(
            "A training window spanning at least two Israeli seasons with ten or more observations in each.",
            "חלון אימון שפרוס על שתי עונות ישראליות לפחות, עם עשר תצפיות ומעלה בכל אחת.",
            "kairos/data/israel_calendar.py season bands", found,
            extra={"seasons_in_window": seasons})
    if family in ("operator_events", "event_layer"):
        first = events.get("first_event_free_day_after_window") if events.get("measurable") else None
        found = {"start": first, "end": first, "name_he": "יום ללא אירוע פעיל",
                 "kind": "event_free"} if first else None
        return _condition(
            "Days with no active operator event in the training window, to contrast against the days that have one.",
            "ימים ללא אירוע מפעיל פעיל בחלון האימון, כדי להשוות מולם את הימים שיש בהם אירוע.",
            "data/calendar_events.csv", found,
            extra={"days_in_window": events.get("days_in_window"),
                   "event_free_days_in_window": events.get("event_free_days")})
    return _condition(
        "Observations on both sides of the question, which this window does not hold.",
        "תצפיות בשני צדי השאלה, שאין בחלון הזה.",
        "the training window itself", None)


def _condition(condition_en: str, condition_he: str, source: str,
               found: Optional[dict[str, Any]], extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "condition_en": condition_en,
        "condition_he": condition_he,
        "source": source,
        "earliest": found,
        "earliest_state": "dated" if found and found.get("start") else "unknown",
        "evidence": extra or {},
    }


def coverage(gates: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The whole coverage answer: window, both contrasts, and the blocked register."""
    if gates is None:
        from kairos_api import model_console_gates

        gates = model_console_gates.ledger()["gates"]
    return {
        "window": training_window(),
        "retention": retention_contrast(),
        "audience": audience_contrast(),
        "blocked": blocked_register(gates),
    }
