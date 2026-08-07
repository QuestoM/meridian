"""The comparison as a screen reads it, published where a screen can import it.

**Why this file exists.** Everything this piece measures lives at a terminal or
in ``models/releases/``, and neither is reachable from a browser. The model
console's routes and its panel registry are P7's and frozen, so this piece
cannot publish a route and cannot add a section to that console's rail. What it
can do is write the measurement into its own frontend row, where its own panel
imports it at build time, and cross-check that snapshot against the routes the
console already publishes so a reader is never shown a figure whose subject has
changed under it.

**Three states, and the screen has to be able to reach all three.** The board
carries the digest of every artifact each figure was measured on. The panel asks
the live route for the digests the server is serving now and compares them. Same
digests is a current comparison; different digests is a stale one, named as
stale with the artifact that moved; no answer from the route is unknown, and
unknown is not stale. None of the three is a guess.

**Nothing here is an act and nothing here can name one.** A published board that
carried the terminal's own command lines would put the path into the training act
inside JavaScript that every browser downloads, and the wall on that act is a
route wall, not a bundler one. So the payload is checked before it is written and
the write is refused if any name that reaches this act appears anywhere in it.
That is the same guard as the ownership one, at the same place: the line that
writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scripts import adopt_candidate_ownership as ownership
from scripts import adopt_candidate_registry as registry
from scripts import adopt_candidate_rescore as rescore

# Where the panel imports it from. Inside this piece's own frontend row.
BOARD_DIR = "tv-break-dashboard/src/model/candidates"
BOARD_FILE = "candidate-board.json"

# How many cells travel with each candidate. The full table is 36 rows per
# artifact and the whole point of the ranking is that a handful carry the
# movement, so the screen takes the ranked head and says how many it left.
TOP_CELLS = 8

# Every name a surface would have to say to reach the training act. The same
# three the training-line suite searches the shipped frontend for.
NAMES_THAT_REACH_THE_ACT = ("adopt_candidate", "adopt-candidate", "adoptCandidate")


class ActNamedInAPublishedFile(Exception):
    """Raised when a payload bound for the browser names the training act."""


def _short(digest: Any) -> str:
    return str(digest or "")[:12]


def _money(money: dict[str, Any]) -> dict[str, Any]:
    """The money block a screen reads, tri-state and with its scope.

    ``revenue_delta`` is None on a stale row by construction, and the magnitude
    travels under its own name, so a screen can render "stale, last measured X"
    without a reader being able to mistake X for a current figure.
    """
    scope = money.get("scope") or {}
    return {
        "state": money.get("state"),
        "revenue_delta": money.get("revenue_delta"),
        "last_known_revenue_delta": money.get("last_known_revenue_delta"),
        "measured_at": money.get("measured_at"),
        "rows": scope.get("rows"),
        "basis": scope.get("basis"),
        "whole_plan_delta": money.get("whole_plan_delta"),
        "reason_en": money.get("reason_en"),
        "reason_he": money.get("reason_he"),
    }


def _decision(row: dict[str, Any]) -> dict[str, Any]:
    """The verdict on record, without the free text on it.

    The steward's own sentence is not carried into the bundle. It is unbounded
    text written at a terminal, the console already renders it from the store,
    and a screen that repeats it here would be publishing it twice from two
    sources that can disagree.
    """
    latest = row.get("latest_decision") or {}
    return {
        "state": latest.get("decision"),
        "recorded_at": latest.get("recorded_at"),
        "actor": latest.get("actor"),
        "count": row.get("decisions") or 0,
        "on_rescore": bool(row.get("decision_on_rescore")),
    }


def _cells(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: -abs(float(row.get("squared_error_delta") or 0.0)))
    head = ranked[:TOP_CELLS]
    return {
        "compared": summary.get("cells_compared"),
        "moved": summary.get("cells_moved"),
        "added": summary.get("cells_added") or [],
        "dropped": summary.get("cells_dropped") or [],
        "max_abs_delta": summary.get("max_abs_delta"),
        "max_abs_delta_at": summary.get("max_abs_delta_at"),
        "cancelled_share": summary.get("cancelled_share"),
        "concentrated": summary.get("concentrated"),
        "carries_the_move": summary.get("carries_the_move") or [],
        "carries_share": summary.get("carries_share"),
        "reading_en": summary.get("reading_en"),
        "reading_he": summary.get("reading_he"),
        "key_shape_en": summary.get("key_shape_en"),
        "key_shape_he": summary.get("key_shape_he"),
        "top": [{key: row.get(key) for key in
                 ("cell", "shipped", "candidate", "delta", "breaks",
                  "squared_error_delta", "share_of_absolute")} for row in head],
        "top_of": len(ranked),
    }


def _candidate(row: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    deltas = scored.get("cell_deltas") or {}
    return {
        "id": row.get("id"),
        "file": row.get("file"),
        "bytes": row.get("bytes"),
        "sha256": row.get("sha256"),
        "short": _short(row.get("sha256")),
        "computed_at": row.get("computed_at"),
        "breaks_fitted_on": row.get("breaks_fitted_on"),
        # Both halves of what this row does not share with the rows beside it:
        # how much of the evaluation was in its own fit, and what its producer
        # recorded about adopting it. The screen reads the first as a caveat on
        # the comparison and the second as a note about the artifact alone.
        "fit_basis": row.get("fit_basis"),
        "self_reported": row.get("self_reported"),
        "rmse": row.get("rmse"),
        "rmse_delta": row.get("rmse_delta"),
        "paired_statistic": row.get("paired_statistic"),
        "paired_bar": row.get("paired_bar"),
        "fold_dispersion": row.get("fold_dispersion"),
        "breaks_improved": (scored.get("paired") or {}).get("breaks_improved"),
        "breaks_worsened": (scored.get("paired") or {}).get("breaks_worsened"),
        "verdict": row.get("verdict"),
        "verdict_en": row.get("verdict_en"),
        "verdict_he": row.get("verdict_he"),
        "rule_en": row.get("rule_en"),
        "rule_he": row.get("rule_he"),
        "duplicate_of": row.get("duplicate_of") or [],
        "adopted": bool(row.get("adopted")),
        "money": _money(row.get("money") or {}),
        "decision": _decision(row),
        "cells": _cells((deltas.get("summary") or {}), deltas.get("rows") or []),
    }


def board(paths: Optional[rescore.Paths] = None,
          payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The whole comparison as one payload, with no act and no command in it."""
    paths = paths or rescore.Paths()
    payload = payload if payload is not None else registry.registry(paths)
    stored = rescore.load_rescore(paths) or {}
    scores = {row.get("id"): row for row in stored.get("candidates") or []}
    version = payload.get("live_version") or {}
    retention = ((version.get("artifacts") or {}).get("retention") or {})
    return {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "measured_at": stored.get("measured_at"),
        "rescore_state": payload.get("rescore_state"),
        "fingerprint": stored.get("fingerprint"),
        "evaluation": payload.get("evaluation"),
        "limit": payload.get("limit"),
        "fit_basis": payload.get("fit_basis"),
        "baselines": payload.get("baselines") or [],
        "cell_structure": payload.get("cell_structure"),
        "structure_finding": payload.get("structure_finding"),
        "duplicate_groups": payload.get("duplicate_groups") or [],
        "shipped": {
            "rmse": (payload.get("shipped") or {}).get("rmse"),
            "sha256": (payload.get("shipped") or {}).get("sha256"),
            "short": _short((payload.get("shipped") or {}).get("sha256")),
            "file": (payload.get("shipped") or {}).get("file"),
            "cells": (payload.get("shipped") or {}).get("cells"),
            "version_id": version.get("id"),
            "version_name": version.get("name"),
            "version_short": version.get("short"),
            "computed_at": retention.get("computed_at"),
        },
        "candidates": [_candidate(row, scores.get(row.get("id")) or {})
                       for row in payload.get("candidates") or []],
    }


def board_path(paths: rescore.Paths) -> Path:
    return paths.root / BOARD_DIR / BOARD_FILE


def offending_names(text: str) -> list[str]:
    return [name for name in NAMES_THAT_REACH_THE_ACT if name in text]


def save_board(payload: dict[str, Any], paths: Optional[rescore.Paths] = None) -> Path:
    """Write the board, refusing a payload that names the act or leaves the row."""
    paths = paths or rescore.Paths()
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    named = offending_names(text)
    if named:
        raise ActNamedInAPublishedFile(
            "this payload is imported by a browser bundle and it names the training act: "
            + ", ".join(named) + ". Nothing was written.")
    target = board_path(paths)
    ownership.guard(paths.root, target, paths.releases_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target
