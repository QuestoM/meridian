"""Model console: the measured retention coefficients and their drift.

Moved verbatim from catalog_api.py as part of the wave-zero router split. The
payload reads the trained coefficient artifact and reports the pooling note and
the level drift straight from its metadata; when the artifact predates the drift
monitor the block is an honest unavailable, never a fabricated verdict.

**Walled, and answering rather than refusing.** The payload is the coefficient
artifact's own metadata, every gate reason and the drift monitor, which is
training content, so the content sits behind ``affiliation = company``. It is
one of the four open reads section 4.5 of the rebuild specification names, and
it was the widest of them: the dashboard fetched it on every page load for
every account.

That last sentence is why this route does not answer 403. The shell fetches it
unconditionally alongside ten other endpoints (``src/shell/use-kairos-data.js``,
frozen), and any non-200 among them flips ``partial`` true, which renders "Some
data failed to load" on every page for the whole session. Measured before this
change: of the eleven endpoints the shell fetches, exactly one was non-200 for a
channel account and it was this one, so the wall's first act on the product was
to put a permanent failure banner in front of every operator.

So the wall closes the content and not the door. A channel account gets 200 with
the tri-state and nothing else: ``state = "unavailable"`` and the reason in the
words the refusal itself uses. No coefficient, no drift block, no metadata, and
none of section 4.2's training lexicon, which a test greps for. A company
account gets the same measured payload it got before, plus the same tri-state
field, which reads ``real`` when the artifact was read and ``unknown`` when
there is no artifact to read. One other string moved: the honest unavailable the
drift block carries when the artifact predates the monitor now says "train"
rather than a retired verb, which section 4.2's verb test asks for.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from kairos_api.core import MODELS_DIR, _safe_number, _signature
from kairos_api.model_console_api import MODEL_WALL

# The artifact this route reports on, named once so the honest unknown state can
# say which file is missing rather than gesturing at one.
COEFFICIENTS_ARTIFACT = "models/tv_break_coefficients.json"

logger = logging.getLogger(__name__)

router = APIRouter()


def _segment_key(channel_name: str) -> tuple[str, str, str] | None:
    parts = str(channel_name or "").split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def _weighted_impact_rows(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        segment = str(item.get(key) or "")
        coefficient = _safe_number(item.get("coefficient"), math.nan)
        if not segment or not math.isfinite(coefficient):
            continue
        grouped.setdefault(segment, []).append(item)

    rows: list[dict[str, Any]] = []
    for segment, values in grouped.items():
        total_weight = 0
        weighted_coefficient = 0.0
        weighted_raw = 0.0
        ci_low: list[float] = []
        ci_high: list[float] = []
        for item in values:
            sample_count = max(1, int(_safe_number(item.get("n"), 1)))
            coefficient = _safe_number(item.get("coefficient"), 0.0)
            raw_delta = _safe_number(item.get("raw_delta"), coefficient)
            weighted_coefficient += coefficient * sample_count
            weighted_raw += raw_delta * sample_count
            total_weight += sample_count
            low = _safe_number(item.get("ci_low"), math.nan)
            high = _safe_number(item.get("ci_high"), math.nan)
            if math.isfinite(low):
                ci_low.append(low)
            if math.isfinite(high):
                ci_high.append(high)
        if total_weight <= 0:
            continue
        rows.append(
            {
                "segment": segment,
                "average_coefficient": round(weighted_coefficient / total_weight, 6),
                "average_raw_delta": round(weighted_raw / total_weight, 6),
                "sample_count": total_weight,
                "channel_count": len(values),
                "ci_low": round(min(ci_low), 6) if ci_low else None,
                "ci_high": round(max(ci_high), 6) if ci_high else None,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["average_coefficient"])), reverse=True)


def _pooling_note(metadata: dict[str, Any]) -> str | None:
    """Honest disclosure that the per-cell retention effects collapse toward one
    pooled constant. Empirical Bayes shrinks the programme-type x position x length
    cells because the between-cell variance sits far below the within-cell variance,
    so the cells share almost all of their signal. Numbers are read straight from
    the coefficient artifact metadata, never hand-set."""
    tau2 = _safe_number(metadata.get("between_cell_variance_tau2"), math.nan)
    within = _safe_number(metadata.get("pooled_within_variance"), math.nan)
    if not math.isfinite(tau2) or not math.isfinite(within) or within <= 0:
        return None
    cells = int(_safe_number(metadata.get("channels"), 0)) or None
    method = str(metadata.get("pooling_method") or "empirical_bayes").replace("_", " ")
    cell_phrase = f"{cells} " if cells else ""
    return f"The {cell_phrase}(programme type x position x length) cells pool to approximately one shared constant under {method}: between-cell variance tau^2 = {tau2:.2e} sits far below within-cell variance {within:.3f}, so the per-cell effects collapse toward a single pooled value."


def _load_measured_impact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": "legacy_csv",
            "pooling_note": None,
            "program_type": [],
            "position": [],
            "length": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "source": "legacy_csv",
            "pooling_note": None,
            "program_type": [],
            "position": [],
            "length": [],
        }

    details = payload.get("detail", {})
    items: list[dict[str, Any]] = []
    for name, raw in details.items():
        if not isinstance(raw, dict):
            continue
        segment = _segment_key(str(raw.get("channel_name") or name))
        if not segment:
            continue
        program_type, position, length = segment
        items.append(
            {
                **raw,
                "program_type": program_type,
                "position": position,
                "length": length,
            }
        )

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "source": payload.get("method") or "measured_coefficients",
        "metadata": metadata,
        "pooling_note": _pooling_note(metadata),
        "program_type": _weighted_impact_rows(items, "program_type"),
        "position": _weighted_impact_rows(items, "position"),
        "length": _weighted_impact_rows(items, "length"),
    }


@lru_cache(maxsize=16)
def _impact_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    summary = _load_measured_impact_summary(MODELS_DIR / "tv_break_coefficients.json")
    # Weekly level drift of the coefficient measurement base, measured at
    # rebuild time and carried in the artifact metadata (see
    # kairos.model.drift_monitor and docs/model-validation/
    # uncertainty-calibration.md finding 4). Echoed here for the Data page;
    # when the artifact predates the monitor (or carries no metadata) the
    # block is an honest "unavailable", never a fabricated verdict.
    metadata = summary.get("metadata")
    drift = metadata.get("level_drift") if isinstance(metadata, dict) else None
    if not isinstance(drift, dict) or not drift:
        drift = {
            "status": "unavailable",
            "reason": "the coefficients artifact carries no level-drift measurement; train the model to compute it",
        }
    measured = any(summary.get(axis) for axis in ("program_type", "position", "length"))
    nothing_measured = f"{COEFFICIENTS_ARTIFACT} holds no measured cells to report; train the model to produce them"
    return {
        "state": "real" if measured else "unknown",
        "state_reason": None if measured else nothing_measured,
        "coefficient_impacts": summary,
        "drift": drift,
    }


def _walled_payload() -> dict[str, Any]:
    """What an account on the other side of the line gets: the state, and why.

    A fresh dict per call, because the measured branch is cached and a caller
    that mutated a shared one would poison every later read. Nothing here names
    a coefficient, a gate or a drift measurement, so a channel account's copy of
    this response returns zero hits of section 4.2's lexicon.
    """
    return {
        "state": "unavailable",
        "state_reason": MODEL_WALL.detail,
    }


@router.get("/api/impact", tags=["catalog"])
def impact(request: Request) -> dict[str, Any]:
    """The measured coefficients for a company account, the tri-state for anyone else.

    ``MODEL_WALL.read_reason`` is the same gate ``guard()`` applies, consulted
    directly so the refusal becomes a body rather than a status. There is no
    write on this route, so nothing here carries ``can_edit``: the only question
    it answers is whether this account may see the measurement, and ``state``
    answers it.
    """
    if not MODEL_WALL.allows_read(request):
        return _walled_payload()
    return _impact_cached(
        _signature([MODELS_DIR / "tv_break_coefficients.json"])
    )
