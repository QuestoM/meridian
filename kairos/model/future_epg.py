"""Future-week competitor EPG: the prediction-time input contract.

The counter-programming covariate trains on PAST competitor programming and
ratings. At prediction time the client usually has the competitors' PROGRAMME
schedules for the coming week (published EPG) but never their future ad breaks.
This module defines how that future schedule enters the model, and what happens
honestly when it is absent.

The file contract
-----------------
Path: ``data/reference/CompetitorProgrammes.xlsx`` (preferred) or
``data/reference/CompetitorProgrammes.csv``. Schema: EXACTLY the reference
``Programmes`` schema, parsed by the same loader:

  * ``Channel``      competitor channel name (must match the Dayparts channel
                     names for the audience-strength feature to find history)
  * ``Title``        programme title (classified into a genre by the same
                     :class:`~kairos.data.classifier.ProgramClassifier`)
  * ``Date``         DD/MM/YYYY
  * ``Start time``   HH:MM:SS
  * ``End time``     HH:MM:SS (crossing midnight is handled by the parser)
  * ``Duration``     optional, seconds
  * ``TVR``          optional and IGNORED here: future ratings do not exist,
                     so audience strength always comes from the historical
                     minute-level curve, never from a claimed future number.

The honest absent state (load-bearing)
--------------------------------------
When no future EPG file is present, :func:`load_future_competitor_epg` returns
``(None, status)`` with ``status["present"] is False`` and a reason, and
:func:`forward_adjustment` contributes EXACTLY 0.0 with ``applied False``. The
covariate never guesses what competitors might air.

The information boundary (law)
------------------------------
Only :data:`~kairos.model.competitor_features.EXTENDED_FORWARD_FEATURES` are
computable here; the file carries programmes, not breaks, so the training-only
rival co-breaking signal cannot leak in. :func:`forward_adjustment`
additionally calls :func:`~kairos.model.competitor_features.assert_forward_only`
so a mislabeled training-only beta fails loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import REFERENCE_DIR, load_programmes
from kairos.model.competitor_features import (
    EXTENDED_FORWARD_FEATURES,
    _category_at,
    _genre_contrast,
    _prog_start,
    _strength,
    assert_forward_only,
    _break_minutes,
    _programme_category_lookup,
)

logger = logging.getLogger(__name__)

FUTURE_EPG_XLSX = REFERENCE_DIR / "CompetitorProgrammes.xlsx"
FUTURE_EPG_CSV = REFERENCE_DIR / "CompetitorProgrammes.csv"


def _resolve_future_epg_path(path: str | Path | None = None) -> Optional[Path]:
    """The future-EPG file to read: explicit path, else xlsx, else csv, else None."""
    if path is not None:
        p = Path(path)
        return p if p.exists() else None
    if FUTURE_EPG_XLSX.exists():
        return FUTURE_EPG_XLSX
    if FUTURE_EPG_CSV.exists():
        return FUTURE_EPG_CSV
    return None


def load_future_competitor_epg(
    path: str | Path | None = None,
) -> tuple[Optional[pd.DataFrame], dict[str, Any]]:
    """Load the future-week competitor EPG, or report its absence honestly.

    Returns ``(frame, status)``. ``frame`` is the parsed programmes frame (same
    columns as :func:`kairos.data.loaders.load_programmes`) or ``None``.
    ``status`` always carries ``present`` (bool), ``path`` (str or None),
    ``rows``, ``channels``, ``window_start``/``window_end`` (ISO date strings
    or None) and a one-line ``reason``. No value is fabricated: an unreadable
    or empty file is reported absent, never silently treated as a schedule.
    """
    resolved = _resolve_future_epg_path(path)
    if resolved is None:
        return None, {
            "present": False,
            "path": None,
            "rows": 0,
            "channels": [],
            "window_start": None,
            "window_end": None,
            "reason": (
                "no future competitor EPG file found "
                f"(looked for {FUTURE_EPG_XLSX.name} / {FUTURE_EPG_CSV.name} under "
                f"{REFERENCE_DIR.name}); the counter-programming covariate contributes nothing"
            ),
        }
    try:
        frame = load_programmes(resolved)
    except Exception as exc:  # noqa: BLE001 - any parse failure is an honest absence
        logger.warning("Future competitor EPG at %s unreadable: %s", resolved, exc)
        return None, {
            "present": False,
            "path": str(resolved),
            "rows": 0,
            "channels": [],
            "window_start": None,
            "window_end": None,
            "reason": f"future competitor EPG at {resolved.name} could not be parsed: {exc}",
        }
    usable = frame[frame["start_dt"].notna()]
    if usable.empty:
        return None, {
            "present": False,
            "path": str(resolved),
            "rows": 0,
            "channels": [],
            "window_start": None,
            "window_end": None,
            "reason": f"future competitor EPG at {resolved.name} carries no parseable rows",
        }
    channels = sorted(str(c) for c in usable["Channel"].dropna().unique())
    return usable.reset_index(drop=True), {
        "present": True,
        "path": str(resolved),
        "rows": int(len(usable)),
        "channels": channels,
        "window_start": usable["start_dt"].min().date().isoformat(),
        "window_end": usable["start_dt"].max().date().isoformat(),
        "reason": f"future competitor EPG loaded from {resolved.name}",
    }


def counterprogramming_features_for_window(
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    epg: Optional[pd.DataFrame],
    classifier: ProgramClassifier,
    baseline: Mapping[tuple[str, int], float],
    own_channel: str,
    own_category: Optional[str] = None,
) -> Optional[dict[str, float]]:
    """The three FORWARD features for one future break window, or None when absent.

    ``epg`` is the frame from :func:`load_future_competitor_epg` (None when the
    file is missing, in which case this returns None so the caller applies no
    adjustment). ``baseline`` is the HISTORICAL audience curve from
    :func:`kairos.model.measure._baseline_levels` (typical rival TVR by
    broadcast minute), the only audience source allowed for a future week.
    ``own_category`` is the genre of the programme the break sits in (the
    planner knows its own schedule); when None the genre-contrast feature is
    0.0, exactly as the training extractor treats an unmatched programme. For
    exact parity with training, anchor it at the break's MIDDLE minute (the
    trainer's convention in
    :func:`~kairos.model.competitor_features.attach_competitor_features`). A
    rival channel present in the EPG but absent from the daypart history adds
    0.0 strength (honest absence, nothing invented).
    """
    if epg is None:
        return None
    rivals = tuple(
        c for c in sorted(str(x) for x in epg["Channel"].dropna().unique())
        if c != own_channel
    )
    if not rivals:
        return None
    lookup = _programme_category_lookup(epg, classifier)
    minutes = _break_minutes(window_start, window_end)
    return {
        "competitor_strength": _strength(minutes, rivals, dict(baseline)),
        "competitor_genre_contrast": _genre_contrast(
            minutes, own_category, rivals, lookup
        ),
        "competitor_prog_start": _prog_start(window_start, window_end, rivals, lookup),
    }


def forward_adjustment(
    features: Optional[Mapping[str, float]],
    betas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The log-effect adjustment a live decision may apply, with honest absence.

    ``betas`` is the ``competitor_betas`` block persisted in the coefficients
    JSON metadata (feature -> {beta, reference, role, ...}). Only ``forward``
    roles contribute; the boundary is enforced by
    :func:`~kairos.model.competitor_features.assert_forward_only`. When
    ``features`` is None (no future EPG) the adjustment is exactly 0.0 and the
    payload says so, so a missing schedule can never move a plan silently.
    """
    # Filter by ROLE only, then assert: a training-only feature mislabeled as
    # forward must fail loudly (ForwardBoundaryError), never be silently used
    # or silently dropped. Names outside the known forward set contribute
    # nothing below because the features payload never carries them.
    forward = {
        name: spec for name, spec in betas.items()
        if str(spec.get("role", "")) == "forward"
    }
    assert_forward_only(forward.keys())
    forward = {
        name: spec for name, spec in forward.items()
        if name in EXTENDED_FORWARD_FEATURES
    }
    if features is None or not forward:
        return {
            "adjustment": 0.0,
            "applied": False,
            "reason": (
                "future competitor EPG absent; covariate contributes nothing"
                if features is None
                else "no forward competitor betas available"
            ),
        }
    adjustment = 0.0
    for name, spec in forward.items():
        value = features.get(name)
        if value is None:
            continue
        adjustment += float(spec["beta"]) * (float(value) - float(spec["reference"]))
    return {
        "adjustment": float(adjustment),
        "applied": True,
        "reason": "forward counter-programming adjustment from the future competitor EPG",
    }
