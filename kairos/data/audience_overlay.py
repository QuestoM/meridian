"""Audience-model overlay: forward-dated segments take modeled ratings.

The transform's ``baseline_tvr`` is the historical rating from the data. Once
the operator activates the audience model (the ``audience_model_activation``
settings flag, shipped OFF), FORWARD-dated segments (broadcast day today or
later) replace that baseline with the trained model's expected rating
(:func:`kairos.model.audience_model.predict_tvr`), so the coming week's
forecast is priced on expected audience rather than last month's mean.

Honesty rules, in order:

  * OFF (the shipped default, and an absent flag reads OFF) is an exact no-op:
    the segment list is returned untouched, the same objects, and no clock is
    read. Historical dates and every measurement path never see a prediction.
  * The overlay is tolerant of the model being absent: no artifact on disk, no
    ``kairos.model.audience_model`` module, or a prediction failure all leave
    the historical baselines in place (logged, never raised, never guessed).
  * Every segment the overlay actually processes carries a basis marker (the
    dynamic ``tvr_basis`` attribute): the model's own per-row basis for a
    predicted forward segment, ``"historical"`` otherwise, so downstream
    surfaces can label each number's source. A prediction that comes back
    missing or negative keeps the historical value and says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kairos.optimize.optimizer import ProgramSegment

logger = logging.getLogger(__name__)

# Repo root: this file is kairos/data/audience_overlay.py, two parents up.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"
AUDIENCE_MODEL_PATH = ROOT / "models" / "audience_model.json"

# The dynamic attribute name carrying each processed segment's rating basis.
BASIS_ATTR = "tvr_basis"
BASIS_HISTORICAL = "historical"
BASIS_MODEL = "model"

# The columns predict_tvr's frozen row contract expects, in order.
_PREDICT_COLUMNS = ("date", "channel", "program_title", "start_seconds", "duration_seconds")


def audience_model_active(settings_path: Path | None = None) -> bool:
    """Whether the operator has switched the audience model on.

    Reads the saved settings file the same way the freshness guard does: parse
    through ``KairosSettings`` when the API model is importable (so coercion
    matches exactly what the API reads), falling back to a strict boolean read
    of the raw key when the engine runs standalone. A missing file, a missing
    key, or an unreadable file all read OFF; activation is never guessed.
    """
    path = settings_path if settings_path is not None else SETTINGS_PATH
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict):
        return False
    try:
        from kairos_api.core import KairosSettings

        return bool(getattr(KairosSettings(**raw), "audience_model_activation", False))
    except Exception:
        return raw.get("audience_model_activation") is True


def _segment_date(day: object) -> date | None:
    """The segment's broadcast date, or None for an undated segment."""
    try:
        return date.fromisoformat(str(day)[:10])
    except ValueError:
        return None


def _mark(segment: "ProgramSegment", basis: str) -> None:
    """Record the rating basis on a (frozen) segment as a dynamic attribute."""
    object.__setattr__(segment, BASIS_ATTR, basis)


def apply_audience_model(
    segments: list["ProgramSegment"],
    *,
    today: date | None = None,
    settings_path: Path | None = None,
    artifact_path: Path | None = None,
) -> list["ProgramSegment"]:
    """Replace forward-dated segments' baselines with the model's expected TVR.

    Returns ``segments`` untouched (the same list object, no clock read) when
    the activation flag is off, the artifact is absent, the prediction module
    is unavailable, or the prediction fails; the transform stays byte-identical
    to the pre-model behavior in every one of those states. When the overlay
    runs, only segments dated ``today`` or later are re-based; each processed
    segment carries its basis marker (see module docstring).
    """
    if not segments:
        return segments
    if not audience_model_active(settings_path):
        return segments
    artifact = artifact_path if artifact_path is not None else AUDIENCE_MODEL_PATH
    if not artifact.exists():
        return segments
    try:
        from kairos.model.audience_model import predict_tvr
    except Exception:
        logger.warning(
            "audience_model_activation is on but kairos.model.audience_model is "
            "unavailable; keeping historical baselines"
        )
        return segments

    reference = today if today is not None else date.today()
    forward: list[int] = []
    for index, segment in enumerate(segments):
        segment_day = _segment_date(segment.day)
        if segment_day is not None and segment_day >= reference:
            forward.append(index)
    if not forward:
        for segment in segments:
            _mark(segment, BASIS_HISTORICAL)
        return segments

    rows = pd.DataFrame(
        [
            {
                "date": str(segments[index].day),
                "channel": segments[index].channel,
                "program_title": segments[index].program_title,
                "start_seconds": segments[index].start_seconds,
                "duration_seconds": segments[index].duration_seconds,
            }
            for index in forward
        ],
        columns=list(_PREDICT_COLUMNS),
    )
    try:
        predicted = predict_tvr(rows, path=artifact)
    except Exception:
        logger.exception("audience model prediction failed; keeping historical baselines")
        return segments
    if not isinstance(predicted, pd.DataFrame) or "predicted_tvr" not in predicted.columns:
        logger.warning(
            "audience model returned no predicted_tvr column; keeping historical baselines"
        )
        return segments

    values = pd.to_numeric(predicted["predicted_tvr"], errors="coerce").tolist()
    bases: Sequence[object] = (
        predicted["basis"].tolist()
        if "basis" in predicted.columns
        else [BASIS_MODEL] * len(values)
    )

    result = list(segments)
    for position, index in enumerate(forward):
        segment = segments[index]
        value = values[position] if position < len(values) else None
        if value is None or pd.isna(value) or float(value) < 0:
            # An unpredictable row keeps its honest historical value and says so.
            _mark(segment, BASIS_HISTORICAL)
            continue
        # A forecast produced by the audience model is not the input file's
        # observed rating currency. Clear settlement provenance so a modeled
        # baseline can never masquerade as overnight+1 Jewish-household data.
        rebased = replace(
            segment,
            baseline_tvr=float(value),
            rating_audience_basis="",
            rating_vintage="",
            rating_source="",
        )
        basis = bases[position] if position < len(bases) else BASIS_MODEL
        _mark(rebased, str(basis) if basis else BASIS_MODEL)
        result[index] = rebased
    for index, segment in enumerate(result):
        if not hasattr(segment, BASIS_ATTR):
            _mark(segment, BASIS_HISTORICAL)
    return result
