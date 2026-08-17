"""The rating forecast: one programme, its expected TVR, its range, its drivers.

The audience model already predicts an expected rating for every forward segment
the plan prices (:mod:`kairos.data.audience_overlay`). That number was never a
stage of its own: it moved money silently, with no range on it and no account of
what produced it. This module is that stage. It answers one programme at a time,
and every answer carries four things the plain prediction never did.

**The number.** Identical, to the float, to what
:meth:`kairos.model.audience_model.AudienceModel.predict_tvr` puts on the same
row -- this surface explains the shipped prediction, it does not compute a
second, prettier one.

**A range that was measured.** The bands come from the scatter the model was
fitted from (:mod:`kairos.model.forecast_dispersion`) and from the shrinkage
weight the fit actually applied, so a cell with three observations is visibly
wider than a cell with three hundred. On the real artifact that inequality is
measured, not asserted: the owned channel's 74-observation news series returns a
log-space spread of 0.596 against 0.942 for a 3-observation reality series.
Where the scatter cannot be measured the band is withheld with the reason. There
is no invented interval anywhere.

**The drivers.** The model is multiplicative in log space, so the prediction
decomposes exactly (:mod:`kairos.model.forecast_drivers`): a base level, the
channel, the genre, and one factor per activated family, multiplying back to the
point forecast. Families that are off are listed as not applied with the verdict
their own held-out measurement returned.

**The currency it is in, and the currency it is not.**
:mod:`kairos.model.forecast_basis` owns that refusal, and the window refusal
beside it.

History rides on every payload next to the forecast -- the plain historical mean
is what this product priced on before the model existed, so the two numbers
together are the honest comparison rather than a claim that the model is better.

**And on the real window the historical mean is the more accurate of the two in
points.** :mod:`kairos.model.forecast_backtest` measured it walk-forward: the
model wins on the log-space objective its gates were scored on (RMSE 0.683
against 0.707) and loses in arithmetic rating points (MAE 1.188 against 0.898,
bias -0.249). Read that verdict before quoting a gate's held-out percentage as
evidence the forecast is more accurate -- the percentage is real and it is about
a different objective. Nothing here presents the model as the better number; it
presents both, with the measurement that says which is which.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from statistics import NormalDist
from typing import Any, Callable, Optional

import pandas as pd

from kairos.model.audience_factors import cell_key
from kairos.model.audience_frame import (
    PREDICTION_COLUMNS,
    build_training_frame,
    prediction_frame,
    slot_band_of_hour,
)
from kairos.model.audience_model import ARTIFACT_PATH, AudienceModel, load_audience_model
from kairos.model.forecast_basis import (
    DEFAULT_LEVEL,
    LEVEL_LABELS_HE,
    MAX_HORIZON_DAYS,
    audience_basis_block,
    audience_is_servable,
    audience_refusal,
    classify_audience,
    horizon_for,
    unavailable,
)
from kairos.model.forecast_dispersion import Dispersion, build_dispersion, predictive_sd
from kairos.model.forecast_drivers import (
    apply_families,
    base_terms,
    drivers_for,
    historical_level,
    not_applied_for,
    series_answered_at,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LEVEL",
    "MAX_HORIZON_DAYS",
    "ForecastService",
    "audience_basis_block",
    "audience_is_servable",
    "classify_audience",
    "default_service",
    "factor_cell_key",
    "forecast_programme",
    "slot_band_for_seconds",
]

_INTERVAL_METHOD_EN = (
    "empirical-Bayes predictive band in log space: "
    "sd^2 = sigma_within^2 + (k/(n+k))^2 * tau^2 + n*sigma_within^2/(n+k)^2, "
    "with sigma_within the pooled within-cell spread and tau the "
    "DerSimonian-Laird between-cell spread measured at the level that answered; "
    "exponentiated back, so the band is asymmetric in points"
)


@lru_cache(maxsize=8)
def _z_for(level: float) -> float:
    return float(NormalDist().inv_cdf(0.5 + float(level) / 2.0))


@dataclass(frozen=True)
class ForecastService:
    """The model, the scatter behind it, and the window both were measured in."""

    model: AudienceModel
    dispersion: Optional[Dispersion]
    dispersion_reason: str = ""

    @classmethod
    def load(
        cls, *, artifact_path: Any = ARTIFACT_PATH,
        frame: Optional[pd.DataFrame] = None, with_frame: bool = True,
    ) -> "ForecastService":
        """Load the shipped artifact and, when asked, the frame behind the bands.

        The artifact alone gives every point forecast. The training frame is
        needed only for the ranges, and it is the one expensive read here (the
        spots history), so a caller that wants points only can skip it and the
        payloads will say the bands are unavailable and why.
        """
        model = load_audience_model(artifact_path)
        if frame is None and not with_frame:
            return cls(model=model, dispersion=None, dispersion_reason=(
                "the training frame was not loaded in this process, so the scatter "
                "behind each level is unmeasured and no range is published"
            ))
        if frame is None:
            try:
                frame = build_training_frame()
            except Exception as exc:  # noqa: BLE001 - absent history is an honest absence
                logger.warning("forecast: training frame unavailable: %s", exc)
                return cls(model=model, dispersion=None, dispersion_reason=(
                    f"the training frame could not be built ({exc}); no range is published"
                ))
        return cls(
            model=model,
            dispersion=build_dispersion(frame, tvr_floor=model.base.tvr_floor),
            dispersion_reason="",
        )

    # ----------------------------------------------------------------- window

    def measured_window(self) -> dict[str, Any]:
        """The dates the model's observations actually span, or an honest gap."""
        if self.dispersion is None:
            return {"from": None, "to": None, "available": False,
                    "reason": self.dispersion_reason}
        return {
            "from": self.dispersion.window_from, "to": self.dispersion.window_to,
            "available": self.dispersion.window_from is not None,
            "n_observations": self.dispersion.n_observations,
        }

    # --------------------------------------------------------------- forecast

    def forecast_rows(
        self, rows: pd.DataFrame, *, level: float = DEFAULT_LEVEL,
        audience: str = "", lineup_frame_fn: Optional[Callable] = None,
    ) -> list[dict[str, Any]]:
        """One payload per input row, in row order.

        ``rows`` carries :data:`kairos.model.audience_frame.PREDICTION_COLUMNS`.
        Batched on purpose: the competitor lineup is one build for the whole
        request rather than one per programme.
        """
        missing = [c for c in PREDICTION_COLUMNS if c not in rows.columns]
        if missing:
            raise KeyError(f"forecast rows missing required column(s): {missing}")
        refusal = audience_refusal(audience)
        if refusal is not None:
            return [dict(refusal) for _ in range(len(rows))]
        if len(rows) == 0:
            return []

        scored = prediction_frame(rows).reset_index(drop=True)
        base_log = self.model.base.log_base(scored)
        historical = self.model.base.historical(scored)
        effects = apply_families(self.model, scored, lineup_frame_fn=lineup_frame_fn)
        window = self.measured_window()

        return [
            self._row_payload(
                scored.iloc[position], position, level=level, audience=audience,
                base_log=float(base_log[position]),
                historical=float(historical[position]),
                total_delta=float(effects.deltas[position]),
                model_basis=bool(effects.applied[position]),
                contributions=effects.contributions, window=window,
            )
            for position in range(len(scored))
        ]

    def forecast_programme(
        self, *, channel: str, program_title: str, day: Any,
        start_seconds: float = 0.0, duration_seconds: float = 0.0,
        level: float = DEFAULT_LEVEL, audience: str = "",
        lineup_frame_fn: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """One programme's forecast, with its range, drivers and provenance."""
        rows = pd.DataFrame([{
            "date": str(day), "channel": str(channel),
            "program_title": str(program_title),
            "start_seconds": float(start_seconds),
            "duration_seconds": float(duration_seconds),
        }], columns=list(PREDICTION_COLUMNS))
        return self.forecast_rows(
            rows, level=level, audience=audience, lineup_frame_fn=lineup_frame_fn,
        )[0]

    # ---------------------------------------------------------------- payload

    def _row_payload(
        self, row: pd.Series, position: int, *, level: float, audience: str,
        base_log: float, historical: float, total_delta: float, model_basis: bool,
        contributions: dict[str, dict[str, Any]], window: dict[str, Any],
    ) -> dict[str, Any]:
        raw_day = row["date"]
        day: Optional[date] = (
            None if pd.isna(raw_day) else getattr(raw_day, "date", lambda: None)()
        )
        if day is None:
            return unavailable(
                "לא ניתן לקרוא את תאריך השידור המבוקש",
                "the requested date cannot be read",
            )
        refusal, horizon = horizon_for(day, window)
        if refusal is not None:
            return refusal

        channel = str(row["channel"])
        base_level, terms = base_terms(
            self.model.base, channel, str(row["genre"]), str(row["slot_band"])
        )
        hist_level = historical_level(self.model.base, channel, row)
        if not model_basis:
            resolved = hist_level
        elif series_answered_at(contributions, position):
            resolved = "series"
        else:
            resolved = base_level

        expected = math.exp(base_log + total_delta) if model_basis else historical
        drivers, fallbacks = drivers_for(
            row, position, terms=terms, base_level=base_level,
            contributions=contributions, model_basis=model_basis,
            expected=expected, resolved=resolved,
        )
        return {
            "available": True,
            "expected_tvr": round(float(expected), 4),
            "interval": self._interval(
                channel=channel, resolved=resolved, row=row,
                expected=expected, level=level,
            ),
            "audience_basis": audience_basis_block(audience),
            "history": {
                "historical_tvr": round(float(historical), 4),
                "level": hist_level,
                "level_he": LEVEL_LABELS_HE.get(hist_level, hist_level),
                "label_he": "ממוצע היסטורי (הבסיס שקדם למודל)",
            },
            "drivers": drivers,
            "not_applied": not_applied_for(contributions, position),
            "resolution": {
                "level": resolved,
                "level_he": LEVEL_LABELS_HE.get(resolved, resolved),
                "basis": "model" if model_basis else "historical_mean",
                "fallbacks": fallbacks,
                "series_key": str(row["series_key"]),
                "genre": str(row["genre"]),
                "slot_band": str(row["slot_band"]),
            },
            "programme": {
                "title": str(row["title"]), "date": day.isoformat(),
                "start_seconds": int(row["start_seconds"]),
                "slot_hour": int(row["slot_hour"]),
            },
            "horizon": horizon,
            "provenance": {
                "computed_at": self.model.computed_at,
                "n_observations": self.model.base.n_observations,
                "shrinkage_k": self.model.base.shrinkage_k,
                "artifact_kind": "eb_log_multiplicative",
                "level_that_answered": resolved,
            },
        }

    def _interval(
        self, *, channel: str, resolved: str, row: pd.Series,
        expected: float, level: float,
    ) -> dict[str, Any]:
        """The measured range, or an honest absence with the reason."""
        block: dict[str, Any] = {
            "level": round(float(level), 4), "low": None, "high": None,
            "available": False,
            "method_he": "רצועת חיזוי אמפירית-בייסיאנית במרחב הלוג, מומרת חזרה בחזקה",
            "method_en": _INTERVAL_METHOD_EN,
        }
        if self.dispersion is None:
            block["reason"] = self.dispersion_reason or "no measurement frame in this process"
            return block
        cell = {
            "series": str(row["series_key"]), "genre": str(row["genre"]),
            "slot": str(row["slot_band"]), "channel": channel, "global": "*",
        }.get(resolved, "")
        table = self.dispersion.table(resolved, channel)
        sd, components = predictive_sd(table, cell, self.model.base.shrinkage_k)
        block["components"] = components
        if sd is None:
            block["reason"] = components.get("reason", "the level's scatter is not measured")
            return block
        z = _z_for(level)
        centre = math.log(max(float(expected), self.model.base.tvr_floor))
        block.update({
            "available": True,
            "low": round(math.exp(centre - z * sd), 4),
            "high": round(math.exp(centre + z * sd), 4),
            "sd_log": round(sd, 6),
            "z": round(z, 4),
            "n_observations": components.get("n_observations"),
        })
        return block


@lru_cache(maxsize=1)
def default_service() -> ForecastService:
    """The process-wide service over the shipped artifact and the real frame."""
    return ForecastService.load()


def forecast_programme(**kwargs: Any) -> dict[str, Any]:
    """One programme's forecast over the shipped model (see :class:`ForecastService`)."""
    return default_service().forecast_programme(**kwargs)


def slot_band_for_seconds(start_seconds: float) -> str:
    """The slot band a clock time falls in, for callers building request rows."""
    return slot_band_of_hour(int(float(start_seconds) // 3600) % 24)


def factor_cell_key(*parts: object) -> str:
    """The factor-table cell key, re-exported so callers need not import two modules."""
    return cell_key(*parts)
