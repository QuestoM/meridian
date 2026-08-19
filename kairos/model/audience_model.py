"""The audience model: expected TVR per programme slot, gated by measurement.

Forecasting the coming week for the exact same programmes differs between a
week with two Hanukkah days in winter and an August summer week, in expected
rating points, on top of which programme it is, what the rivals air opposite,
and where the break sits. This is that expected-TVR surface, distinct from
the retention model that prices interruptions.

Structure: a pooled multiplicative base in log space (series falling to genre
falling to the slot grand mean, pseudo-count shrinkage like the retention
pooling), plus one gated factor per family
(:data:`~kairos.model.audience_factors.FAMILIES`) fitted on base residuals
and admitted only past the five-fold +2 percent held-out bar. The base also
keeps the plain historical means (series -> genre -> slot -> channel ->
global): with every gate off, prediction reproduces the current historical
baseline_tvr semantics (a plain historical mean) exactly, and the per-row
``basis`` says which path spoke ("model" only where an activated factor
applies, else "base"). NaN competitor pressure means the family is not
applicable for that row. A family whose source is absent or contrast-free
records an honest off verdict with the reason, never an error. Activation
rides settings (the top-level ``audience_model_activation`` key, absent reads
False); the transform seam that swaps forward baselines is wired by the
engine owner, and historical or measurement paths never use predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from kairos.model.audience_factors import (
    FAMILIES,
    AUDIENCE_GATE_MIN_RELATIVE_IMPROVEMENT,
    cell_deltas_for,
    family_cells,
    fit_cell_deltas,
    fit_pressure_beta,
    gate_family,
    gate_off,
    pressure_deltas_for,
)
from kairos.model.audience_frame import (
    attach_pressure,
    build_training_frame,
    prediction_frame,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "models" / "audience_model.json"
_SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"

# The top-level settings key carrying the activation flag (absent reads False).
ACTIVATION_SETTINGS_KEY = "audience_model_activation"

# Log-space floor: a measured zero rating enters the multiplicative model at
# this floor (its plain mean stays untouched on the historical path).
TVR_FLOOR = 0.01
# Pseudo-count pooling strength, mirroring the retention layers' default.
DEFAULT_SHRINKAGE_K = 20.0


def _shrunk(n: int, mean: float, k: float, parent: float) -> float:
    return (n * mean + k * parent) / (n + k)


@dataclass(frozen=True)
class AudienceBase:
    """The pooled base: log-space shrunk levels plus the plain historical means."""

    tvr_floor: float
    shrinkage_k: float
    n_observations: int
    global_log: float
    channel_log: dict[str, float]
    genre_log: dict[str, dict[str, float]]
    slot_log: dict[str, dict[str, float]]
    hist_global: float
    hist_channel: dict[str, float]
    hist_slot: dict[str, dict[str, float]]
    hist_genre: dict[str, dict[str, float]]
    hist_series: dict[str, dict[str, float]]

    @classmethod
    def empty(
        cls, *, tvr_floor: float = TVR_FLOOR, shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    ) -> "AudienceBase":
        """The degenerate no-data base: NaN levels, empty tables, honest."""
        return cls(
            tvr_floor=tvr_floor, shrinkage_k=shrinkage_k, n_observations=0,
            global_log=float("nan"), channel_log={}, genre_log={}, slot_log={},
            hist_global=float("nan"), hist_channel={}, hist_slot={},
            hist_genre={}, hist_series={},
        )

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, *,
        tvr_floor: float = TVR_FLOOR, shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    ) -> "AudienceBase":
        if frame.empty:
            return cls.empty(tvr_floor=tvr_floor, shrinkage_k=shrinkage_k)
        work = frame[["channel", "genre", "slot_band", "series_key", "tvr"]].copy()
        work["log_tvr"] = np.log(np.maximum(work["tvr"].astype(float), tvr_floor))
        mu = float(work["log_tvr"].mean())

        channel_log: dict[str, float] = {}
        genre_log: dict[str, dict[str, float]] = {}
        slot_log: dict[str, dict[str, float]] = {}
        for channel, group in work.groupby("channel", sort=True):
            name = str(channel)
            theta = _shrunk(len(group), float(group["log_tvr"].mean()), shrinkage_k, mu)
            channel_log[name] = theta
            genre_log[name] = {
                str(genre): _shrunk(len(sub), float(sub["log_tvr"].mean()), shrinkage_k, theta)
                for genre, sub in group.groupby("genre", sort=True)
            }
            slot_log[name] = {
                str(band): _shrunk(len(sub), float(sub["log_tvr"].mean()), shrinkage_k, theta)
                for band, sub in group.groupby("slot_band", sort=True)
            }

        def _nested_means(key: str) -> dict[str, dict[str, float]]:
            out: dict[str, dict[str, float]] = {}
            for (channel, value), sub in work.groupby(["channel", key], sort=True):
                if key == "series_key" and not value:
                    continue
                out.setdefault(str(channel), {})[str(value)] = float(sub["tvr"].mean())
            return out

        return cls(
            tvr_floor=tvr_floor, shrinkage_k=shrinkage_k,
            n_observations=int(len(work)), global_log=mu,
            channel_log=channel_log, genre_log=genre_log, slot_log=slot_log,
            hist_global=float(work["tvr"].mean()),
            hist_channel={
                str(c): float(g["tvr"].mean()) for c, g in work.groupby("channel", sort=True)
            },
            hist_slot=_nested_means("slot_band"),
            hist_genre=_nested_means("genre"),
            hist_series=_nested_means("series_key"),
        )

    def log_base(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-row pooled log level: genre, falling to slot, channel, global."""
        values: list[float] = []
        for channel, genre, band in zip(frame["channel"], frame["genre"], frame["slot_band"]):
            channel = str(channel)
            by_genre = self.genre_log.get(channel, {})
            if str(genre) in by_genre:
                values.append(by_genre[str(genre)])
                continue
            by_slot = self.slot_log.get(channel, {})
            if str(band) in by_slot:
                values.append(by_slot[str(band)])
                continue
            values.append(self.channel_log.get(channel, self.global_log))
        return np.array(values)

    def historical(self, frame: pd.DataFrame) -> np.ndarray:
        """The plain historical mean path: series -> genre -> slot -> channel -> global."""
        values: list[float] = []
        rows = zip(frame["channel"], frame["series_key"], frame["genre"], frame["slot_band"])
        for channel, series, genre, band in rows:
            channel = str(channel)
            for table, key in (
                (self.hist_series, str(series)),
                (self.hist_genre, str(genre)),
                (self.hist_slot, str(band)),
            ):
                by_channel = table.get(channel, {})
                if key and key in by_channel:
                    values.append(by_channel[key])
                    break
            else:
                values.append(self.hist_channel.get(channel, self.hist_global))
        return np.array(values)

    def to_payload(self) -> dict[str, Any]:
        """The base tables, flat under the artifact's ``base`` block."""
        return {"kind": "eb_log_multiplicative", **asdict(self)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AudienceBase":
        """Rebuild from :meth:`to_payload`; unknown keys ignored, missing
        keys degrade to the honest empty base, never a fabricated level."""
        data = asdict(cls.empty())
        data.update({key: payload[key] for key in data if key in payload})
        return cls(**data)

    def summary(self) -> dict[str, Any]:
        """A compact, human-scale description for status payloads."""
        return {
            "kind": "eb_log_multiplicative",
            "n_observations": self.n_observations,
            "channels": sorted(self.hist_channel),
            "n_series": sum(len(v) for v in self.hist_series.values()),
            "n_genres": sum(len(v) for v in self.hist_genre.values()),
            "tvr_floor": self.tvr_floor, "shrinkage_k": self.shrinkage_k,
        }


@dataclass
class AudienceModel:
    """The fitted audience model: base, gate verdicts, activated factors."""

    base: AudienceBase
    gates: dict[str, dict[str, Any]]
    factors: dict[str, Any]
    computed_at: str
    owned_channel: str = ""
    activation_default: bool = False
    source_fingerprints: dict[str, str] = field(default_factory=dict)

    def predict_tvr(
        self, rows: pd.DataFrame, *,
        classifier=None, calendar_path=None, events_path=None,
        lineup_frame_fn: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """Rows plus ``predicted_tvr`` and per-row ``basis``: ``"model"`` only
        where a gate-on factor applies, else the historical mean path."""
        out = rows.copy()
        if out.empty:
            out["predicted_tvr"] = pd.Series(dtype=float)
            out["basis"] = pd.Series(dtype=object)
            return out
        scored = prediction_frame(
            out, classifier=classifier, calendar_path=calendar_path,
            events_path=events_path,
        ).reset_index(drop=True)

        base_log = self.base.log_base(scored)
        deltas = np.zeros(len(scored))
        applied = np.zeros(len(scored), dtype=bool)
        for family in FAMILIES:
            if self.gates.get(family, {}).get("verdict") != "on":
                continue
            payload = self.factors.get(family)
            if payload is None:
                continue
            if family == "competitor_lineup":
                pressure, _reason = attach_pressure(
                    scored, self.owned_channel, lineup_frame_fn=lineup_frame_fn,
                )
                if pressure is None:
                    continue
                deltas += pressure_deltas_for(pressure, payload)
                applied |= np.isfinite(pressure)
            else:
                cells = family_cells(scored, family)
                deltas += cell_deltas_for(cells, payload["cells"])
                applied |= np.array([cell is not None for cell in cells])

        out["predicted_tvr"] = np.where(
            applied, np.exp(base_log + deltas), self.base.historical(scored)
        )
        out["basis"] = np.where(applied, "model", "base")
        return out

    def to_artifact(self) -> dict[str, Any]:
        """The frozen artifact shape: exactly these five top-level keys."""
        base_payload = self.base.to_payload()
        base_payload["owned_channel"] = self.owned_channel
        base_payload["factors"] = self.factors
        return {
            "computed_at": self.computed_at,
            "activation_default": self.activation_default,
            "base": base_payload,
            "gates": self.gates,
            "source_fingerprints": self.source_fingerprints,
        }

    def write_artifact(self, path: str | Path = ARTIFACT_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_artifact(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "AudienceModel":
        base_payload = dict(payload.get("base", {}))
        owned_channel = str(base_payload.pop("owned_channel", ""))
        factors = base_payload.pop("factors", {})
        return cls(
            base=AudienceBase.from_payload(base_payload),
            gates={k: dict(v) for k, v in payload.get("gates", {}).items()},
            factors=dict(factors),
            computed_at=str(payload.get("computed_at", "")),
            owned_channel=owned_channel,
            activation_default=bool(payload.get("activation_default", False)),
            source_fingerprints=dict(payload.get("source_fingerprints", {})),
        )


def _settings_owned_channel() -> str:
    try:
        raw = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return str(raw.get("operator_channel", "") or "")
    except (OSError, json.JSONDecodeError, ValueError):
        return ""


def fit_audience_model(
    frame: Optional[pd.DataFrame] = None, *,
    spots: Optional[pd.DataFrame] = None,
    classifier=None, calendar_path=None, events_path=None,
    owned_channel: Optional[str] = None,
    tvr_floor: float = TVR_FLOOR,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    min_relative_improvement: float = AUDIENCE_GATE_MIN_RELATIVE_IMPROVEMENT,
    computed_at: Optional[str] = None,
    source_fingerprints: Optional[dict[str, str]] = None,
    lineup_frame_fn: Optional[Callable] = None,
) -> AudienceModel:
    """Fit the base, run every family gate, and fit factors for the on
    families. Deterministic given the data (``computed_at`` is only a stamp);
    a family whose source is absent or contrast-free records verdict off with
    the honest reason, and nothing is ever forced on."""
    computed_at = computed_at or datetime.now(timezone.utc).isoformat()
    if frame is None:
        frame = build_training_frame(
            spots, classifier=classifier, calendar_path=calendar_path,
            events_path=events_path,
        )
    frame = frame.reset_index(drop=True)
    owned = _settings_owned_channel() if owned_channel is None else owned_channel
    base = AudienceBase.fit(frame, tvr_floor=tvr_floor, shrinkage_k=shrinkage_k)

    if frame.empty:
        no_data = (
            "no training observations available; the audience model has "
            "nothing to measure"
        )
        return AudienceModel(
            base=base, gates={f: gate_off(no_data, computed_at) for f in FAMILIES},
            factors={}, computed_at=computed_at, owned_channel=owned,
            source_fingerprints=dict(source_fingerprints or {}),
        )

    log_tvr = np.log(np.maximum(frame["tvr"].to_numpy(float), tvr_floor))
    pressure, pressure_reason = attach_pressure(frame, owned, lineup_frame_fn=lineup_frame_fn)

    def base_fit(train: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
        return AudienceBase.fit(train, tvr_floor=tvr_floor, shrinkage_k=shrinkage_k).log_base

    gates: dict[str, dict[str, Any]] = {}
    factors: dict[str, Any] = {}
    full_residuals: Optional[np.ndarray] = None
    for family in FAMILIES:
        if family == "competitor_lineup" and pressure is None:
            gates[family] = gate_off(
                pressure_reason or "competitor lineup unavailable", computed_at,
            )
            continue
        # Measured against the model as it stands, not against the bare base.
        # The families already in `factors` are what prediction will add to this
        # one, so they are what it has to beat.
        gates[family] = gate_family(
            frame, family, base_fit,
            log_tvr=log_tvr, shrinkage_k=shrinkage_k,
            pressure=pressure if family == "competitor_lineup" else None,
            active=tuple(factors),
            min_relative_improvement=min_relative_improvement,
            measured_at=computed_at,
        )
        if gates[family]["verdict"] != "on":
            continue
        if full_residuals is None:
            full_residuals = log_tvr - base.log_base(frame)
        if family == "competitor_lineup":
            payload = fit_pressure_beta(full_residuals, pressure)
            if payload is None:
                gates[family] = gate_off(
                    "the gate passed but no pressure slope could be fitted on "
                    "the full data; factor stays off",
                    computed_at,
                )
                continue
            factors[family] = payload
        else:
            factors[family] = {
                "cells": fit_cell_deltas(full_residuals, family_cells(frame, family), shrinkage_k)
            }

    return AudienceModel(
        base=base, gates=gates, factors=factors, computed_at=computed_at,
        owned_channel=owned, source_fingerprints=dict(source_fingerprints or {}),
    )


def load_audience_model(path: str | Path = ARTIFACT_PATH) -> AudienceModel:
    """Read the shipped artifact; raises FileNotFoundError when absent."""
    return AudienceModel.from_artifact(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def predict_tvr(
    rows: pd.DataFrame, *,
    model: Optional[AudienceModel] = None,
    path: str | Path = ARTIFACT_PATH,
    **kwargs: Any,
) -> pd.DataFrame:
    """Module-level prediction surface: rows plus predicted_tvr and basis."""
    model = load_audience_model(path) if model is None else model
    return model.predict_tvr(rows, **kwargs)


def audience_model_activation(settings: Optional[Mapping[str, Any]] = None) -> bool:
    """The operator's activation flag; absent, unreadable or falsy reads False."""
    if settings is None:
        try:
            settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return False
    if not isinstance(settings, Mapping):
        return False
    return bool(settings.get(ACTIVATION_SETTINGS_KEY, False))


def audience_model_status(
    path: str | Path = ARTIFACT_PATH, settings: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The honest tri-state status payload for the API surface:
    ``{available, computed_at, activation, gates, base_summary}``, with
    available False and everything else honestly empty when the artifact is
    absent or unreadable, never a fabricated model description."""
    activation = audience_model_activation(settings)
    try:
        model = load_audience_model(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {
            "available": False, "computed_at": None, "activation": activation,
            "gates": {}, "base_summary": None,
        }
    return {
        "available": True, "computed_at": model.computed_at,
        "activation": activation, "gates": model.gates,
        "base_summary": model.base.summary(),
    }
