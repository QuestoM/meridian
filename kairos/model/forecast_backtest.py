"""Walk-forward accuracy of the rating forecast, measured rather than claimed.

A forecast surface that reports its own confidence band has made a testable
promise, and this module is where the promise is tested. The observation frame is
cut into contiguous date blocks; each block is forecast by a model fitted on the
blocks BEFORE it and on nothing else. Every number reported here therefore comes
from a model that had not seen the row it was scored on.

What is refitted per fold: the pooled base, all eight family gates (each running
its own five inner folds on the fold's training rows), and the factor tables --
that is, ``fit_audience_model`` itself, the same call
``scripts/compute_audience_model.py`` makes. The dispersion behind the interval
is refitted on the same training rows. Nothing is reused from the shipped
artifact, so a fold's band is the band that fold's model would actually have
published.

**Four numbers, and what each is for.**

``mae`` and ``rmse`` are in rating points, the unit the plan prices in. ``bias``
is the MEAN SIGNED error, kept separate because a forecast that is 0.4 points
high on everything is a different failure from one that is 0.4 points off in
both directions, and only the first is fixable by a constant. ``mape`` is
reported over observations at or above :data:`MAPE_TVR_FLOOR` only -- a percentage
error against a measured rating of 0.0 is unbounded, so those rows are excluded,
COUNTED, and named. ``interval_coverage`` is the honesty check on the band: at
the published level the observed rating should fall inside it that often, and a
coverage far below the level means the band is too narrow no matter how good the
point forecast is.

**The historical mean is scored beside the model, on the same rows.** It is what
this product priced on before the model existed. Reporting the model's error
without it would be a number with nothing to be better than.

**What this measurement cannot cover, and says so.** The first block has no prior
data and reports UNAVAILABLE rather than a number. The competitor family is
fitted per fold but is NOT APPLIED to test rows: its feature is a rival title's
shrunk mean rating, which only training data may supply, so a test date's lineup
is honestly unknown here. Its own gate measured the smallest held-out gain of the
three active families, and the exclusion is reported on every payload rather than
buried. Absence of contrast is not failure, and a fold the data cannot support
reports the reason instead of a figure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from kairos.model.audience_frame import PREDICTION_COLUMNS, build_training_frame
from kairos.model.audience_model import TVR_FLOOR, fit_audience_model, load_audience_model
from kairos.model.forecast import ForecastService
from kairos.model.forecast_basis import DEFAULT_LEVEL
from kairos.model.forecast_dispersion import build_dispersion

logger = logging.getLogger(__name__)

# How many contiguous date blocks the window is cut into. The first is training
# seed only and is reported UNAVAILABLE, so this yields ``n - 1`` scored folds.
DEFAULT_BLOCKS = 6

# Observations below this measured rating are excluded from MAPE (a percentage
# error against a measured 0.0 is unbounded), counted, and named on the payload.
MAPE_TVR_FLOOR = 0.05

# Fewest training observations a fold will fit on. Below this the base's own
# pooling has nothing to pool and the fold reports UNAVAILABLE.
MIN_TRAIN_OBSERVATIONS = 100

# Fewest test observations a fold will score on.
MIN_TEST_OBSERVATIONS = 10

_COMPETITOR_NOTE = (
    "the competitor_lineup family is fitted on each fold's training rows but is "
    "not applied to its test rows: the feature is a rival title's shrunk mean "
    "measured rating, which only prior data may supply, so a test date's lineup "
    "is honestly unknown here rather than back-filled from the future"
)


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


# ------------------------------------------------------------------- the blocks

def date_blocks(frame: pd.DataFrame, n_blocks: int) -> list[list[pd.Timestamp]]:
    """Cut the frame's distinct dates into ``n_blocks`` contiguous groups.

    Split on DATES, not rows, so no fold is ever trained on part of a day and
    tested on the rest of it -- the same day's observations share a calendar
    context and a lineup, and splitting inside one would leak both.
    """
    days = sorted(pd.to_datetime(frame["date"], errors="coerce").dropna().unique())
    if not days or n_blocks < 2:
        return []
    n_blocks = min(int(n_blocks), len(days))
    edges = np.array_split(np.arange(len(days)), n_blocks)
    return [[days[i] for i in block] for block in edges if len(block)]


# ------------------------------------------------------------------ the metrics

def _metrics(
    observed: np.ndarray, predicted: np.ndarray, historical: np.ndarray,
    low: np.ndarray, high: np.ndarray, level: float,
) -> dict[str, Any]:
    """Point error, bias, MAPE with its exclusions, and interval coverage."""
    n = int(len(observed))
    if n == 0:
        return _unavailable("no scored observations in this cell", n=0)
    error = predicted - observed
    hist_error = historical - observed
    scorable = observed >= MAPE_TVR_FLOOR
    banded = np.isfinite(low) & np.isfinite(high)
    inside = banded & (observed >= low) & (observed <= high)
    # The gates admitted each family on held-out RMSE IN LOG SPACE, so the
    # comparison against the pre-model historical mean is reported on BOTH
    # objectives. They can disagree, and on this window they do: see the module
    # docstring. Reporting only the one the model was tuned on would be picking
    # the scoreboard after the game.
    log_observed = np.log(np.maximum(observed, TVR_FLOOR))
    log_error = np.log(np.maximum(predicted, TVR_FLOOR)) - log_observed
    log_hist_error = np.log(np.maximum(historical, TVR_FLOOR)) - log_observed
    out: dict[str, Any] = {
        "available": True,
        "n": n,
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "bias": round(float(np.mean(error)), 4),
        "mean_observed": round(float(np.mean(observed)), 4),
        "historical_mae": round(float(np.mean(np.abs(hist_error))), 4),
        "historical_rmse": round(float(np.sqrt(np.mean(hist_error**2))), 4),
        "historical_bias": round(float(np.mean(hist_error)), 4),
        "log_rmse": round(float(np.sqrt(np.mean(log_error**2))), 4),
        "historical_log_rmse": round(float(np.sqrt(np.mean(log_hist_error**2))), 4),
        "mape": None,
        "mape_n": int(scorable.sum()),
        "mape_excluded_n": int(n - scorable.sum()),
        "mape_excluded_reason": (
            f"observations with a measured rating below {MAPE_TVR_FLOOR} points are "
            "excluded from MAPE because a percentage error against them is unbounded"
        ),
        "interval_level": round(float(level), 4),
        "interval_n": int(banded.sum()),
        "interval_missing_n": int(n - banded.sum()),
        "interval_coverage": None,
    }
    if int(scorable.sum()):
        out["mape"] = round(float(
            100.0 * np.mean(np.abs(error[scorable]) / observed[scorable])
        ), 3)
    if int(banded.sum()):
        out["interval_coverage"] = round(float(inside.sum() / banded.sum()), 4)
        out["interval_mean_width"] = round(
            float(np.mean(high[banded] - low[banded])), 4
        )
    return out


def verdict_for(overall: dict[str, Any]) -> dict[str, Any]:
    """Where the model beats the pre-model historical mean, and where it does not.

    Two objectives, reported side by side because they can disagree and on the
    real window they DO. The gates admitted families on held-out RMSE in log
    space; the plan prices in arithmetic rating points. A log-space win does not
    carry over: ``exp`` of a mean of logs estimates the geometric centre, which
    sits below the arithmetic mean of a right-skewed rating distribution, so a
    model fitted and gated in logs can be honest in logs and systematically low
    in points. A negative ``bias`` beside a log-space win is the signature of
    exactly that, and this block names it rather than reporting only the
    objective the model was tuned on.
    """
    if not overall.get("available"):
        return {"available": False, "reason": overall.get("reason", "nothing scored")}
    log_gain = overall["historical_log_rmse"] - overall["log_rmse"]
    points_gain = overall["historical_mae"] - overall["mae"]
    beats_log = log_gain > 0
    beats_points = points_gain > 0
    return {
        "available": True,
        "beats_historical_in_log_space": bool(beats_log),
        "beats_historical_in_points": bool(beats_points),
        "log_rmse_gain": round(float(log_gain), 4),
        "log_rmse_gain_pct": round(
            100.0 * log_gain / overall["historical_log_rmse"], 2
        ) if overall["historical_log_rmse"] else None,
        "points_mae_gain": round(float(points_gain), 4),
        "points_mae_gain_pct": round(
            100.0 * points_gain / overall["historical_mae"], 2
        ) if overall["historical_mae"] else None,
        "interval_is_conservative": bool(
            overall.get("interval_coverage") is not None
            and overall["interval_coverage"] > overall["interval_level"]
        ),
        "headline_en": (
            ("the model beats" if beats_log else "the model does not beat")
            + " the pre-model historical mean on the log-space objective its gates "
            + "were measured on, and "
            + ("beats" if beats_points else "does NOT beat")
            + " it in arithmetic rating points, the unit the plan prices in"
        ),
        "headline_he": (
            ("המודל מנצח את" if beats_log else "המודל אינו מנצח את")
            + " הממוצע ההיסטורי במרחב הלוג, המדד שעל פיו נמדדו השערים, ו"
            + ("מנצח" if beats_points else "אינו מנצח")
            + " אותו בנקודות רייטינג, היחידה שבה התוכנית מתומחרת"
        ),
        "mechanism_note_en": (
            "a negative bias beside a log-space win is the retransformation "
            "shortfall: the exponential of a log-space level estimates the "
            "geometric centre, which sits below the arithmetic mean of a "
            "right-skewed rating distribution"
        ) if (beats_log and not beats_points and overall["bias"] < 0) else None,
    }


def _breakdown(
    keys: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
    historical: np.ndarray, low: np.ndarray, high: np.ndarray, level: float,
) -> dict[str, Any]:
    """Per-cell metrics, ``n`` always reported, thin cells never hidden."""
    out: dict[str, Any] = {}
    for key in sorted({str(k) for k in keys}):
        mask = keys == key
        out[key] = _metrics(
            observed[mask], predicted[mask], historical[mask],
            low[mask], high[mask], level,
        )
    return out


# --------------------------------------------------------------------- the walk

@dataclass
class _Scored:
    """Aligned arrays of everything one fold produced, plus its refusals."""

    observed: list[float]
    predicted: list[float]
    historical: list[float]
    low: list[float]
    high: list[float]
    genre: list[str]
    slot: list[str]
    refusals: list[str]

    @classmethod
    def empty(cls) -> "_Scored":
        return cls([], [], [], [], [], [], [], [])

    def extend(self, other: "_Scored") -> None:
        for name in ("observed", "predicted", "historical", "low", "high",
                     "genre", "slot", "refusals"):
            getattr(self, name).extend(getattr(other, name))

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(self.observed, dtype=float),
            np.asarray(self.predicted, dtype=float),
            np.asarray(self.historical, dtype=float),
            np.asarray(self.low, dtype=float),
            np.asarray(self.high, dtype=float),
            np.asarray(self.genre, dtype=object),
            np.asarray(self.slot, dtype=object),
        )


def _prediction_rows(test: pd.DataFrame) -> pd.DataFrame:
    """Test observations as prediction-surface rows (the frozen input contract)."""
    return pd.DataFrame({
        "date": test["date"].astype(str).str.slice(0, 10),
        "channel": test["channel"].astype(str),
        "program_title": test["title"].astype(str),
        "start_seconds": test["start_seconds"].astype(float),
        "duration_seconds": 0.0,
    }, columns=list(PREDICTION_COLUMNS)).reset_index(drop=True)


def _train_lineup_fn(spots: Optional[pd.DataFrame], cutoff: pd.Timestamp) -> Callable:
    """A lineup builder that may only see airings BEFORE ``cutoff``.

    The lineup carries rival title strengths, which are measured mean ratings.
    Restricting the source to prior airings is what makes the fold's competitor
    factor leak-free; the cost is that test dates fall outside the coverage and
    the family reports not-applicable there, which is stated on the payload.
    """
    def build(dates: Any, owned_channel: str) -> Any:
        from kairos.model.competitor_lineup import lineup_frame

        return lineup_frame(dates, owned_channel, spots=spots, epg=None)

    if spots is None:
        return build
    prior = spots[pd.to_datetime(spots["air_dt"], errors="coerce") < cutoff]

    def build_prior(dates: Any, owned_channel: str) -> Any:
        from kairos.model.competitor_lineup import lineup_frame

        return lineup_frame(dates, owned_channel, spots=prior, epg=None)

    return build_prior


def _score_fold(
    train: pd.DataFrame, test: pd.DataFrame, *, level: float,
    owned_channel: str, spots: Optional[pd.DataFrame], cutoff: pd.Timestamp,
    stamp: str,
) -> tuple[dict[str, Any], _Scored]:
    """Refit on ``train``, forecast ``test``, and score what came back."""
    lineup_fn = _train_lineup_fn(spots, cutoff)
    model = fit_audience_model(
        frame=train, owned_channel=owned_channel, computed_at=stamp,
        lineup_frame_fn=lineup_fn,
    )
    service = ForecastService(
        model=model,
        dispersion=build_dispersion(train, tvr_floor=model.base.tvr_floor),
    )
    payloads = service.forecast_rows(
        _prediction_rows(test), level=level, lineup_frame_fn=lineup_fn,
    )
    scored = _Scored.empty()
    observed = test["tvr"].astype(float).to_numpy()
    genres = test["genre"].astype(str).to_numpy()
    slots = test["slot_band"].astype(str).to_numpy()
    for position, payload in enumerate(payloads):
        if not payload.get("available"):
            scored.refusals.append(str(payload.get("reason_en") or "refused"))
            continue
        interval = payload.get("interval", {})
        scored.observed.append(float(observed[position]))
        scored.predicted.append(float(payload["expected_tvr"]))
        scored.historical.append(float(payload["history"]["historical_tvr"]))
        scored.low.append(
            float(interval["low"]) if interval.get("available") else math.nan
        )
        scored.high.append(
            float(interval["high"]) if interval.get("available") else math.nan
        )
        scored.genre.append(str(genres[position]))
        scored.slot.append(str(slots[position]))
    fold_gates = {family: gate.get("verdict") for family, gate in model.gates.items()}
    return {"gates": fold_gates, "n_refused": len(scored.refusals)}, scored


def walk_forward(
    frame: Optional[pd.DataFrame] = None, *,
    spots: Optional[pd.DataFrame] = None,
    n_blocks: int = DEFAULT_BLOCKS,
    level: float = DEFAULT_LEVEL,
    owned_channel: Optional[str] = None,
) -> dict[str, Any]:
    """Walk the window forward, forecasting each block from its own past only.

    ``owned_channel`` defaults to the shipped artifact's, so the backtest scopes
    the competitor family exactly as the live model does without reading
    operator settings. ``frame`` and ``spots`` default to the real history.
    """
    if frame is None:
        frame = build_training_frame()
    if owned_channel is None:
        try:
            owned_channel = load_audience_model().owned_channel
        except (OSError, ValueError):
            owned_channel = ""
    if spots is None:
        try:
            from kairos.data.loaders import load_spots

            spots = load_spots()
        except Exception as exc:  # noqa: BLE001 - absent history is an honest absence
            logger.warning("backtest: spots history unavailable: %s", exc)
            spots = None

    work = frame.reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    blocks = date_blocks(work, n_blocks)
    method = {
        "kind": "walk_forward_by_date_block",
        "n_blocks": len(blocks),
        "requested_blocks": int(n_blocks),
        "interval_level": round(float(level), 4),
        "mape_tvr_floor": MAPE_TVR_FLOOR,
        "min_train_observations": MIN_TRAIN_OBSERVATIONS,
        "min_test_observations": MIN_TEST_OBSERVATIONS,
        "refit_per_fold": "fit_audience_model (pooled base, all eight gates, factor tables) plus the dispersion behind the interval",
        "competitor_note": _COMPETITOR_NOTE,
    }
    if not blocks:
        return {
            "available": False, "method": method, "folds": [],
            "reason": (
                f"the frame carries too few distinct dates to cut into {n_blocks} "
                "walk-forward blocks; no out-of-sample fold can be formed"
            ),
        }

    folds: list[dict[str, Any]] = []
    pooled = _Scored.empty()
    for index, block in enumerate(blocks):
        cutoff = min(block)
        test = work[work["date"].isin(block)]
        train = work[work["date"] < cutoff]
        window = {
            "test_from": str(min(block).date()), "test_to": str(max(block).date()),
            "train_from": (
                str(pd.Timestamp(train["date"].min()).date()) if len(train) else None
            ),
            "train_to": (
                str(pd.Timestamp(train["date"].max()).date()) if len(train) else None
            ),
        }
        head = {"fold": index + 1, "n_train": int(len(train)), "n_test": int(len(test)),
                **window}
        if len(train) == 0:
            folds.append({**head, **_unavailable(
                "the first block has no prior observations to fit on; it is the "
                "training seed and is never scored"
            )})
            continue
        if len(train) < MIN_TRAIN_OBSERVATIONS:
            folds.append({**head, **_unavailable(
                f"only {len(train)} prior observations, fewer than the "
                f"{MIN_TRAIN_OBSERVATIONS} a fold will fit on; no figure is reported"
            )})
            continue
        if len(test) < MIN_TEST_OBSERVATIONS:
            folds.append({**head, **_unavailable(
                f"only {len(test)} observations in this block, fewer than the "
                f"{MIN_TEST_OBSERVATIONS} needed to score a fold"
            )})
            continue
        detail, scored = _score_fold(
            train, test, level=level, owned_channel=owned_channel,
            spots=spots, cutoff=cutoff, stamp=f"fold-{index + 1}-through-{window['train_to']}",
        )
        pooled.extend(scored)
        observed, predicted, historical, low, high, _genre, _slot = scored.arrays()
        folds.append({
            **head, **detail,
            **_metrics(observed, predicted, historical, low, high, level),
        })

    observed, predicted, historical, low, high, genre, slot = pooled.arrays()
    overall = _metrics(observed, predicted, historical, low, high, level)
    scored_folds = [f for f in folds if f.get("available")]
    return {
        "available": bool(scored_folds) and bool(overall.get("available")),
        "method": method,
        "overall": overall,
        "verdict": verdict_for(overall),
        "folds": folds,
        "by_genre": _breakdown(genre, observed, predicted, historical, low, high, level),
        "by_slot": _breakdown(slot, observed, predicted, historical, low, high, level),
        "gaps": [
            {"kind": "fold_unavailable", "fold": f["fold"], "reason": f["reason"]}
            for f in folds if not f.get("available")
        ] + [{"kind": "family_excluded", "family": "competitor_lineup",
              "reason": _COMPETITOR_NOTE}],
        "n_folds_scored": len(scored_folds),
        "n_observations_scored": int(len(observed)),
        "window": {
            "from": str(pd.Timestamp(work["date"].min()).date()) if len(work) else None,
            "to": str(pd.Timestamp(work["date"].max()).date()) if len(work) else None,
            "n_observations": int(len(work)),
        },
    }
