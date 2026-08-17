"""Measured dispersion behind each pooling level of the audience model.

The audience model (:mod:`kairos.model.audience_model`) is a shrunk log-space
level per cell: ``theta = (n*mean + k*parent) / (n + k)``. The artifact keeps
the fitted levels but not the scatter they were fitted from, so a confidence
range cannot be read off the artifact alone. This module measures that scatter
from the same training frame the model was fitted on, at each level the model
actually resolves through (series, genre, slot, channel, global), and hands the
forecast service two numbers per level plus the count behind the exact cell:

  * ``sigma_within`` -- the pooled spread of single observations inside their
    own cell, in log space. This is the irreducible noise a forecast of one
    programme-slot faces even when the level is known perfectly.
  * ``tau`` -- the DerSimonian-Laird between-cell spread: how much the cells at
    this level genuinely differ from one another once their own sampling noise
    is subtracted. This is what the shrinkage costs a thin cell, because a thin
    cell is pulled toward the parent by weight ``k / (n + k)`` and the parent is
    ``tau`` away from where the cell truly sits.

From those the forecast service forms the predictive variance actually implied
by the estimator the model uses (see :func:`predictive_sd`). Nothing here is
assumed: a level with no cell carrying two observations reports unavailable with
the reason, and the band is then withheld rather than invented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The levels the pooled base and the series factor resolve through, deepest
# first, and the training-frame column that keys each level's cells. ``channel``
# and ``global`` are scoped across the whole frame rather than inside a channel.
LEVEL_ORDER = ("series", "genre", "slot", "channel", "global")
LEVEL_COLUMNS = {"series": "series_key", "genre": "genre", "slot": "slot_band"}
PARENT_LEVEL = {
    "series": "genre",
    "genre": "channel",
    "slot": "channel",
    "channel": "global",
    "global": "",
}

# The scope key standing for "not inside one channel" (the channel and global
# levels), so one table dict serves every level.
GLOBAL_SCOPE = ""

# The single cell key of the global level: everything measured, one cell.
GLOBAL_CELL = "*"


@dataclass(frozen=True)
class LevelDispersion:
    """One ``(level, channel)`` table of measured scatter, or an honest absence."""

    level: str
    channel: str
    sigma_within: float
    tau: float
    n_cells: int
    n_observations: int
    counts: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    @property
    def available(self) -> bool:
        """Whether a band can be computed from this table at all."""
        return bool(math.isfinite(self.sigma_within))

    def summary(self) -> dict[str, Any]:
        """The compact, channel-name-free description for a payload."""
        return {
            "level": self.level,
            "sigma_within": None if not self.available else round(self.sigma_within, 6),
            "tau": None if not math.isfinite(self.tau) else round(self.tau, 6),
            "n_cells": self.n_cells,
            "n_observations": self.n_observations,
            "available": self.available,
            "reason": self.reason,
        }


def _unavailable(level: str, channel: str, reason: str) -> LevelDispersion:
    return LevelDispersion(
        level=level, channel=channel, sigma_within=float("nan"), tau=float("nan"),
        n_cells=0, n_observations=0, counts={}, reason=reason,
    )


def _measure(level: str, channel: str, keys: pd.Series, values: np.ndarray) -> LevelDispersion:
    """Pooled within-cell and DerSimonian-Laird between-cell spread of ``values``.

    ``keys`` labels each observation's cell. ``sigma_within`` pools the
    within-cell sums of squares over their degrees of freedom, so cells of one
    observation contribute nothing to it and cannot pretend to zero variance.
    ``tau`` is the classic one-way random-effects moment estimator, floored at
    zero: cells that differ by no more than their own sampling noise honestly
    measure no between-cell spread at all.
    """
    frame = pd.DataFrame({"cell": keys.astype(str).to_numpy(), "y": values})
    frame = frame[frame["cell"] != ""]
    if frame.empty:
        return _unavailable(level, channel, f"no observations carry a {level} key")
    grouped = frame.groupby("cell", sort=True)["y"]
    counts = grouped.size()
    means = grouped.mean()
    # Sum of squared deviations inside each cell; a one-observation cell is 0
    # with 0 degrees of freedom, so it drops out of both sums below.
    sums_of_squares = grouped.apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    degrees = float((counts - 1).clip(lower=0).sum())
    total = int(counts.sum())
    n_cells = int(len(counts))
    if degrees <= 0:
        return _unavailable(
            level, channel,
            f"no {level} cell carries two observations, so the within-cell "
            "spread cannot be measured",
        )
    sigma_within = math.sqrt(float(sums_of_squares.sum()) / degrees)

    tau = float("nan")
    if n_cells >= 2 and total > 0:
        weights = counts.to_numpy(dtype=float)
        grand = float((weights * means.to_numpy(dtype=float)).sum() / weights.sum())
        q_statistic = float((weights * (means.to_numpy(dtype=float) - grand) ** 2).sum())
        denominator = float(weights.sum() - (weights**2).sum() / weights.sum())
        if denominator > 0:
            tau_squared = max(
                0.0, (q_statistic - (n_cells - 1) * sigma_within**2) / denominator
            )
            tau = math.sqrt(tau_squared)
    if level == "global":
        # The global level has no parent to be shrunk toward, so the
        # between-cell term is zero by construction rather than unmeasured.
        tau = 0.0
    return LevelDispersion(
        level=level, channel=channel, sigma_within=sigma_within, tau=tau,
        n_cells=n_cells, n_observations=total,
        counts={str(cell): int(n) for cell, n in counts.items()},
        reason="" if math.isfinite(tau) else (
            f"fewer than two {level} cells; the between-cell spread cannot be measured"
        ),
    )


@dataclass(frozen=True)
class Dispersion:
    """Every level's measured scatter, keyed ``(level, channel scope)``."""

    tables: dict[tuple[str, str], LevelDispersion]
    n_observations: int
    tvr_floor: float
    window_from: Optional[str] = None
    window_to: Optional[str] = None

    def table(self, level: str, channel: str) -> LevelDispersion:
        """The table for one level, or an honest absence (never a guess)."""
        scope = GLOBAL_SCOPE if level in ("channel", "global") else str(channel)
        found = self.tables.get((level, scope))
        if found is not None:
            return found
        return _unavailable(
            level, scope,
            f"the training frame carries no {level} observations for this scope",
        )

    def summary(self) -> dict[str, Any]:
        """Level-by-level scatter with no channel named: a disclosure payload.

        Channel-scoped levels are folded to their observation-weighted mean so
        the surface can state the spread behind the model without listing the
        channels it was measured on (the competitor boundary).
        """
        out: dict[str, Any] = {}
        for level in LEVEL_ORDER:
            tables = [t for (lvl, _), t in self.tables.items() if lvl == level and t.available]
            if not tables:
                out[level] = {"available": False, "reason": "not measured in this frame"}
                continue
            weights = float(sum(t.n_observations for t in tables)) or 1.0
            out[level] = {
                "available": True,
                "sigma_within": round(
                    sum(t.sigma_within * t.n_observations for t in tables) / weights, 6
                ),
                "tau": round(
                    sum((t.tau if math.isfinite(t.tau) else 0.0) * t.n_observations
                        for t in tables) / weights, 6,
                ),
                "n_cells": int(sum(t.n_cells for t in tables)),
                "n_observations": int(sum(t.n_observations for t in tables)),
                "scopes": len(tables),
            }
        return out


def build_dispersion(frame: pd.DataFrame, *, tvr_floor: float) -> Dispersion:
    """Measure every level's scatter from the model's own training frame.

    ``frame`` is a :func:`kairos.model.audience_frame.build_training_frame`
    result. Log space matches the model: a measured zero enters at the same
    floor the fit used, so the scatter is the scatter the levels were fitted
    from and not a differently-transformed one.
    """
    tables: dict[tuple[str, str], LevelDispersion] = {}
    if frame is None or len(frame) == 0:
        return Dispersion(tables={}, n_observations=0, tvr_floor=tvr_floor)
    work = frame.reset_index(drop=True)
    values = np.log(np.maximum(work["tvr"].astype(float).to_numpy(), tvr_floor))

    for channel, positions in work.groupby("channel", sort=True).groups.items():
        index = np.asarray(positions)
        rows = work.loc[index]
        subset = values[index]
        for level, column in LEVEL_COLUMNS.items():
            tables[(level, str(channel))] = _measure(level, str(channel), rows[column], subset)
    tables[("channel", GLOBAL_SCOPE)] = _measure(
        "channel", GLOBAL_SCOPE, work["channel"], values
    )
    tables[("global", GLOBAL_SCOPE)] = _measure(
        "global", GLOBAL_SCOPE, pd.Series([GLOBAL_CELL] * len(work)), values
    )

    dates = pd.to_datetime(work["date"], errors="coerce").dropna()
    return Dispersion(
        tables=tables, n_observations=int(len(work)), tvr_floor=float(tvr_floor),
        window_from=dates.min().date().isoformat() if len(dates) else None,
        window_to=dates.max().date().isoformat() if len(dates) else None,
    )


def predictive_sd(
    table: LevelDispersion, cell_key: str, shrinkage_k: float
) -> tuple[Optional[float], dict[str, Any]]:
    """The log-space predictive spread implied by the estimator the model uses.

    Three measured terms, and no fourth:

    ``sigma_within^2``
        one observation's scatter around its own cell.
    ``(k / (n + k))^2 * tau^2``
        the level error the shrinkage buys: the estimate sits ``k / (n + k)`` of
        the way toward the parent, and cells are ``tau`` apart, so a cell with
        three observations carries most of that gap and a cell with three
        hundred carries almost none. This term is why a thin cell's band is
        visibly wider, and it is the shrinkage weight the fit actually applied.
    ``n * sigma_within^2 / (n + k)^2``
        the sampling error of the cell's own mean inside the shrunk estimate.

    Returns ``(sd, components)``; ``sd`` is None when the level's scatter could
    not be measured, and ``components`` then carries the reason.
    """
    n = int(table.counts.get(str(cell_key), 0))
    weight_on_parent = float(shrinkage_k) / (n + float(shrinkage_k))
    components: dict[str, Any] = {
        "level": table.level,
        "n_observations": n,
        "n_cells_at_level": table.n_cells,
        "shrinkage_k": float(shrinkage_k),
        "weight_on_parent": round(weight_on_parent, 6),
    }
    if not table.available:
        components["reason"] = table.reason or "the level's scatter is not measured"
        return None, components
    tau = table.tau if math.isfinite(table.tau) else None
    if tau is None:
        components["reason"] = (
            table.reason
            or f"the between-cell spread at the {table.level} level is not measured"
        )
        return None, components
    within = table.sigma_within**2
    level_error = (weight_on_parent**2) * (tau**2) + n * within / (n + float(shrinkage_k)) ** 2
    components.update({
        "sigma_within": round(table.sigma_within, 6),
        "tau": round(tau, 6),
        "var_observation": round(within, 8),
        "var_level": round(level_error, 8),
    })
    return math.sqrt(within + level_error), components
