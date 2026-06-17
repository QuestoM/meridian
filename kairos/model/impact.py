"""The retention-impact contract between a trained posterior and the optimizer.

The optimizer values every break with a per-segment ``impact_coefficient``: the
retention change a single break causes, in the same units as
:class:`kairos.optimize.optimizer.ProgramSegment.impact_coefficient` (a delta on
the [0, 1] retention multiplier, normally negative because breaks shed audience).
See :func:`kairos.optimize.objective.predicted_retention`, where it is applied as
``baseline + impact_coefficient * k`` for ``k`` breaks.

This module defines where that number comes from:

  * :class:`ImpactModel` is the abstract contract: given a break's program type,
    position and length, return its retention impact coefficient.
  * :class:`AssumptionImpactModel` is the honest fallback used until Meridian is
    trained. It returns the single declared assumption,
    :attr:`kairos.optimize.pricing.OptimizerAssumptions.retention_impact_per_break`,
    for every segment, and labels itself as an assumption so nothing pretends to
    be a fitted result.
  * :class:`PosteriorImpactModel` is the trained path: it reads per-channel
    coefficients extracted from a fitted Meridian posterior. It is only built
    when such a posterior is loaded.
  * :func:`load_impact_model` chooses between them: a real posterior when the pkl
    exists and Meridian is available, the assumption fallback otherwise, with a
    log line stating which was used.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from kairos.model.spec import ChannelDescriptor, meridian_available
from kairos.optimize.pricing import OptimizerAssumptions

logger = logging.getLogger(__name__)

_CHANNEL_SEPARATOR = "_"


@dataclass(frozen=True)
class RetentionEstimate:
    """A per-cell retention-impact estimate with its uncertainty.

    ``coefficient`` is the point retention delta the optimizer consumes (<= 0).
    ``ci_low``/``ci_high`` bound it (a measured credible interval, or the point
    itself when no interval is known), ``n`` is the number of breaks behind it, and
    ``confidence`` is the ``high``/``medium``/``low`` label from
    :func:`kairos.model.measure.confidence_label`. This is the full posterior the
    Stage 1 design carries end to end so the decision can be uncertainty-aware.
    """

    coefficient: float
    ci_low: float
    ci_high: float
    n: int
    confidence: str


class ImpactModel(ABC):
    """Contract for supplying a per-segment retention impact coefficient.

    Implementations map a break's attributes to the retention change per break
    (the optimizer's ``impact_coefficient``). The value is a delta on the [0, 1]
    retention multiplier and is normally <= 0.
    """

    #: Honest label of where the coefficients come from: "measured" (the real
    #: detrended per-break effect), "trained" (a fitted Meridian posterior), or
    #: "assumption" (the declared fallback).
    source: str = "unknown"

    @abstractmethod
    def coefficient_for(
        self,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> float:
        """Return the retention impact coefficient for one break attribute cell."""
        raise NotImplementedError

    def estimate_for(
        self,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> RetentionEstimate:
        """Return the full retention estimate (point + interval + n + confidence).

        The default wraps :meth:`coefficient_for` into a degenerate estimate with
        no interval (``ci_low == ci_high == coefficient``), ``n`` 0 and ``low``
        confidence, so every model satisfies the richer contract even when it only
        knows a point. Models that carry real uncertainty override this.
        """
        point = self.coefficient_for(program_type, break_position, break_length)
        return RetentionEstimate(
            coefficient=point, ci_low=point, ci_high=point, n=0, confidence="low",
        )

    @property
    def is_trained(self) -> bool:
        """True when the coefficients are a fitted or measured result, not the fallback."""
        return self.source != "assumption"


class AssumptionImpactModel(ImpactModel):
    """Fallback model that returns the declared retention assumption for any break.

    Until Meridian is trained, every segment uses the single owner-declared
    number, :attr:`OptimizerAssumptions.retention_impact_per_break`. This is an
    assumption, not a measurement, and ``source`` says so. It keeps the whole
    pipeline runnable with no trained model present.
    """

    source = "assumption"

    def __init__(self, assumptions: OptimizerAssumptions | None = None) -> None:
        self._assumptions = assumptions or OptimizerAssumptions()

    @property
    def assumptions(self) -> OptimizerAssumptions:
        return self._assumptions

    @property
    def coefficient(self) -> float:
        """The single declared retention impact per break."""
        return self._assumptions.retention_impact_per_break

    def coefficient_for(
        self,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> float:
        # The assumption is flat across attributes by design: it is one declared
        # number, not a per-channel estimate.
        return self._assumptions.retention_impact_per_break


class PosteriorImpactModel(ImpactModel):
    """Trained model backed by per-channel coefficients from a fitted posterior.

    ``coefficients`` maps a channel name (``program_type_position_length``) to its
    retention impact coefficient, as extracted from a Meridian posterior. Lookups
    fall back to ``default`` (the declared assumption) for any channel the
    posterior did not estimate, so an unseen break attribute never raises.
    """

    source = "trained"

    def __init__(
        self,
        coefficients: Mapping[str, float],
        *,
        default: float,
        source: str = "trained",
        detail: Optional[Mapping[str, RetentionEstimate]] = None,
        series: Optional[Mapping[tuple[str, str], RetentionEstimate]] = None,
    ) -> None:
        if not coefficients:
            raise ValueError("PosteriorImpactModel needs at least one fitted coefficient")
        self._coefficients = dict(coefficients)
        self._default = float(default)
        self.source = source
        # Optional per-cell uncertainty (interval + n + confidence). When present
        # the model exposes the full posterior through estimate_for; when absent
        # estimate_for degrades to the point coefficient with low confidence.
        self._detail = dict(detail) if detail else {}
        # Optional series layer: (genre cell name, canonical series key) -> estimate.
        # When a segment carries a program_title whose series sits in this map under
        # its genre cell, the series-aware coefficient is returned; otherwise the
        # lookup falls back to the genre cell (honest cold-start).
        self._series = dict(series) if series else {}

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(self._coefficients)

    @property
    def has_detail(self) -> bool:
        """True when this model carries per-cell credible intervals and counts."""
        return bool(self._detail)

    def coefficient_for(
        self,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> float:
        name = ChannelDescriptor.from_parts(program_type, break_position, break_length).name
        if name in self._coefficients:
            return self._coefficients[name]
        logger.debug("No fitted coefficient for channel %s; using declared default", name)
        return self._default

    def estimate_for(
        self,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> RetentionEstimate:
        """Return the full retention estimate, with interval and confidence when known.

        When detail was loaded for this cell, the credible interval, sample count
        and confidence label travel with the point coefficient. For a cell the
        data did not estimate, this falls back to the declared default with no
        interval and ``low`` confidence, so an unseen break attribute never raises
        and never claims a confidence it does not have.
        """
        name = ChannelDescriptor.from_parts(program_type, break_position, break_length).name
        if name in self._detail:
            return self._detail[name]
        point = self.coefficient_for(program_type, break_position, break_length)
        return RetentionEstimate(
            coefficient=point, ci_low=point, ci_high=point, n=0, confidence="low",
        )

    @property
    def has_series(self) -> bool:
        """True when this model carries a series-aware layer."""
        return bool(self._series)

    def coefficient_for_title(
        self,
        program_title: str,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> float:
        """Return the series-aware coefficient for a break, else the genre cell.

        The programme ``program_title`` is canonicalized to a series key (the same
        deterministic key the trainer used), and if that series exists under this
        break's genre cell in the trained series layer, its coefficient is returned.
        Otherwise the lookup falls back to :meth:`coefficient_for`, the genre cell
        coefficient, so a NEW title never seen in training still gets an honest
        genre-level answer (cold-start). The genre effect always backs the result.
        """
        name = ChannelDescriptor.from_parts(program_type, break_position, break_length).name
        key = self._series_key(program_title)
        if key and (name, key) in self._series:
            return self._series[(name, key)].coefficient
        return self.coefficient_for(program_type, break_position, break_length)

    def estimate_for_title(
        self,
        program_title: str,
        program_type: str,
        break_position: str,
        break_length: str,
    ) -> RetentionEstimate:
        """Full series-aware estimate (interval + n + confidence), else genre cell.

        Like :meth:`coefficient_for_title` but returns the whole estimate, so the
        decision stays uncertainty-aware on the series layer too. Falls back to the
        genre-cell estimate for an unseen series (honest cold-start).
        """
        name = ChannelDescriptor.from_parts(program_type, break_position, break_length).name
        key = self._series_key(program_title)
        if key and (name, key) in self._series:
            return self._series[(name, key)]
        return self.estimate_for(program_type, break_position, break_length)

    @staticmethod
    def _series_key(program_title: str) -> str:
        """Canonicalize a programme title to its series key (empty on failure).

        Lazy import keeps the impact <-> data import graph free of a hard cycle and
        lets impact.py import on a desktop Python even if the title module's
        optional deps are missing.
        """
        try:
            from kairos.data.title_features import canonicalize_series

            return canonicalize_series(program_title)
        except Exception:  # noqa: BLE001 - any failure degrades to the genre cell
            return ""


def load_impact_model(
    path: str | Path,
    *,
    assumptions: OptimizerAssumptions | None = None,
    coefficients_path: str | Path | None = None,
) -> ImpactModel:
    """Load the best available impact model for ``path``, or fall back honestly.

    Resolution order, most authoritative first:

      1. Measured coefficients JSON (``models/tv_break_coefficients.json`` next to
         the pkl, or ``coefficients_path``). These are the real detrended per-break
         retention effects from :mod:`kairos.model.measure`, in the optimizer's
         units. They need no Meridian, so they drive the optimizer even on a
         desktop Python. ``source`` is "measured".
      2. A fitted Meridian posterior pkl, interpreted into retention deltas, when
         the pkl and Meridian are both present. ``source`` is "trained".
      3. The declared assumption, labelled "assumption", otherwise.

    Every path is logged, so a caller always gets a working, clearly-labelled
    model rather than an error.
    """
    assumptions = assumptions or OptimizerAssumptions()
    model_path = Path(path)

    # 1. Measured coefficients (preferred, no Meridian needed). Lazy import keeps
    # the impact <-> measure <-> prepare <-> transform import cycle from forming.
    # We read the full per-cell detail (point + credible interval + n + confidence)
    # so the uncertainty reaches the optimizer in a real run, not only in unit
    # construction. ``read_coefficients_json`` stays the back-compat flat reader.
    from kairos.model.measure import read_coefficients_detail, read_coefficients_json
    from kairos.model.series import read_series_coefficients

    coeff_path = Path(coefficients_path) if coefficients_path else model_path.with_name(
        "tv_break_coefficients.json"
    )
    detail = read_coefficients_detail(coeff_path)
    if detail:
        coefficients = {name: d.coefficient for name, d in detail.items()}
        estimates = {
            name: RetentionEstimate(
                coefficient=d.coefficient,
                ci_low=d.ci_low,
                ci_high=d.ci_high,
                n=d.n,
                confidence=d.confidence,
            )
            for name, d in detail.items()
        }
        # Additive series layer (empty for a pre-series JSON, leaving behaviour
        # identical). Keyed (cell, series) -> estimate for the title-aware lookup.
        series = {
            key: RetentionEstimate(
                coefficient=d.coefficient,
                ci_low=d.ci_low,
                ci_high=d.ci_high,
                n=d.n,
                confidence=d.confidence,
            )
            for key, d in read_series_coefficients(coeff_path).items()
        }
        logger.info(
            "Loaded %d measured retention coefficients (with uncertainty) from %s (%d series).",
            len(coefficients),
            coeff_path,
            len(series),
        )
        return PosteriorImpactModel(
            coefficients,
            default=assumptions.retention_impact_per_break,
            source="measured",
            detail=estimates,
            series=series,
        )

    # Back-compat fallback: a coefficients file that carries only the flat map and
    # no detail block still loads as a point-estimate model.
    measured = read_coefficients_json(coeff_path)
    if measured:
        logger.info("Loaded %d measured retention coefficients from %s.", len(measured), coeff_path)
        return PosteriorImpactModel(
            measured, default=assumptions.retention_impact_per_break, source="measured"
        )

    if not model_path.exists():
        logger.info(
            "No trained posterior at %s; using the declared retention assumption (%.4f per break).",
            model_path,
            assumptions.retention_impact_per_break,
        )
        return AssumptionImpactModel(assumptions)

    if not meridian_available():
        logger.info(
            "Found %s but meridian is not installed; using the declared retention assumption "
            "(%.4f per break) instead of the trained posterior.",
            model_path,
            assumptions.retention_impact_per_break,
        )
        return AssumptionImpactModel(assumptions)

    # Both the file and Meridian are present: read the raw posterior coefficients
    # and map them into the optimizer's retention-delta units.
    raw = _extract_coefficients(model_path)  # pragma: no cover - needs meridian
    if not raw:  # pragma: no cover - needs meridian
        logger.info(
            "Posterior at %s carried no usable coefficients; using the declared assumption.",
            model_path,
        )
        return AssumptionImpactModel(assumptions)
    coefficients = _to_retention_deltas(raw, anchor=assumptions.retention_impact_per_break)
    logger.info("Loaded trained impact coefficients for %d channels from %s.", len(coefficients), model_path)
    return PosteriorImpactModel(  # pragma: no cover - needs meridian
        coefficients, default=assumptions.retention_impact_per_break
    )


def _extract_coefficients(model_path: Path) -> dict[str, float]:  # pragma: no cover - needs meridian
    """Extract the raw per-channel media coefficients from a fitted posterior.

    Reads Meridian's media coefficient (``beta_m``) from the posterior, averages
    across chains and draws, and keys it by channel name. These are the model's
    own units (a positive media-response scale), not the optimizer's retention
    delta; :func:`_to_retention_deltas` performs that mapping. No value is
    fabricated here. Guarded behind :func:`load_impact_model`.
    """
    from meridian.model.model import load_mmm  # type: ignore

    fitted = load_mmm(str(model_path))
    posterior = fitted.inference_data.posterior
    param = "beta_m" if "beta_m" in posterior else "beta_media"
    mean_coefs = posterior[param].mean(dim=("chain", "draw"))

    coefficients: dict[str, float] = {}
    for channel in mean_coefs["media_channel"].values:
        coefficients[str(channel)] = float(mean_coefs.sel(media_channel=channel).values)
    return coefficients


def _to_retention_deltas(raw: Mapping[str, float], *, anchor: float) -> dict[str, float]:
    """Map raw Meridian media coefficients into the optimizer's retention deltas.

    The optimizer consumes a per-break retention delta that is, by the engine's
    own premise, negative (a break sheds audience, see
    :func:`kairos.optimize.objective.predicted_retention`). Meridian's ``beta_m``
    is on a different, positive media-response scale and cannot be used directly.
    This applies a documented, principled normalization: each channel keeps the
    relative impact the posterior measured, scaled so the average channel equals
    the declared ``anchor`` magnitude and every channel takes the engine's
    negative sign. So the trained model drives which break attributes shed more
    or less retention, while the overall magnitude stays in the declared, sane
    range. With a degenerate posterior (all zeros) the anchor is returned for
    every channel, which is the honest assumption fallback.
    """
    magnitudes = {channel: abs(value) for channel, value in raw.items()}
    mean_magnitude = (sum(magnitudes.values()) / len(magnitudes)) if magnitudes else 0.0
    if mean_magnitude <= 0.0:
        return {channel: anchor for channel in raw}
    sign_anchor = -abs(anchor)
    return {
        channel: sign_anchor * (magnitude / mean_magnitude)
        for channel, magnitude in magnitudes.items()
    }
