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
from pathlib import Path
from typing import Mapping

from kairos.model.spec import ChannelDescriptor, meridian_available
from kairos.optimize.pricing import OptimizerAssumptions

logger = logging.getLogger(__name__)

_CHANNEL_SEPARATOR = "_"


class ImpactModel(ABC):
    """Contract for supplying a per-segment retention impact coefficient.

    Implementations map a break's attributes to the retention change per break
    (the optimizer's ``impact_coefficient``). The value is a delta on the [0, 1]
    retention multiplier and is normally <= 0.
    """

    #: Honest label of where the coefficients come from ("assumption" or "trained").
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

    @property
    def is_trained(self) -> bool:
        """True only when the coefficients come from a fitted posterior."""
        return self.source == "trained"


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

    def __init__(self, coefficients: Mapping[str, float], *, default: float) -> None:
        if not coefficients:
            raise ValueError("PosteriorImpactModel needs at least one fitted coefficient")
        self._coefficients = dict(coefficients)
        self._default = float(default)

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(self._coefficients)

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


def load_impact_model(
    path: str | Path,
    *,
    assumptions: OptimizerAssumptions | None = None,
) -> ImpactModel:
    """Load a fitted impact model from ``path``, or fall back honestly.

    Returns a :class:`PosteriorImpactModel` only when the pkl exists and Meridian
    is available to interpret it. In every other case (missing file, or Meridian
    not installed) it returns the :class:`AssumptionImpactModel` and logs which
    path was taken, so a caller without a trained model still gets a working,
    clearly-labelled model rather than an error.
    """
    assumptions = assumptions or OptimizerAssumptions()
    model_path = Path(path)

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

    # Both the file and Meridian are present: read the per-channel coefficients.
    # Owner-flagged future work; never reached in the deps-absent environment.
    coefficients = _extract_coefficients(model_path)  # pragma: no cover - needs meridian
    if not coefficients:  # pragma: no cover - needs meridian
        logger.info(
            "Posterior at %s carried no usable coefficients; using the declared assumption.",
            model_path,
        )
        return AssumptionImpactModel(assumptions)
    logger.info("Loaded trained impact coefficients for %d channels from %s.", len(coefficients), model_path)
    return PosteriorImpactModel(  # pragma: no cover - needs meridian
        coefficients, default=assumptions.retention_impact_per_break
    )


def _extract_coefficients(model_path: Path) -> dict[str, float]:  # pragma: no cover - needs meridian
    """Extract per-channel retention coefficients from a fitted posterior pkl.

    This mirrors the legacy extraction: read the media coefficients from the
    posterior, average across chains and draws, and key them by channel name.
    Guarded behind :func:`load_impact_model` so it only runs with Meridian
    present and a real posterior on disk; no value is fabricated here.
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
