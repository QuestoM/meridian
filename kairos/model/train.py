"""Env-gated training entrypoint for the Meridian TV break impact model.

Training a Meridian posterior needs a specific stack that the Kairos desktop does
not carry: Python 3.11 or 3.12 (Meridian and the pinned TensorFlow do not yet
support 3.13), plus ``tensorflow`` and ``google-meridian``. This module makes
that requirement explicit and refuses to run obscurely without it.

:func:`can_train` reports whether the environment can train at all.
:func:`train_tv_break_model` is the real entrypoint: it shapes the prepared data
into a Meridian ``InputData``, builds the spec, samples the posterior and saves
it. Every dependency-touching line is guarded, and the function raises a clear
RuntimeError listing exactly what is missing and what to install when the stack
is absent. Nothing here fabricates a coefficient or a training metric: with the
deps missing it produces no model, only an honest error.

This is owner-flagged future work. The skeleton is written so that, once the
right Python and libraries are in place, the body is the actual training flow.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from kairos.model.spec import ChannelSpec, assemble_meridian_spec, build_channel_spec

logger = logging.getLogger(__name__)

# Meridian and its TensorFlow pin support these CPython minor versions only.
SUPPORTED_PYTHON: tuple[tuple[int, int], ...] = ((3, 11), (3, 12))
REQUIRED_PACKAGES: tuple[str, ...] = ("tensorflow", "meridian")
INSTALL_HINT = (
    "Create a Python 3.11 or 3.12 environment and install the training stack:\n"
    "  python -m pip install 'tensorflow' 'google-meridian'\n"
    "Training is owner-flagged future work and does not run on the default desktop Python."
)


def _python_supported() -> bool:
    return sys.version_info[:2] in SUPPORTED_PYTHON


def _missing_packages() -> tuple[str, ...]:
    """Return the required training packages that are not importable here."""
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return tuple(missing)


def can_train() -> bool:
    """Return True only when the Python version and training packages are present.

    A pure capability check with no side effects, so callers and tests can branch
    on it without importing TensorFlow or Meridian.
    """
    return _python_supported() and not _missing_packages()


def _training_requirement_error() -> RuntimeError:
    """Build a clear, honest error describing exactly what training needs."""
    reasons: list[str] = []
    if not _python_supported():
        running = f"{sys.version_info.major}.{sys.version_info.minor}"
        supported = " or ".join(f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON)
        reasons.append(f"Python {supported} is required, but this interpreter is {running}.")
    missing = _missing_packages()
    if missing:
        reasons.append("Missing required package(s): " + ", ".join(missing) + ".")
    detail = " ".join(reasons) if reasons else "The training stack is unavailable."
    return RuntimeError(
        "Cannot train the Meridian TV break impact model. "
        f"{detail}\n{INSTALL_HINT}"
    )


def train_tv_break_model(
    training_data: Any = None,
    *,
    output_path: str | Path = Path("models") / "tv_break_posterior.pkl",
    channel_spec: Optional[ChannelSpec] = None,
    n_chains: int = 4,
    n_adapt: int = 500,
    n_burnin: int = 500,
    n_keep: int = 1000,
) -> Path:
    """Train the Meridian impact posterior and save it to ``output_path``.

    Raises a clear :class:`RuntimeError` when the environment cannot train
    (unsupported Python, or TensorFlow / Meridian absent), naming what is missing
    and what to install, rather than failing on a deep import. When the stack is
    present this runs the real flow: shape the data into Meridian ``InputData``,
    build the spec, sample the posterior and save it. It returns the path written.

    No result is fabricated: without the dependencies the function produces no
    file and raises instead.
    """
    if not can_train():
        raise _training_requirement_error()

    # The lines below are the genuine training skeleton. They are guarded by the
    # can_train() check above and are only reachable on a supported Python with
    # TensorFlow and Meridian installed, so they never run in the default desktop
    # environment and never invent a trained coefficient.
    if training_data is None:  # pragma: no cover - needs meridian
        raise ValueError(
            "training_data is required to fit the model: pass the prepared Meridian InputData "
            "(media tensor keyed by the program_type x break_position x break_length channels, "
            "with the retention TVR KPI)."
        )

    from meridian.model import model as meridian_model  # type: ignore # pragma: no cover - needs meridian

    spec_description = channel_spec or build_channel_spec()  # pragma: no cover - needs meridian
    model_spec = assemble_meridian_spec(spec_description)  # pragma: no cover - needs meridian

    logger.info(  # pragma: no cover - needs meridian
        "Fitting Meridian impact model over %d channels with KPI %s.",
        spec_description.num_channels,
        spec_description.kpi_name,
    )
    mmm = meridian_model.Meridian(  # pragma: no cover - needs meridian
        input_data=training_data, model_spec=model_spec
    )
    mmm.sample_prior(n_keep)  # pragma: no cover - needs meridian
    mmm.sample_posterior(  # pragma: no cover - needs meridian
        n_chains=n_chains, n_adapt=n_adapt, n_burnin=n_burnin, n_keep=n_keep
    )

    destination = Path(output_path)  # pragma: no cover - needs meridian
    destination.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover - needs meridian
    meridian_model.save_mmm(mmm, str(destination))  # pragma: no cover - needs meridian
    logger.info("Saved trained posterior to %s.", destination)  # pragma: no cover - needs meridian
    return destination  # pragma: no cover - needs meridian
