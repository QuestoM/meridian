"""Meridian-compatible model specification for the TV break impact model.

This module describes the media-mix model Kairos will fit once a trained
Meridian posterior is available, and it does so without requiring Meridian or
TensorFlow to be installed. The KPI is viewer retention (expressed in TVR rating
points); the media channels are the cross-product of three break attributes the
channel actually controls:

  * ``program_type``   the genre of the surrounding programme (News, Drama, ...),
  * ``break_position`` where the break sits in the programme (first, middle, last),
  * ``break_length``   the length bucket of the break (short, standard, long).

A channel name is the three attributes joined with ``_`` (``News_first_short``),
matching the ``program_type_position_length`` convention the legacy model used
when it split posterior coefficients back into break attributes.

Meridian is imported behind a guard so this description stays pure Python and
fully testable. :func:`build_channel_spec` builds the channel descriptors with no
dependency on Meridian at all. :func:`assemble_meridian_spec` is the seam that
would build the real ``meridian.model.spec.ModelSpec`` once the library and a
trained dataset are present; it is never called at import time.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

# Guarded optional import: the model layer must import on a desktop Python that
# has neither Meridian nor TensorFlow. ``meridian`` stays None when absent, and
# every Meridian-touching function checks it before use.
try:  # pragma: no cover - exercised only where Meridian is installed
    import meridian
except Exception:  # noqa: BLE001 - any import failure means "not available"
    meridian = None

# The default attribute vocabularies. Program types mirror the classifier's
# pricing classes; positions and lengths are the buckets the optimizer reasons
# about. They are declared here so the channel grid is explicit and editable.
DEFAULT_PROGRAM_TYPES: tuple[str, ...] = ("News", "PrimeShow1", "PrimeShow2", "Other")
DEFAULT_BREAK_POSITIONS: tuple[str, ...] = ("first", "middle", "last")
DEFAULT_BREAK_LENGTHS: tuple[str, ...] = ("short", "standard", "long")

KPI_NAME = "retention_tvr"
KPI_DESCRIPTION = "Viewer retention measured in TVR rating points"
_CHANNEL_SEPARATOR = "_"


def meridian_available() -> bool:
    """Return True when the Meridian library imported successfully.

    This is a pure runtime check with no side effects, so callers can branch on
    it cheaply (the optional import already ran at module load).
    """
    return meridian is not None


@dataclass(frozen=True)
class ChannelDescriptor:
    """One media channel: a (program_type x break_position x break_length) cell.

    ``name`` is the joined channel id used as the Meridian media-channel
    coordinate; the three attribute fields keep the parts addressable so a fitted
    coefficient can be mapped straight back to a break decision.
    """

    name: str
    program_type: str
    break_position: str
    break_length: str

    @classmethod
    def from_parts(cls, program_type: str, break_position: str, break_length: str) -> "ChannelDescriptor":
        name = _CHANNEL_SEPARATOR.join((program_type, break_position, break_length))
        return cls(
            name=name,
            program_type=program_type,
            break_position=break_position,
            break_length=break_length,
        )


@dataclass(frozen=True)
class ChannelSpec:
    """A pure-Python description of the Meridian model's channels and KPI.

    This is the testable contract: it captures every media channel the model
    would carry and the KPI it predicts, without constructing any Meridian
    object. :func:`assemble_meridian_spec` consumes it when the real library is
    present.
    """

    kpi_name: str
    kpi_description: str
    program_types: tuple[str, ...]
    break_positions: tuple[str, ...]
    break_lengths: tuple[str, ...]
    channels: tuple[ChannelDescriptor, ...] = field(default_factory=tuple)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view, handy for logging or JSON serialisation."""
        return {
            "kpi_name": self.kpi_name,
            "kpi_description": self.kpi_description,
            "program_types": list(self.program_types),
            "break_positions": list(self.break_positions),
            "break_lengths": list(self.break_lengths),
            "channels": [
                {
                    "name": channel.name,
                    "program_type": channel.program_type,
                    "break_position": channel.break_position,
                    "break_length": channel.break_length,
                }
                for channel in self.channels
            ],
        }


def build_channel_spec(
    program_types: tuple[str, ...] = DEFAULT_PROGRAM_TYPES,
    break_positions: tuple[str, ...] = DEFAULT_BREAK_POSITIONS,
    break_lengths: tuple[str, ...] = DEFAULT_BREAK_LENGTHS,
    *,
    kpi_name: str = KPI_NAME,
    kpi_description: str = KPI_DESCRIPTION,
) -> ChannelSpec:
    """Build the (program_type x break_position x break_length) channel spec.

    The channel grid is the full cross-product of the three vocabularies, in a
    deterministic order (program type outermost, length innermost). The result is
    pure data: no Meridian object is created and nothing needs Meridian or
    TensorFlow installed, so this is the function tests assert against.
    """
    if not program_types or not break_positions or not break_lengths:
        raise ValueError("each of program_types, break_positions and break_lengths must be non-empty")

    channels = tuple(
        ChannelDescriptor.from_parts(program_type, position, length)
        for program_type, position, length in itertools.product(
            program_types, break_positions, break_lengths
        )
    )
    return ChannelSpec(
        kpi_name=kpi_name,
        kpi_description=kpi_description,
        program_types=tuple(program_types),
        break_positions=tuple(break_positions),
        break_lengths=tuple(break_lengths),
        channels=channels,
    )


def assemble_meridian_spec(channel_spec: ChannelSpec) -> Any:
    """Assemble the real ``meridian.model.spec.ModelSpec`` from a ChannelSpec.

    This is the seam to the trained-model world. It is never run at import time
    and it refuses to run without Meridian present, so the pure-Python path above
    stays usable everywhere. The body is a faithful skeleton: it imports the
    Meridian spec module and builds a ModelSpec keyed to the KPI and channels. It
    is owner-flagged future work and is not exercised until Meridian is installed.
    """
    if not meridian_available():
        raise RuntimeError(
            "assemble_meridian_spec requires the meridian library, which is not installed. "
            "Install google-meridian (with tensorflow) before assembling the real model spec."
        )

    # Imported lazily so the module imports cleanly without Meridian. The exact
    # priors and knots belong with the trained dataset; this records the contract
    # (KPI plus the cross-product channels) the trained spec must honour.
    from meridian.model import spec as meridian_spec  # pragma: no cover - needs meridian

    return meridian_spec.ModelSpec()  # pragma: no cover - needs meridian
