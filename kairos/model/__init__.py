"""Kairos model layer: the env-gated seam to a trained Meridian posterior.

This package describes the Meridian impact model (channels and KPI), defines the
contract for turning a fitted posterior into the optimizer's per-segment
retention impact coefficient, and provides the env-gated training entrypoint. It
imports cleanly without Meridian or TensorFlow: the trained paths are guarded and
an honest, declared-assumption fallback keeps the pipeline runnable until the
model is fitted.
"""

from kairos.model.impact import (
    AssumptionImpactModel,
    ImpactModel,
    PosteriorImpactModel,
    load_impact_model,
)
from kairos.model.spec import (
    ChannelDescriptor,
    ChannelSpec,
    assemble_meridian_spec,
    build_channel_spec,
    meridian_available,
)
from kairos.model.train import can_train, train_tv_break_model

__all__ = [
    "AssumptionImpactModel",
    "ChannelDescriptor",
    "ChannelSpec",
    "ImpactModel",
    "PosteriorImpactModel",
    "assemble_meridian_spec",
    "build_channel_spec",
    "can_train",
    "load_impact_model",
    "meridian_available",
    "train_tv_break_model",
]
