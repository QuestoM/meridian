"""Settings models for the optional airtime caps, and their one translation.

Two caps beyond the hourly guardrail can be configured here: a total across a
wall-clock window of the day, and a total expressed as a fraction of the
broadcast day. Both are ABSENT unless an operator configures them, and both
ship switched off even once configured, because the commercial channels do not
uniformly work to a single regulatory regime and the product's job is to be
able to apply a rule and to say plainly whether it did.

Nothing in this module is a regulatory figure. No window bounds, no minute
count and no fraction are defaulted to a legal value, because the applicable
text may have been amended and an operator's licence may differ. A cap that
nobody configured does not exist; it is not a cap of zero.

Split out of :mod:`kairos_api.core` under the 450-line law: that module is the
settings home but is already well over the limit, so the models and their
translation live here and core imports them.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from kairos.optimize.guardrails import (
    SECONDS_PER_CALENDAR_DAY,
    AirtimeCaps,
    airtime_caps_from_mapping,
)

__all__ = [
    "DayFractionAdCapSettings",
    "WindowAdCapSettings",
    "airtime_caps_from_settings",
]


class WindowAdCapSettings(BaseModel):
    """A cap on total ad minutes inside a wall-clock window of the broadcast day.

    Every field but ``enabled`` is REQUIRED: there is no default window and no
    default limit, so a half-written cap is refused rather than silently
    completed with a number nobody chose. Hours are half-open,
    ``[start_hour, end_hour)``, with 24 meaning midnight.
    """

    enabled: bool = False
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)
    max_ad_minutes: float = Field(ge=0)

    @model_validator(mode="after")
    def _window_must_be_forward(self) -> "WindowAdCapSettings":
        if self.end_hour <= self.start_hour:
            raise ValueError("end_hour must be after start_hour")
        return self


class DayFractionAdCapSettings(BaseModel):
    """A cap on a channel-day's ad time as a fraction of the broadcast day.

    ``max_fraction_of_day`` is a fraction, not a percentage: a tenth of the day
    is 0.1. ``day_seconds`` is the denominator and defaults to a full calendar
    day; an operator that does not broadcast around the clock sets its own.
    """

    enabled: bool = False
    max_fraction_of_day: float = Field(gt=0, le=1)
    day_seconds: float = Field(default=SECONDS_PER_CALENDAR_DAY, gt=0)


def airtime_caps_from_settings(
    window: Optional[WindowAdCapSettings],
    day_fraction: Optional[DayFractionAdCapSettings],
) -> AirtimeCaps:
    """Translate the settings models into the engine's caps.

    Routes through :func:`kairos.optimize.guardrails.airtime_caps_from_mapping`,
    the same function the plain-dict service path uses, so the API model and the
    service cannot drift into disagreeing about what a stored cap means.
    """
    return airtime_caps_from_mapping({
        "window_ad_cap": window.model_dump() if window is not None else None,
        "day_fraction_ad_cap": day_fraction.model_dump() if day_fraction is not None else None,
    })
