"""How long a commercial pod actually is, measured instead of declared.

The optimizer has always planned in two-minute breaks. That number was never
measured; it is a round figure that entered as ``a two-minute break, a common
unit`` and then silently decided what every regulatory cap in the engine could
mean. Measured against the broadcaster's own numbering of its own log
(:mod:`kairos.data.as_aired`), one month of the operator channel:

    829 pods, mean 190.7 seconds, median 181, quartiles 109 and 249.
    Eighteen of 760 pods in the earlier survey held exactly 120 seconds.

The consequence is not a rounding error, it is a currency error. Four pods an
hour reads as 480 seconds to the engine and is 763 seconds on air, so the
twelve-minute cap that looks comfortably clear is breached, and the eight-minute
cap protecting news and children's programming is exceeded by half again. The
engine's plan is compliant in a unit that does not exist.

**Why the mean and not the median.** These pods are the planning unit for a
total: N pods times L seconds has to reproduce the airtime a schedule really
consumes, and only the mean satisfies that by construction, because the mean is
the total divided by the count. The median is the better description of a
typical pod and the worse basis for a cap.

**Why one number and not a daypart model.** Pod length looks strongly
daypart-shaped: by hour it ranges from 42 seconds overnight to 371 at ten in the
evening, and hour alone explains 29.8 percent of the in-sample variance, hour
crossed with weekday 41.0 percent. It does not survive contact with held-out
data. Trained on the first 21 days and tested on the last 9, an hour-conditioned
mean with the best shrinkage beats a single global constant by 4.1 percent of
mean absolute error -- 76.2 seconds against 79.5. That is thin, the cells behind
it are thin (sixteen pods in the one o'clock hour), and this codebase does not
ship a model that cannot beat a constant out of sample. One measured number is
what the evidence supports, and the seam below takes a richer estimate the day a
larger sample earns one.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from kairos.data.as_aired import identify_aired_pods

# The engine's declared default, kept here so a reader can see what the measured
# value replaces without opening the optimizer.
DECLARED_BREAK_LENGTH_SECONDS = 120.0

# Planning on the measured length moves real money and real ad load, so it is an
# owner decision rather than a consequence of the data arriving. Absent, this
# reads off and the engine keeps its declared default exactly.
ACTIVATION_SETTINGS_KEY = "measured_pod_length_activation"

# A measurement below this many pods is not a measurement. One thin week of a
# minor channel must not be allowed to restate a plan's airtime.
MIN_PODS_FOR_A_MEASUREMENT = 100

# Sanity bounds. A pod outside these is a parsing failure, not a broadcast.
_MIN_PLAUSIBLE_SECONDS = 30.0
_MAX_PLAUSIBLE_SECONDS = 600.0


def measure_pod_length(
    spots: pd.DataFrame,
    *,
    channel: Optional[str] = None,
) -> dict[str, Any]:
    """The measured pod length for one channel, with everything needed to judge it.

    Returns ``seconds`` (the mean, the planning value), alongside ``median``,
    ``pods``, the quartiles and a ``usable`` flag. ``seconds`` is ``None`` when
    the sample is too small or the result implausible, so a caller can never
    accidentally plan on a number this module would not stand behind.
    """
    pods = identify_aired_pods(spots, channel=channel)
    if pods.empty:
        return {"seconds": None, "pods": 0, "usable": False, "reason": "no pods in the source"}

    seconds = pd.to_numeric(pods["seconds"], errors="coerce").dropna()
    seconds = seconds[seconds > 0]
    count = int(len(seconds))
    if count < MIN_PODS_FOR_A_MEASUREMENT:
        return {
            "seconds": None, "pods": count, "usable": False,
            "reason": f"only {count} pods, fewer than the {MIN_PODS_FOR_A_MEASUREMENT} this module requires",
        }

    mean = float(seconds.mean())
    result = {
        "seconds": round(mean, 1),
        "median": round(float(seconds.median()), 1),
        "p25": round(float(seconds.quantile(0.25)), 1),
        "p75": round(float(seconds.quantile(0.75)), 1),
        "pods": count,
        "total_seconds": round(float(seconds.sum()), 1),
        "channel": channel,
        "usable": True,
        "reason": None,
    }
    if not (_MIN_PLAUSIBLE_SECONDS <= mean <= _MAX_PLAUSIBLE_SECONDS):
        result["seconds"] = None
        result["usable"] = False
        result["reason"] = (
            f"the measured mean of {mean:.0f}s falls outside the plausible "
            f"{_MIN_PLAUSIBLE_SECONDS:.0f}-{_MAX_PLAUSIBLE_SECONDS:.0f}s band"
        )
    return result


def pods_per_hour_under_cap(pod_seconds: float, cap_seconds: float) -> float:
    """How many pods of this length fit under a seconds-per-hour cap.

    The arithmetic every minutes cap in :mod:`kairos.optimize.guardrails` rests
    on, written down once. At the declared 120 seconds a twelve-minute cap admits
    six pods and the four-pod rule binds first, which is why the seconds cap has
    never bitten. At the measured length it admits under four, so the seconds cap
    becomes the binding rule and the pod count falls out of it rather than being
    set by a separate ceiling.
    """
    if pod_seconds <= 0:
        raise ValueError("pod_seconds must be positive")
    return cap_seconds / float(pod_seconds)


def cap_reading(pod_seconds: float, guardrails: Any) -> dict[str, Any]:
    """What the shipped caps mean at a given pod length, stated plainly.

    Written for the disclosure surface rather than for the solver: an operator
    reading a plan's revenue is entitled to know how many pods an hour the rules
    actually allow at the length the plan is priced on, and whether the pod-count
    rule or the seconds rule is the one doing the work.
    """
    general = float(getattr(guardrails, "max_ad_seconds_per_hour", 0.0) or 0.0)
    protected = float(getattr(guardrails, "protected_max_ad_seconds_per_hour", 0.0) or 0.0)
    per_hour = int(getattr(guardrails, "max_breaks_per_hour", 0) or 0)
    fits_general = pods_per_hour_under_cap(pod_seconds, general) if general > 0 else None
    fits_protected = pods_per_hour_under_cap(pod_seconds, protected) if protected > 0 else None
    binding = "pod count"
    if fits_general is not None and fits_general < per_hour:
        binding = "ad seconds"
    return {
        "pod_seconds": round(float(pod_seconds), 1),
        "max_breaks_per_hour": per_hour,
        "seconds_at_the_pod_ceiling": round(per_hour * float(pod_seconds), 1),
        "pods_under_the_general_cap": round(fits_general, 2) if fits_general is not None else None,
        "pods_under_the_protected_cap": round(fits_protected, 2) if fits_protected is not None else None,
        "binding_rule": binding,
        "general_cap_breached_at_the_pod_ceiling": (
            per_hour * float(pod_seconds) > general if general > 0 else None
        ),
        "protected_cap_breached_at_the_pod_ceiling": (
            per_hour * float(pod_seconds) > protected if protected > 0 else None
        ),
    }


def measured_length_from_settings(
    settings: Optional[Mapping[str, Any]],
    measurement: Mapping[str, Any],
) -> Optional[float]:
    """The pod length to plan on, or ``None`` to leave the declared default alone.

    Activation is a settings decision and never inferred: a missing key, a
    non-``True`` value, or an unusable measurement all read as off. Returning
    ``None`` is the off state, and callers must treat it as "change nothing"
    rather than as a zero.
    """
    if not settings:
        return None
    if settings.get(ACTIVATION_SETTINGS_KEY) is not True:
        return None
    if not measurement.get("usable"):
        return None
    seconds = measurement.get("seconds")
    return float(seconds) if seconds else None
