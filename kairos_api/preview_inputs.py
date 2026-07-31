"""Segment construction for the operator previews, on the commit path's seams.

This is :func:`preview_inputs`, moved out of :mod:`kairos_api.overrides` where it
was a module-private helper that two other modules imported anyway. One function
imported across three modules is not private, and the ambiguity about who owned
it is what put it here: it is now one module, with one owner, that both effect
previews and any later placement surface read.

Every dependent imports it the same way::

    from kairos_api.preview_inputs import preview_inputs

The behaviour is the function's behaviour, unchanged. Every import stays inside
the function body exactly as before, for two reasons: the ``kairos_api.core``
import would otherwise close a cycle, and the impact model pulls in the whole
scientific stack, which measured at 3.4 s of one-time module import. Keeping it
lazy means importing this module stays cheap for a caller that never builds
segments.

The read cache, and why it ships switched off. Segment construction is a pure
function of files and settings and is independent of any placement, so the same
channel-day rebuilt twice in a process is the same list, and it is memoizable
through :mod:`kairos_api.read_cache`. The wiring is here and is tested in both
states. It defaults to off because of what the attribution measured, not because
of a doubt about the mechanism: on `רשת 13` / 2024-11-01 the whole of this
function is 14.3 ms of an 1,871 ms response, the cacheable part of it is 11.3 ms,
and the two optimizer legs are 1,807 ms. A cache here buys 0.6 percent of the
response and takes on a permanent obligation, which is that every future input to
segment construction must be reflected in the fingerprint or a money number goes
stale. That trade is not worth taking for 11 ms, so the honest default is off and
the measurement is written down rather than the hope. The full attribution is in
``docs/ux-gauntlet/contracts/W0-5.md``.

When it is on, the fingerprint carries every input the build reads: the saved
settings, the live pricing model, the folded assumptions (which is where the
measured first-break multiplier lands, so a change to it can never be served
from cache), today's date, the signatures of every config, model and reference
file, and the seam functions themselves by identity.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

from kairos_api import read_cache

ROOT = Path(__file__).resolve().parents[1]

# One namespace, keyed by scope. A handful of channel-days is all any surface
# holds at once, and the bound stops a per-day key space from growing forever.
CACHE_NAMESPACE = "preview_segments"

# Off by default, for the measured reason in the module docstring. True makes
# every call consult the cache; False rebuilds every time, which is exactly what
# this function did before the cache existed.
USE_READ_CACHE = False

read_cache.configure(CACHE_NAMESPACE, capacity=8)


def _stable(value: Any) -> Any:
    """A hashable, order-independent summary of a nested settings-like value.

    Total by construction: mappings become sorted key-value tuples with string
    keys (a settings map mixes int and str keys, which json's sort_keys cannot
    order), sequences become tuples, and anything else becomes its repr. It
    never raises, because a fingerprint that raises turns a cache into an
    outage.
    """
    if isinstance(value, dict):
        return tuple(sorted((str(key), _stable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _stable(asdict(value))
    return repr(value)


def _seams() -> dict[str, Callable[..., Any]]:
    """The engine functions this module calls, bound once per call.

    The same bound objects feed the fingerprint and the build, so a monkeypatch
    that lands between the two cannot produce a value stamped with the wrong
    fingerprint, and a patched seam always misses the cache.
    """
    from kairos.data.loaders import load_daily_input, load_programmes
    from kairos.data.transform import (
        build_segments_from_daily_input,
        build_segments_from_programmes,
    )
    from kairos.model.impact import load_impact_model
    from kairos.service import _apply_first_break_multiplier, _build_classifier

    return {
        "apply_first_break_multiplier": _apply_first_break_multiplier,
        "build_classifier": _build_classifier,
        "build_segments_from_daily_input": build_segments_from_daily_input,
        "build_segments_from_programmes": build_segments_from_programmes,
        "load_daily_input": load_daily_input,
        "load_impact_model": load_impact_model,
        "load_programmes": load_programmes,
    }


def _folded_assumptions(seams: dict[str, Callable[..., Any]]) -> Any:
    """The optimizer assumptions with the measured first-break multiplier folded in.

    Computed before the cache is consulted, and it is cheap (it reads one small
    JSON), because this is where the measured multiplier lands. Fingerprinting
    the derived value rather than the function that derives it means a test or a
    rebuild that changes the multiplier misses the cache even though no file
    signature and no seam identity moved.
    """
    from kairos.optimize.pricing import OptimizerAssumptions

    return seams["apply_first_break_multiplier"](OptimizerAssumptions())


def _fingerprint(
    settings_map: dict[str, Any],
    pricing: Any,
    assumptions: Any,
    daily_input: Optional[str],
    seams: dict[str, Callable[..., Any]],
) -> tuple:
    """Everything that can change the built segments, in one hashable value.

    Conservative on purpose. Today's date is included even though the pricing
    model already carries the event map it dates, because an input that is
    silently time-dependent is the classic stale-cache defect and one rebuild a
    day costs 12 ms. The seam functions are included by identity, which is also
    what keeps them alive, so an id can never be recycled under the cache.
    """
    paths = [
        ROOT / "data" / "kairos_settings.json",
        ROOT / "data" / "reference" / "Programmes.xlsx",
        ROOT / "data" / "Programmes.csv",
        ROOT / "data" / "calendar_events.csv",
    ]
    if daily_input:
        paths.append(Path(daily_input))
    return (
        _stable(settings_map),
        _stable(pricing),
        _stable(assumptions),
        date.today().isoformat(),
        read_cache.directory_signatures(ROOT / "config", "*.yaml"),
        read_cache.directory_signatures(ROOT / "models", "*"),
        read_cache.file_signatures(paths),
        tuple(seams[name] for name in sorted(seams)),
    )


def _build_segments(
    channel: Optional[str],
    day: Optional[str],
    daily_input: Optional[str],
    pricing: Any,
    assumptions: Any,
    seams: dict[str, Callable[..., Any]],
) -> list:
    """Build the real ProgramSegments, in the export path's own order.

    The impact model is loaded on the RAW assumptions and the measured
    first-break multiplier is folded in afterwards, which is the ordering
    :func:`kairos.export.schedule.build_weekly_schedule` uses. Reversing it would
    change every retention number, so the order is load-bearing: ``assumptions``
    arrives already folded and is used for the segments, while the impact model
    is loaded on a raw set, exactly as before.
    """
    from kairos.optimize.pricing import OptimizerAssumptions

    impact = seams["load_impact_model"](
        ROOT / "models" / "tv_break_posterior.pkl", assumptions=OptimizerAssumptions(),
    )
    classifier = seams["build_classifier"]()
    if daily_input:
        daily = seams["load_daily_input"](daily_input)
        return seams["build_segments_from_daily_input"](
            daily, classifier, pricing, assumptions=assumptions, impact_model=impact,
        )
    programmes = seams["load_programmes"]()
    return seams["build_segments_from_programmes"](
        programmes, classifier, pricing,
        assumptions=assumptions, impact_model=impact, channel=channel, day=day,
    )


def preview_inputs(
    channel: Optional[str], day: Optional[str], daily_input: Optional[str],
) -> tuple[list, dict[str, Any]]:
    """Build real ProgramSegments plus the exact engine kwargs the commit path uses.

    Mirrors :func:`kairos.export.schedule.build_weekly_schedule` seam for seam:
    pricing from the SAVED settings, the impact model loaded on the raw assumptions
    and THEN the measured first-break multiplier folded in (the export's ordering),
    and the service's wrapped classifier. The returned kwargs feed
    :func:`kairos.optimize.day_core._optimize_one_day` with the saved guardrails,
    weights, objective mode, pacing reference and operator channel, so a preview
    optimized with these inputs is the plan the weekly recompute would write, not a
    parallel engine. Raises when the data to build real segments is absent.

    The engine kwargs are rebuilt on every call, never cached: the pacing
    reference date rolls over at midnight and the guardrails follow the saved
    settings, so they are cheap and must stay live. Only the segments, which are
    placement-independent, are cacheable, and only when ``USE_READ_CACHE`` is on.
    """
    from kairos.service import (
        _pacing_knobs_from_settings,
        guardrails_from_settings,
        pricing_from_settings,
    )
    from kairos_api.core import _load_settings, _model_dump, _reference_today

    saved = _load_settings()
    settings_map = _model_dump(saved)
    pricing = pricing_from_settings(settings_map)
    seams = _seams()
    assumptions = _folded_assumptions(seams)

    def build() -> list:
        return _build_segments(channel, day, daily_input, pricing, assumptions, seams)

    if USE_READ_CACHE:
        segments = read_cache.cached(
            CACHE_NAMESPACE,
            key=(str(channel or ""), str(day or ""), str(daily_input or "")),
            fingerprint=_fingerprint(settings_map, pricing, assumptions, daily_input, seams),
            build=build,
        )
    else:
        segments = build()
    engine_kwargs: dict[str, Any] = {
        "guardrails": guardrails_from_settings(settings_map),
        "revenue_weight": saved.revenue_weight / 100.0,
        "risk_lambda": saved.risk_lambda,
        "objective_mode": getattr(saved, "objective_mode", "blend"),
        "pacing_today": _reference_today(saved),
        "pacing_knobs": _pacing_knobs_from_settings(settings_map),
        "operator_channel": str(saved.operator_channel or ""),
        "pricing": pricing,
    }
    # A fresh list every call, so a caller that sorts or trims its own copy
    # cannot reach into the cached one. ProgramSegment is a frozen dataclass, so
    # the segments themselves are safe to share.
    return list(segments), engine_kwargs
