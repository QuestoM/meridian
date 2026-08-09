"""A reusable probe for THE INERT LEVER, the defect class named 2026-08-09.

The class
---------
A value the operator sets or supplies is computed, carried, and then
structurally cannot reach the decision. The signature is the same every time:
no error, no warning, a surface that reads as live, and an output that provably
cannot change. Three instances found so far:

  1. Demand weights (advertiser demand, inventory awareness, delivery pacing).
     Read once in the greedy ranking step, then optimised back out by the F1
     refiner and the exact DP tier, neither of which reads them. Measured over
     30 real operator channel-days: refine=False moved 754 segment break-counts
     on 30 of 30 days; refine=True moved 2, on 1 of 30.
  2. The hourly ad-minutes cap. Break length is a hardcoded 120 seconds and at
     most 4 breaks may fall in an hour, so the real ceiling is 8 minutes under a
     configured 12-minute cap. The configured value can never bind.
  3. Inventory awareness, a second and independent time. 994 rows are read and
     100% of them are discarded on an unparseable hour, silently.

The guard
---------
Vary a lever across its full configured range and assert the output changes. If
it cannot, the lever is inert.

THE PART THAT IS EASY TO GET WRONG, and the reason this helper exists rather
than an ad-hoc assert per site: a fixture that is not at a BINDING constraint
passes vacuously, in BOTH directions. Where nothing is scarce, the engine simply
gives every candidate its maximum and no ranking preference can change anything
-- so "the lever does nothing" is true of the fixture rather than of the lever.
The first draft of the demand-weight test was written that way and passed while
proving nothing; it was only sized to bind after the vacuity was caught by hand.

So ``binds`` is a REQUIRED argument here, never defaulted. It is the difference
between a measurement and a comfortable silence.

Adding instance four
--------------------
Write a ``run`` that maps one lever setting to a comparable output, a ``binds``
that is true only when the output sits against a real constraint, and a
``settings`` list spanning the lever's configured range. Then call
:func:`assert_lever_bites` for a lever that is supposed to work, or
:func:`assert_lever_is_inert` to PIN one that is known not to, with a reason.
Pinning is not blessing: an inert pin fails the moment the lever starts working,
which is the alarm you want, because starting to work moves real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class LeverProbe:
    """One lever, run across a range of settings, with the outputs it produced."""

    name: str
    settings: tuple[Any, ...]
    outputs: tuple[Any, ...]
    unbound: tuple[Any, ...]

    @property
    def bound(self) -> bool:
        """Did every probed run sit against a real constraint?"""
        return not self.unbound

    @property
    def distinct_outputs(self) -> int:
        """How many genuinely different outputs the settings produced."""
        seen: list[Any] = []
        for output in self.outputs:
            if output not in seen:
                seen.append(output)
        return len(seen)

    @property
    def moved(self) -> bool:
        return self.distinct_outputs > 1


def probe_lever(
    *,
    name: str,
    run: Callable[[Any], Any],
    settings: Sequence[Any],
    binds: Callable[[Any], bool],
) -> LeverProbe:
    """Run ``name`` across ``settings`` and record what each setting produced.

    ``run`` maps one setting to a comparable output (break counts, a plan, a
    total -- anything ``==`` compares meaningfully). ``binds`` answers, for one
    output, whether it sat against a real constraint; see the module docstring
    for why it has no default. ``settings`` must span the lever's configured
    range and hold at least two distinct values, or there is nothing to compare.
    """
    values = tuple(settings)
    if len(values) < 2:
        raise ValueError(
            f"lever {name!r} was probed with {len(values)} setting(s); "
            "a range needs at least two to compare"
        )
    outputs = tuple(run(setting) for setting in values)
    unbound = tuple(
        setting for setting, output in zip(values, outputs) if not binds(output)
    )
    return LeverProbe(name=name, settings=values, outputs=outputs, unbound=unbound)


def _require_binding(probe: LeverProbe) -> None:
    """Refuse to render any verdict on a fixture that was never constrained."""
    if not probe.bound:
        raise AssertionError(
            f"lever {probe.name!r} was probed on an UNCONSTRAINED fixture "
            f"(settings {list(probe.unbound)} produced an output with nothing "
            "scarce forcing a choice). Any verdict here would be vacuous: size "
            "the fixture so a cap or guardrail actually binds, then re-probe."
        )


def assert_lever_bites(probe: LeverProbe) -> None:
    """Assert the lever reaches the decision: its range changes the output."""
    _require_binding(probe)
    if not probe.moved:
        raise AssertionError(
            f"INERT LEVER: {probe.name!r} produced one identical output across "
            f"{len(probe.settings)} settings spanning its configured range, on a "
            "fixture that DOES bind. The value is computed and carried but cannot "
            "reach the decision."
        )


def assert_lever_is_inert(probe: LeverProbe, *, because: str) -> None:
    """Pin a lever KNOWN not to reach the decision, with the reason it does not.

    ``because`` is required and is the measured reason, not a guess. This fails
    if the lever starts working, which is the intended alarm: a lever that begins
    to bite moves real money and the change must be measured, not assumed.
    """
    if not because.strip():
        raise ValueError("assert_lever_is_inert requires a measured reason")
    _require_binding(probe)
    if probe.moved:
        raise AssertionError(
            f"{probe.name!r} is pinned inert ({because}) but it MOVED the output: "
            f"{probe.distinct_outputs} distinct results across {len(probe.settings)} "
            "settings. If this is deliberate it is a money change; measure the "
            "revenue effect before adopting it, then update this pin."
        )
