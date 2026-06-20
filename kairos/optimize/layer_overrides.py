"""Per-layer and per-final advertiser overrides on the rate-card breakdown.

The owner's pricing model: a slot's price is base CPP times a stack of named
premium layers (prime, show, position, ad-type), and a scoped advertiser or
campaign rule may REPLACE exactly one named layer (for example "advertiser X gets
a low position-2 price that replaces the general position-2 premium, while prime
and show still stack") or ADJUST the whole final price.

This module is the resolution + application step that makes that real, composing
:class:`~kairos.optimize.advertiser_rules.AdvertiserRuleEngine` conditions onto a
:class:`~kairos.optimize.pricing.PriceBreakdown`:

  * Per-layer REPLACE: a PREMIUM condition whose ``target_layer`` names a layer
    swaps that layer's multiplier for the matched scope. Other layers still stack.
    ``target_layer == "final"`` adjusts the whole composed price instead.
  * Campaign scope: conditions carry ``scope_campaigns``; a campaign always belongs
    to one advertiser, so a campaign-scoped rule narrows that advertiser's rule to
    specific campaigns.
  * Most-specific-wins: when several REPLACE rules target the same layer for one
    slot, the most specific (most matched scope dimensions) wins; ties break on an
    explicit ``priority`` then CSV row order. Shadowed rules are reported, not
    silently dropped, so the dashboard precedence preview can grey them out.

Honesty: a rule with an empty ``target_layer`` is a legacy whole-stack premium and
is NOT handled here (it stays on AdvertiserRuleEngine.effective_premium), so adding
this module changes no existing price until a rule is explicitly given a
target_layer. Every override line keeps ``source="override:<rule_id>"`` provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kairos.optimize._rule_helpers import CPP_MODES, compute_premium_factor
from kairos.optimize.advertiser_rules import PREMIUM, AdvertiserRuleEngine, Condition
from kairos.optimize.pricing import PriceBreakdown, PriceLayer

FINAL = "final"

# Canonical re-assembly order so a breakdown reads base -> prime -> ... every time.
_LAYER_ORDER = ("program", "prime", "day", "show", "position", "ad_type")


@dataclass(frozen=True)
class ResolvedOverride:
    """One advertiser/campaign override that won resolution for a slot.

    ``multiplier`` is the factor the named layer is replaced with (or, for a
    ``final`` override, multiplied into the whole price). ``rule_id`` is kept for
    provenance and for the precedence preview.
    """

    rule_id: str
    target_layer: str
    multiplier: float


@dataclass(frozen=True)
class ShadowedOverride:
    """An override that matched the slot but lost to a more specific rule."""

    rule_id: str
    target_layer: str
    winner_rule_id: str
    reason: str


@dataclass(frozen=True)
class OverrideResolution:
    """The resolved overrides for one slot: winners per layer plus shadowed rules."""

    layer_overrides: tuple[ResolvedOverride, ...] = ()
    final_overrides: tuple[ResolvedOverride, ...] = ()
    shadowed: tuple[ShadowedOverride, ...] = ()


def _override_multiplier(condition: Condition, base_cpp: Optional[float]) -> float:
    """The replacement factor a targeted PREMIUM rule contributes."""
    return compute_premium_factor(condition.value, condition.mode, base_cpp)


def resolve_layer_overrides(
    engine: AdvertiserRuleEngine,
    advertiser_id: str,
    *,
    position: Optional[int] = None,
    genre: Optional[str] = None,
    daypart: Optional[str] = None,
    programme: Optional[str] = None,
    campaign: Optional[str] = None,
    base_cpp: Optional[float] = None,
) -> OverrideResolution:
    """Resolve targeted overrides for one slot, most-specific-wins per layer.

    Only PREMIUM conditions with a non-empty ``target_layer`` that match the slot's
    scope take part; legacy whole-stack rules are ignored here. For each named
    layer the most specific matching rule wins (specificity, then priority, then
    later CSV row order); the losers become ``shadowed`` entries. ``final`` rules
    do not compete: they all apply, composed multiplicatively.
    """
    matched = [
        c for c in engine._conditions_for(advertiser_id)
        if c.target_layer and c.effect == PREMIUM and c.matches(
            position=position, genre=genre, daypart=daypart,
            programme=programme, campaign=campaign,
        )
    ]
    by_layer: dict[str, list[Condition]] = {}
    finals: list[Condition] = []
    for condition in matched:
        if condition.target_layer == FINAL:
            finals.append(condition)
        else:
            by_layer.setdefault(condition.target_layer, []).append(condition)

    winners: list[ResolvedOverride] = []
    shadowed: list[ShadowedOverride] = []
    for layer, group in by_layer.items():
        # Most specific first; ties to higher priority; final tie to later row order.
        ordered = sorted(
            enumerate(group),
            key=lambda pair: (pair[1].specificity(), pair[1].priority, pair[0]),
            reverse=True,
        )
        win_idx, win = ordered[0]
        winners.append(ResolvedOverride(win.rule_id, layer, _override_multiplier(win, base_cpp)))
        for _, lost in ordered[1:]:
            shadowed.append(ShadowedOverride(
                rule_id=lost.rule_id, target_layer=layer, winner_rule_id=win.rule_id,
                reason=(f"a more specific override ({win.rule_id}) wins the {layer} layer"
                        if win.specificity() > lost.specificity()
                        else f"override {win.rule_id} wins the {layer} layer on priority/order"),
            ))

    final_resolved = tuple(
        ResolvedOverride(c.rule_id, FINAL, _override_multiplier(c, base_cpp)) for c in finals
    )
    return OverrideResolution(
        layer_overrides=tuple(winners), final_overrides=final_resolved,
        shadowed=tuple(shadowed),
    )


def apply_overrides(breakdown: PriceBreakdown, resolution: OverrideResolution) -> PriceBreakdown:
    """Return a new breakdown with the resolved overrides applied.

    A per-layer override replaces that named layer's multiplier in place (or injects
    the layer in canonical order if it is not currently active, so an advertiser can
    be given a position price even when the general position layer is off). Final
    overrides are appended as ``final`` layers that multiply the whole price. Every
    overridden or injected layer is tagged ``source="override:<rule_id>"`` so the
    line traces to its rule (Law 9). With an empty resolution this is the identity.
    """
    by_name: dict[str, PriceLayer] = {layer.name: layer for layer in breakdown.layers}
    extra_names = [name for name in by_name if name not in _LAYER_ORDER]

    for override in resolution.layer_overrides:
        by_name[override.target_layer] = PriceLayer(
            name=override.target_layer, multiplier=override.multiplier,
            source=f"override:{override.rule_id}",
        )

    ordered_names = [n for n in _LAYER_ORDER if n in by_name] + [
        n for n in extra_names if n in by_name
    ]
    layers = [by_name[name] for name in ordered_names]
    for override in resolution.final_overrides:
        layers.append(PriceLayer(
            name=FINAL, multiplier=override.multiplier,
            source=f"override:{override.rule_id}",
        ))
    return PriceBreakdown(base_cpp=breakdown.base_cpp, layers=tuple(layers))


__all__ = [
    "FINAL",
    "ResolvedOverride",
    "ShadowedOverride",
    "OverrideResolution",
    "resolve_layer_overrides",
    "apply_overrides",
]
