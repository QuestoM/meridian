"""Position payloads for the pricing API: the vocabulary and the preferred set.

Split out of :mod:`kairos_api.pricing_api` so that module stays under the project
line limit. The engine vocabulary itself lives in
:mod:`kairos.optimize.positions`; nothing is defined twice here.
"""

from __future__ import annotations

from typing import Any

from kairos.optimize.positions import (
    COUNTING_METHODS,
    MIDDLE_TOKEN,
    POSITION_TOKENS,
    label as position_label,
    method_label,
)
from kairos.optimize.pricing import PricingModel


def position_vocabulary(values: dict[str, Any]) -> list[dict[str, Any]]:
    """The six trade positions plus the middle default, each with its set state.

    The shipped rate card prices 1, 2, 3, L and the middle default; ordinals 4
    and 5 ship UNSET, which is a no-op (a spot at 4 or 5 still prices at L when
    it is the tail of its break, otherwise at the middle default). Listing all of
    them lets an operator set 4 or 5 deliberately instead of the vocabulary
    pretending they do not exist, and ``configured`` keeps an unset ordinal from
    reading as a premium of 1.0 that somebody chose.
    """
    tokens = [*POSITION_TOKENS, MIDDLE_TOKEN]
    extra = [key for key in values if key not in tokens]
    return [
        {
            "key": key,
            "he": position_label(key, "he"),
            "en": position_label(key, "en"),
            "configured": key in values,
        }
        for key in (*tokens, *extra)
    ]


def preferred_payload(model: PricingModel) -> dict[str, Any]:
    """The preferred-position configuration and the counting methods, tri-state.

    ``positions`` is null when nobody has configured a preferred set, which is
    the shipped state: no preferred-position percentage may be computed anywhere
    until an operator sets one, per client or per agreement. Every method a
    percentage could be quoted under is named here, bilingually, because two
    parties audit each other with that number and an unlabelled percentage is
    worse than none (docs/media-domain-from-the-trade.md).
    """
    positions, scope = model.preferred_positions()
    return {
        "positions": None if positions is None else sorted(positions),
        "scope": scope,
        "basis": "unset" if positions is None else "configured",
        "per_advertiser": {
            key: sorted(value) for key, value in model.preferred_positions_by_advertiser.items()
        },
        "counting_methods": [
            {"key": key, "en": method_label(key, "en"), "he": method_label(key, "he")}
            for key in COUNTING_METHODS
        ],
    }
