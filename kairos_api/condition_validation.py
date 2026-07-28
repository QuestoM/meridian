"""Shared payload validation for the advertiser and agency condition CRUDs.

Both condition stores carry the same scope and mode vocabulary, so the checks
that guard the new custom-pricing fields live once here: the weekday scope
(ISO tokens 1..7, Monday=1, Saturday=6, Sunday=7) and the premium_discount
mode (a percent 0..100 off the premium surcharge). The weekday option list the
dashboard renders also lives here so both condition builders offer the same
vocabulary. Everything raises plain HTTP 400s; nothing is coerced silently.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException

from kairos.optimize._rule_helpers import PREMIUM_DISCOUNT, invalid_weekday_tokens, parse_float
from kairos.optimize.advertiser_rules import normalize_scope


def validate_weekday_scope(raw: object) -> str:
    """Normalize a scope_weekdays payload string, rejecting non-ISO tokens."""
    bad = invalid_weekday_tokens(raw)
    if bad:
        raise HTTPException(
            status_code=400,
            detail=(
                "scope_weekdays accepts ANY or a comma-joined list of ISO weekday "
                f"numbers 1..7 (Monday=1, Saturday=6, Sunday=7); invalid: {', '.join(bad)}"
            ),
        )
    return normalize_scope(raw)


def validate_mode_value(mode: str, value: float) -> None:
    """Reject a premium_discount whose percent is outside 0..100."""
    if mode == PREMIUM_DISCOUNT and not 0.0 <= float(value) <= 100.0:
        raise HTTPException(
            status_code=400,
            detail="premium_discount value is the percent taken off the premium surcharge and must be between 0 and 100",
        )


def validate_effective_mode_value(
    normalized_mode: str,
    payload_value: Optional[float],
    stored_value: Any,
) -> None:
    """Validate the (mode, value) pair an update would leave in the store.

    An update may change the mode, the value, or both, so the pair that must
    hold together is the payload field when given, else the stored cell.
    """
    effective = float(payload_value) if payload_value is not None else parse_float(stored_value, 1.0)
    validate_mode_value(normalized_mode, effective)


def weekday_options() -> list[dict[str, str]]:
    """The weekday vocabulary for the condition builders, ISO-keyed, bilingual.

    Keys stay ISO (Monday=1 .. Sunday=7); only the presentation ORDER follows
    the Israeli week convention, Sunday first through Saturday last, so the
    dashboard renders the chips as the operator reads a week.
    """
    labels = [
        ("7", "יום ראשון", "Sunday"),
        ("1", "יום שני", "Monday"),
        ("2", "יום שלישי", "Tuesday"),
        ("3", "יום רביעי", "Wednesday"),
        ("4", "יום חמישי", "Thursday"),
        ("5", "יום שישי", "Friday"),
        ("6", "שבת", "Saturday"),
    ]
    return [{"key": key, "he": he, "en": en} for key, he, en in labels]
