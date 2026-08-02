"""Pure scope-matching helpers and CSV loaders for advertiser_rules.py.

Split out so advertiser_rules.py stays under the project 450-line limit. The
public names (without leading underscore) in this module are re-imported by
advertiser_rules; nothing else in the codebase should import from here directly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kairos.optimize._rule_models import Baseline, Condition

ANY = "ANY"

# Premium mode constants (kept here so the pure helpers are self-contained;
# advertiser_rules imports them back).
MULTIPLIER = "multiplier"
PERCENT = "percent"
CPP_ABSOLUTE = "cpp_absolute"
CPP_ADD = "cpp_add"
CPP_DISCOUNT = "cpp_discount"
# PREMIUM_DISCOUNT takes a percent 0..100 off the premium SURCHARGE only (the
# part of the composed premium above 1.0). It cannot be resolved standalone, so
# compute_premium_factor treats it as 1.0; the real math is
# apply_surcharge_discount, applied by the engine AFTER every other premium mode.
PREMIUM_DISCOUNT = "premium_discount"
PREMIUM_MODES = (MULTIPLIER, PERCENT, CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT, PREMIUM_DISCOUNT)
CPP_MODES = (CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)

# The only legal weekday scope tokens: ISO weekday numbers, Monday=1 .. Sunday=7
# (Saturday, Hebrew שבת, is 6).
WEEKDAY_TOKENS = frozenset(str(day) for day in range(1, 8))


def scope_tokens(raw: object) -> frozenset[str]:
    """Split a comma-joined scope string into a token set.

    An empty value or the literal ``ANY`` (case-insensitive) becomes the empty
    set, which the engine reads as "matches everything in this dimension".
    """
    text = str(raw or "").strip()
    if not text or text.upper() == ANY:
        return frozenset()
    return frozenset(part.strip() for part in text.split(",") if part.strip())


def position_scope_tokens(raw: object) -> frozenset[str]:
    """Split a position scope, then canonicalise every token onto 1..5 / L / gold.

    The stores grew three parallel position vocabularies (``1,2,3`` in the rate
    card, ``first,last`` in advertiser_rules.csv, ``first,middle,last`` in the
    conditions dropdown), and a word-form token could never match an observed
    ``position_in_break`` integer, so a rule written in words silently matched
    nothing. Canonicalising here gives one vocabulary at the door, so scope
    matching, overlap detection and specificity all read the same tokens.
    """
    from kairos.optimize.positions import canonical_token

    tokens = {canonical_token(token) for token in scope_tokens(raw)}
    return frozenset(token for token in tokens if token)


def dimension_matches(scope: frozenset[str], value: Optional[str]) -> bool:
    """True when a single observed ``value`` falls inside a scope token set."""
    if not scope:
        return True
    if value is None:
        return False
    return str(value) in scope


def scopes_intersect(a: frozenset[str], b: frozenset[str]) -> bool:
    """True when two scope token sets can describe the same value."""
    if not a or not b:
        return True
    return bool(a & b)


def parse_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_opt_float(raw: object):
    """Parse an optional non-negative float; blank or invalid yields ``None``.

    Used for the advertiser baseline's optional pacing-strength columns, where a
    blank cell must mean "use the global default", not zero. A negative value is
    rejected (returned as ``None``) so a typo never inverts the pacing steer.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


def parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


def parse_mode(raw: object) -> str:
    """Read a premium mode cell, falling back to MULTIPLIER for unknown values."""
    text = str(raw or "").strip().lower()
    return text if text in PREMIUM_MODES else MULTIPLIER


def compute_premium_factor(value: float, mode: str, base_cpp: Optional[float]) -> float:
    """Convert (value, mode) into a revenue multiplier.

    CPP modes return 1.0 when ``base_cpp`` is not a positive number, rather than
    guess a conversion. PERCENT reads ``value`` as a signed percent. MULTIPLIER
    (the default) reads ``value`` as the multiplier itself. PREMIUM_DISCOUNT has
    no standalone factor (its math needs the running premium, see
    :func:`apply_surcharge_discount`), so it resolves to 1.0 here; that keeps
    every consumer that converts rules one at a time (segment demand, layer
    overrides) an honest no-op instead of misreading a percent as a multiplier.
    """
    if mode == PREMIUM_DISCOUNT:
        return 1.0
    if mode == PERCENT:
        return 1.0 + value / 100.0
    if mode in CPP_MODES:
        if base_cpp is None or base_cpp <= 0:
            return 1.0
        if mode == CPP_ABSOLUTE:
            effective_cpp = value
        elif mode == CPP_ADD:
            effective_cpp = base_cpp + value
        else:  # CPP_DISCOUNT
            effective_cpp = base_cpp - value
        return max(0.0, effective_cpp) / base_cpp
    return value  # MULTIPLIER


def apply_surcharge_discount(premium: float, percent: float) -> float:
    """Discount the premium SURCHARGE (the part above 1.0) by ``percent``.

    The frozen semantics of the premium_discount mode:
    ``1 + (premium - 1) * (1 - percent/100)``. It composes AFTER the other
    premium modes, sequential discounts stack multiplicatively on the surcharge,
    and the result can never fall below 1.0 or rise above the pre-discount
    premium. A premium at or below 1.0 carries no surcharge to discount, so the
    rule is then a no-op rather than a guess. ``percent`` is clamped to 0..100
    so a hand-edited out-of-range cell can never invert into an uplift.
    """
    if premium <= 1.0:
        return premium
    kept = 1.0 - min(100.0, max(0.0, percent)) / 100.0
    return 1.0 + (premium - 1.0) * kept


def invalid_weekday_tokens(raw: object) -> list[str]:
    """The tokens in a weekday scope string that are not ISO weekdays 1..7."""
    return sorted(token for token in scope_tokens(raw) if token not in WEEKDAY_TOKENS)


def load_baselines(path: Path) -> dict[str, "Baseline"]:
    """Read advertiser_rules.csv, returning Baseline objects keyed by id."""
    from kairos.optimize._rule_models import Baseline

    if not path.exists():
        return {}
    out: dict[str, Baseline] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            advertiser_id = str(row.get("advertiser_id", "")).strip()
            if not advertiser_id:
                continue
            out[advertiser_id] = Baseline(
                advertiser_id=advertiser_id,
                default_premium=parse_float(row.get("default_premium"), 1.0),
                allow_positions=position_scope_tokens(row.get("allow_positions")),
                allow_genres=scope_tokens(row.get("allow_genres")),
                prime_time_only=parse_bool(row.get("prime_time_only")),
                urgency_k=parse_opt_float(row.get("urgency_k")),
                ahead_k=parse_opt_float(row.get("ahead_k")),
            )
    return out


def load_conditions(path: Path) -> dict[str, list["Condition"]]:
    """Read advertiser_conditions.csv, returning Condition lists keyed by advertiser.

    Rows whose effect is not in the known effect set are skipped, so a malformed
    line never silently changes pricing.
    """
    if not path.exists():
        return {}
    out: dict[str, list] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            condition = condition_from_row(row)
            if condition is not None:
                out.setdefault(condition.advertiser_id, []).append(condition)
    return out


def condition_from_row(row: dict[str, str]) -> "Optional[Condition]":
    """Parse one CSV row into a Condition, or None when the row is malformed.

    A missing ``scope_weekdays`` column reads as ANY, so every legacy store
    (and every legacy row in an upgraded store) keeps its exact meaning. A
    premium_discount row never carries a ``target_layer``: the discount is
    defined only on the composed whole-stack premium, so the layer-override
    resolver must not see it (a targeted discount would otherwise silently
    REPLACE a rate-card layer with the 1.0 standalone factor).
    """
    from kairos.optimize._rule_models import Condition, _EFFECTS

    advertiser_id = str(row.get("advertiser_id", "")).strip()
    rule_id = str(row.get("rule_id", "")).strip()
    effect = str(row.get("effect", "")).strip().lower()
    if not advertiser_id or not rule_id or effect not in _EFFECTS:
        return None
    mode = parse_mode(row.get("mode"))
    target_layer = str(row.get("target_layer", "")).strip().lower()
    if mode == PREMIUM_DISCOUNT:
        target_layer = ""
    return Condition(
        advertiser_id=advertiser_id,
        rule_id=rule_id,
        effect=effect,
        value=parse_float(row.get("value"), 1.0),
        mode=mode,
        scope_positions=position_scope_tokens(row.get("scope_positions")),
        scope_genres=scope_tokens(row.get("scope_genres")),
        scope_dayparts=scope_tokens(row.get("scope_dayparts")),
        scope_programmes=scope_tokens(row.get("scope_programmes")),
        scope_campaigns=scope_tokens(row.get("scope_campaigns")),
        scope_weekdays=scope_tokens(row.get("scope_weekdays")),
        target_layer=target_layer,
        priority=int(parse_float(row.get("priority"), 0.0)),
        notes=str(row.get("notes", "")),
    )
