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
    from kairos.optimize.advertiser_rules import Baseline, Condition

ANY = "ANY"

# Premium mode constants (kept here so the pure helpers are self-contained;
# advertiser_rules imports them back).
MULTIPLIER = "multiplier"
PERCENT = "percent"
CPP_ABSOLUTE = "cpp_absolute"
CPP_ADD = "cpp_add"
CPP_DISCOUNT = "cpp_discount"
PREMIUM_MODES = (MULTIPLIER, PERCENT, CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)
CPP_MODES = (CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)


def scope_tokens(raw: object) -> frozenset[str]:
    """Split a comma-joined scope string into a token set.

    An empty value or the literal ``ANY`` (case-insensitive) becomes the empty
    set, which the engine reads as "matches everything in this dimension".
    """
    text = str(raw or "").strip()
    if not text or text.upper() == ANY:
        return frozenset()
    return frozenset(part.strip() for part in text.split(",") if part.strip())


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
    (the default) reads ``value`` as the multiplier itself.
    """
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


def load_baselines(path: Path) -> dict[str, "Baseline"]:
    """Read advertiser_rules.csv, returning Baseline objects keyed by id."""
    from kairos.optimize.advertiser_rules import Baseline

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
                allow_positions=scope_tokens(row.get("allow_positions")),
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
    """Parse one CSV row into a Condition, or None when the row is malformed."""
    from kairos.optimize.advertiser_rules import Condition, _EFFECTS  # type: ignore[attr-defined]

    advertiser_id = str(row.get("advertiser_id", "")).strip()
    rule_id = str(row.get("rule_id", "")).strip()
    effect = str(row.get("effect", "")).strip().lower()
    if not advertiser_id or not rule_id or effect not in _EFFECTS:
        return None
    return Condition(
        advertiser_id=advertiser_id,
        rule_id=rule_id,
        effect=effect,
        value=parse_float(row.get("value"), 1.0),
        mode=parse_mode(row.get("mode")),
        scope_positions=scope_tokens(row.get("scope_positions")),
        scope_genres=scope_tokens(row.get("scope_genres")),
        scope_dayparts=scope_tokens(row.get("scope_dayparts")),
        scope_programmes=scope_tokens(row.get("scope_programmes")),
        notes=str(row.get("notes", "")),
    )
