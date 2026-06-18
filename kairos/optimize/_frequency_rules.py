"""Loader, parser and scope-resolution for ad-frequency / separation rules.

This is the pure data layer for :mod:`kairos.optimize.frequency`. It defines the
rule vocabulary, parses ``data/frequency_rules.csv`` into typed
:class:`FrequencyRule` objects, and resolves the MOST-SPECIFIC effective rule for
a given (limit_type, advertiser, campaign, ad) target. Keeping this here keeps
the enforcement pass in ``frequency.py`` under the 450-line cap.

Where these rules live and bite. The WEEKLY break-count optimizer decides break
COUNTS per programme segment and has NO advertiser attribution, so it cannot
honor a per-advertiser frequency rule. Real attribution (advertiser, campaign,
position-in-break, break_start) exists only on the DAILY spot path
(:func:`kairos.data.loaders.load_daily_input`), so frequency and competitive
separation are enforced THERE, by :func:`kairos.optimize.frequency.enforce`, over
the ordered priced spots. This module never enforces; it only describes.

Scoping and composition. A limit can be authored at four scopes:

  * ``default``     -> applies to every advertiser unless a finer rule overrides,
  * ``advertiser``  -> applies to one ``advertiser_id``,
  * ``campaign``    -> applies to one ``advertiser_id`` + ``campaign``,
  * ``ad``          -> applies to one ``advertiser_id`` + ``campaign`` + ``ad``.

For one limit_type the EFFECTIVE rule is the most specific match
(ad > campaign > advertiser > default). :func:`resolve_effective` implements that
single, clean precedence. COMPETITIVE_SEPARATION is different: it is keyed by a
named ``competing_group`` (a set of advertisers that compete, e.g. two banks) and
is resolved per group, not per advertiser, by :func:`competitive_groups`.

Honesty rules: an empty file (header only) yields no rules, so with no authored
rule the spot log is unchanged (identity). A malformed row is skipped with a
recorded reason rather than silently bending the plan; the reasons are returned
so the caller can surface them. Nothing is invented to fill an empty file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_FREQUENCY_PATH = DATA_DIR / "frequency_rules.csv"

# Limit-type vocabulary. Each is a hard cap enforced deterministically.
MAX_PER_BREAK = "max_per_break"            # at most N spots of the target in one break
MAX_CONSECUTIVE = "max_consecutive"        # no same target in N adjacent break positions
MIN_SEPARATION = "min_separation"          # min gap (minutes OR positions) between two
MAX_PER_DAY = "max_per_day"                # at most N spots of the target across the day
COMPETITIVE_SEPARATION = "competitive_separation"  # keep two competitors apart
_LIMIT_TYPES = (
    MAX_PER_BREAK,
    MAX_CONSECUTIVE,
    MIN_SEPARATION,
    MAX_PER_DAY,
    COMPETITIVE_SEPARATION,
)

# Scope vocabulary, ordered least-to-most specific (index = specificity rank).
DEFAULT = "default"
ADVERTISER = "advertiser"
CAMPAIGN = "campaign"
AD = "ad"
_SCOPES = (DEFAULT, ADVERTISER, CAMPAIGN, AD)
_SPECIFICITY = {DEFAULT: 0, ADVERTISER: 1, CAMPAIGN: 2, AD: 3}

# Unit vocabulary for MIN_SEPARATION (and ignored for the others).
MINUTES = "minutes"
POSITIONS = "positions"
_UNITS = (MINUTES, POSITIONS)

COLUMNS = [
    "rule_id",
    "limit_type",
    "scope",
    "advertiser_id",
    "campaign",
    "ad",
    "competing_group",
    "members",
    "value",
    "unit",
    "enabled",
    "notes",
]


@dataclass(frozen=True)
class FrequencyRule:
    """One authored frequency / separation rule.

    ``value`` is the numeric cap (N spots, N positions or N minutes depending on
    ``limit_type`` and ``unit``). For COMPETITIVE_SEPARATION, ``members`` is the
    set of advertiser ids that compete and ``value``/``unit`` say how far apart to
    keep them (0 positions = not in the same break; N minutes = at least N apart).
    """

    rule_id: str
    limit_type: str
    scope: str
    advertiser_id: str = ""
    campaign: str = ""
    ad: str = ""
    competing_group: str = ""
    members: frozenset[str] = frozenset()
    value: float = 0.0
    unit: str = ""
    enabled: bool = True
    notes: str = ""

    @property
    def specificity(self) -> int:
        return _SPECIFICITY.get(self.scope, 0)

    def targets(self, advertiser: str, campaign: str, ad: str) -> bool:
        """True when this scoped rule applies to the given spot identity.

        ``default`` targets everything. ``advertiser`` matches the advertiser id;
        ``campaign`` also matches the campaign; ``ad`` also matches the ad. A blank
        authored dimension on a finer scope is treated as ``ANY`` for that
        dimension so a half-filled row still binds at its declared scope.
        """
        if self.scope == DEFAULT:
            return True
        if self.advertiser_id and self.advertiser_id != advertiser:
            return False
        if self.scope in (CAMPAIGN, AD) and self.campaign and self.campaign != campaign:
            return False
        if self.scope == AD and self.ad and self.ad != ad:
            return False
        return True


@dataclass
class FrequencyRuleSet:
    """All authored rules plus any parse-skip reasons (honest diagnostics)."""

    rules: list[FrequencyRule] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def by_limit(self, limit_type: str) -> list[FrequencyRule]:
        return [r for r in self.rules if r.enabled and r.limit_type == limit_type]


def _to_float(value: object, default: float = 0.0) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_bool(value: object, default: bool = True) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    return text in ("1", "true", "yes", "y", "on")


def _members(value: object) -> frozenset[str]:
    text = str(value if value is not None else "").strip()
    if not text:
        return frozenset()
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return frozenset(p for p in parts if p)


def rule_from_row(row: dict[str, str]) -> tuple[Optional[FrequencyRule], str]:
    """Parse one CSV row into a rule, or return (None, reason) when malformed."""
    rule_id = str(row.get("rule_id", "") or "").strip()
    limit_type = str(row.get("limit_type", "") or "").strip().lower()
    scope = str(row.get("scope", "") or "").strip().lower() or DEFAULT
    if limit_type not in _LIMIT_TYPES:
        return None, f"{rule_id or '<no id>'}: unknown limit_type '{limit_type}'"
    if scope not in _SCOPES:
        return None, f"{rule_id or '<no id>'}: unknown scope '{scope}'"
    value = _to_float(row.get("value"))
    if value is None:
        return None, f"{rule_id or '<no id>'}: non-numeric value '{row.get('value')}'"
    unit = str(row.get("unit", "") or "").strip().lower()
    if limit_type == MIN_SEPARATION and unit not in _UNITS:
        return None, f"{rule_id or '<no id>'}: min_separation needs unit minutes|positions"
    if limit_type == COMPETITIVE_SEPARATION:
        members = _members(row.get("members"))
        if len(members) < 2:
            return None, f"{rule_id or '<no id>'}: competitive_separation needs >=2 members"
        if unit not in _UNITS:
            unit = POSITIONS
    else:
        members = frozenset()
    rule = FrequencyRule(
        rule_id=rule_id or limit_type,
        limit_type=limit_type,
        scope=scope,
        advertiser_id=str(row.get("advertiser_id", "") or "").strip(),
        campaign=str(row.get("campaign", "") or "").strip(),
        ad=str(row.get("ad", "") or "").strip(),
        competing_group=str(row.get("competing_group", "") or "").strip(),
        members=members,
        value=value,
        unit=unit,
        enabled=_to_bool(row.get("enabled")),
        notes=str(row.get("notes", "") or "").strip(),
    )
    return rule, ""


def load_frequency_rules(path: str | Path | None = None) -> FrequencyRuleSet:
    """Load and parse the frequency-rules CSV; an absent/empty file yields none."""
    target = Path(path) if path else DEFAULT_FREQUENCY_PATH
    result = FrequencyRuleSet()
    if not target.exists():
        return result
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rule, reason = rule_from_row(row)
            if rule is None:
                result.skipped.append(reason)
            else:
                result.rules.append(rule)
    return result


def resolve_effective(
    rules: list[FrequencyRule], advertiser: str, campaign: str, ad: str
) -> Optional[FrequencyRule]:
    """The single most-specific rule that targets this spot identity, or None.

    Ad beats campaign beats advertiser beats default. Among rules of equal
    specificity the first authored wins (stable, deterministic). This is the clean
    composition rule: the operator never reasons about additive stacking, only
    about which scope is most specific.
    """
    best: Optional[FrequencyRule] = None
    for rule in rules:
        if not rule.targets(advertiser, campaign, ad):
            continue
        if best is None or rule.specificity > best.specificity:
            best = rule
    return best


def competitive_groups(rules: list[FrequencyRule]) -> list[FrequencyRule]:
    """The enabled competitive-separation rules (each carries its member set)."""
    return [r for r in rules if r.enabled and r.limit_type == COMPETITIVE_SEPARATION]
