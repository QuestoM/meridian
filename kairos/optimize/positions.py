"""The spot-position vocabulary: ordinals 1 to 5 and L, where L is LAST.

Source of truth: docs/media-domain-from-the-trade.md, section "Positions: the
product is wrong today". Preferred positions in the Israeli trade are first,
second, third, fourth, fifth and Last. Last is written ``L`` and is a distinct
position, not the fifth ordinal: a break with three spots has a first and a last
and they can be the same campaign. Which of the six count as preferred is per
client and per agreement, so it is configuration and never a constant.

Two axes are easy to confuse and this module owns exactly one of them:

  * the position of a SPOT inside a BREAK (``position_in_break``) is what the
    trade calls a position, and is what this module models;
  * the position of a BREAK inside a PROGRAMME (first / middle / last break) is
    the retention model's channel grid and lives in :mod:`kairos.model.spec`.
    Nothing here touches it.

Occupancy versus pricing. A spot can occupy TWO positions at once: the only spot
in a one-spot break is both position 1 and L, and the third spot of a three-spot
break is both position 3 and L. :func:`occupied_tokens` returns every position a
spot holds, which is what the counting methods below need. Pricing has to choose
a single multiplier, so :func:`premium_token` picks one key: the ordinal when the
rate card prices that ordinal explicitly, otherwise L when the spot is the tail
of its break, otherwise the middle default. On the shipped rate card (ordinals 1,
2 and 3 priced, 4 and 5 unset) that reproduces the previous derived rule exactly
for every position and break size; ``tests/test_positions.py`` proves the grid.

Counting a preferred-position percentage. The trade runs two live methods and
they are used by two parties to audit each other, so an unlabelled percentage is
worse than no percentage. :func:`preferred_position_rate` therefore takes the
method as a required argument and returns it, bilingually labelled, welded to the
number.

  * :data:`AGENCY_METHOD`, preferred by the trade: the numerator is the number of
    preferred positions obtained and the denominator is the number of breaks the
    campaign appeared in, counting a break twice if it appeared twice.
  * :data:`CHANNEL_METHOD`: measured out of total broadcasts, so one broadcast
    counts once whether or not it holds two preferred positions at the same time.

Honest limit of the extraction. On the transcript's own wording the two
denominators coincide (a campaign's appearances in breaks are its broadcasts),
so as implemented here the methods diverge only in the numerator, when a single
broadcast occupies two preferred positions at once. The transcript asserts the
two methods give different percentages in practice; the rest of that difference
is not derivable from the extraction and is not invented here. What the product
guarantees is the part that matters for an audit: every percentage names its
method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

# The six positions of the trade. Ordinals are stored as their digit strings so a
# scope token, a rate-card key and a counted position all speak one language.
ORDINAL_POSITIONS: tuple[int, ...] = (1, 2, 3, 4, 5)
LAST_TOKEN = "L"
POSITION_TOKENS: tuple[str, ...] = tuple(str(n) for n in ORDINAL_POSITIONS) + (LAST_TOKEN,)

# The rate card's catch-all key for a position that is neither an explicitly
# priced ordinal nor the tail of its break.
MIDDLE_TOKEN = "default_middle"

# The premium gold break is a property of the BREAK, not a position inside it,
# but the rule engine has always let an operator scope a rule to it through the
# position dimension, so it stays a legal token here.
GOLD_POSITION = "gold"

# Word forms seen in the stores and in operator input, mapped onto the canonical
# token. "last" is the rate card's own legacy key for L and stays readable for
# ever, so an override saved before this vocabulary landed still applies.
_WORD_TO_TOKEN: dict[str, str] = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "last": LAST_TOKEN,
    "l": LAST_TOKEN,
    "ראשון": "1",
    "שני": "2",
    "שלישי": "3",
    "רביעי": "4",
    "חמישי": "5",
    "אחרון": LAST_TOKEN,
}

# Bilingual labels for the six positions plus the two non-ordinal keys an
# operator meets on the rate card. Hebrew is the default locale.
_LABELS: dict[str, tuple[str, str]] = {
    "1": ("First", "ראשון"),
    "2": ("Second", "שני"),
    "3": ("Third", "שלישי"),
    "4": ("Fourth", "רביעי"),
    "5": ("Fifth", "חמישי"),
    LAST_TOKEN: ("Last (L)", "אחרון (L)"),
    MIDDLE_TOKEN: ("Middle default", "אמצע, ברירת מחדל"),
    GOLD_POSITION: ("Gold break", "ברייק זהב"),
}

# The two live counting methods. A percentage is never returned without one.
AGENCY_METHOD = "agency"
CHANNEL_METHOD = "channel"
COUNTING_METHODS: tuple[str, ...] = (AGENCY_METHOD, CHANNEL_METHOD)

_METHOD_LABELS: dict[str, tuple[str, str]] = {
    AGENCY_METHOD: (
        "agency method, out of breaks appeared in",
        "שיטת המשרד, מתוך הפסקות שבהן הופיע",
    ),
    CHANNEL_METHOD: (
        "channel method, out of total broadcasts",
        "שיטת הערוץ, מתוך סך השידורים",
    ),
}


def canonical_token(value: Any) -> Optional[str]:
    """Normalise any spelling of a position onto its canonical token.

    Accepts a digit, a digit string, the English or Hebrew word form, ``L`` in
    either case, and the rate card's legacy ``last``. Returns ``None`` for an
    empty value so a caller can tell "not given" from "given and unrecognised";
    an unrecognised token is passed through unchanged rather than dropped, so a
    stored scope this vocabulary does not know is never silently widened.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return str(int(text))
    lowered = text.lower()
    return _WORD_TO_TOKEN.get(lowered, _WORD_TO_TOKEN.get(text, text))


def label(token: Any, locale: str = "he") -> str:
    """The operator-facing name of a position token, Hebrew by default."""
    key = canonical_token(token)
    pair = _LABELS.get(key or "")
    if pair is None:
        return str(token)
    return pair[1] if str(locale).lower().startswith("he") else pair[0]


def position_options() -> list[dict[str, str]]:
    """The position vocabulary as bilingual options for a scope or rate-card list.

    The six trade positions in order, then the gold break, which is a break-level
    token the rule engine has always accepted in the position dimension.
    """
    options = [
        {"key": token, "he": _LABELS[token][1], "en": _LABELS[token][0]}
        for token in POSITION_TOKENS
    ]
    options.append({
        "key": GOLD_POSITION,
        "he": _LABELS[GOLD_POSITION][1],
        "en": _LABELS[GOLD_POSITION][0],
    })
    return options


def normalize_position_scope(raw: Any) -> str:
    """Serialize a position scope back to a canonical comma string, or ``ANY``.

    The store-facing twin of :func:`canonical_token`, so an operator who writes
    ``first,last`` gets ``1,L`` back and the store carries one vocabulary. Empty
    or the literal ``ANY`` stays ``ANY``, the engine's "matches everything".
    """
    text = str(raw or "").strip()
    if not text or text.upper() == "ANY":
        return "ANY"
    tokens = {canonical_token(part) for part in text.split(",")}
    cleaned = sorted(token for token in tokens if token)
    return ",".join(cleaned) if cleaned else "ANY"


def occupied_tokens(position: Any, break_size: Any = None) -> tuple[str, ...]:
    """Every position a single spot holds: its ordinal, and L when it is the tail.

    The trade's point exactly: a break with three spots has a first and a last,
    and a one-spot break's only spot is both. Returns the ordinal alone when the
    break size is unknown, because L cannot be asserted without knowing which
    spot is last, and guessing would be a fabricated position.
    """
    ordinal = canonical_token(position)
    if ordinal is None or not ordinal.lstrip("-").isdigit():
        return () if ordinal is None else (ordinal,)
    tokens = [ordinal]
    size = canonical_token(break_size)
    if size is not None and size.lstrip("-").isdigit() and int(ordinal) == int(size):
        tokens.append(LAST_TOKEN)
    return tuple(tokens)


def premium_token(position: int, break_size: Optional[int], configured: Iterable[Any]) -> str:
    """The single rate-card key that prices one spot.

    An ordinal the rate card prices explicitly wins, because that is the operator
    naming a price for that exact slot. Otherwise the tail of the break is L.
    Otherwise the middle default. ``configured`` is the set of keys the rate card
    actually carries, so an ordinal the operator has not priced stays a no-op and
    the spot prices exactly as it did before that ordinal was addressable.
    """
    if int(position) < 1:
        raise ValueError("position must be >= 1")
    keys = {canonical_token(key) for key in configured}
    ordinal = str(int(position))
    if ordinal in keys:
        return ordinal
    if break_size is not None and int(position) == int(break_size):
        return LAST_TOKEN
    return MIDDLE_TOKEN


def parse_preferred(raw: Any) -> Optional[frozenset[str]]:
    """Read a configured preferred-position set, tri-state.

    ``None`` (or a missing key) means the set is UNSET: nobody has said which
    positions this client or agreement treats as preferred, so no percentage may
    be computed. An empty list means the set is set and empty. A string is
    comma-joined tokens; a list is the tokens themselves. Every token is
    canonicalised, so ``first,last``, ``1,L`` and ``1, אחרון`` are one set.
    """
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            return None
        parts: Sequence[Any] = [p for p in text.split(",")]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = list(raw)
    else:
        parts = [raw]
    tokens = {canonical_token(part) for part in parts}
    return frozenset(token for token in tokens if token)


def resolve_preferred(
    *,
    agreement: Any = None,
    per_advertiser: Optional[dict[str, Any]] = None,
    advertiser: Optional[str] = None,
    channel_default: Any = None,
) -> tuple[Optional[frozenset[str]], str]:
    """The preferred set that applies, and the name of the scope it came from.

    Most specific wins, because the trade sets this per agreement first and per
    client second: agreement, then advertiser, then the channel default. The
    second element is the scope name so a surface can say whose rule it is
    reading. When nothing is configured the answer is ``(None, "unset")`` and no
    percentage may be shown.
    """
    from_agreement = parse_preferred(agreement)
    if from_agreement is not None:
        return from_agreement, "agreement"
    if advertiser and per_advertiser:
        from_advertiser = parse_preferred(per_advertiser.get(str(advertiser)))
        if from_advertiser is not None:
            return from_advertiser, "advertiser"
    from_channel = parse_preferred(channel_default)
    if from_channel is not None:
        return from_channel, "channel_default"
    return None, "unset"


@dataclass(frozen=True)
class Appearance:
    """One broadcast of a campaign: which break, which position, how long."""

    break_id: str
    position: int
    break_size: Optional[int] = None


@dataclass(frozen=True)
class PreferredPositionRate:
    """A preferred-position percentage that cannot be read without its method.

    ``percent`` is ``None`` when the preferred set is unset or there is nothing
    to measure, and ``basis`` says which of those it is, so a surface shows real,
    unavailable or unknown rather than a zero that looks like a result.
    """

    method: str
    method_label_en: str
    method_label_he: str
    numerator: int
    denominator: int
    percent: Optional[float]
    basis: str
    preferred: tuple[str, ...]
    breaks_appeared_in: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_label_en": self.method_label_en,
            "method_label_he": self.method_label_he,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "percent": self.percent,
            "basis": self.basis,
            "preferred": list(self.preferred),
            "breaks_appeared_in": self.breaks_appeared_in,
        }


def method_label(method: str, locale: str = "he") -> str:
    """The bilingual name of a counting method, Hebrew by default."""
    pair = _METHOD_LABELS.get(method)
    if pair is None:
        raise ValueError(f"unknown counting method: {method!r}")
    return pair[1] if str(locale).lower().startswith("he") else pair[0]


def preferred_position_rate(
    appearances: Iterable[Appearance],
    preferred: Optional[Iterable[Any]],
    method: str,
) -> PreferredPositionRate:
    """Count a campaign's preferred-position percentage under a NAMED method.

    ``method`` is required and is carried back on the result, because two parties
    audit each other with this number. See the module docstring for what each
    method counts and for the honest limit of the extraction.
    """
    if method not in COUNTING_METHODS:
        raise ValueError(f"method must be one of {list(COUNTING_METHODS)}")
    rows = list(appearances)
    preferred_set = parse_preferred(list(preferred)) if preferred is not None else None
    breaks = len({row.break_id for row in rows})
    empty = PreferredPositionRate(
        method=method,
        method_label_en=method_label(method, "en"),
        method_label_he=method_label(method, "he"),
        numerator=0,
        denominator=0,
        percent=None,
        basis="unset",
        preferred=(),
        breaks_appeared_in=breaks,
    )
    if preferred_set is None:
        return empty
    ordered = tuple(sorted(preferred_set))
    if not rows:
        return PreferredPositionRate(
            method=method,
            method_label_en=empty.method_label_en,
            method_label_he=empty.method_label_he,
            numerator=0,
            denominator=0,
            percent=None,
            basis="unavailable",
            preferred=ordered,
            breaks_appeared_in=breaks,
        )
    # Agency: count the preferred POSITIONS obtained, so a broadcast holding two
    # preferred positions at once counts twice, against one denominator entry per
    # appearance in a break. Channel: count the BROADCASTS that obtained at least
    # one preferred position, out of total broadcasts.
    numerator = 0
    for row in rows:
        held = [t for t in occupied_tokens(row.position, row.break_size) if t in preferred_set]
        numerator += len(held) if method == AGENCY_METHOD else (1 if held else 0)
    denominator = len(rows)
    return PreferredPositionRate(
        method=method,
        method_label_en=empty.method_label_en,
        method_label_he=empty.method_label_he,
        numerator=numerator,
        denominator=denominator,
        percent=round(100.0 * numerator / denominator, 4),
        basis="real",
        preferred=ordered,
        breaks_appeared_in=breaks,
    )
