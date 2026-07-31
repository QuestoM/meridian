"""Advertiser name identity: the name space, the alias index and the resolver.

Split out of :mod:`kairos.optimize.advertiser_rules` to keep that module under
the project line limit. It carries no rule math; it answers one question, which
the rules engine could not answer before: given the advertiser string a real
daily file carries, which stored record is that.

Why this exists. The rules store is keyed on ``ADV_01..ADV_45`` while every real
spot names its advertiser in Hebrew, so the two key spaces did not intersect at
all and every rule lookup missed in silence. Agencies already solved the same
problem: ``data/agencies.csv`` carries ``name``, ``display_name`` and a
pipe-joined ``aliases`` column, and the daily agency string resolves through
them by string equality. This module gives the advertiser store the same shape
and the same mechanism.

Two stores, two jobs.

  * ``data/advertiser_rules.csv`` gains ``name``, ``display_name`` and
    ``aliases``. Filling one of them BINDS that advertiser to that rules row, so
    the row's premium and its scoped conditions start pricing that advertiser's
    spots. A blank name binds nothing, which is why adding the columns moves no
    money on its own.
  * ``data/advertiser_names.csv`` is the observed name space: one row per
    advertiser seen in the real data, with its aliases and where the name came
    from. It carries no premium and no rule, so reading it can never change a
    price. It exists so an advertiser can be a named record before anybody has
    written a rule for it.

Honesty rules. Nothing here invents a name: an unmatched token resolves to
``None`` and the caller states that it is unresolved. Resolution is exact
equality after a documented, conservative fold (surrounding and repeated
whitespace, letter case, and the Hebrew geresh and gershayim against their
ASCII equivalents), never a fuzzy or edit-distance match, because a wrong
advertiser is a wrong price.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMES_PATH = ROOT / "data" / "advertiser_names.csv"

# The alias separator, identical to the one data/agencies.csv already uses.
ALIAS_SEPARATOR = "|"

# The identity columns the rules store shares with data/agencies.csv.
IDENTITY_COLUMNS = ("name", "display_name", "aliases")

# The observed name space's own columns. ``name`` is the key: identity here is
# semantic (the advertiser's real name, which is what every source carries),
# never an ordinal position in a file.
NAMES_COLUMNS = ("name", "display_name", "aliases", "source", "first_seen", "notes")

# Punctuation that differs only typographically between two spellings of the
# same Hebrew name. Folding these merges "בע"מ" written with a gershayim and
# with an ASCII quote; it never merges two different names.
_PUNCTUATION_FOLD = {
    "׳": "'",   # Hebrew geresh
    "״": '"',   # Hebrew gershayim
    "‘": "'",   # left single quotation mark
    "’": "'",   # right single quotation mark
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
}

# Where a resolved name was matched, so a surface can say why it resolved.
MATCH_ID = "advertiser_id"
MATCH_NAME = "name"
MATCH_DISPLAY_NAME = "display_name"
MATCH_ALIAS = "alias"


def normalize_name(raw: object) -> str:
    """The comparison form of an advertiser name.

    Collapses surrounding and repeated whitespace, folds letter case, and maps
    the Hebrew geresh and gershayim (and the typographic quotes) onto their
    ASCII equivalents. Everything else is left exactly as written, so two
    genuinely different names never collapse into one.
    """
    text = str(raw if raw is not None else "")
    for source, target in _PUNCTUATION_FOLD.items():
        text = text.replace(source, target)
    return " ".join(text.split()).casefold()


def split_aliases(raw: object) -> tuple[str, ...]:
    """Split a pipe-joined alias cell into its trimmed, non-empty parts."""
    text = str(raw if raw is not None else "")
    return tuple(part.strip() for part in text.split(ALIAS_SEPARATOR) if part.strip())


def join_aliases(aliases: Iterable[str]) -> str:
    """Serialize aliases back to the pipe-joined cell the stores hold."""
    return ALIAS_SEPARATOR.join(alias.strip() for alias in aliases if str(alias).strip())


@dataclass(frozen=True)
class AdvertiserName:
    """One row of the observed advertiser name space."""

    name: str
    display_name: str = ""
    aliases: tuple[str, ...] = ()
    source: str = ""
    first_seen: str = ""
    notes: str = ""

    @property
    def shown_name(self) -> str:
        """The name to show: the operator's display name, else the real name."""
        return self.display_name.strip() or self.name


@dataclass(frozen=True)
class ResolvedAdvertiser:
    """One resolved advertiser: who it is, and which rules row it is bound to.

    ``advertiser_id`` is ``None`` when no rules row claims this name, which is
    the shipped state of every advertiser: the record is named and real, and it
    carries no rule until the operator writes one.
    """

    key: str
    name: str
    display_name: str = ""
    aliases: tuple[str, ...] = ()
    source: str = ""
    advertiser_id: Optional[str] = None
    matched_on: str = MATCH_NAME

    @property
    def shown_name(self) -> str:
        return self.display_name.strip() or self.name

    @property
    def has_rules_row(self) -> bool:
        return self.advertiser_id is not None


@dataclass
class NameIndex:
    """Every token that resolves to a rules row, plus the collisions found.

    ``by_token`` maps the normalized form of an id, name, display name or alias
    to its ``advertiser_id``. First row wins, exactly as the agency layer does,
    and every later claim on the same token is recorded in ``collisions`` rather
    than silently overwriting, so a store that binds one name twice is visible
    instead of arbitrary.
    """

    by_token: dict[str, str] = field(default_factory=dict)
    collisions: list[tuple[str, str, str]] = field(default_factory=list)

    def claim(self, token: object, advertiser_id: str) -> None:
        key = normalize_name(token)
        if not key or not advertiser_id:
            return
        held = self.by_token.get(key)
        if held is None:
            self.by_token[key] = advertiser_id
        elif held != advertiser_id:
            self.collisions.append((key, held, advertiser_id))

    def get(self, token: object) -> Optional[str]:
        return self.by_token.get(normalize_name(token))

    def __bool__(self) -> bool:
        return bool(self.by_token)

    def __len__(self) -> int:
        return len(self.by_token)


def build_name_index(rows: Iterable[Mapping[str, object]]) -> NameIndex:
    """Index the identity columns of the rules store rows.

    The raw ``advertiser_id`` is indexed too, so a store keyed on real names
    (the shape one of the two migration options produces) resolves through the
    same path as a store keyed on tokens with names filled in beside them.
    """
    index = NameIndex()
    for row in rows:
        advertiser_id = str(row.get("advertiser_id", "") or "").strip()
        if not advertiser_id:
            continue
        index.claim(advertiser_id, advertiser_id)
        for column in IDENTITY_COLUMNS:
            value = row.get(column, "")
            if column == "aliases":
                for alias in split_aliases(value):
                    index.claim(alias, advertiser_id)
            else:
                index.claim(value, advertiser_id)
    return index


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a csv into dict rows, tolerating a missing file (empty list)."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_name_index(path: Path) -> NameIndex:
    """Build the rules-store alias index from a csv on disk."""
    return build_name_index(read_csv_rows(path))


def load_advertiser_names(path: Path | None = None) -> dict[str, AdvertiserName]:
    """Read the observed name space, keyed by the normalized canonical name.

    A missing file yields an empty map, so every caller degrades to "no name
    space" rather than failing, and no name is ever invented to fill it.
    """
    target = Path(path) if path is not None else DEFAULT_NAMES_PATH
    out: dict[str, AdvertiserName] = {}
    for row in read_csv_rows(target):
        name = str(row.get("name", "") or "").strip()
        if not name:
            continue
        key = normalize_name(name)
        if key in out:
            continue
        out[key] = AdvertiserName(
            name=name,
            display_name=str(row.get("display_name", "") or "").strip(),
            aliases=split_aliases(row.get("aliases", "")),
            source=str(row.get("source", "") or "").strip(),
            first_seen=str(row.get("first_seen", "") or "").strip(),
            notes=str(row.get("notes", "") or "").strip(),
        )
    return out


def _names_token_index(names: Mapping[str, AdvertiserName]) -> dict[str, str]:
    """Map every name and alias in the name space to its canonical key."""
    tokens: dict[str, str] = {}
    for key, record in names.items():
        for token in (record.name, record.display_name, *record.aliases):
            normalized = normalize_name(token)
            if normalized:
                tokens.setdefault(normalized, key)
    return tokens


def resolve_advertiser(
    token: object,
    *,
    names: Mapping[str, AdvertiserName],
    rules_index: NameIndex | None = None,
    names_tokens: Mapping[str, str] | None = None,
) -> Optional[ResolvedAdvertiser]:
    """Resolve one advertiser string to a named record, or ``None``.

    The name space answers who the advertiser is; the rules index answers which
    rules row, if any, is bound to it. A token that matches only a rules row
    still resolves, using that row's own identity columns, so an advertiser the
    operator created by hand is not invisible just because it has never aired.
    """
    normalized = normalize_name(token)
    if not normalized:
        return None
    tokens = names_tokens if names_tokens is not None else _names_token_index(names)
    advertiser_id = rules_index.get(normalized) if rules_index is not None else None
    key = tokens.get(normalized)
    if key is not None:
        record = names[key]
        return ResolvedAdvertiser(
            key=record.name,
            name=record.name,
            display_name=record.display_name,
            aliases=record.aliases,
            source=record.source or "observed",
            advertiser_id=advertiser_id,
            matched_on=_match_kind(normalized, record),
        )
    if advertiser_id is None:
        return None
    return ResolvedAdvertiser(
        key=advertiser_id,
        name=str(token).strip(),
        display_name="",
        aliases=(),
        source="rules",
        advertiser_id=advertiser_id,
        matched_on=MATCH_ID if normalize_name(advertiser_id) == normalized else MATCH_ALIAS,
    )


def _match_kind(normalized: str, record: AdvertiserName) -> str:
    if normalized == normalize_name(record.name):
        return MATCH_NAME
    if record.display_name and normalized == normalize_name(record.display_name):
        return MATCH_DISPLAY_NAME
    return MATCH_ALIAS
