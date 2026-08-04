"""Which column of the operator's OWN file a finding is about.

Split out of ``uploads_validate.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule the package already follows.

**The measured defect this module exists for.** A spots export whose dates will
not parse was refused, and the red Hebrew panel printed a bold Latin chip
``air_dt`` beside the sentence. The file's own header is
``Campaign,Channel,Date,Start time,Duration,TVR``: ``air_dt`` is not in it, is
not in any export this door accepts, and is a column the loader COMPUTES after
the read. The same defect stood on the amber side, over a file that was
accepted: the consequence read "about ``spot_time, date``" of a file whose
headers are ``שעה`` and ``תאריך``. Inside one card this destination spoke both
vocabularies at once, because a missing-column refusal one panel away correctly
named ``מפרסם``, the file's own header.

**The rule, stated once.** A finding names a header the candidate's own header
row really carries, or it names no column at all and says what it IS about
instead. Nothing here guesses: every name returned was read off the header row
of the file in front of it, and a loaded name that traces back to none of them
is answered with a scope, which the surface resolves to a word in the language
it is being read in.

The rename map is the loader's own, read backwards, so a header the engine
starts renaming later is carried here without a second copy of it. The two
datetime columns are not renames at all: ``start_dt`` and ``air_dt`` are built
by combining two headers, so both are named, because a datetime that will not
parse can be wrong in either of them.
"""

from __future__ import annotations

from typing import Iterable

from kairos.data.loaders import DAILY_COLUMN_MAP

# The loader's rename map read backwards: the header each canonical name was
# renamed FROM. Derived and never copied, so the two cannot drift apart.
HEADER_OF: dict[str, str] = {loaded: header for header, loaded in DAILY_COLUMN_MAP.items()}

# The headers each COMPUTED datetime column is built from, which is the case no
# rename map can answer: nothing was renamed, two columns were combined.
BUILT_FROM: dict[str, tuple[str, ...]] = {
    "start_dt": ("Date", "Start time"),
    "air_dt": ("Date", "Start time"),
}

# The kinds whose loaded frame keeps the file's own header names, so a name that
# is not in the header row is a column the engine needs the file to ADD, and
# printing it is the only actionable word the sentence has: measured, a spots
# export with no ``TVR`` column is refused for that column, and answering "the
# table" there deletes the one thing a person could act on. The daily kind is
# absent because every name in its loaded frame is a rename of a Hebrew header,
# and the dayparts kind because its loader melts one row per channel column and
# validates that melt's names, which no export of that kind carries.
HEADER_NAMED_KINDS = frozenset({"programmes", "spots", "advertiser_rules", "rate_card", "campaign_flights"})

# What a finding whose column traces back to no header is about instead: the
# table the loader parsed the file into. One of
# :data:`kairos_api.uploads_messages.SCOPES`, so the surface already resolves it.
FALLBACK_SCOPE = "frame"


def headers_for(loaded: str, headers: Iterable[object]) -> list[str]:
    """The header(s) of THIS file that one loaded column was made from.

    The canonical name itself is the second candidate, because a file that was
    exported with the engine's own names already carries it, and empty is the
    honest answer for a header row that carries neither.
    """
    present = [str(header) for header in headers]
    candidates = BUILT_FROM.get(loaded) or (HEADER_OF.get(loaded, loaded), loaded)
    named = [name for name in dict.fromkeys(candidates) if name and name in present]
    return named if loaded in BUILT_FROM else named[:1]


def place(loaded: str, headers: Iterable[object], kind: str = "") -> tuple[str, str]:
    """``(column, scope)`` for one finding: this file's own header, or the scope.

    Exactly one of the two is filled, which is the rule the surface renders by:
    the chip is a column name the file really carries or the word for what the
    finding is about, and never both, and never a name that is neither.

    The one name printed that the header row does not carry is a column of a
    :data:`HEADER_NAMED_KINDS` export that the engine needs and the file has not
    got, which is the finding that this column is missing: the word to print
    there is the name to add.
    """
    named = headers_for(loaded, headers)
    if named:
        return ", ".join(named), ""
    return (loaded, "") if kind in HEADER_NAMED_KINDS else ("", FALLBACK_SCOPE)
