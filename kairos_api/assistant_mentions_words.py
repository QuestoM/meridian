"""The kinds a mention may name, and the approved word for each.

Every Hebrew word here is READ from somewhere that already shipped it, never
minted for this route. A kind whose word had to be invented does not appear,
which is why this table has four rows and not the twenty-odd of the full
taxonomy: the other kinds arrive as their coverage does, each with the word the
product already uses for it.

Where each word comes from, so the next reader can check rather than trust:

``advertiser``  ``מפרסם``          tv-break-dashboard/src/vocabulary.js, ``object.advertiser``
``agency``      ``סוכנות``          tv-break-dashboard/src/vocabulary.js, ``object.agency``
``program``     ``תוכנית``          shipped on seven surfaces (BreakBoard, BreakInspector,
                                   MoneyDetail, GoldBreakManager, ScheduleInspector,
                                   RestrictionComposer, constraint-predicate) and in
                                   src/history/history-row-words.js
``event``       ``אירוע לוח שנה``    src/history/history-fields.js and
                                   src/rules/pricing-layers-lib.js

The icon key is not a glyph. It names the RAIL DESTINATION whose own icon the
kind borrows, or the lucide name where the kind has no rail item, and the
dashboard resolves it against ``shell/nav.js``. The icon is navigational
identity rather than decoration, so the server states which identity and the
client states how it is drawn.

RANK is the kind order the matcher sorts by before it sorts by score: an exact
identifier first (the route handles that above the table), then the plan spine,
then the commercial objects, then rules and money. A bare number in this product
usually means a time or a date, which is why no kind here is keyed by number.
"""

from __future__ import annotations

# rank, Hebrew, English, icon key. Ranks leave gaps on purpose: the day, break
# and pod kinds land between the programme and the commercial objects when their
# read tools exist, and inserting them should not renumber what is already here.
KINDS: dict[str, dict[str, object]] = {
    "program": {"rank": 20, "he": "תוכנית", "en": "programme", "icon": "MonitorPlay"},
    "advertiser": {"rank": 40, "he": "מפרסם", "en": "advertiser", "icon": "nav:Advertisers"},
    "agency": {"rank": 41, "he": "סוכנות", "en": "agency", "icon": "nav:Agencies"},
    "event": {"rank": 60, "he": "אירוע לוח שנה", "en": "calendar event", "icon": "CalendarDays"},
}

KIND_NAMES: tuple[str, ...] = tuple(KINDS)


def rank(kind: str) -> int:
    """Sort position of a kind. An unknown kind sorts last rather than raising,
    because a kind reaching here that the table does not hold is a bug in the
    caller and should not take an operator's picker down with it."""
    entry = KINDS.get(kind)
    return int(entry["rank"]) if entry else 999


def label(kind: str, locale: str = "he") -> str:
    """The approved word for a kind, in the operator's language. Empty for an
    unknown kind: a gap a reviewer notices beats a snake_case token a person has
    to decode, which is the rule vocabulary.js already states for the same case.
    """
    entry = KINDS.get(kind)
    if not entry:
        return ""
    return str(entry["he" if locale == "he" else "en"])


def icon(kind: str) -> str:
    return str(KINDS.get(kind, {}).get("icon", ""))
