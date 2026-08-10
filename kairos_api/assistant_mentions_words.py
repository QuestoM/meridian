"""The kinds a mention may name, and the approved word for each.

Every Hebrew word here is READ from somewhere that already shipped it, never
minted for this route. A kind whose word had to be invented does not appear,
which is why this table has six rows and not the twenty-odd of the full
taxonomy: the other kinds arrive as their coverage does, each with the word the
product already uses for it.

Where each word comes from, so the next reader can check rather than trust:

``day``         ``יום שידור``        downloads_api_reports.py, constraints_sentence.py,
                                   makegood_store_words.py and campaigns_goal_words.py all
                                   ship this exact phrase for this exact object
``advertiser``  ``מפרסם``          tv-break-dashboard/src/vocabulary.js, ``object.advertiser``
``agency``      ``סוכנות``          tv-break-dashboard/src/vocabulary.js, ``object.agency``
``program``     ``תוכנית``          shipped on seven surfaces (BreakBoard, BreakInspector,
                                   MoneyDetail, GoldBreakManager, ScheduleInspector,
                                   RestrictionComposer, constraint-predicate) and in
                                   src/history/history-row-words.js
``break``       ``ברייק``           tv-break-dashboard/src/vocabulary.js, ``object.break``
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
    "day": {"rank": 10, "he": "יום שידור", "en": "broadcast day", "icon": "nav:Schedule"},
    "program": {"rank": 20, "he": "תוכנית", "en": "programme", "icon": "MonitorPlay"},
    "break": {"rank": 30, "he": "ברייק", "en": "break", "icon": "nav:Break Library"},
    "advertiser": {"rank": 40, "he": "מפרסם", "en": "advertiser", "icon": "nav:Advertisers"},
    "agency": {"rank": 41, "he": "סוכנות", "en": "agency", "icon": "nav:Agencies"},
    "event": {"rank": 60, "he": "אירוע לוח שנה", "en": "calendar event", "icon": "CalendarDays"},
}

KIND_NAMES: tuple[str, ...] = tuple(KINDS)

# THE LADDER, AND IT IS A GRAPH OF TYPED EDGES RATHER THAN A TREE.
#
# A file picker can navigate by path because every leaf has exactly one. This
# product's objects do not: a day holds programmes and a programme runs on days,
# and both directions are the same relation read from the same scoped frame. So
# descending is an EDGE the caller names, not a path segment it appends, and the
# same object is reachable from two parents without being two objects.
#
# Only edges whose CHILD store can be produced under the operator's own scope
# appear here. Break coverage now exists, so the first complete plan-spine path
# is day -> programme -> break. Pod and spot remain absent until their identities
# can be carried across the same scoped graph rather than inferred from labels.
EDGES: dict[str, tuple[str, ...]] = {
    "day": ("program",),
    "program": ("day", "break"),
    "agency": ("advertiser",),
}


# WHAT A REFERENCE CAME BACK AS, in the operator's own language.
#
# Four states and no fewer, because "we looked and it is not there" and "we could
# not look" are different claims and a product that reports them as one thing is
# guessing on the operator's behalf. Every word below is READ from a module that
# already ships it for this exact meaning, and the source is named so the next
# reader checks rather than trusts.
#
# ``resolved``     ``נקרא``            break_api_pod_spots.PREFERRED_BASIS_HE, which
#                                     opens "נקרא מ..." for a figure that was read
# ``changed``      ``השתנה``           break_api_pod_order.STALE_HE, for a store that
#                                     moved under a saved reading
# ``gone``         ``לא נמצא``         campaigns_api_store.py, for an identifier the
#                                     store was read for and does not hold
# ``unavailable``  ``לא ניתן לקרוא``    break_api_states.py, for a store that could
#                                     not be read at all
STATES: dict[str, dict[str, str]] = {
    "resolved": {"he": "נקרא", "en": "read"},
    "changed": {"he": "השתנה", "en": "changed since it was pointed at"},
    "gone": {"he": "לא נמצא", "en": "not found in the store"},
    "unavailable": {"he": "לא ניתן לקרוא", "en": "the store could not be read"},
}

# The one thing a descent with nothing under it says, in both languages. The
# Hebrew is overview_api_drill.py's own sentence for this exact situation, a
# level of a drill with nothing readable beneath it, and it is reused rather than
# rewritten. It names the KIND that was asked for and never the container that
# was asked about, which is what keeps every empty descent identical.
ABSENT_REASON_EN = "nothing of this kind is held under a container of that kind"
ABSENT_REASON_HE = "אין מה לקרוא מתחת לרמה הזאת"


def state_label(state: str, locale: str = "he") -> str:
    """The approved word for a resolution state. Empty for an unknown state, on
    the same rule the kind labels follow: a gap a reviewer notices beats a
    snake_case token an operator has to decode."""
    entry = STATES.get(state)
    if not entry:
        return ""
    return entry["he" if locale == "he" else "en"]


def child_kinds(kind: str) -> tuple[str, ...]:
    """The kinds a container of this kind descends into, in ladder order."""
    return EDGES.get(kind, ())


def is_container(kind: str) -> bool:
    """Whether a row of this kind may be entered rather than only accepted."""
    return bool(EDGES.get(kind))


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
