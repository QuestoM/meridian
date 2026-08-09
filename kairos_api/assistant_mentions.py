"""The mention index and its one route: ``GET /api/assistant/mentions``.

WHAT THIS IS FOR. The composer had no way to point at a thing. It carried
``page_context``, which says WHERE THE OPERATOR IS at rail-label granularity and
never WHAT THEY MEAN, so a second thing, a thing on another page, or anything
below entity granularity could not be referred to at all. Typing ``@`` now opens
a picker over the objects that actually exist, and choosing one puts the store's
own name into the question.

WHAT IT IS NOT. It is not a gate. The measurement this whole piece turns on is
that the other product's structured mention path was exercised ZERO times in
10,952 recorded turns: a mention system that is the only way to name something
does not get used. So every free-text route into the model is untouched by this
file. ``_question_dates`` still parses a date out of prose, ``find_advertiser``
still fuzzy-matches a typed name, the keyword sections still fire. This route
adds a way of asking and removes none.

THE BOUNDARY, AND IT HAS THREE RULES HERE RATHER THAN ONE.

The candidate index is SERVER-SIDE and that is the whole reason this module
exists rather than a client-side array. The saved weekly plan holds every
channel, because the retention model is measured against the competitive lineup;
shipping the index to the browser would put rival rows in it.

1. THE CAP IS APPLIED AFTER SCOPING. ``omitted`` counts matches that survived
   the scope and lost to the cap. An omitted count computed before scoping is
   itself a rival count, which is the reason ``_section_counts`` already gives
   for its own ordering.
2. NO-MATCH AND NOT-OURS ARE INDISTINGUISHABLE. A rival channel's name, a rival
   programme's title and an outright typo all return the same empty result with
   the same absent reason. "None on your channel" would confirm the name exists,
   which is the disclosure the boundary exists to prevent.
3. NO KIND IS SEARCHABLE WHOSE STORE CANNOT BE SCOPED. Of the four kinds here,
   the programme comes out of ``assistant_context._owned_frame()`` and is scoped
   by construction. The other three are operator-owned rule stores with no
   channel column at all: ``advertisers.COLUMNS``, ``agencies.COLUMNS`` and
   ``events_api.COLUMNS`` are checked against that claim at import-safe call
   time by the test, so the claim is measured rather than asserted.

HEBREW. ``בחדשות`` must match ``חדשות`` or the picker is useless in the language
the product is written in, which is the requirement neither reference product
has. The one-letter prefixes are stripped on BOTH sides using
``assistant_context._strip_hebrew_prefixes`` -- the existing one. A second home
for one idea is the defect this campaign has spent the day removing.

MATCHING takes the reference matcher's shape: kind rank first, then a
case-insensitive subsequence over label and identifier with a prefix bonus. The
caps are its caps too, 20 fetched and 8 shown, because those numbers were
arrived at against a popup of the same size.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from kairos_api import assistant_mentions_words as words
from kairos_api import read_cache

router = APIRouter(tags=["assistant"])

CACHE_NAMESPACE = "assistant_mentions"
read_cache.configure(CACHE_NAMESPACE, capacity=4)

# The reference product's own two numbers. Twenty is what a keystroke ranks
# over; eight is what a popup shows without becoming a page.
FETCH_CAP = 20
SHOW_CAP = 8
QUERY_MAX = 80

# The prefix bonus, and it is a bonus because lower sorts first. A candidate
# whose match starts at character zero beats one that matches in the middle by
# more than any gap penalty can close, which is what makes typing the first
# letters of a name feel like the name.
PREFIX_BONUS = 100


# --- the query, folded the way a label is folded ----------------------------------
def _strip_prefixes(text: str) -> str:
    """Every Hebrew word in the text with its one-letter prefixes removed.

    The stripper itself is assistant_context's. This wraps it over a phrase
    rather than a word, and it is the only thing this module adds to it.
    """
    from kairos_api.assistant_context import _strip_hebrew_prefixes

    return " ".join(_strip_hebrew_prefixes(word) for word in text.split())


def _fold(text: str) -> str:
    """The comparable form of a label or a query: prefix-stripped, case-folded,
    whitespace collapsed. Applied to both sides, so the stripping is symmetric
    and בחדשות finds חדשות whichever of the two was typed."""
    return _strip_prefixes(str(text or "").strip().casefold())


def _subsequence_score(needle: str, haystack: str) -> int | None:
    """How well a folded needle matches a folded haystack, lower being better,
    or None when it does not match at all.

    A subsequence rather than a substring, so ``חדשות ערב`` finds
    ``מהדורת חדשות הערב``. The score is the distance the match had to travel:
    where it started plus the gaps it jumped. A match at position zero takes the
    prefix bonus.
    """
    if not needle:
        return 0
    if not haystack:
        return None
    position = 0
    first = -1
    score = 0
    for character in needle:
        found = haystack.find(character, position)
        if found < 0:
            return None
        if first < 0:
            first = found
        score += found - position
        position = found + 1
    score += first
    return score - PREFIX_BONUS if first == 0 else score


def _best_score(needle: str, row: dict[str, Any]) -> int | None:
    """The better of the label match and the identifier match.

    Both are searched because an operator who knows a record by its stored id
    should reach it by typing the id, and one who knows it by name should reach
    it by name, without choosing a mode first.
    """
    scores = [
        value
        for value in (
            _subsequence_score(needle, row["_fold_label"]),
            _subsequence_score(needle, row["_fold_id"]),
        )
        if value is not None
    ]
    return min(scores) if scores else None


# --- the candidate index ----------------------------------------------------------
def _row(kind: str, ident: str, label: str, parent: list[dict[str, Any]]) -> dict[str, Any]:
    label = str(label or "").strip()
    ident = str(ident or "").strip()
    return {
        "type": kind,
        "id": ident,
        "label": label or ident,
        "icon": words.icon(kind),
        "parent": parent,
        "_fold_label": _fold(label or ident),
        "_fold_id": _fold(ident),
    }


def _programs() -> list[dict[str, Any]]:
    """Programmes of the saved plan, on the operator's own channel only.

    ``_owned_frame`` is the scope. It returns None with a reason whenever no
    owned rows can be produced, and this returns nothing at all in that case:
    there is no unscoped fallback here and there must never be one.

    The dim second line is the disambiguator that matters most in this product,
    because a programme recurs every weekday and a code editor has no analogue
    of that. It carries the days the programme actually runs and the channel,
    as two parts rather than one string, so the client isolates each and never
    wraps a phrase.
    """
    from kairos_api import assistant_context

    frame, owned, _competitors, _reason = assistant_context._owned_frame()
    if frame is None:
        return []
    column = "program_title" if "program_title" in frame.columns else "program_type"
    rows: list[dict[str, Any]] = []
    for title, group in frame.groupby(frame[column].astype(str).str.strip(), sort=False):
        name = str(title).strip()
        if not name or name.lower() == "nan":
            continue
        days = sorted({str(value) for value in group["date_text"]})
        parent: list[dict[str, Any]] = []
        if days:
            parent.append({"kind": "span", "from": days[0], "to": days[-1]})
        parent.append({"kind": "name", "text": owned})
        rows.append(_row("program", name, name, parent))
    return rows


def _advertisers() -> list[dict[str, Any]]:
    """The advertiser rules store. No channel column exists in it: an advertiser
    is a commercial counterparty of this operator, not a market-wide row, so
    there is nothing here to scope and nothing to leak."""
    from kairos_api.advertisers import _load_frame, _row_to_record

    rows = []
    for _, raw in _load_frame().iterrows():
        record = _row_to_record(raw)
        ident = str(record.get("advertiser_id") or "").strip()
        if not ident:
            continue
        label = str(record.get("display_name") or record.get("name") or "").strip() or ident
        parent = [] if label == ident else [{"kind": "code", "text": ident}]
        rows.append(_row("advertiser", ident, label, parent))
    return rows


def _agencies() -> list[dict[str, Any]]:
    """The agencies store, same argument as the advertisers store above."""
    from kairos_api import agencies

    rows = []
    for _, raw in agencies._load_frame().iterrows():
        record = agencies._row_to_record(raw)
        ident = str(record.get("agency_id") or "").strip()
        if not ident:
            continue
        label = str(record.get("name") or "").strip() or ident
        parent = [] if label == ident else [{"kind": "code", "text": ident}]
        rows.append(_row("agency", ident, label, parent))
    return rows


def _events() -> list[dict[str, Any]]:
    """Calendar events. The dim line is the event's own window, which is the
    real disambiguator: a holiday recurs and the two occurrences are two rows."""
    from kairos_api import events_api

    rows = []
    for _, raw in events_api._load_frame().iterrows():
        ident = str(raw.get("event_id") or "").strip()
        if not ident:
            continue
        label = str(raw.get("name") or "").strip() or ident
        start = str(raw.get("start_date") or "").strip()
        end = str(raw.get("end_date") or "").strip()
        parent = [{"kind": "span", "from": start, "to": end}] if start else []
        rows.append(_row("event", ident, label, parent))
    return rows


_BUILDERS = {
    "program": _programs,
    "advertiser": _advertisers,
    "agency": _agencies,
    "event": _events,
}


def _fingerprint() -> tuple[Any, ...]:
    """What the index is a function of. A run rewrites the plan under the
    operator, so the index has to fall over when any of these files move rather
    than serve a programme that no longer exists."""
    from kairos_api import agencies, channel_scope, events_api
    from kairos_api.advertisers import RULES_PATH
    from kairos_api.core import ROOT

    return (
        read_cache.file_signatures(
            [
                RULES_PATH,
                agencies.AGENCIES_PATH,
                events_api.EVENTS_PATH,
                ROOT / "output" / "weekly_break_schedule.csv",
            ]
        ),
        channel_scope.operator_channel(),
    )


def build_index() -> list[dict[str, Any]]:
    """Every candidate, already scoped, cached on the fingerprint above.

    A builder that raises takes its own kind out of the index and leaves the
    rest standing. An unreadable store is an absent kind, never an invented row:
    the operator who cannot find an advertiser learns that from an empty picker,
    which is the same thing the picker says about a name that is not theirs.
    """

    def _build() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for kind in words.KIND_NAMES:
            try:
                rows.extend(_BUILDERS[kind]())
            except Exception:  # noqa: BLE001 - an unreadable store is an absent kind
                continue
        return rows

    return read_cache.cached(CACHE_NAMESPACE, "index", _fingerprint(), _build)


# --- the search -------------------------------------------------------------------
def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def search(q: str = "", types: str = "", limit: int = SHOW_CAP) -> dict[str, Any]:
    """Rank the scoped index against a query and return at most ``limit`` rows.

    The order of operations is the boundary: the index is scoped before it is
    matched, matched before it is capped, and the omitted count is taken from
    the capped end of an already-scoped list. Nothing in the payload is derived
    from a row that did not survive the scope.
    """
    from kairos_api import channel_scope

    query = str(q or "")[:QUERY_MAX].strip()
    wanted = {name for name in str(types or "").split(",") if name.strip() in words.KINDS}
    shown = max(1, min(int(limit or SHOW_CAP), FETCH_CAP))

    index = [row for row in build_index() if not wanted or row["type"] in wanted]
    needle = _fold(query)
    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for row in index:
        score = _best_score(needle, row)
        if score is None:
            continue
        # An exact identifier sorts above every kind, which is the one case where
        # the operator has already told us precisely what they mean.
        rank = -1 if needle and needle == row["_fold_id"] else words.rank(row["type"])
        scored.append((rank, score, row["label"], row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))

    matched = len(scored)
    rows = [_public(item[3]) for item in scored[:shown]]
    # The query is NOT echoed. Boundary rule two wants a rival name and a typo to
    # be indistinguishable, and an echo makes the two payloads differ in a field
    # a scanner has to be told to ignore. The client already knows what it typed
    # and tracks staleness against its own copy, so the echo bought nothing and
    # cost the strongest form of the property: the two answers are byte-identical.
    return {
        "rows": rows,
        "count": len(rows),
        # After scoping and after matching, never before either.
        "omitted": max(0, matched - len(rows)),
        "limit": shown,
        "kinds": list(words.KIND_NAMES),
        "scope": {
            "scope_channel": channel_scope.operator_channel() or None,
            "scoped": bool(channel_scope.operator_channel()),
        },
    }


@router.get("/mentions")
def get_mentions(q: str = "", types: str = "", limit: int = SHOW_CAP) -> dict[str, Any]:
    """Candidates for the composer's ``@`` picker.

    Advisory in exactly the sense ``page_context`` is advisory: nothing about
    the ask changes because this route exists, no tool is restricted by it, and
    a client that never calls it behaves precisely as the console did before.
    """
    return search(q=q, types=types, limit=limit)
