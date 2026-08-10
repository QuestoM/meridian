"""Descending a container: ``GET /api/assistant/mentions/children``.

WHY THIS EXISTS AT ALL, AND WHY BOTH REFERENCE PRODUCTS COULD DECLINE IT.

A flat fuzzy search substitutes for navigation exactly when every leaf has a
unique typeable path. In a code editor it does, so one product kills drill-down
with a trailing space and the other collapses the directory kind into the file
kind, and both are right for what they hold. This product's leaves do not have
that property: a spot has no name, a break is a clock reading that recurs, and a
programme title repeats every weekday. Something has to be enterable or the
objects below the top of the ladder are unreachable by typing.

THE LADDER IS A GRAPH, NOT A TREE, and that is why this is an edge route rather
than a path walk. ``assistant_mentions_words.EDGES`` holds it. A day descends to
its programmes and a programme ascends to its days, and neither is the parent of
the other: they are the same relation read in two directions off one scoped
frame. The same object reached down one ladder and up another is one object,
which is the thing a file tree cannot express.

THE BOUNDARY, WITH THE SAME THREE RULES AS THE SEARCH, and one consequence that
is specific to descending.

1. THE CAP IS APPLIED AFTER SCOPING. The children are produced from the scoped
   frame and only then counted and cut, so ``omitted`` is never a rival count.
2. NO-MATCH AND NOT-OURS ARE INDISTINGUISHABLE, and here that is sharper than in
   the search: descending into an object that is not ours must answer exactly
   what descending into an object that does not exist answers. So this route
   ECHOES NOTHING -- not the parent id, not the parent label, not the edge that
   was refused. Every empty descent returns the same bytes, whatever was asked
   for. A reason that named the parent would confirm the parent exists.
3. NO EDGE IS TRAVERSABLE WHOSE CHILD STORE CANNOT BE SCOPED. The two plan-spine
   edges read ``assistant_context._owned_frame()``, which is the scope itself.
   The agency-to-advertiser edge reads two operator-owned rule stores with no
   channel column in either.

HONEST EMPTIES. A container with no children returns a STATED absence and never
a bare empty list, because an empty list in a picker reads as "zero of them" and
the two are different claims. What it may not do is state a reason that
distinguishes the cases above, so the stated reason names the EDGE and not the
parent: there is nothing of this kind under a container of that kind here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from kairos_api import assistant_mentions as index
from kairos_api import assistant_mentions_words as words

router = APIRouter(tags=["assistant"])

# The same two numbers the search uses, for the same reason: the popup is the
# same popup and the drill is a mode of it rather than a second surface.
SHOW_CAP = index.SHOW_CAP
FETCH_CAP = index.FETCH_CAP
ID_MAX = 200
# A programme reached through a day needs that day only for its next edge. Its
# public mention id stays the programme's store key, so accepting it still
# resolves exactly as a programme mention. This separate drill id preserves the
# graph context without corrupting the typed reference.
DRILL_SEPARATOR = "␟"


# --- the edges ---------------------------------------------------------------------
def _owned() -> tuple[Any, str]:
    """The scoped plan frame and the operator's own channel, or (None, "")."""
    from kairos_api import assistant_context

    frame, owned, _competitors, _reason = assistant_context._owned_frame()
    return frame, owned


def _program_column(frame: Any) -> str:
    return "program_title" if "program_title" in frame.columns else "program_type"


def _day_to_program(parent_id: str) -> list[dict[str, Any]]:
    """The programmes that run on one broadcast day of the operator's channel.

    The dim line is the day's own earliest start for that programme plus how
    many of the day's segments it holds, because within a single day the
    disambiguator is the clock and not the date.
    """
    frame, owned = _owned()
    if frame is None:
        return []
    matched = frame[frame["date_text"].astype(str) == parent_id]
    if matched.empty:
        return []
    column = _program_column(frame)
    rows: list[dict[str, Any]] = []
    for title, group in matched.groupby(matched[column].astype(str).str.strip(), sort=False):
        name = str(title).strip()
        if not name or name.lower() == "nan":
            continue
        starts = sorted(value for value in group.get("start_norm", []) if str(value).strip())
        parent: list[dict[str, Any]] = []
        if starts:
            parent.append({"kind": "figure", "text": str(starts[0])})
        parent.append({"kind": "figure", "text": str(int(len(group)))})
        parent.append({"kind": "name", "text": owned})
        row = index._row("program", name, name, parent)
        row["drill_id"] = f"{parent_id}{DRILL_SEPARATOR}{name}"
        row["drill_edge"] = "break"
        rows.append(row)
    return rows


def _program_to_day(parent_id: str) -> list[dict[str, Any]]:
    """The broadcast days one programme runs on, on the operator's channel.

    This is the ascending half of the same edge and it reads the same frame. A
    programme value that exists only on a rival channel matches nothing here,
    which is not a special case: the frame it is matched against never held a
    rival row to begin with.
    """
    frame, owned = _owned()
    if frame is None:
        return []
    column = _program_column(frame)
    matched = frame[frame[column].astype(str).str.strip() == parent_id]
    if matched.empty:
        return []
    rows: list[dict[str, Any]] = []
    for date_text, group in matched.groupby(matched["date_text"].astype(str), sort=True):
        iso = str(date_text).strip()
        if not iso or iso.lower() == "nan":
            continue
        parent = [{"kind": "name", "text": owned},
                  {"kind": "figure", "text": str(int(len(group)))}]
        rows.append(index._row("day", iso, iso, parent))
    return rows


def _program_to_break(parent_id: str) -> list[dict[str, Any]]:
    """The real breaks inside one programme on one owned broadcast day.

    The day context comes from the row produced by ``_day_to_program``. A flat
    programme search still descends to its days first, because a recurring
    programme has no single set of breaks until a day is selected.
    """
    day, separator, programme = str(parent_id or "").partition(DRILL_SEPARATOR)
    if not separator or not day or not programme:
        return []
    from kairos_api import break_store
    from kairos_api.break_api_detail import _clock

    frame, _owned_channel = _owned()
    if frame is None:
        return []
    match_field = "programme" if _program_column(frame) == "program_title" else "genre"
    try:
        records = break_store.break_records(break_store.day_plan(day))
    except LookupError:
        return []
    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get(match_field) or "").strip() != programme:
            continue
        label = _clock(float(record.get("start_seconds") or 0))
        parent = [
            {"kind": "span", "from": day, "to": day},
            {"kind": "name", "text": programme},
            {"kind": "figure", "text": str(record.get("ordinal") or "")},
        ]
        rows.append(index._row("break", str(record.get("break_id") or ""), label, parent))
    return rows


def _agency_to_advertiser(parent_id: str) -> list[dict[str, Any]]:
    """The advertisers effectively linked to one agency.

    ``links_for`` already resolves the three-way question of observed, stored
    and manual links and hands back the effective set, so this route asks it
    rather than re-deriving a fourth answer to the same question. A linked name
    with no row in the advertiser rules store is dropped rather than invented
    into one: a mention needs a typed identifier the resolver can read back, and
    a bare string from a link file is not that.
    """
    from kairos_api import agency_conditions

    linked = {str(name).strip() for name in agency_conditions.links_for(parent_id)["effective"]}
    if not linked:
        return []
    return [row for row in index.build_index()
            if row["type"] == "advertiser" and (row["id"] in linked or row["label"] in linked)]


_EDGE_BUILDERS = {
    ("day", "program"): _day_to_program,
    ("program", "day"): _program_to_day,
    ("program", "break"): _program_to_break,
    ("agency", "advertiser"): _agency_to_advertiser,
}


# --- the route ---------------------------------------------------------------------
def _absent(edge: tuple[str, str] | None) -> dict[str, Any]:
    """The one empty answer, and it says the same thing every time.

    It names the EDGE that was asked for and never the container that was asked
    about, so descending into a rival's programme, into a programme that was
    deleted this morning, and into an outright typo all return these exact
    bytes. An unknown edge returns them too: refusing an edge by name would
    disclose which edges exist for kinds the caller has no row of.
    """
    child = edge[1] if edge else ""
    kind_he = words.label(child, "he") if child else ""
    return {
        "rows": [],
        "count": 0,
        "omitted": 0,
        "limit": SHOW_CAP,
        "child_type": child or None,
        "absent": {
            "reason_code": "no_children",
            "reason": words.ABSENT_REASON_EN + (f" ({child})" if child else ""),
            # A STATED absence in the operator's own language, because an empty
            # list in a picker reads as "zero of them" and the two are different
            # claims. Both sentences name the KIND and never the container, so
            # they are the same bytes for a rival's object, a deleted one and an
            # outright typo.
            "reason_he": words.ABSENT_REASON_HE + (f" ({kind_he})" if kind_he else ""),
        },
    }


def children(type: str = "", id: str = "", edge: str = "",
             limit: int = SHOW_CAP) -> dict[str, Any]:
    """One step down the ladder, already scoped, capped after scoping.

    ``edge`` names the child kind. Absent, it takes the first edge the taxonomy
    declares for the parent kind, which is the ladder's own order rather than a
    guess made here.
    """
    parent_kind = str(type or "").strip()
    parent_id = str(id or "").strip()[:ID_MAX]
    wanted = str(edge or "").strip()
    shown = max(1, min(int(limit or SHOW_CAP), FETCH_CAP))

    available = words.child_kinds(parent_kind)
    child_kind = wanted if wanted in available else (available[0] if available and not wanted else "")
    key = (parent_kind, child_kind)
    build = _EDGE_BUILDERS.get(key)
    if build is None or not parent_id:
        return _absent(key if child_kind else None)

    try:
        rows = build(parent_id)
    except Exception:  # noqa: BLE001 - an unreadable store is an absent edge, never an invented row
        rows = []
    if not rows:
        return _absent(key)

    rows.sort(key=lambda row: (words.rank(row["type"]), row["label"]))
    shown_rows = [index._public(row) for row in rows[:shown]]
    return {
        "rows": shown_rows,
        "count": len(shown_rows),
        # After scoping, after building, never before either.
        "omitted": max(0, len(rows) - len(shown_rows)),
        "limit": shown,
        "child_type": child_kind,
        "absent": None,
    }


@router.get("/mentions/children")
def get_mention_children(type: str = "", id: str = "", edge: str = "",
                         limit: int = SHOW_CAP) -> dict[str, Any]:
    """The children of one container, for the picker's drill mode.

    Advisory in exactly the sense the search route is advisory: it adds a way of
    reaching an object and takes none away, and a client that never calls it
    behaves precisely as the console did before drill mode existed.
    """
    return children(type=type, id=id, edge=edge, limit=limit)
