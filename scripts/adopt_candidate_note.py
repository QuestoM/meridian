"""The release note: the one training-authored sentence that crosses the line.

Section 4.6 of the specification makes this the single exception to the wall
this piece exists to hold. Everything else a model steward writes stays on the
training side. The release note is written for the operator, in plain language,
and it is the only reason a plan that moves money afterwards has a legible
cause. So it is the most load-bearing string on this surface and it was the
least visible one.

**Measured before this module existed.** ``models/releases/decisions.jsonl``
holds seven records and exactly one of them carries a note: a ship verdict on
``afterwindow``, 92 characters of Hebrew written by a steward. That sentence
appeared zero times at this terminal, zero times in the published board and
zero times in ``checks``. What both surfaces carried instead was a boolean.

**The falsehood underneath it.** ``decide`` answered one thing about a note,
that it is not empty, and the store answers a second thing at the moment of the
write: that it carries no gate verdict, no p-value and no coefficient. Those two
run at different times, so a plan run printed **every check passed and nothing
has been written, add --perform to record it** for a note the store then
refused. Reproduced before this module, on ``competitor`` with a no-ship verdict
and an English note reading "the retention gate cleared at p=0.004": five checks
passed, the plan reported ready, and ``--perform`` came back with the store's
Hebrew refusal naming ``gate`` and ``p=``. The check runs here now, on both
languages, over the store's own guard rather than a second copy of its word
list, so the two can never disagree about one sentence.

**The second measurement, and it is about the other side of the wall.** A note
is written for an operator and no operator reads one today. The operator's own
disclosure resolves its release note through ``current_release_note`` on the
version store, that attribute does not exist, and the disclosure therefore
answers ``unknown`` with ``models/releases/`` as the path that would supply it.
So the chip a steward reads beside a verdict, "carries a release note for the
operator side", is a promise about a reader that is not there. It is a
measurement here now, tri-state, and it names what would carry one.

**What is not here.** The word list itself. ``check_release_note`` is the
store's own guard, called and never reimplemented, on the same argument the
provenance measurement uses for the freshness guard: a second copy of a rule is
a second rule, and the one that matters is the one that runs at the write.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# What a note is, as a state rather than as a boolean. A verdict that needs no
# note and a verdict missing the one it needs are two different facts, and a
# boolean renders both as false.
NOTE_STATE: dict[str, dict[str, str]] = {
    "written": {
        "en": "This verdict carries the sentence the operator side reads.",
        "he": "ההכרעה הזו נושאת את המשפט שהצד התפעולי קורא.",
    },
    "absent": {
        "en": "This verdict carries no release note, and a no-ship verdict is not required to carry one.",
        "he": "ההכרעה הזו אינה נושאת הערת גרסה, והכרעה שלא להשיק אינה מחויבת לשאת אחת.",
    },
    "missing": {
        "en": "This verdict ships a model and carries no release note, which is the one sentence an operator would read about the change.",
        "he": "ההכרעה הזו משיקה מודל ואינה נושאת הערת גרסה, שהיא המשפט היחיד שמפעיל היה קורא על השינוי.",
    },
    "withheld": {
        "en": "The note on record names the training act itself, so it is not carried into a file a browser downloads. It is read at the terminal and on the model versions screen.",
        "he": "ההערה שברישום נוקבת בשם פעולת האימון עצמה, ולכן היא אינה נישאת אל קובץ שדפדפן מוריד. היא נקראת במסוף ובמסך גרסאות מודל.",
    },
}

# What the store's own guard says about a sentence, before the write rather
# than during it. The refusal itself is the store's and travels verbatim.
CROSSING: dict[str, dict[str, str]] = {
    "clean": {
        "en": "The store accepts this note: it carries no gate verdict, no p-value and no coefficient.",
        "he": "החנות מקבלת את ההערה הזו: אין בה הכרעת שער, אין ערך מובהקות ואין מקדם.",
    },
    "refused": {
        "en": "The store refuses this note. Its refusal, verbatim: ",
        "he": "החנות מסרבת להערה הזו. הסירוב שלה, כלשונו: ",
    },
    "absent": {
        "en": "There is no note to check.",
        "he": "אין הערה לבדוק.",
    },
}

# Whether a note reaches the side it was written for. Measured against the
# store rather than assumed from the fact that one is on record.
REACH: dict[str, dict[str, str]] = {
    "no_reader": {
        "en": "No operator surface reads a release note today. The version store carries no reader for the note of the version in force, so the operator side answers that it does not know what changed.",
        "he": "שום מסך תפעולי אינו קורא היום הערת גרסה. מאגר הגרסאות אינו נושא קורא להערה של הגרסה שבתוקף, ולכן הצד התפעולי משיב שאינו יודע מה השתנה.",
    },
    "none_recorded": {
        "en": "The operator side can read a release note and none is recorded for the model version in force.",
        "he": "הצד התפעולי יכול לקרוא הערת גרסה ואף אחת אינה רשומה לגרסת המודל שבתוקף.",
    },
    "published": {
        "en": "The operator side reads this sentence about the model version in force.",
        "he": "הצד התפעולי קורא את המשפט הזה על גרסת המודל שבתוקף.",
    },
    "unavailable": {
        "en": "The version store holds a reader for the operator's release note and it could not be read.",
        "he": "מאגר הגרסאות נושא קורא להערת הגרסה של המפעיל ולא ניתן היה לקרוא אותו.",
    },
}

# What would close the gap, named where the gap is stated and never as an act
# this terminal can run.
REACH_UNBLOCKED: dict[str, str] = {
    "en": "A reader on the version store that answers with the note of the version in force. The operator side already asks for one by name and already says models/releases/ is where it would come from.",
    "he": "קורא במאגר הגרסאות שמשיב בהערה של הגרסה שבתוקף. הצד התפעולי כבר מבקש אותו בשמו וכבר אומר שהמקור שלו הוא תיקיית הגרסאות.",
}

# The rule itself, stated wherever a note is shown, because a steward writing
# one needs the boundary and not only the refusal that follows breaking it.
CROSSING_RULE: dict[str, str] = {
    "en": "A release note is plain language for the operator side. It names what changed and in which direction, and it carries no gate verdict, no p-value and no coefficient.",
    "he": "הערת גרסה היא שפה פשוטה לצד התפעולי. היא נוקבת במה שהשתנה ובאיזה כיוון, ואינה נושאת הכרעת שער, ערך מובהקות או מקדם.",
}

# Where a steward reads the note that is on a record, rather than a second copy
# of it living in a bundled file.
NOTE_LIVES: dict[str, str] = {
    "en": "The note on a verdict is stored with the record and is read here and on the model versions screen of this console.",
    "he": "ההערה שעל הכרעה נשמרת עם הרישום ונקראת כאן ובמסך גרסאות מודל של המסוף הזה.",
}


def _store():
    """The version store, behind a seam so a test can stand one up.

    The same seam the history reading uses, and for the same reason: this piece
    has exactly one place where it asks another piece what it holds.
    """
    from kairos_api import model_version_store as store

    return store


def crossing(text: str) -> dict[str, Any]:
    """What the store's own guard says about one sentence, without writing it.

    ``check_release_note`` raises on a note it will not take, so the raise is
    caught and turned into a state. The refusal travels verbatim in both
    halves, because it is the store's sentence and naming the words a second
    time here would be a second word list.
    """
    body = str(text or "").strip()
    if not body:
        return dict(state="absent", refusal="",
                    reading_en=CROSSING["absent"]["en"], reading_he=CROSSING["absent"]["he"])
    store = _store()
    try:
        store.check_release_note(body)
    except store.ModelVersionError as refusal:
        return dict(state="refused", refusal=str(refusal),
                    reading_en=CROSSING["refused"]["en"] + str(refusal),
                    reading_he=CROSSING["refused"]["he"] + str(refusal))
    return dict(state="clean", refusal="",
                reading_en=CROSSING["clean"]["en"], reading_he=CROSSING["clean"]["he"])


def note_state(text: str, *, ships: bool) -> str:
    if str(text or "").strip():
        return "written"
    return "missing" if ships else "absent"


def note_block(note_he: str, note_en: str, *, ships: bool) -> dict[str, Any]:
    """One verdict's note, both languages, with what the store makes of each.

    Both halves are checked. The English one was checked by nothing here and by
    the store at the write, which is how a plan came to report ready for a
    record the store then refused.
    """
    state = note_state(note_he, ships=ships)
    return {
        "he": str(note_he or "").strip(),
        "en": str(note_en or "").strip(),
        "state": state,
        "reading_en": NOTE_STATE[state]["en"],
        "reading_he": NOTE_STATE[state]["he"],
        "crossing_he": crossing(note_he),
        "crossing_en": crossing(note_en),
        "rule_en": CROSSING_RULE["en"],
        "rule_he": CROSSING_RULE["he"],
    }


def refused_halves(block: dict[str, Any]) -> list[str]:
    """Which language of a note the store will not take, named rather than counted."""
    return [half for half in ("he", "en")
            if (block.get(f"crossing_{half}") or {}).get("state") == "refused"]


def verdict_checks(decision: str, note_he: str, note_en: str) -> list[dict[str, Any]]:
    """The condition a verdict plan was answering only at the moment of the write.

    One check, on both languages at once, because a steward reads a list of
    conditions and not a list of fields. It is appended for every verdict rather
    than only for a ship, since the store checks a no-ship verdict's note too
    and that is the case the falsehood was reproduced on.
    """
    block = note_block(note_he, note_en, ships=decision == "shipped")
    refused = refused_halves(block)
    if not refused:
        return [{
            "id": "release_note_crossing",
            "passed": True,
            "reason_en": CROSSING_RULE["en"] if block["state"] != "written"
            else block["crossing_he"]["reading_en"],
            "reason_he": CROSSING_RULE["he"] if block["state"] != "written"
            else block["crossing_he"]["reading_he"],
            "how_en": "", "how_he": "",
        }]
    named = " and ".join("Hebrew" if half == "he" else "English" for half in refused)
    detail = (block[f"crossing_{refused[0]}"]).get("refusal") or ""
    return [{
        "id": "release_note_crossing",
        "passed": False,
        "reason_en": f"The store will refuse this record for the {named} note. Its refusal, verbatim: {detail}",
        "reason_he": f"החנות תסרב לרישום הזה בגלל ההערה. הסירוב שלה, כלשונו: {detail}",
        "how_en": CROSSING_RULE["en"],
        "how_he": CROSSING_RULE["he"],
    }]


def operator_reads() -> dict[str, Any]:
    """Whether a note reaches the side it was written for, measured, tri-state.

    Read off the version store directly rather than through the operator's own
    disclosure, which costs nearly nine seconds because it loads the audience
    model on the way. A test holds the two equal in every state, so the cheap
    reading here and the sentence an operator is actually served cannot come
    apart. The same trade the provenance measurement makes with the freshness
    guard, and for the same measured reason.
    """
    reader = getattr(_store(), "current_release_note", None)
    if not callable(reader):
        return _reach("no_reader", "")
    try:
        text = str(reader() or "").strip()
    except Exception:  # noqa: BLE001 - an unreadable store is a state, never a guess
        return _reach("unavailable", "")
    return _reach("published", text) if text else _reach("none_recorded", "")


def _reach(state: str, text: str) -> dict[str, Any]:
    return {
        "state": state,
        "text": text,
        "reading_en": REACH[state]["en"],
        "reading_he": REACH[state]["he"],
        "supplied_by": "models/releases/",
        "unblocked_by_en": REACH_UNBLOCKED["en"] if state == "no_reader" else "",
        "unblocked_by_he": REACH_UNBLOCKED["he"] if state == "no_reader" else "",
    }


def withheld(block: dict[str, Any]) -> dict[str, Any]:
    """The same block with the sentence taken off it and the absence named.

    Called by the publisher when a note names the training act. The published
    board is imported by a browser bundle and the wall on this act is a route
    wall, so a sentence that names the act may not travel in one; a publish that
    died on a steward's own words would be worse than either.
    """
    return dict(block, he="", en="", state="withheld",
                reading_en=NOTE_STATE["withheld"]["en"],
                reading_he=NOTE_STATE["withheld"]["he"])


def for_the_board(block: dict[str, Any],
                  offends: Optional[Callable[[str], list[str]]] = None) -> dict[str, Any]:
    """One note as a published file may carry it, scrubbed by the caller's guard."""
    block = block or {}
    text = f"{block.get('he') or ''} {block.get('en') or ''}"
    return withheld(block) if offends and offends(text) else dict(block)


def render_operator_reads(reach: dict[str, Any]) -> list[str]:
    """What the other side of the wall is reading right now, beside the version.

    Printed under the live model version because that is its subject. A note is
    written for an operator, so whether an operator reads one is a fact about
    this act and not a footnote about the store.
    """
    if not reach:
        return []
    lines = [f"  what the operator side reads about this version: {reach['reading_en']}"]
    if reach.get("text"):
        lines.append(f"    its own words: {reach['text']}")
    if reach.get("unblocked_by_en"):
        lines.append(f"    supplied by: {reach['unblocked_by_en']}")
    return lines


def render_note(block: dict[str, Any], indent: int) -> list[str]:
    """The sentence itself, at the terminal that authors it and reads it back.

    Every surface of this piece carried a boolean where this sentence was, on
    the one string the specification lets across the line.
    """
    if not block:
        return []
    pad = " " * indent
    if block["state"] != "written":
        return [f"{pad}release note: {block['reading_en']}"]
    lines = [f"{pad}release note, what the operator side reads: {block['he']}"]
    if block.get("en"):
        lines.append(f"{pad}  the same in English, carried on the record: {block['en']}")
    for half in refused_halves(block):
        lines.append(f"{pad}  {(block[f'crossing_{half}']).get('reading_en')}")
    return lines
