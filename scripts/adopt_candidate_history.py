"""Every verdict on record, in the order it was taken, and the ones the join dropped.

JS-19 ends with "the verdict is stored and a later reader can see what was
tried". The first half was built in round 2 and the second half was not, because
every read this piece makes of the decision log keeps one record and discards
the rest.

**Measured before this module existed.** ``models/releases/decisions.jsonl``
holds seven records. The registry filters them to ``subject == "candidate"`` and
keeps ``taken[0]``, so five reach the published board as a date on a row and two
reach no surface of this piece at all: the earlier of the two verdicts on
``calibrated``, and the one verdict whose subject is the live model itself. The
terminal is worse than the board: it prints the word and a count in brackets and
**no date and no actor for any verdict at all**, on the surface where the next
verdict is taken.

**The reading a count cannot give.** ``calibrated`` carries two verdicts, both
``not_shipped``, taken twenty-three minutes apart. A column reading "no ship (2)"
says it was refused twice. What actually happened is that the first refusal was
for want of a current measurement and the second was on the measurement, which
are two different kinds of no, and only the second is a verdict about the model.
So the count is split three ways here: a word that changed between verdicts is a
reversal, the same word with the same stated reason is a repeat, and the same
word with a different stated reason is a restatement. The state is measured from
the records rather than named by hand.

**What the live model carries.** A decision record may be about the shipped
model rather than about a candidate, and one on this tree is: a no-ship recorded
against the version in force. Every read in this piece dropped it at the filter,
so the shelf showed five candidates each with a verdict and said nothing about a
standing verdict on the artifact all five are measured against.

**The steward's own sentence does not travel to the browser.** The reason on a
record is unbounded text written at a terminal. It is rendered here, at that
terminal, and the model console renders it from the store; it is not copied into
the published board, for the same reason the board carries no command. What the
board carries is that the sentence exists and where it is read.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from scripts import adopt_candidate_note as note
from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_state import decision_rests_on_rescore

# What a list of verdicts on one artifact amounts to, chosen by measuring the
# records against each other rather than by counting them.
HISTORY_READING: dict[str, dict[str, str]] = {
    "none": {
        "en": "No verdict has been recorded on this artifact.",
        "he": "לא נרשמה שום הכרעה על הקובץ הזה.",
    },
    "one": {
        "en": "One verdict on record.",
        "he": "הכרעה אחת רשומה.",
    },
    "repeated": {
        "en": "{count} verdicts on record and every one of them is the same word for the same stated reason.",
        "he": "{count} הכרעות רשומות וכל אחת מהן היא אותה מילה מאותו נימוק רשום.",
    },
    "restated": {
        "en": "{count} verdicts on record, all of them the same word, each for a different stated reason. The newest is the one in force and the earlier ones are what was tried before it.",
        "he": "{count} הכרעות רשומות, כולן אותה מילה, כל אחת מנימוק רשום אחר. החדשה היא זו שבתוקף והקודמות הן מה שנוסה לפניה.",
    },
    "reversed": {
        "en": "{count} verdicts on record and they do not all say the same thing, so this artifact was decided one way and then the other.",
        "he": "{count} הכרעות רשומות והן אינן אומרות את אותו הדבר, ולכן הקובץ הזה הוכרע לכיוון אחד ואז לכיוון האחר.",
    },
}

# What is on record about the live artifact itself, which is a different subject
# from every candidate row and was filtered out of all of them.
LIVE_MODEL_READING: dict[str, dict[str, str]] = {
    "none": {
        "en": "No verdict has been recorded on the shipped model itself.",
        "he": "לא נרשמה שום הכרעה על המודל המשודר עצמו.",
    },
    "standing": {
        "en": "A verdict is on record on the shipped model itself, against the model version in force, and every candidate below is measured against that same version.",
        "he": "רשומה הכרעה על המודל המשודר עצמו, מול גרסת המודל שבתוקף, וכל מועמד למטה נמדד מול אותה גרסה בדיוק.",
    },
    "superseded": {
        "en": "Every verdict on record on the shipped model itself was taken against an earlier model version, so none of them is about the artifact in force.",
        "he": "כל ההכרעות הרשומות על המודל המשודר עצמו התקבלו מול גרסת מודל קודמת, ולכן אף אחת מהן אינה עוסקת בקובץ שבתוקף.",
    },
}

# Where the sentence itself is read. The console renders every record with the
# words its steward wrote, and this piece names that screen in the console's own
# words rather than in a second set of its own.
REASON_LIVES: dict[str, str] = {
    "en": "The sentence each verdict was taken for is stored with the record and is read on the model versions screen of this console. It is not copied into this file.",
    "he": "המשפט שבגללו התקבלה כל הכרעה נשמר עם הרשומה ונקרא במסך גרסאות מודל של המסוף הזה. הוא אינו מועתק לקובץ הזה.",
}

# The console's own name for a record whose subject is the shipped model, from
# console-words.js:194. Two surfaces naming one subject two ways is a divergence
# a steward walks into inside one session.
LIVE_SUBJECT: dict[str, str] = {"en": "the shipped model", "he": "המודל המשודר"}


def _decisions() -> list[dict[str, Any]]:
    """The whole append-only log, newest first, through P7's own reader.

    A seam rather than a direct call, so a test can stand a log up without a
    store and so this piece has exactly one place where it asks another piece
    what has been decided.
    """
    from kairos_api import model_version_store as store

    return store.decisions()


def _row(record: dict[str, Any], *, version_id: str, in_force: bool,
         superseded_by: Optional[str]) -> dict[str, Any]:
    """One record as this piece reads it, with the reason still on it.

    ``decision``, ``recorded_at`` and ``actor`` keep the store's own key names,
    because three callers already read a raw record under those names and a
    renamed key would be a second vocabulary for one fact.
    """
    state = str(record.get("decision") or "")
    stated = str(record.get("reason") or "").strip()
    return {
        "decision_id": record.get("decision_id"),
        "recorded_at": record.get("recorded_at"),
        "actor": record.get("actor"),
        "decision": state,
        "decision_en": (words.DECISION_WORDS.get(state) or {}).get("en", state),
        "decision_he": (words.DECISION_WORDS.get(state) or {}).get("he", state),
        "model_version_id": record.get("model_version_id"),
        "model_version_name": record.get("model_version_name"),
        # A verdict recorded against an earlier version is not a verdict about
        # the artifact in force. The adoption act has always matched the version
        # before honouring a ship verdict and the registry column never did, so
        # the column could show a verdict about a model that is no longer there.
        "against_version_in_force": bool(version_id)
        and str(record.get("model_version_id") or "") == version_id,
        "on_rescore": decision_rests_on_rescore(record),
        "in_force": in_force,
        "superseded_by": superseded_by,
        # The sentence the operator side was given, rather than whether one
        # exists. This was a boolean, on the one string the specification lets
        # across the line: measured on this tree, exactly one record carries a
        # note, and the 92 characters a steward wrote reached no surface of this
        # piece at all while a chip beside it said a note had been carried.
        "note": note.note_block(record.get("release_note_he"),
                                record.get("release_note_en"),
                                ships=state == "shipped"),
        "money_direction": record.get("money_direction"),
        "reason": stated,
    }


def _reading(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    if len(rows) == 1:
        return "one"
    if len({row["decision"] for row in rows}) > 1:
        return "reversed"
    return "repeated" if len({row["reason"] for row in rows}) == 1 else "restated"


def _block(rows: list[dict[str, Any]], table: dict[str, dict[str, str]],
           state: str) -> dict[str, Any]:
    reading = table[state]
    return {
        "rows": rows,
        "count": len(rows),
        "state": state,
        # How many of these a surface showing only the newest would not show. It
        # is the figure this whole module exists for and it is never inferred
        # from the count by a reader.
        "not_shown_by_the_latest": max(len(rows) - 1, 0),
        "reading_en": reading["en"].format(count=len(rows)),
        "reading_he": reading["he"].format(count=len(rows)),
    }


def _rows_for(records: list[dict[str, Any]], version_id: str) -> list[dict[str, Any]]:
    """The records of one subject, newest first, marked with what supersedes what."""
    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        out.append(_row(record, version_id=version_id, in_force=index == 0,
                        superseded_by=str(records[index - 1].get("decision_id") or "")
                        if index else None))
    return out


def decision_log(known_ids: Optional[Iterable[str]] = None, *, version_id: str = "",
                 version_name: str = "",
                 records: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The whole log, grouped by what each record is about.

    One reader for one file. The registry used to walk the log itself for the
    latest verdict per candidate while this reading walked it again, and two
    readers of one append-only log is how two surfaces come to disagree about
    what was decided.
    """
    log = list(records if records is not None else _decisions())
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    live: list[dict[str, Any]] = []
    for record in log:
        if record.get("subject") == "candidate":
            by_candidate.setdefault(str(record.get("candidate_id") or ""), []).append(record)
        else:
            live.append(record)
    known = set(known_ids or by_candidate)
    candidates: dict[str, dict[str, Any]] = {}
    for identifier, records_for_it in by_candidate.items():
        rows = _rows_for(records_for_it, version_id)
        candidates[identifier] = _block(rows, HISTORY_READING, _reading(rows))
    for identifier in known:
        candidates.setdefault(identifier, _block([], HISTORY_READING, "none"))
    live_rows = _rows_for(live, version_id)
    live_state = ("none" if not live_rows else
                  "standing" if any(row["against_version_in_force"] for row in live_rows)
                  else "superseded")
    every_row = [row for block in candidates.values() for row in block["rows"]] + live_rows
    return {
        "records": len(log),
        "version_id": version_id,
        "version_name": version_name,
        "candidates": candidates,
        # The subject every read on this piece filtered out. It is not a
        # candidate row and it is not a footnote either: it is a recorded verdict
        # about the artifact the whole table is measured against.
        "live_model": _block(live_rows, LIVE_MODEL_READING, live_state),
        # Verdicts about a candidate whose file is not on this shelf. Measured on
        # this tree: none. It is reported rather than assumed, because a log is
        # append-only and a candidate file is not.
        "off_the_shelf": sorted(set(by_candidate) - known),
        # Where every record in the log went, so a reader can add it up. A
        # surface that shows some of an append-only log and says nothing about
        # the rest is the defect this module was written for.
        "tally": {
            "on_the_shelf": sum(len(candidates[identifier]["rows"])
                                for identifier in sorted(known & set(candidates))),
            "on_the_live_model": len(live_rows),
            "off_the_shelf": sum(len(candidates[identifier]["rows"])
                                 for identifier in sorted(set(by_candidate) - known)),
        },
        # Whether any verdict on record was taken against a model version that is
        # no longer in force. Measured rather than assumed: a tree where this is
        # false carries verdicts about a model that is gone, and the column that
        # shows them says nothing about it.
        "against_another_version": [row["decision_id"] for row in every_row
                                    if not row["against_version_in_force"]],
        "all_against_version_in_force": all(
            row["against_version_in_force"] for row in every_row),
    }


def history_for(log: dict[str, Any], identifier: str) -> dict[str, Any]:
    """One artifact's verdicts, or the empty state, never a missing key."""
    return (log.get("candidates") or {}).get(identifier) or _block([], HISTORY_READING, "none")


def _public(row: dict[str, Any],
            offends: Optional[Callable[[str], list[str]]]) -> dict[str, Any]:
    row = {key: value for key, value in row.items() if key != "reason"}
    return dict(row, note=note.for_the_board(row.get("note") or {}, offends))


def for_the_board(block: dict[str, Any],
                  offends: Optional[Callable[[str], list[str]]] = None) -> dict[str, Any]:
    """The same block with the steward's own sentence taken off it, and the
    operator's sentence kept.

    The two are not the same kind of text and this is the line the whole piece
    is about. ``reason`` is written for this side of the wall: unbounded, in
    whatever language the steward types, already rendered by the console from
    the store, so a copy of it in a bundled file is a second source that can
    disagree with the first. The release note is written FOR the other side, it
    is the one training-authored sentence section 4.6 lets cross, and a steward
    choosing between five candidates cannot read what the last shipment told an
    operator unless it is here.

    The one sentence that still does not travel is one that names this act, and
    the caller's own guard decides that, so a note is scrubbed rather than a
    publish dying on a steward's own words.
    """
    block = block or {}
    return dict(block, rows=[_public(row, offends) for row in block.get("rows") or []],
                reason_en=REASON_LIVES["en"], reason_he=REASON_LIVES["he"],
                note_en=note.NOTE_LIVES["en"], note_he=note.NOTE_LIVES["he"])


def for_the_board_live(block: dict[str, Any],
                       offends: Optional[Callable[[str], list[str]]] = None) -> dict[str, Any]:
    """The same, plus the console's own name for the subject of these records.

    A candidate's history is about the artifact whose name is beside it. These
    are about the live model, which is a subject and not a row, so the block
    carries the word for it rather than a screen inventing a second one.
    """
    return dict(for_the_board(block, offends),
                subject_en=LIVE_SUBJECT["en"], subject_he=LIVE_SUBJECT["he"])


def render_live_model(log: dict[str, Any]) -> list[str]:
    """What is on record about the live artifact, beside the live artifact.

    Printed under the version block rather than under the candidates, because
    its subject is that version. On this tree it is a no-ship recorded against
    the version in force, and before this it was on no surface of this piece.
    """
    block = (log or {}).get("live_model") or {}
    if not block.get("rows"):
        return []
    lines = [f"  verdicts on {LIVE_SUBJECT['en']} itself: {block['reading_en']}"]
    for row in block["rows"]:
        lines.extend(_render_row(row, indent=4))
    return lines


def _render_row(row: dict[str, Any], indent: int) -> list[str]:
    pad = " " * indent
    marks = [row["decision_en"]]
    marks.append("in force" if row["in_force"] else f"superseded by {row['superseded_by']}")
    marks.append("on this comparison" if row["on_rescore"] else "before this comparison existed")
    if not row["against_version_in_force"]:
        marks.append(f"taken against model version {row['model_version_name']}, not the one in force")
    lines = [f"{pad}{words.when(row['recorded_at'])}  by {row['actor'] or 'not recorded'}  {', '.join(marks)}"]
    if row["reason"]:
        lines.append(f"{pad}  its own words: {row['reason']}")
    # The note itself rather than a mark saying one exists. It is the only
    # sentence on this record written for anybody outside this room. A no-ship
    # verdict that carries none says nothing here rather than saying so on every
    # row, because six of the seven records on this tree are exactly that and a
    # line repeated six times is what buries the one line that matters.
    if (row.get("note") or {}).get("state") != "absent":
        lines.extend(note.render_note(row["note"], indent=indent + 2))
    return lines


def render_history(payload: dict[str, Any]) -> list[str]:
    """Every verdict on every candidate, in the order it was taken.

    The table three blocks up prints one word and a count in brackets. It cannot
    say when any of them was taken, by whom, on what, or that two of them say the
    same word for two different reasons, and on this tree all four of those are
    facts about the shelf.
    """
    rows = [row for row in payload.get("candidates") or [] if (row.get("history") or {}).get("rows")]
    if not rows:
        return []
    lines = ["Verdicts on record, in the order they were taken"]
    for row in rows:
        history = row["history"]
        lines.append(f"  {row['id']:20s} {history['reading_en']}")
        for record in history["rows"]:
            lines.extend(_render_row(record, indent=4))
    lines.append("")
    log = payload.get("decision_log") or {}
    tally = log.get("tally") or {}
    if log.get("off_the_shelf"):
        lines.append(f"  verdicts on record about artifacts that are not on this shelf: {', '.join(log['off_the_shelf'])}")
    lines.append(f"  {log.get('records', 0)} records in the decision log: {tally.get('on_the_shelf', 0)} on the artifacts above, {tally.get('on_the_live_model', 0)} on {LIVE_SUBJECT['en']} itself, {tally.get('off_the_shelf', 0)} on artifacts that are not on this shelf")
    if not log.get("all_against_version_in_force"):
        lines.append(f"  taken against a model version that is not the one in force: {', '.join(log.get('against_another_version') or [])}")
    lines.append("")
    return lines
