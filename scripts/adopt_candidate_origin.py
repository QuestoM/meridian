"""What each artifact was built for, and what data it was built from.

**The absence this module closes, measured before it was written.** Every
candidate's own metadata may carry a one-line ``purpose``, and no surface this
piece owns carried it. Measured on this tree: ``afterwindow`` records
``after-window de-bias verification recompute (post-clip code, reference data)``
and that sentence appeared zero times in the published board, zero times in
``show``, zero times in ``checks`` and zero times in ``diff``. The word
``purpose`` appeared once on the whole surface, as one of forty-three metadata
key NAMES in the list of keys an adoption would add. So a steward opened a shelf
of five opaque identifiers and inferred what each one was trying to do from its
coefficient table.

The reference this piece is measured against puts it first: an experiment
tracker's run overview leads with the notes, then the exact checkout and the
command that produced the run. Two of those three are answerable from this tree
and the third is not, and the difference is stated rather than papered over.

**Purpose is a stored value, not authored copy.** It is the producer's own
sentence in the producer's own language, so it is rendered as a value with a
caption saying whose words they are, exactly as an artifact's self-test reason
already is. Two of the five candidates record none, and that is an honest
absence naming the field that would supply it, never an inferred purpose. A
purpose guessed from a coefficient table is the same defect as a figure nobody
measured.

**What the artifact was fitted from, checked rather than quoted.** Every
artifact records ``source_fingerprints``, one digest per source file, and the
product already has a guard that answers this question:
``kairos.model.freshness.coefficient_freshness`` decides whether stored
coefficients still match the data on disk, in three states.

**It is held to that guard by a test rather than by an import, and the reason is
a measurement.** Importing anything under ``kairos.model`` costs **17 seconds**
on this machine, because that package's own ``__init__`` probes TensorFlow
through ``kairos.model.train``. The registry is the first command of this
steward's job and it read in 0.6 seconds; paying that import to answer a
question about three digests would have made the opening command of the story
thirty times slower. So the comparison is four lines here, over the same hasher
the guard itself uses, ``kairos.observability.run_log.checksum_file``, and
``tests/test_p12_origin.py`` asserts that this module and that guard reach the
same verdict in every one of its three states. A duplicated rule held by a test
is a different thing from a duplicated rule held by hope.

Measured on this tree: all six artifacts record the same three files at the same
digests, and all three files are on disk with those exact bytes. So the source
data is not one of the differences between these rows, and the fit basis
finding in ``adopt_candidate_basis.py`` is about which breaks each fit covered
and not about which data it read.

**The half that cannot be answered, said plainly.** No artifact here records the
command that produced it. Not one metadata key on any of the six names a script,
a flag or a command line. So an artifact on this shelf can be identified exactly
and cannot be rebuilt from what it carries, and that is stated as the gap with
the thing that would close it rather than left for a reader to discover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from kairos.observability.run_log import checksum_file

from scripts import adopt_candidate_words as words

# Where a producer records what it built the artifact for. Three of the five
# candidates on this tree carry it and the shipped artifact does not.
PURPOSE_KEY = "purpose"

# Where a producer records the data it read, one digest per file. The same key
# the engine's own freshness guard reads.
FINGERPRINTS_KEY = "source_fingerprints"

# The three files every artifact in this tree was measured from. Digested so a
# stored re-score can say the data moved rather than being served as current.
# It lives here rather than beside the re-score because it is the same question
# this module asks per artifact: what was this made from.
SOURCE_FILES = ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx")


def data_fingerprint(paths: Any) -> dict[str, str]:
    """One digest per measured source, so a stale re-score can name what moved.

    ``absent`` rather than nothing for a file that is not there, because the
    fingerprint this feeds is a string and a missing input has to change it.
    """
    return {name: checksum_file(Path(paths.reference_dir) / name) or "absent"
            for name in SOURCE_FILES}


# Whose sentence the purpose line is, said where it is printed. The text is the
# producer's own and it is not translated, so the caption is what carries the
# language and the absence is a state with the field that would fill it.
PURPOSE: dict[str, dict[str, str]] = {
    "recorded": {
        "en": "What its producer recorded it was built for, in its own words.",
        "he": "מה שהמפיק שלו רשם שהוא נבנה בשבילו, במילותיו שלו.",
    },
    "absent": {
        "en": "This artifact records no purpose, so what it was built for is written down nowhere. It would be supplied by a purpose line in the artifact metadata, written by whatever produces it.",
        "he": "הקובץ הזה אינו רושם ייעוד, ולכן מה שהוא נבנה בשבילו אינו כתוב בשום מקום. הוא יסופק על ידי שורת ייעוד במטא של הקובץ, שתיכתב על ידי מה שמפיק אותו.",
    },
}


# Whether the data it records reading is the data on disk now, which is also the
# data this evaluation rebuilt its breaks from. The states are the freshness
# guard's own three, renamed to the words this surface uses for them.
SOURCES: dict[str, dict[str, str]] = {
    "verified": {
        "en": "Every source file it records is on disk with the digest it recorded, so it was fitted from the same bytes this evaluation rebuilt its breaks from.",
        "he": "כל קובץ מקור שהוא רושם נמצא על הדיסק עם טביעת האצבע שרשם, ולכן הוא אומן מאותם בייטים שהמדידה הזו בנתה מהם מחדש את הברייקים שלה.",
    },
    "moved": {
        "en": "The bytes on disk are not the bytes it records reading, so it was not fitted from the data this evaluation rebuilt its breaks from. What moved: {changed}.",
        "he": "הבייטים שעל הדיסק אינם הבייטים שהוא רושם שקרא, ולכן הוא לא אומן מהנתונים שהמדידה הזו בנתה מהם מחדש את הברייקים שלה. מה שזז: {changed}.",
    },
    "unverifiable": {
        "en": "A source file it records is not on disk, so what it was fitted from cannot be checked from here at all.",
        "he": "קובץ מקור שהוא רושם אינו נמצא על הדיסק, ולכן לא ניתן לבדוק מכאן כלל ממה הוא אומן.",
    },
    "absent": {
        "en": "This artifact records no source files, so the data it was fitted from is not identified anywhere in it.",
        "he": "הקובץ הזה אינו רושם קבצי מקור, ולכן הנתונים שהוא אומן מהם אינם מזוהים בו בשום מקום.",
    },
}


# And whether that data is the same data the shipped artifact read. This is the
# confound the fit-basis measurement cannot see: two artifacts fitted on the same
# NUMBER of breaks out of two different source files are not comparable at all.
AGREEMENT: dict[str, dict[str, str]] = {
    "same": {
        "en": "It records the same source files at the same digests as the shipped artifact, so the data is not one of the differences between them.",
        "he": "הוא רושם את אותם קבצי מקור ובאותן טביעות אצבע כמו הקובץ המשודר, ולכן הנתונים אינם אחד ההבדלים ביניהם.",
    },
    "differs": {
        "en": "It records different source data from the shipped artifact, so part of the difference between them is the data and not the model. Where they differ: {files}.",
        "he": "הוא רושם נתוני מקור אחרים מהקובץ המשודר, ולכן חלק מההבדל ביניהם הוא הנתונים ולא המודל. במה הם נבדלים: {files}.",
    },
    "unknown": {
        "en": "One of the two records no source fingerprints, so whether both were fitted from the same data is unknown here rather than established.",
        "he": "אחד מהשניים אינו רושם טביעות אצבע של מקורות, ולכן השאלה אם שניהם אומנו מאותם נתונים אינה ידועה כאן ולא הוכחה.",
    },
}


# The half of the provenance this tree cannot answer. Measured: no metadata key
# on any of the six artifacts names a script, a flag or a command line.
RECIPE: dict[str, str] = {
    "en": "What every artifact here records is the data it read, identified by digest. None of them records the command that produced it, so an artifact on this shelf can be identified exactly and cannot be rebuilt from anything it carries.",
    "he": "מה שכל קובץ כאן רושם הוא הנתונים שקרא, מזוהים בטביעת אצבע. אף אחד מהם אינו רושם את הפקודה שהפיקה אותו, ולכן קובץ על המדף הזה ניתן לזיהוי מדויק ולא ניתן לבנייה מחדש מכל מה שהוא נושא.",
    "unblocked_by_en": "The producing script recording its own command line and its flags into the artifact metadata, beside the source fingerprints it already writes there.",
    "unblocked_by_he": "הסקריפט המפיק ירשום את שורת הפקודה ואת הדגלים שלו לתוך המטא של הקובץ, לצד טביעות האצבע של המקורות שהוא כבר כותב שם.",
}


def _fingerprints(metadata: Optional[Mapping[str, Any]]) -> dict[str, str]:
    stored = (metadata or {}).get(FINGERPRINTS_KEY)
    if not isinstance(stored, Mapping):
        return {}
    return {str(key): str(value) for key, value in stored.items()}


def _purpose(metadata: Mapping[str, Any]) -> Optional[str]:
    value = metadata.get(PURPOSE_KEY)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _source_rows(root: Path, fingerprints: Mapping[str, str]) -> list[dict[str, Any]]:
    """One row per file the artifact records reading, against the file now.

    ``matches`` is tri-state: true, false, or None when the file is not on disk,
    because a file that is gone did not fail the comparison, nobody could make
    it.
    """
    rows = []
    for name in sorted(fingerprints):
        recorded = str(fingerprints[name])
        current = checksum_file(root / name)
        rows.append({
            "file": name,
            "sha256": recorded,
            "short": recorded[:12],
            "on_disk": current is not None,
            "matches": None if current is None else current == recorded,
        })
    return rows


def _sources_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The freshness verdict on this artifact's recorded sources, in four states.

    The same three the engine's own guard reaches, over the same hasher, with a
    test holding the two equal. The fourth is the one that guard folds into its
    ``unknown``: an artifact that records no fingerprints at all is a producer
    that wrote nothing down, and an artifact whose fingerprinted file is gone is
    a tree that has lost an input. The acts that fix those are different, so they
    are not one state here.

    A missing file wins over a moved one. Once a fingerprinted input cannot be
    hashed, nothing about the rest of the set can be called verified or moved
    without stating more than was measured.
    """
    if not rows:
        return {"state": "absent", "changed": []}
    if any(not row["on_disk"] for row in rows):
        return {"state": "unverifiable", "changed": []}
    changed = sorted(row["file"] for row in rows if not row["matches"])
    return {"state": "moved" if changed else "verified", "changed": changed}


def _agreement(mine: Mapping[str, str],
               theirs: Mapping[str, str]) -> tuple[str, list[str]]:
    if not mine or not theirs:
        return "unknown", []
    differ = sorted(name for name in set(mine) | set(theirs)
                    if mine.get(name) != theirs.get(name))
    return ("differs", differ) if differ else ("same", [])


def origin_row(identifier: str, metadata: Optional[Mapping[str, Any]], *,
               root: Path,
               shipped_metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What one artifact was for and what it was made from, with both limits.

    ``shipped_metadata`` is optional and the agreement keys are absent without
    it, rather than a row agreeing with itself. The shipped artifact has nothing
    to be compared against here, and emitting ``same`` for it would be a true
    sentence about nothing.
    """
    metadata = metadata if isinstance(metadata, Mapping) else {}
    purpose = _purpose(metadata)
    fingerprints = _fingerprints(metadata)
    source_rows = _source_rows(Path(root), fingerprints)
    sources = _sources_state(source_rows)
    row: dict[str, Any] = {
        "id": identifier,
        "purpose": purpose,
        "purpose_state": "recorded" if purpose else "absent",
        **words.pair(PURPOSE, "recorded" if purpose else "absent", "purpose_reading"),
        "sources": source_rows,
        "sources_recorded": len(fingerprints),
        "sources_state": sources["state"],
        "sources_changed": sources["changed"],
        **words.pair(SOURCES, sources["state"], "sources_reading",
                     changed=", ".join(sources["changed"])),
        "recipe_state": "not_recorded",
        "recipe_en": RECIPE["en"],
        "recipe_he": RECIPE["he"],
        "recipe_unblocked_by_en": RECIPE["unblocked_by_en"],
        "recipe_unblocked_by_he": RECIPE["unblocked_by_he"],
    }
    if shipped_metadata is None:
        return row
    state, differ = _agreement(fingerprints, _fingerprints(shipped_metadata))
    row["same_sources_as_shipped"] = state
    row["differs_on"] = differ
    row.update(words.pair(AGREEMENT, state, "agreement_reading",
                          files=", ".join(differ)))
    return row


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The shipped row first, then every candidate, each with its origin block."""
    shipped = (payload.get("shipped") or {}).get("origin") or {}
    rows = [dict(shipped, id="shipped (live)")] if shipped else []
    return rows + [row["origin"] for row in payload.get("candidates") or []
                   if row.get("origin")]


def render_purposes(payload: Mapping[str, Any]) -> list[str]:
    """What each artifact was for, before anything is ranked.

    First on the terminal for the reason the reference puts its notes first: a
    reader who does not know what an artifact was trying to do cannot read a
    table of five of them at all. The producer's own sentence is printed
    verbatim; a row that records none prints the absence and the field that
    would fill it, and never an inference from the numbers below.
    """
    rows = _rows(payload)
    if not rows:
        return []
    lines = ["What each artifact was built for, as its own producer recorded it"]
    for row in rows:
        lines.append(f"  {str(row.get('id')):20s} {row.get('purpose') or 'no purpose recorded'}")
    absent = [row for row in rows if row.get("purpose_state") == "absent"]
    lines.append("")
    lines.append(f"  {PURPOSE['recorded']['en']}")
    if absent:
        lines.append(f"  {PURPOSE['absent']['en']}")
    lines.append("")
    return lines


def _signature(row: Mapping[str, Any]) -> tuple:
    """What would have to differ between two rows for them to read differently."""
    return (tuple((item["file"], item["sha256"]) for item in row.get("sources") or []),
            row.get("sources_state"))


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def render_provenance(payload: Mapping[str, Any]) -> list[str]:
    """What each artifact was fitted from, and the half nobody recorded.

    Printed under the artifact files, because it is the same question one level
    down: the block above says which file each row is, and this says which data
    that file read and whether that data is still here.

    Written the way ``adopt_candidate_basis.py`` writes its own block, and for
    the same reason. When every row says the same thing, six rows saying it six
    times buries the one that would not, so the shared answer is stated once
    with the files themselves under it, and only a row that reads differently
    is named. On this tree nothing reads differently, and that identity is the
    finding rather than a repetition of it.
    """
    rows = _rows(payload)
    if not rows:
        return []
    lines = ["", "What each artifact was fitted from"]
    common = _signature(rows[0])
    deviating = [row for row in rows if _signature(row) != common]
    if not deviating:
        head = rows[0]
        lines.append(f"  every one of the {_plural(len(rows), 'artifact')} here records the same {_plural(len(head.get('sources') or []), 'source file')} at the same digests")
        for item in head.get("sources") or []:
            lines.append(f"    {item['file']}  {item['short']}  on disk: {str(bool(item['on_disk'])).lower()}, digest matches: {str(item['matches']).lower()}")
        lines.append(f"  {head.get('sources_reading_en')}")
    else:
        for row in rows:
            files = ", ".join(f"{item['file']} {item['short']}" for item in row.get("sources") or [])
            lines.append(f"  {str(row.get('id')):20s} {files or 'no source files recorded'}")
            lines.append(f"  {'':20s} {row.get('sources_reading_en')}")
    # The cross-artifact reading is separate from the on-disk one and is stated
    # for the rows that do not agree with the shipped artifact, because that is
    # the confound the fit-basis count cannot see.
    apart = [row for row in rows if row.get("same_sources_as_shipped") in ("differs", "unknown")]
    for row in apart:
        lines.append(f"  {str(row.get('id')):20s} {row.get('agreement_reading_en')}")
    if not apart:
        lines.append("  every candidate records the same source data as the shipped artifact, so the data is not one of the differences between any two rows above")
    lines.append("")
    lines.append(f"  {RECIPE['en']}")
    lines.append(f"  lifted by: {RECIPE['unblocked_by_en']}")
    return lines
