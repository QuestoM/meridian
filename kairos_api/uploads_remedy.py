"""Every sentence the source card says about a state, an upload and a file.

Split out of ``uploads_status.py`` at the 450-line law and named by the
``<parent stem>_<role>.py`` rule the package already follows eleven times. The
split is by subject as well as by size: what state an input is in is derived
there, and the words that state is read in are here, so a sentence can be
corrected without touching a derivation and a derivation cannot be changed by
editing prose.

**The remedy vocabulary is seven words for six states.** Six of them are the
state's own name. The seventh, ``shadowed_by_a_later_day``, is the one case
where the state's own sentence would be false: a kind the engine reads out of a
directory whose own last upload lost the resolver, where "keep uploading here
and the plan will not change" is wrong because the next upload is read or not
read by the day its name carries.

**The eighth is a live file that is read, carries rows, and is not clean.**
``in_use`` is true of it and reads as nothing to do, and it is the last thing a
steward sees after committing a file the door warned about. Measured on the
shipped surface: a daily log whose 20 of 20 rows carried a clock the loader
cannot read was accepted by the door, and the card it landed on printed the teal
"in use" chip and "nothing to do. upload a new file when the next one lands",
with the warning that no spot in the live input has a clock four lines below it.
The state stays what it is, because the engine really does read that file; what
to do about it stops being nothing.

**A count is said the way each language counts one of a thing.** The English
sentence read "1 warning(s)" before, which is a developer's shorthand on an
operator's screen; both languages now name one warning as one and two as two.
"""

from __future__ import annotations

from typing import Any

REMEDIES: dict[str, dict[str, str]] = {
    "in_use": {
        "en": "Nothing to do. Upload a new file when the next one lands, then run the plan.",
        "he": "אין מה לעשות. העלו קובץ חדש כשיגיע הבא, ואז הריצו את התוכנית.",
    },
    "shadowed": {
        "en": "Remove the file the engine reads first, or keep uploading here and the plan will not change.",
        "he": "הסירו את הקובץ שהמנוע קורא קודם, או המשיכו להעלות כאן והתוכנית לא תשתנה.",
    },
    # The same state, and not the same news. The one above is a kind another
    # file is read INSTEAD of, where nothing this operator uploads is ever read
    # until that file goes; this one is a kind whose own file is read and whose
    # last upload lost the resolver, where the next upload is read or not read
    # depending on the day its name carries. One sentence for both is how the
    # remedy came to say "the plan will not change" over an input where it can.
    "shadowed_by_a_later_day": {
        "en": "The file that arrived last is not the one the engine reads. Upload the day you meant to change, or remove {live} and the engine reads the newest day left on disk.",
        "he": "הקובץ שהגיע אחרון אינו הקובץ שהמנוע קורא. העלו את היום שהתכוונתם לשנות, או הסירו את {live} והמנוע יקרא את היום החדש ביותר שנותר על הדיסק.",
    },
    # The one the six states cannot say, because it is not a state: the engine
    # reads this file, it carries rows, and the last check of it came back with
    # a warning that is now true of every figure downstream of it.
    "in_use_with_warnings": {
        "en": "The last check of this file came back with {warnings}, about {fields}. Fix that in the export and upload it again, or every figure from this input goes on carrying what the check found.",
        "he": "הבדיקה האחרונה של הקובץ הזה החזירה {warnings}, על {fields}. תקנו את זה בייצוא והעלו שוב, אחרת כל נתון מהקלט הזה ימשיך לשאת את מה שהבדיקה מצאה.",
    },
    "not_read": {
        "en": "Change this input where the engine reads it instead. Uploading here stores the file and changes no number.",
        "he": "שנו את הקלט הזה במקום שבו המנוע קורא אותו. העלאה כאן שומרת את הקובץ ולא משנה אף מספר.",
    },
    "empty": {
        "en": "Upload a file that carries rows. The engine reads this file and it has none, so no figure can be computed from this input yet.",
        "he": "העלו קובץ שיש בו שורות. המנוע קורא את הקובץ הזה ואין בו אף שורה, ולכן עדיין לא ניתן לחשב ממנו שום נתון.",
    },
    "invalid": {
        "en": "Fix the named columns in the export and upload it again. Nothing was replaced.",
        "he": "תקנו את העמודות שצוינו בייצוא והעלו שוב. שום דבר לא הוחלף.",
    },
    "missing": {
        "en": "No file has been uploaded for this input yet.",
        "he": "עדיין לא הועלה קובץ עבור הקלט הזה.",
    },
}

CONSEQUENCES: dict[str, dict[str, str]] = {
    "replaces_live_input": {
        "en": "This is the live input. Uploading replaces what the plan is computed from.",
        "he": "זהו הקלט החי. העלאה מחליפה את מה שהתוכנית מחושבת ממנו.",
    },
    "changes_model_basis": {
        "en": "This is the live input and the model version was measured on it, so a new file makes that measurement out of date.",
        "he": "זהו הקלט החי וגרסת המודל נמדדה עליו, ולכן קובץ חדש הופך את המדידה הזו ללא עדכנית.",
    },
    "stored_not_read": {
        "en": "Stored and validated. Nothing reads this file, so uploading changes no number.",
        "he": "נשמר ונבדק. שום דבר לא קורא את הקובץ הזה, ולכן העלאה לא משנה אף מספר.",
    },
    # The conditional one, and the condition is named rather than left for the
    # operator to infer from a filename. Neither "it replaces the live input"
    # nor "it changes no number" is true of this kind on its own.
    "replaces_only_a_later_day": {
        "en": "The engine reads the daily file whose name carries the newest airing date, currently {live}. An upload here becomes that file only when its own name carries a later day, and is stored and read by nothing when it does not.",
        "he": "המנוע קורא את קובץ היום ששמו נושא את תאריך השידור החדש ביותר, כרגע {live}. העלאה כאן הופכת לקובץ הזה רק אם שמה נושא יום מאוחר יותר, ואם לא, היא נשמרת ואינה נקראת.",
    },
    "stored_without_replacing": {
        "en": "This file will be stored and validated, and it will replace nothing. The engine will go on reading {live}, so the plan will not change.",
        "he": "הקובץ הזה יישמר וייבדק, והוא לא יחליף דבר. המנוע ימשיך לקרוא את {live}, ולכן התוכנית לא תשתנה.",
    },
    # The one thing the five above cannot say, and the one an operator was left
    # to find out after the click: this file wins the read path AND it carries
    # no rows, so what it replaces the live input with is nothing.
    "replaces_live_input_with_no_rows": {
        "en": "This file will become the live input and it carries no data rows, so every figure computed from this input will be empty until a file with rows replaces it.",
        "he": "הקובץ הזה יהפוך לקלט החי ואין בו אף שורת נתונים, ולכן כל נתון שמחושב מהקלט הזה יהיה ריק עד שיוחלף בקובץ שיש בו שורות.",
    },
    # The same shape as the one above and far more common. Measured on the
    # shipped card: a daily file whose 20 of 20 rows carried a clock the loader
    # cannot read was answered "this is the live input", in the teal tone, under
    # the heading that says it passed every check. The count and the field are
    # in the sentence, because a consequence that does not name what is wrong
    # leaves the reader to go and find it.
    "replaces_live_input_with_warnings": {
        "en": "This file will become the live input and it carries {warnings}, about {fields}, so every figure computed from this input carries what the check found.",
        "he": "הקובץ הזה יהפוך לקלט החי ויש בו {warnings}, על {fields}, ולכן כל נתון שמחושב מהקלט הזה נושא את מה שהבדיקה מצאה.",
    },
}

# Why a file this product stored is not the one the engine reads. Only the
# daily kind can hold more than one file, so both sentences name that resolver:
# every other kind lands on exactly one path and a second file of it cannot
# exist. The second code is the dangerous one, and it is the one an operator
# hits: the file they sent last is not the file any number rests on.
STORED_REASONS: dict[str, dict[str, str]] = {
    "another_day_is_read": {
        "en": "Stored here and not read. The engine reads the newest daily file by the airing date in its name, currently {live}.",
        "he": "נשמר כאן ואינו נקרא. המנוע קורא את קובץ היום החדש ביותר לפי תאריך השידור שבשמו, כרגע {live}.",
    },
    "arrived_after_the_file_that_is_read": {
        "en": "Uploaded after the file the engine reads, and not read. The engine picks the newest daily file by the airing date in its name, currently {live}, which names a later day than this one.",
        "he": "הועלה אחרי הקובץ שהמנוע קורא, ואינו נקרא. המנוע בוחר את קובץ היום החדש ביותר לפי תאריך השידור שבשמו, כרגע {live}, שנושא יום מאוחר יותר מזה.",
    },
}

# What a warning is about when it names no column at all. Every warning this
# door can raise about a file the engine will read names one today, so this is
# the honest fallback rather than a case, in the surface's own words for it.
WHOLE_FILE = ("the whole file", "הקובץ כולו")

# The warning a file the engine reads with no data rows in it carries, for the
# readers that render warnings and nothing else.
NO_ROWS_WARNING = "The engine reads this file and it carries no data rows, so no figure can be computed from this input."


def warned_summary(findings: Any) -> tuple[int, str | tuple[str, str]]:
    """How many warn-severity findings a report carries, and the fields they name.

    One derivation for both sentences that quote it, so the consequence said
    before the click and the remedy said after it cannot count differently. The
    fields are the operator's own column names, de-duplicated in the order the
    findings raised them, and the fallback when a warning is about no column at
    all is the surface's own word for the whole file rather than a blank.
    """
    warned = [item for item in (findings or []) if str((item or {}).get("severity") or "") == "warning"]
    named = [column for column in dict.fromkeys(str((item or {}).get("column") or "").strip() for item in warned) if column]
    return len(warned), ", ".join(named) or WHOLE_FILE


def warning_count(count: int) -> tuple[str, str]:
    """How many warnings, said the way each language counts one of them."""
    if count == 1:
        return ("1 warning", "אזהרה אחת")
    return (f"{count} warnings", f"{count} אזהרות")
