"""Every sentence this destination writes itself, in both languages.

Split out of ``uploads_validate.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule the package already follows.

**The refusal is the one sentence this destination exists for, so it is the one
sentence that may not arrive in the wrong language.** Measured on the shipped
surface before this module existed: a daily file with two unreadable dates and
two unreadable clocks was refused with a Hebrew heading, Hebrew row labels and
two English reasons, and a dayparts file with renamed channel columns was
refused with one English reason. Every other word on that card was Hebrew.

The division of labour is exact. A code this destination raises ITSELF is
authored here in both languages, with its own measured numbers formatted into
both. A violation raised by the frozen data contracts
(:mod:`kairos.data.contracts`) keeps its English detail, because the counts and
column names inside that sentence are the contract's own and re-authoring it in
Hebrew from the code alone would drop them. A record carries ``message_he`` only
when there is a real Hebrew sentence behind it, so a surface falls back to the
English detail rather than to a blank line.

A left-to-right run inside a right-to-left sentence is wrapped in a first-strong
isolate, the way :mod:`kairos_api.downloads_api` wraps a path: without it the
Hebrew sentence's own punctuation is reordered to the wrong side of a column
name. Each run is its own placeholder so each isolate resolves on its own
content. The English sentence needs no isolate and carries none, so every
English string here renders byte-identical to the one the door shipped before.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

# First-strong isolate, around any field that can carry a left-to-right run.
ISOLATE_START = "⁨"
ISOLATE_END = "⁩"

# What a finding is ABOUT when it is about no single column. A refusal at the
# door is often about the whole file, about its header row, or about the table
# the loader parsed it into, and the record still owes a ``column`` key, so this
# module used to put an internal token in that key: ``<file>``, ``<header>`` or
# ``<frame>``. Measured on the shipped card: the surface prints that key as the
# chip beside the reason, so a daily export missing one column rendered a bold
# Latin ``<header>`` on a Hebrew screen, above the sentence naming the column
# that was actually missing. A code travels on ``scope`` instead and the surface
# resolves it to a word in the language it is being read in, or to nothing; the
# ``column`` key stays and is empty, because there is no column to name.
SCOPES = ("file", "header", "frame")

# What a warn-severity finding actually cost the engine. Two warnings on one
# card are not the same news: a field the engine did not get leaves part of the
# file unusable, and a value it could read two ways is fully read, on one of the
# two readings, so every figure from the file rests on that reading. The surface
# heads the two differently and neither of them is a clean pass. A code absent
# from this table is a loss, which is the safe default and is what every warning
# the frozen contracts raise gets.
FIELD_LOST = "field_lost"
VALUE_INTERPRETED = "value_interpreted"
EFFECTS: dict[str, str] = {
    "unreadable_times": FIELD_LOST,
    "skipped_campaign_rows": FIELD_LOST,
    "no_data_rows": FIELD_LOST,
    "ambiguous_day_month": VALUE_INTERPRETED,
}

# One entry per sentence this destination raises itself, keyed by the finding's
# own code. Several keys are one code in two situations, and the caller passes
# the situation it means; the entries that are not a whole sentence are
# fragments built into one, which is how a list of headers, or the word for what
# a row of one kind is, stays in the language the sentence around it is read in.
MESSAGES: dict[str, dict[str, str]] = {
    "unreadable_file": {
        "en": "The uploaded file could not be read as a CSV table. Check that it is a UTF-8 CSV export with a single header row and try again.",
        "he": "לא ניתן היה לקרוא את הקובץ שהועלה כטבלת CSV. ודאו שזהו ייצוא CSV בקידוד UTF-8 עם שורת כותרת אחת, ונסו שוב.",
    },
    "empty_file": {
        "en": "Uploaded file is empty",
        "he": "הקובץ שהועלה ריק, ואין בו מה לקרוא",
    },
    "too_large": {
        "en": "Uploaded file exceeds the {megabytes} MB upload limit",
        "he": "הקובץ שהועלה חורג ממגבלת ההעלאה של {megabytes} MB",
    },
    "missing_columns": {
        "en": "Missing required columns for '{kind}': {columns}",
        "he": "חסרות בקובץ הזה עמודות נדרשות: {columns}. תקנו את הכותרות בייצוא והעלו שוב.",
    },
    "contract_refusal": {
        "en": "Upload rejected by the {dataset} data contract: {reasons}",
        "he": "הקובץ נדחה על ידי חוזה הנתונים ושום דבר לא הוחלף: {reasons}",
    },
    "no_parseable_dates": {
        "en": "none of the {rows} row(s) has a parseable date, so the engine would build zero segments from this file",
        "he": "באף אחת מ־{rows} השורות אין תאריך שהמנוע יכול לקרוא, ולכן הוא יבנה מהקובץ הזה אפס מקטעים",
    },
    "unparseable_dates": {
        "en": "{unreadable} of {rows} row(s) carry a date the loader cannot read; the engine keeps those rows with an empty date, so nothing places them on the day they belong to",
        "he": "ב־{unreadable} מתוך {rows} שורות יש תאריך שהמנוע אינו יכול לקרוא; השורות האלה נשמרות עם תאריך ריק, ולכן שום דבר לא ממקם אותן ביום שאליו הן שייכות",
    },
    "unreadable_times": {
        "en": "{unreadable} of {rows} row(s) carry a time of day the loader cannot read, so those spots reach the engine with no clock: their daypart and their separation from the next showing cannot be worked out",
        "he": "ב־{unreadable} מתוך {rows} שורות יש שעה שהמנוע אינו יכול לקרוא, ולכן התשדירים האלה מגיעים אליו בלי שעה: אי אפשר לחשב את רצועת השידור שלהם ואת ההפרדה מהשידור הבא",
    },
    "ambiguous_day_month": {
        "en": "{ambiguous} of {rows} row(s) have a day/month-ambiguous slash date; the loader reads them month-first ({pattern})",
        "he": "ב־{ambiguous} מתוך {rows} שורות יש תאריך עם לוכסנים שאי אפשר לדעת בו מה היום ומה החודש; המנוע קורא אותם כשהחודש ראשון ({pattern})",
    },
    "no_loadable_campaigns": {
        "en": "none of the {rows} row(s) carries {columns} and a positive target, so the pacing engine would read zero campaigns from this file",
        "he": "באף אחת מ־{rows} השורות אין {columns} ויעד חיובי, ולכן מנוע הקצב יקרא מהקובץ הזה אפס קמפיינים",
    },
    "skipped_campaign_rows": {
        "en": "{skipped} of {rows} row(s) will be skipped by the pacing loader (missing {columns}, or a positive target)",
        "he": "{skipped} מתוך {rows} שורות יידלגו על ידי מנוע הקצב, מפני שחסר בהן {columns} או יעד חיובי",
    },
    "no_data_rows": {
        "en": "the file yields zero audience rows: the recognized channel columns are present but carry no data rows",
        "he": "הקובץ מניב אפס שורות רייטינג: עמודות הערוץ המוכרות קיימות, אך אין מתחתיהן אף שורת נתונים",
    },
    # The same code for every other kind, said twice, because a file with no
    # rows is not the same news everywhere. The first is a refusal: there is no
    # world in which an operator meant to publish an empty lineup or an empty
    # rate card. The second is accepted and names the outcome, because a
    # broadcast day with nothing booked on it is a state that really occurs.
    "no_data_rows_refused": {
        "en": "the file carries its header and no data rows at all, so the engine reads zero {rows_are} from it and no figure anywhere can be computed from this input",
        "he": "בקובץ יש שורת כותרת ואין בו אף שורת נתונים, ולכן המנוע קורא ממנו אפס {rows_are} ואי אפשר לחשב ממנו שום נתון בשום מסך",
    },
    "no_data_rows_accepted": {
        "en": "the file carries its header and no data rows at all, so the engine reads zero {rows_are} from it and every figure that comes from this input will be empty",
        "he": "בקובץ יש שורת כותרת ואין בו אף שורת נתונים, ולכן המנוע קורא ממנו אפס {rows_are} וכל נתון שמגיע מהקלט הזה יהיה ריק",
    },
    # What one row of each kind is, as a fragment built into the two sentences
    # above, which is how a list of words stays in the language it is read in.
    "rows_of_programmes": {
        "en": "programmes",
        "he": "תוכניות",
    },
    "rows_of_spots": {
        "en": "historical spots",
        "he": "תשדירים היסטוריים",
    },
    "rows_of_advertiser_rules": {
        "en": "advertiser rules",
        "he": "כללי מפרסמים",
    },
    "rows_of_rate_card": {
        "en": "rate card rows",
        "he": "שורות של כרטיס התעריפים",
    },
    "rows_of_daily": {
        "en": "spots",
        "he": "תשדירים",
    },
    "rows_of_campaign_flights": {
        "en": "campaigns",
        "he": "קמפיינים",
    },
    "no_recognized_channel_columns": {
        "en": "the file yields zero audience rows because no column here is named for a channel; the loader matches a column header to a channel name exactly and knows {count} such names, one for each channel the audience export carries, of which the only one this account may be shown is your own channel {owned}, so re-export this file with the headers the audience export writes rather than renamed ones; the unrecognized columns found were {found}{clause}",
        "he": "הקובץ מניב אפס שורות רייטינג מפני שאף עמודה כאן אינה נושאת שם של ערוץ; המנוע משווה כותרת עמודה לשם ערוץ בדיוק ומכיר {count} שמות כאלה, אחד לכל ערוץ שייצוא הרייטינג נושא, ומתוכם היחיד שמותר להציג לחשבון הזה הוא הערוץ שלכם {owned}, ולכן ייצאו את הקובץ מחדש עם הכותרות שייצוא הרייטינג כותב במקום כותרות ששמן שונה; העמודות שלא זוהו הן {found}{clause}",
    },
    "no_recognized_channel_columns_unset": {
        "en": "the file yields zero audience rows because no column here is named for a channel; the loader matches a column header to a channel name exactly and knows {count} such names, but no operator channel is configured in settings, so there is no name this account may be shown to check these headers against; set the operator channel on the settings screen. The unrecognized columns found were {found}{clause}",
        "he": "הקובץ מניב אפס שורות רייטינג מפני שאף עמודה כאן אינה נושאת שם של ערוץ; המנוע משווה כותרת עמודה לשם ערוץ בדיוק ומכיר {count} שמות כאלה, אך לא מוגדר ערוץ מפעיל בהגדרות, ולכן אין שם שמותר להציג לחשבון הזה כדי לבדוק מולו את הכותרות; הגדירו את ערוץ המפעיל במסך ההגדרות. העמודות שלא זוהו הן {found}{clause}",
    },
    "withheld_columns": {
        "en": ", and {withheld} further column(s) naming a channel you do not own, which are not listed here",
        "he": ", ולצידן עמודות שנושאות שם של ערוץ שאינו שלכם ואינן מפורטות כאן, {withheld} במספר",
    },
    "no_columns_found": {
        "en": "none",
        "he": "אין",
    },
    "campaign_requirements": {
        "en": "a campaign_id, parseable flight_start/flight_end dates",
        "he": "מזהה קמפיין campaign_id ותאריכי flight_start/flight_end שניתן לקרוא",
    },
    "campaign_fields": {
        "en": "campaign_id, flight dates",
        "he": "מזהה קמפיין campaign_id או תאריכי קמפיין",
    },
}


def _rendered(fields: dict[str, object], index: int, isolate: bool) -> dict[str, str]:
    """One language's view of the fields, left-to-right runs isolated for Hebrew.

    A field whose value is a two-item tuple carries its own two languages, which
    is how a fragment assembled before the sentence stays in one language.
    """
    rendered: dict[str, str] = {}
    for name, value in fields.items():
        text = str(value[index]) if isinstance(value, tuple) else str(value)
        rendered[name] = f"{ISOLATE_START}{text}{ISOLATE_END}" if isolate and text else text
    return rendered


def say(code: str, **fields: object) -> tuple[str, str]:
    """This code's sentence in English and in Hebrew, with its own numbers in it."""
    words = MESSAGES.get(code)
    if words is None:
        return "", ""
    english = words["en"].format(**_rendered(fields, 0, False))
    hebrew = words["he"].format(**_rendered(fields, 1, True))
    return english, hebrew


def effect_of(code: str, severity: str) -> dict[str, str]:
    """The ``effect`` key of one finding, or nothing at all when it has none.

    An error is refused at the door and no part of that file reaches the engine,
    so only a warning, which rides along with the file, has an effect to state.
    """
    return {"effect": EFFECTS.get(code, FIELD_LOST)} if severity == "warning" else {}


def record(code: str, column: str, severity: str, scope: str = "", **fields: object) -> dict[str, Any]:
    """One finding this destination authors itself, as the record a surface renders.

    ``key`` and ``fields`` ride along, which is what a stored report keeps in
    place of the two sentences: rendered once and kept, a sentence carrying the
    operator's channel is that channel frozen at the moment it was written.

    ``scope`` is one of :data:`SCOPES` and is present only on a finding that is
    about no column, which is where the internal token used to go.
    """
    english, hebrew = say(code, **fields)
    entry: dict[str, Any] = {"column": column, "code": code, "message": english, "severity": severity}
    if scope:
        entry["scope"] = scope
    if hebrew:
        entry["message_he"] = hebrew
    entry["key"] = code
    entry["fields"] = dict(fields)
    return entry


def flat(severity: Any, where: Any, code: Any, message: Any) -> str:
    """One finding as the flat line every existing reader of a report parses.

    A finding about no column states what it IS about in that slot, which is its
    scope, rather than leaving the slot empty. Measured before this: the English
    half of a commit refusal quotes these lines, so a dayparts file with no data
    rows was refused with ``[error] : no_data_rows - the file yields zero...``.
    """
    return f"[{severity}] {where or ''}: {code} - {message}"


def flat_finding(finding: Any) -> str:
    """One finding record as that same line, every slot taken from the record.

    The record is where a column name is resolved to a header the operator's own
    file carries and where a finding about no column carries its scope, so a
    line built from anything else would name the column by another word than the
    chip beside it. One rule for the live payload and the stored report both.
    """
    entry = finding if isinstance(finding, dict) else {}
    return flat(entry.get("severity"), entry.get("column") or entry.get("scope"), entry.get("code"), entry.get("message"))


def reject(message: str, errors: list[str] | None = None, message_he: str = "", findings: list[dict[str, Any]] | None = None) -> JSONResponse:
    """The 400 an upload gets when it must not replace the live input.

    ``detail_he`` rides beside ``detail`` and each finding carries its own
    ``message_he``, because the panel this refusal lands in is a Hebrew screen,
    and a refusal nobody reading that screen can read is not a refusal.
    """
    content: dict[str, Any] = {"detail": message, "errors": errors or [message], "valid": False}
    if message_he:
        content["detail_he"] = message_he
    if findings:
        content["findings"] = findings
    return JSONResponse(status_code=400, content=content)


def refuse(code: str, errors: list[str] | None = None, column: str = "", scope: str = "file", **fields: object) -> JSONResponse:
    """The 400 for a refusal this destination writes itself, in both languages.

    Every refusal raised before a loader runs is about the file itself, so the
    scope defaults to ``file`` and the column to nothing. A caller that means
    the header row says so.
    """
    english, hebrew = say(code, **fields)
    return reject(english, errors, hebrew, [record(code, column, "error", scope, **fields)])
