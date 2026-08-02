"""What the door checks, and what it cannot check, per input kind.

Split out of ``uploads.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule.

The reference for this module is Frame.io publishing its proxy ladder exactly:
the rungs, the codecs, the ceilings, and the plain statement that files above
eight audio channels will not play at all. A product that says what it cannot
do, in exact terms, is trusted on what it says it can.

So every input declares three things, derived from the code that really runs
rather than from a hand-written list that would drift on the first change: the
columns the header gate requires, the engine loader the file is parsed with,
and the data contract the loaded frame is checked against. Beside them sits the
short list of things the check genuinely does not verify, because an operator
who believes a green tick means the numbers are right will trust a file nobody
checked the meaning of.
"""

from __future__ import annotations

from typing import Any

# The three sentences a card renders under "what this check does not verify".
# They are the same for every kind because they are true of every kind: the
# gate proves a file loads and satisfies its contract, and it can prove nothing
# about whether the values in it are the right values.
CANNOT_VERIFY: tuple[dict[str, str], ...] = (
    {
        "code": "values_are_true",
        "en": "Whether the values are the right values. A file that loads cleanly can still carry last week's numbers.",
        "he": "האם הערכים הם הערכים הנכונים. קובץ שנטען כראוי עדיין יכול לשאת את המספרים של השבוע שעבר.",
    },
    {
        "code": "completeness",
        "en": "Whether the file is complete. A day or a programme missing from the export is missing here too.",
        "he": "האם הקובץ שלם. יום או תוכנית שחסרים בייצוא חסרים גם כאן.",
    },
    {
        "code": "duplicates_across_files",
        "en": "Whether it repeats something an earlier file already carried. Each upload is checked on its own.",
        "he": "האם הוא חוזר על משהו שקובץ קודם כבר נשא. כל העלאה נבדקת בפני עצמה.",
    },
)

LOADER_NAMES: dict[str, str] = {
    "programmes": "kairos.data.loaders.load_programmes",
    "spots": "kairos.data.loaders.load_spots",
    "dayparts": "kairos.data.loaders.load_dayparts",
    "daily": "kairos.data.loaders.load_daily_input",
    "campaign_flights": "kairos.optimize.pacing.load_campaigns",
}

CONTRACT_NAMES: dict[str, str] = {
    "programmes": "kairos.data.contracts.validate_programmes",
    "spots": "kairos.data.contracts.validate_spots",
    "dayparts": "kairos.data.contracts.validate_dayparts",
    "daily": "kairos.data.contracts.validate_daily_input",
}

# A kind with no loader-backed contract gets the header gate and nothing more,
# and the card says exactly that rather than implying a check that never runs.
HEADER_ONLY = {
    "en": "The required columns, and nothing else. This kind has no loader-backed contract, so a row with a wrong value is stored as it is.",
    "he": "העמודות הנדרשות, ותו לא. לסוג הזה אין חוזה מבוסס טוען, ולכן שורה עם ערך שגוי נשמרת כפי שהיא.",
}

WITH_CONTRACT = {
    "en": "The required columns, then the file is parsed with the engine's own loader and checked against its data contract. An error-severity finding refuses the file before it replaces anything.",
    "he": "העמודות הנדרשות, ואז הקובץ נקרא עם הטוען של המנוע עצמו ונבדק מול חוזה הנתונים שלו. ממצא בחומרת שגיאה דוחה את הקובץ לפני שהוא מחליף משהו.",
}


def checks_for(kind: str, required_columns: list[str]) -> dict[str, Any]:
    """What the door runs for this kind, and what it does not answer.

    ``loader`` and ``contract`` are the real dotted names of the code that
    runs, so a reader who wants to know exactly what was checked can go and
    read it, and a rename that is not reflected here is visible.
    """
    loader = LOADER_NAMES.get(kind)
    contract = CONTRACT_NAMES.get(kind)
    checked = WITH_CONTRACT if loader else HEADER_ONLY
    return {
        "required_columns": list(required_columns),
        "loader": loader,
        "contract": contract,
        "checked_en": checked["en"],
        "checked_he": checked["he"],
        "cannot_verify": [dict(entry) for entry in CANNOT_VERIFY],
    }
