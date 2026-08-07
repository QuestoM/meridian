"""Every authored string this piece emits, in both languages, in one place.

Split out of ``adopt_candidate_rescore.py`` and ``adopt_candidate_adoption.py``
under the naming rule of section 8.2, for a reason worth stating: five display
fields in the frozen payload shape existed in English only. ``how_en`` on every
adoption check, ``rule_en`` on every verdict, ``metric_en`` in the evaluation
block, ``basis_en`` on both baselines and ``next_act.en`` in the registry. The
campaign law is that an authored string is two strings, and neither parent file
had the room to hold the second half: adoption was at 448 lines of a 450 cap.

Keeping the pairs here rather than beside their use has a second effect that is
worth more than the line count. A pair that lives in a table cannot drift into
one language, because adding a key without its partner is visible in one look
at one file, and a test walks this table and asserts both halves of every entry.

The terminal renders the English half, because the terminal is a company-side
tool and every other line on it is English. The payloads carry both halves, so
the Hebrew console that a route will one day serve has the string it needs
without a translation step and without this piece guessing at one.
"""

from __future__ import annotations

from typing import Any

# The verdict the common-basis re-score reaches, and what each one means.
VERDICTS: dict[str, dict[str, str]] = {
    "identical": {
        "en": "Predicts exactly what the shipped model predicts, break for break.",
        "he": "חוזה בדיוק את מה שהמודל המשודר חוזה, ברייק אחר ברייק.",
    },
    "better": {
        "en": "Closer to the measured effects than the shipped model, by more than the fold dispersion.",
        "he": "קרוב יותר לאפקטים הנמדדים מהמודל המשודר, ביותר מפיזור הקיפולים.",
    },
    "worse": {
        "en": "Further from the measured effects than the shipped model, by more than the fold dispersion.",
        "he": "רחוק יותר מהאפקטים הנמדדים מהמודל המשודר, ביותר מפיזור הקיפולים.",
    },
    "not_distinguishable": {
        "en": "Not distinguishable from the shipped model on this evaluation. The movement is inside the noise this data carries.",
        "he": "אינו ניתן להבחנה מהמודל המשודר במדידה הזו. התנועה נמצאת בתוך הרעש שהנתונים האלה נושאים.",
    },
}

# The sentence that decides how every figure on this surface may be read. There
# are three of them and which one is emitted is a measurement, taken in
# adopt_candidate_basis.py against what each artifact records it was fitted on.
# It was one constant until round 6, which asserted the first of these three of
# every tree, including the tree it is on, where it is false.
IN_SAMPLE_LIMIT: dict[str, str] = {
    "state": "in_sample",
    "en": "Every artifact scored here was fitted on all of these breaks, so each absolute figure is optimistic. Only the difference between two rows is readable, because both carry the same optimism.",
    "he": "כל קובץ שנמדד כאן אומן על כל הברייקים האלה, ולכן כל מספר מוחלט הוא אופטימי. רק ההפרש בין שתי שורות ניתן לקריאה, כי שתיהן נושאות את אותה אופטימיות.",
    "unblocked_by_en": "A second month of measured breaks that no artifact here was fitted on.",
    "unblocked_by_he": "חודש נוסף של ברייקים נמדדים שאף קובץ כאן לא אומן עליו.",
}

LIMIT_UNEVEN: dict[str, str] = {
    "en": "Not every artifact was fitted on all of these breaks, so the optimism is not the same in every row and a difference against one of the named rows carries a confound on top of the noise.",
    "he": "לא כל קובץ אומן על כל הברייקים האלה, ולכן האופטימיות אינה זהה בכל שורה, והפרש מול אחת השורות שנקובות כאן נושא הטיה נוספת מעבר לרעש.",
    "unblocked_by_en": "The identity of the breaks each artifact was fitted on, recorded in the artifact rather than the count alone. The counts are recorded here and the identities are not, so the size of this confound is not computable from anything on disk.",
    "unblocked_by_he": "זהות הברייקים שכל קובץ אומן עליהם, רשומה בתוך הקובץ ולא רק המניין. המניינים רשומים כאן והזהויות לא, ולכן גודל ההטיה הזו אינו ניתן לחישוב מדבר שנמצא על הדיסק.",
}

LIMIT_UNKNOWN: dict[str, str] = {
    "en": "At least one artifact records nothing about how many breaks it was fitted on, so whether the optimism is common to every row is unknown here rather than established.",
    "he": "לפחות קובץ אחד אינו רושם דבר על מספר הברייקים שאומן עליהם, ולכן השאלה אם האופטימיות משותפת לכל שורה אינה ידועה כאן ולא הוכחה.",
    "unblocked_by_en": "The number of breaks each artifact was fitted on, recorded in the artifact by whatever produced it.",
    "unblocked_by_he": "מספר הברייקים שכל קובץ אומן עליהם, רשום בתוך הקובץ על ידי מה שהפיק אותו.",
}

# What an artifact's own producer recorded about adopting it. Carried per row and
# never ranked: a self-test is its own split under its own fit, and ranking two
# of those against each other is the mistake the common-basis re-score exists to
# stop. A recommendation is not a comparison, and it is the one thing about an
# artifact that only the person who produced it knows.
SELF_TEST: dict[str, dict[str, str]] = {
    "advised_against": {
        "en": "Whatever produced this artifact recorded its own out-of-sample test and advised against adopting it.",
        "he": "מה שהפיק את הקובץ הזה רשם בדיקה מחוץ למדגם משלו והמליץ שלא לאמץ אותו.",
    },
    "recommended": {
        "en": "Whatever produced this artifact recorded its own out-of-sample test and recommended adopting it.",
        "he": "מה שהפיק את הקובץ הזה רשם בדיקה מחוץ למדגם משלו והמליץ לאמץ אותו.",
    },
    "recorded_without_a_verdict": {
        "en": "This artifact records its own out-of-sample test and reaches no recommendation either way.",
        "he": "הקובץ הזה רושם בדיקה מחוץ למדגם משלו ואינו מגיע להמלצה לכאן או לכאן.",
    },
    "absent": {
        "en": "This artifact records no out-of-sample test of its own, so there is no recommendation from whatever produced it.",
        "he": "הקובץ הזה אינו רושם בדיקה מחוץ למדגם משלו, ולכן אין המלצה ממה שהפיק אותו.",
    },
}

# The sentence that keeps a self-test from being read as a rank. It travels with
# every self-test the surface prints, in both halves.
SELF_TEST_BASIS: dict[str, str] = {
    "en": "A self-test is the artifact's own split under its own fit, so it is readable about that artifact alone and is never comparable with another row here.",
    "he": "בדיקה עצמית היא הפיצול של הקובץ עצמו תחת האימון של עצמו, ולכן היא ניתנת לקריאה על אותו קובץ בלבד ולעולם אינה בת השוואה לשורה אחרת כאן.",
}

# The fit basis as a line under the table, filled from the measurement.
FIT_BASIS: dict[str, dict[str, str]] = {
    "fewer": {
        "en": "fitted on {fitted} of the {scored} breaks it is scored on, so {shortfall} of them were never in its fit and are in the fit of every row it is compared against",
        "he": "אומן על {fitted} מתוך {scored} הברייקים שהוא נמדד עליהם, ולכן {shortfall} מהם מעולם לא היו באימון שלו והם באימון של כל שורה שהוא מושווה מולה",
    },
    "all": {
        "en": "fitted on {fitted} breaks, as many as it is scored on",
        "he": "אומן על {fitted} ברייקים, כמספר הברייקים שהוא נמדד עליהם",
    },
    "unknown": {
        "en": "records no fit basis, so whether it was fitted on these breaks is unknown rather than assumed",
        "he": "אינו רושם בסיס אימון, ולכן השאלה אם אומן על הברייקים האלה אינה ידועה ולא מונחת",
    },
}

METRIC: dict[str, str] = {
    "en": "Root mean squared error against that measured effect, over the same breaks for every row.",
    "he": "שורש שגיאת הריבוע הממוצעת מול אותו אפקט נמדד, על אותם ברייקים בכל שורה.",
}

# What the spread of the target itself is, which is the figure that says how
# much of it any of these artifacts explains. Rendered beside the metric.
TARGET_SD: dict[str, str] = {
    "en": "The standard deviation of the measured effect itself. An rmse near this figure is a model that explains little beyond the mean.",
    "he": "סטיית התקן של האפקט הנמדד עצמו. שגיאה קרובה למספר הזה היא מודל שמסביר מעט מעבר לממוצע.",
}

BASIS: dict[str, dict[str, str]] = {
    "global_mean_loo": {
        "en": "Predicts each break from the other breaks, never from itself.",
        "he": "חוזה כל ברייק מתוך שאר הברייקים, לעולם לא מתוך עצמו.",
    },
    "cell_mean_loo": {
        "en": "Predicts each break from the other breaks in its own cell, never from itself.",
        "he": "חוזה כל ברייק מתוך שאר הברייקים בתא שלו, לעולם לא מתוך עצמו.",
    },
}

# The rule that decided a verdict, stated as a rule and then as the measurement
# that ran through it. ``{fields}`` are filled by the caller from the same
# figures the payload carries, so the sentence cannot say one thing and the
# table another.
RULE = {
    "identical": {
        "en": "Both artifacts predict the same value for every break.",
        "he": "שני הקבצים חוזים את אותו ערך בכל ברייק.",
    },
    "two_bars": {
        "en": "Distinguishable only when the paired statistic reaches {bar} and the movement in RMSE exceeds the fold dispersion. Measured: statistic {statistic}, movement {moved}, dispersion {dispersion}.",
        "he": "ניתן להבחנה רק כאשר הסטטיסטי המזווג מגיע ל {bar} והתנועה בשגיאה עולה על פיזור הקיפולים. נמדד: סטטיסטי {statistic}, תנועה {moved}, פיזור {dispersion}.",
    },
}

# What would clear a failed adoption check. Keyed by the check that failed, so
# a check carries a key rather than a sentence and the sentence has one home.
HOW: dict[str, dict[str, str]] = {
    "rescore": {
        "en": "python scripts/adopt_candidate.py rescore",
        "he": "python scripts/adopt_candidate.py rescore",
    },
    "measure": {
        "en": "python scripts/adopt_candidate.py measure {id}",
        "he": "python scripts/adopt_candidate.py measure {id}",
    },
    "record_verdict": {
        "en": "python scripts/adopt_candidate.py decide {id} --decision shipped --actor \"<name>\" --reason \"<sentence>\" --release-note-he \"<sentence>\"",
        "he": "python scripts/adopt_candidate.py decide {id} --decision shipped --actor \"<name>\" --reason \"<sentence>\" --release-note-he \"<sentence>\"",
    },
    "adopted_by": {
        "en": "--adopted-by \"<name>\"",
        "he": "--adopted-by \"<name>\"",
    },
    "reason": {
        "en": "--reason \"<sentence>\"",
        "he": "--reason \"<sentence>\"",
    },
    "actor": {
        "en": "--actor \"<name>\"",
        "he": "--actor \"<name>\"",
    },
    "release_note": {
        "en": "--release-note-he \"<sentence the operator side reads>\"",
        "he": "--release-note-he \"<המשפט שהצד התפעולי קורא>\"",
    },
    "rebuild_candidate": {
        "en": "Rebuild the candidate with the layers the shipped artifact carries.",
        "he": "יש לבנות מחדש את המועמד עם השכבות שהקובץ המשודר נושא.",
    },
    "owner_approval": {
        "en": "models/releases/owner_approvals/{id}.json with approved_revenue_delta set to the measured figure.",
        "he": "models/releases/owner_approvals/{id}.json עם approved_revenue_delta שנקוב בדיוק במספר הנמדד.",
    },
    "ownership_ruling": {
        "en": "The lead adds {path} to the row, and records the ruling at {file} with path set to that exact path.",
        "he": "ראש הריצה מוסיף את הקובץ לשורת הבעלות, ורושם את הפסיקה בקובץ {file} עם path שנקוב בדיוק באותו נתיב.",
    },
}

# What a money escalation says. It lives here with every other authored pair,
# and the act module reads it under its old name.
ESCALATION: dict[str, str] = {
    "en": "This adoption would move a shipped figure, so it stops here. Record the measured movement and the reason with the owner, then place the owner's approval under models/releases/owner_approvals/ naming that exact movement in shekels.",
    "he": "ההטמעה הזו תזיז מספר משודר, ולכן היא נעצרת כאן. יש לרשום מול הבעלים את התנועה הנמדדת ואת הסיבה, ואז להניח את אישור הבעלים תחת models/releases/owner_approvals/ עם אותה תנועה מדויקת בשקלים.",
}

# The one act that would move a candidate forward, per state it is in. Never a
# suggestion to adopt: adoption runs its own checks and this table would be
# guessing at their outcome.
NEXT_ACT: dict[str, dict[str, str]] = {
    "rescore": {
        "en": "Re-score it against the shipped model on the common set of breaks.",
        "he": "יש למדוד אותו מחדש מול המודל המשודר על אותה קבוצת ברייקים.",
        "command": "python scripts/adopt_candidate.py rescore",
    },
    "measure": {
        "en": "Measure the money adopting it would move.",
        "he": "יש למדוד את הכסף שהטמעתו תזיז.",
        "command": "python scripts/adopt_candidate.py measure {id}",
    },
    "decide": {
        "en": "Record a ship or no-ship verdict against the model version on disk.",
        "he": "יש לרשום הכרעה להשיק או לא להשיק מול גרסת המודל שעל הדיסק.",
        "command": "python scripts/adopt_candidate.py decide {id} --decision <shipped|not_shipped> --actor \"<name>\" --reason \"<sentence>\"",
    },
    "redecide": {
        "en": "The verdict on record was taken before this comparison existed. Record one on the common-basis re-score, or read the adoption checks.",
        "he": "ההכרעה הרשומה התקבלה לפני שההשוואה הזו התקיימה. יש לרשום הכרעה על בסיס המדידה המשותפת, או לקרוא את בדיקות ההטמעה.",
        "command": "python scripts/adopt_candidate.py decide {id} --decision <shipped|not_shipped> --actor \"<name>\" --reason \"<sentence>\"",
    },
    "checks": {
        "en": "Read the adoption checks.",
        "he": "יש לקרוא את בדיקות ההטמעה.",
        "command": "python scripts/adopt_candidate.py adopt {id} --adopted-by \"<name>\" --reason \"<sentence>\"",
    },
}

# What the verdict recorded from this terminal was taken on, carried into the
# decision record itself so a later reader knows which comparison it rests on.
DECISION_BASIS: dict[str, str] = {
    "en": "Taken on the common-basis re-score: every artifact scored on one identical set of breaks with one metric, not on each artifact's own self-reported held-out figures, which come from different splits and are not comparable.",
    "he": "התקבלה על בסיס המדידה המשותפת: כל קובץ נמדד על אותה קבוצת ברייקים בדיוק ובאותו מדד, ולא על המספרים המוחזקים שכל קובץ מדווח על עצמו, שמגיעים מפיצולים שונים ואינם ברי השוואה.",
}


# The two verdicts as words rather than as the store's keys. ``shipped`` and
# ``not_shipped`` are what the decision log holds and what the flag takes, and
# they were being printed back on the one screen where a steward is choosing
# between exactly these two, with the English key sitting inside the Hebrew
# sentence.
#
# **The Hebrew is the console's, verbatim, and it used to be this piece's own.**
# This terminal said שיגור and אי שיגור while the model console it writes into
# says להשיק and לא להשיק on its own controls and הושקה and לא הושקה on the
# record, at ``console-words.js:180`` and ``:190``. Two surfaces naming one act
# two ways is a divergence a steward walks straight into inside one session,
# because the verdict recorded here is the verdict that console renders. The
# console's file is frozen, so this piece adopts the console's pair rather than
# asking for an edit to a file it may not write.
DECISION_WORDS: dict[str, dict[str, str]] = {
    "shipped": {"en": "ship", "he": "להשיק"},
    "not_shipped": {"en": "no ship", "he": "לא להשיק"},
}


# What the stat column is and what would have to happen for a row to be called
# better or worse. The rule was computed on every row and rendered nowhere, so
# the table carried a bare statistic with no bar beside it and a word, "no
# difference", with nothing on screen saying what decided it.
PAIRED_LEGEND: dict[str, str] = {
    "en": "stat is the paired statistic on the per-break squared error. A row is called better or worse only when it reaches {bar} and its movement in rmse also exceeds its own fold dispersion, and anything else reads as no difference.",
    "he": "stat הוא הסטטיסטי המזווג על שגיאת הריבוע לכל ברייק. שורה נקראת טובה יותר או גרועה יותר רק כאשר הוא מגיע ל {bar} והתנועה בשגיאה עולה גם על פיזור הקיפולים שלה, וכל מצב אחר נקרא ללא הבדל.",
}


# What a cell key is, said wherever one is printed. The keys look like
# PrimeShow2_first_short and a reader who has not been told will read the first
# part as a channel. Measured on this tree: 36 keys, four programme classes,
# three break positions, three break lengths, and no channel name in any of them.
CELL_KEY_SHAPE: dict[str, str] = {
    "en": "A cell key is a programme class, a break position and a break length. It names no channel.",
    "he": "מפתח תא הוא סוג תוכנית, מיקום ברייק ואורך ברייק. הוא אינו נוקב בשום ערוץ.",
}

# The line that reads the point coefficient, so a moved number is reported with
# what moving it does, exactly as the credible bound already is.
CELL_READ_BY: dict[str, str] = {
    "en": "kairos/model/impact.py:321 reads the point coefficient as the retention cost itself.",
    "he": "השורה kairos/model/impact.py:321 קוראת את המקדם עצמו כעלות השימור.",
}

# What the coefficient delta amounts to. The cancelling reading is the one this
# whole block exists for: thirty-six cells that each move and net almost nothing
# is a different artifact from one cell that was fixed, and the metric alone
# cannot tell them apart.
CELL_READING: dict[str, dict[str, str]] = {
    "none": {
        "en": "No coefficient moves. The two artifacts hold the same number in every cell they share.",
        "he": "אף מקדם אינו זז. שני הקבצים מחזיקים את אותו מספר בכל תא משותף.",
    },
    "no_effect": {
        "en": "{moved} of {compared} cells hold a different number and none of them changes the squared error on any break scored here.",
        "he": "ב {moved} מתוך {compared} תאים יש מספר אחר, ואף אחד מהם אינו משנה את שגיאת הריבוע באף ברייק שנמדד כאן.",
    },
    "cancelling": {
        "en": "{moved} of {compared} cells hold a different number, and {cancelled} percent of the movement they make cancels between them. {named} cells carry {share} percent of it.",
        "he": "ב {moved} מתוך {compared} תאים יש מספר אחר, ו {cancelled} אחוזים מהתנועה שהם עושים מתקזזים ביניהם. {named} תאים נושאים {share} אחוזים ממנה.",
    },
    "spread": {
        "en": "{moved} of {compared} cells hold a different number, and {cancelled} percent of the movement they make cancels between them. It takes {named} cells to reach {share} percent of it, so no small set of cells carries this.",
        "he": "ב {moved} מתוך {compared} תאים יש מספר אחר, ו {cancelled} אחוזים מהתנועה שהם עושים מתקזזים ביניהם. צריך {named} תאים כדי להגיע ל {share} אחוזים ממנה, ולכן אין קבוצה קטנה של תאים שנושאת את זה.",
    },
}


# What a gate cell says when one artifact does not carry the key at all. A
# different fact from a key it carries with a different value, so it is a
# sentence and never a blank.
GATE_ABSENT: dict[str, str] = {
    "en": "does not carry it",
    "he": "אינו נושא אותו",
}


# What each held-out block counted. Three gates on this tree record their size
# under three different key names and they are three different things, so the
# unit travels with the figure and 34,560 is never read as 34,560 breaks.
HELD_OUT_UNITS: dict[str, str] = {
    "n_test": "breaks",
    "n_test_minutes": "minutes",
    "n_test_days": "days",
}


def gate_cell(value: Any, absent: bool, language: str = "en") -> str:
    """One side of a gate row: its value, or that this side has no such key.

    A boolean is lowered and a count is grouped, because ``True`` and ``2532``
    are how Python prints them and not how a person reads them, and a float is
    cut at six decimals so a p-value does not arrive with seventeen.
    """
    if absent:
        return GATE_ABSENT[language]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    return str(round(value, 6)) if isinstance(value, float) else str(value)


def size_cell(size: Any, unit: str, absent: bool, language: str = "en") -> str:
    """One side of a held-out row: how much it was measured on, and of what."""
    if absent:
        return GATE_ABSENT[language]
    return f"{gate_cell(size, False)} {HELD_OUT_UNITS.get(unit, unit or '')}".strip()


def when(value: Any) -> str:
    """A stored timestamp as a line of text: to the second, and nothing after.

    The store writes microseconds and an offset because it is a record. Three
    lines on this surface printed the same kind of stamp three different ways,
    one of them truncated and two of them carrying six digits nobody reads, so
    the cut happens once and in one place.
    """
    return str(value or "")[:19] or "not recorded"


def _fill(text: str, fields: dict[str, Any]) -> str:
    return text.format(**fields) if fields else text


def pair(table: dict[str, dict[str, str]], key: str, prefix: str,
         **fields: Any) -> dict[str, str]:
    """One entry of a table as two payload fields, ``<prefix>_en`` and ``_he``.

    An unknown or empty key returns empty strings rather than raising, because a
    check that has nothing to suggest is a real state and not a defect.
    """
    entry = table.get(str(key or ""))
    if not entry:
        return {f"{prefix}_en": "", f"{prefix}_he": ""}
    return {f"{prefix}_en": _fill(entry["en"], fields),
            f"{prefix}_he": _fill(entry["he"], fields)}


def next_act(key: str, **fields: Any) -> dict[str, str]:
    """The next act as the registry emits it: both languages and the command."""
    entry = NEXT_ACT[key]
    return {"en": entry["en"], "he": entry["he"],
            "command": _fill(entry["command"], fields)}
