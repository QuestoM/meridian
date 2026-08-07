"""Clients, pacing: every word and every published threshold the board prints.

The strings live here rather than in the board so that the computation carries no
copy and the copy carries no arithmetic. Two rules hold for everything below.

**A threshold is a policy, not a measurement.** The two pace triggers are the
product's stated rule and they are published on the payload so the reader can see
the number that decided the verdict. They are not a commercial term anybody
signed, and the payload says so; the commercial trigger is owner-pending.

**A reason is a route, not an apology.** Every unavailable and unknown state
carries the sentence that names what is missing and the sentence that names how
to supply it, in both languages, so a surface can render an empty state as a
control rather than as prose.
"""

from __future__ import annotations

from typing import Any

# The published pace triggers. counted-through-today divided by the even-flight
# reference for the same window. A ratio at or above ON_PACE is on pace; below
# AT_RISK is behind; between them is at risk.
ON_PACE_RATIO = 0.95
AT_RISK_RATIO = 0.85

ON_PACE = "on_pace"
AT_RISK = "at_risk"
BEHIND = "behind"
UNKNOWN = "unknown"

# What the flight's remaining days say, independently of the pace to date.
COVERED = "covered"
SHORT_CERTAIN = "short_certain"
NOT_BOOKED_YET = "not_booked_yet"

RATING_POINTS = "rating_points"
ILS = "ils"

TRIGGER_EN = (
    "Counted through the end of the last sourced broadcast day, divided by an even share of the goal "
    "over the flight's broadcast days. At or above 95 percent is on pace, below 85 percent is behind."
)
TRIGGER_HE = (
    "הנספר עד סוף יום השידור האחרון שיש לו מקור, חלקי חלק שווה מהיעד על פני ימי השידור של הטיסה. "
    "95 אחוז ומעלה בקצב, מתחת ל-85 אחוז מפגר."
)
TRIGGER_OWNER_EN = (
    "These two triggers are the product's stated rule and not a commercial term. The commercial "
    "trigger is owner-pending."
)
TRIGGER_OWNER_HE = (
    "שני הספים האלה הם הכלל המוצהר של המוצר ואינם תנאי מסחרי. הסף המסחרי ממתין להחלטת הבעלים."
)

EVEN_REFERENCE_EN = "An even share of the goal across the flight's broadcast days."
EVEN_REFERENCE_HE = "חלק שווה מהיעד על פני ימי השידור של הטיסה."

COUNTED_BASIS_EN = (
    "Rating points are the planned break rating the traffic log carries, on the all-viewers base. "
    "Money is engine-priced from the same per-spot ledger the money board reads. Nothing is invoiced."
)
COUNTED_BASIS_HE = (
    "נקודות הרייטינג הן רייטינג הברייק המתוכנן שיומן השידור נושא, על בסיס כלל הצופים. הכסף מתומחר "
    "במנוע מאותו ספר תשדירים שלוח הכספים קורא. דבר אינו מחויב בחשבונית."
)

NO_GOAL_EN = "This campaign carries no goal in this unit, so there is nothing to pace against."
NO_GOAL_HE = "הקמפיין הזה אינו נושא יעד ביחידה הזו, ולכן אין מול מה למדוד קצב."
NO_GOAL_PATH_EN = "Open the campaign and set a goal on its flight."
NO_GOAL_PATH_HE = "פתחו את הקמפיין וקבעו יעד על הטיסה שלו."

UNMEASURABLE_EN = (
    "The goal names a target audience this product holds no panel breakdown for, so pace against it "
    "is unknown rather than measured against a different base."
)
UNMEASURABLE_HE = (
    "היעד נוקב בקהל יעד שאין למערכת עבורו פילוח פאנל, ולכן הקצב מולו אינו ידוע ואינו נמדד מול בסיס אחר."
)
UNMEASURABLE_PATH_EN = "Supply a panel breakdown for that audience, or restate the goal on the all-viewers base."
UNMEASURABLE_PATH_HE = "ספקו פילוח פאנל לקהל הזה, או נסחו את היעד מחדש על בסיס כלל הצופים."

NO_SOURCE_EN = "No broadcast day of this flight carries a per-spot source, so what it delivered is unknown."
NO_SOURCE_HE = "אף יום שידור של הטיסה הזו אינו נושא מקור ברמת התשדיר, ולכן מה שסופק אינו ידוע."
NO_SOURCE_PATH_EN = (
    "Upload a daily traffic file for the flight days, then run scripts/seed_campaigns.py to build the "
    "delivery ledger from it."
)
NO_SOURCE_PATH_HE = (
    "העלו קובץ שידור יומי לימי הטיסה, ואז הריצו את scripts/seed_campaigns.py כדי לבנות ממנו את ספר "
    "האספקה."
)

GAP_IN_ELAPSED_EN = (
    "Broadcast days that have already run carry no per-spot source, so what was delivered to date is "
    "a floor and no pace can be stated against it."
)
GAP_IN_ELAPSED_HE = (
    "ימי שידור שכבר חלפו אינם נושאים מקור ברמת התשדיר, ולכן מה שסופק עד כה הוא רף תחתון ואי אפשר "
    "לקבוע מולו קצב."
)

NOT_STARTED_EN = "The flight has not started at the instant this ledger was counted at."
NOT_STARTED_HE = "הטיסה טרם החלה ברגע שבו נספר ספר האספקה הזה."
NOT_STARTED_PATH_EN = "Upload a traffic file for a day inside the flight to start counting it."
NOT_STARTED_PATH_HE = "העלו קובץ שידור ליום שנמצא בתוך הטיסה כדי להתחיל לספור אותה."

NO_FLIGHT_DATES_EN = "This campaign carries no start or end date, so it has no flight to pace across."
NO_FLIGHT_DATES_HE = "הקמפיין הזה אינו נושא תאריך התחלה או סיום, ולכן אין לו טיסה שעליה נמדד קצב."
NO_FLIGHT_DATES_PATH_EN = "Open the campaign and set its flight dates."
NO_FLIGHT_DATES_PATH_HE = "פתחו את הקמפיין וקבעו את תאריכי הטיסה שלו."

FORWARD_COVERED_EN = "The spots already on the traffic log reach the goal."
FORWARD_COVERED_HE = "התשדירים שכבר ביומן השידור מגיעים ליעד."
FORWARD_SHORT_EN = (
    "Every remaining broadcast day of this flight has a source and the spots on them do not reach the "
    "goal, so the shortfall is measured rather than projected."
)
FORWARD_SHORT_HE = (
    "לכל יום שידור שנותר בטיסה הזו יש מקור והתשדירים שבהם אינם מגיעים ליעד, ולכן החוסר נמדד ואינו תחזית."
)
FORWARD_OPEN_EN = "Remaining broadcast days carry no source, so the rest of the flight cannot be projected."
FORWARD_OPEN_HE = "ימי שידור שנותרו אינם נושאים מקור, ולכן אי אפשר לחזות את המשך הטיסה."
FORWARD_OPEN_PATH_EN = "Book the remaining days, or upload the traffic file that already holds them."
FORWARD_OPEN_PATH_HE = "הזמינו את הימים שנותרו, או העלו את קובץ השידור שכבר מחזיק אותם."

PACE_VERDICTS = (
    {
        "value": ON_PACE,
        "label_en": "On pace",
        "label_he": "בקצב",
        "meaning_en": "Counted delivery is at or above the even-flight reference for the days counted.",
        "meaning_he": "האספקה הנספרת שווה או גבוהה מהייחוס של טיסה אחידה לימים שנספרו.",
    },
    {
        "value": AT_RISK,
        "label_en": "At risk",
        "label_he": "בסיכון",
        "meaning_en": "Counted delivery is below the reference and above the behind trigger.",
        "meaning_he": "האספקה הנספרת נמוכה מהייחוס וגבוהה מסף הפיגור.",
    },
    {
        "value": BEHIND,
        "label_en": "Behind",
        "label_he": "מפגר",
        "meaning_en": "Counted delivery is below the behind trigger for the days counted.",
        "meaning_he": "האספקה הנספרת נמוכה מסף הפיגור לימים שנספרו.",
    },
    {
        "value": UNKNOWN,
        "label_en": "Unknown",
        "label_he": "לא ידוע",
        "meaning_en": "Something the pace needs is missing, and the row names which one.",
        "meaning_he": "משהו שהקצב זקוק לו חסר, והשורה נוקבת במה שחסר.",
    },
)

FORWARD_STATES = (
    {
        "value": COVERED,
        "label_en": "Booked to goal",
        "label_he": "מוזמן עד היעד",
        "meaning_en": "The spots already on the traffic log reach the goal.",
        "meaning_he": "התשדירים שכבר ביומן השידור מגיעים ליעד.",
    },
    {
        "value": SHORT_CERTAIN,
        "label_en": "Short by a measured amount",
        "label_he": "חסר בסכום נמדד",
        "meaning_en": "Every remaining day has a source and the flight still does not reach the goal.",
        "meaning_he": "לכל יום שנותר יש מקור והטיסה עדיין אינה מגיעה ליעד.",
    },
    {
        "value": NOT_BOOKED_YET,
        "label_en": "Remaining days not sourced",
        "label_he": "לימים שנותרו אין מקור",
        "meaning_en": "Remaining broadcast days carry no source, so the rest cannot be projected.",
        "meaning_he": "ימי שידור שנותרו אינם נושאים מקור, ולכן אי אפשר לחזות את ההמשך.",
    },
)

UNITS = (
    {
        "value": RATING_POINTS,
        "label_en": "rating points",
        "label_he": "נקודות רייטינג",
    },
    {
        "value": ILS,
        "label_en": "ILS",
        "label_he": "שקלים",
    },
)


def reason(code: str) -> dict[str, Any]:
    """One unavailable or unknown state, as the four strings a surface renders."""
    table = {
        "no_goal": (NO_GOAL_EN, NO_GOAL_HE, NO_GOAL_PATH_EN, NO_GOAL_PATH_HE),
        "unmeasurable": (UNMEASURABLE_EN, UNMEASURABLE_HE, UNMEASURABLE_PATH_EN, UNMEASURABLE_PATH_HE),
        "no_source": (NO_SOURCE_EN, NO_SOURCE_HE, NO_SOURCE_PATH_EN, NO_SOURCE_PATH_HE),
        "gap_in_elapsed": (GAP_IN_ELAPSED_EN, GAP_IN_ELAPSED_HE, NO_SOURCE_PATH_EN, NO_SOURCE_PATH_HE),
        "not_started": (NOT_STARTED_EN, NOT_STARTED_HE, NOT_STARTED_PATH_EN, NOT_STARTED_PATH_HE),
        "no_flight_dates": (NO_FLIGHT_DATES_EN, NO_FLIGHT_DATES_HE, NO_FLIGHT_DATES_PATH_EN, NO_FLIGHT_DATES_PATH_HE),
    }
    entry = table.get(code)
    if entry is None:
        return {"code": "", "reason_en": "", "reason_he": "", "path_forward_en": "", "path_forward_he": ""}
    return {
        "code": code,
        "reason_en": entry[0],
        "reason_he": entry[1],
        "path_forward_en": entry[2],
        "path_forward_he": entry[3],
    }


def trigger_block() -> dict[str, Any]:
    """The two published pace triggers, and the sentence that says what they are not."""
    return {
        "on_pace_ratio": ON_PACE_RATIO,
        "at_risk_ratio": AT_RISK_RATIO,
        "rule_en": TRIGGER_EN,
        "rule_he": TRIGGER_HE,
        "not_a_commercial_term_en": TRIGGER_OWNER_EN,
        "not_a_commercial_term_he": TRIGGER_OWNER_HE,
    }


def vocabularies() -> dict[str, Any]:
    """Every controlled word this surface prints, so no screen invents a label."""
    return {
        "pace_verdicts": [dict(entry) for entry in PACE_VERDICTS],
        "forward_states": [dict(entry) for entry in FORWARD_STATES],
        "units": [dict(entry) for entry in UNITS],
    }
