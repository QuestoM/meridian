"""The words a goal-based order is spoken in, in both languages.

Written beside :mod:`kairos_api.campaigns_api_words` and read the same way: a
surface renders these sentences and never composes its own. Every refusal this
piece shares with the pacing board is READ from
:mod:`kairos_api.pacing_alerts_api_words` rather than restated here, because the
product must say the same thing about an unmeasurable audience whichever screen
asks. Only the sentences that are new to the goal-based order live in this file.

Three things are named here and nothing else.

**What kind of order this is.** A goal-based order states an outcome and carries
no spot list. A spot-list order states the lines. An order that states neither is
not yet an order, and the product says so rather than treating a blank as a
booking of nothing.

**Whether the booked goal fits inside the channel's expected rating.** This is a
supply verdict, not a promise. The product holds no per-campaign spot allocation
in the weekly plan, so it can say what the goal needs per day against what the
channel has per day, and it must not say what this campaign will receive.

**What the remaining figure rests on.** Measured, a ceiling, or the booked goal
itself, said plainly every time the number is shown.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# What kind of order this is
# ---------------------------------------------------------------------------

GOAL_BASED = "goal_based"
SPOT_LIST = "spot_list"
NOT_AN_ORDER_YET = "not_an_order_yet"

ORDER_KINDS = (
    {
        "value": GOAL_BASED,
        "label_en": "Goal based",
        "label_he": "מבוססת יעד",
        "meaning_en": "The agency stated the outcome and the channel owns the placement. This order carries no spot list, and none is invented for it.",
        "meaning_he": "הסוכנות נקבה בתוצאה והערוץ אחראי לשיבוץ. ההזמנה הזו אינה נושאת רשימת תשדירים, ואף אחת אינה מומצאת עבורה.",
    },
    {
        "value": SPOT_LIST,
        "label_en": "Spot list",
        "label_he": "רשימת תשדירים",
        "meaning_en": "The lines booked on this order are what the channel is accountable for.",
        "meaning_he": "השורות שהוזמנו בהזמנה הזו הן מה שהערוץ אחראי לו.",
    },
    {
        "value": NOT_AN_ORDER_YET,
        "label_en": "Not an order yet",
        "label_he": "עדיין לא הזמנה",
        "meaning_en": "This campaign states neither a rating-point goal nor a booked line, so there is nothing for the channel to be accountable for yet.",
        "meaning_he": "הקמפיין הזה אינו נוקב ביעד נקודות רייטינג ואינו נושא שורה שהוזמנה, ולכן אין עדיין דבר שהערוץ אחראי לו.",
    },
)

ORDER_KIND_VALUES = tuple(entry["value"] for entry in ORDER_KINDS)

COMPLETE_PATH_EN = "Set a rating-point goal and the audience it counts against, or add a flight with the lines that were booked."
COMPLETE_PATH_HE = "קבעו יעד נקודות רייטינג ואת הקהל שמולו הוא נספר, או הוסיפו טיסה עם השורות שהוזמנו."

NO_SPOT_LIST_EN = "A goal-based order carries no spot list. The channel places against the goal, so nothing here stands in for lines the agency did not book."
NO_SPOT_LIST_HE = "הזמנה מבוססת יעד אינה נושאת רשימת תשדירים. הערוץ משבץ מול היעד, ולכן דבר כאן אינו בא במקום שורות שהסוכנות לא הזמינה."

MEANS_WHAT_IT_SAYS_EN = "A quantity on a spot-list order is a negotiating position, because an agency asks for more prime than it expects to get. A goal-based order has no such position in it, so the number a trader types means what it says and the channel plans to deliver it rather than to discount it."
MEANS_WHAT_IT_SAYS_HE = "כמות בהזמנה מבוססת רשימת תשדירים היא עמדת מיקוח, כי סוכנות מבקשת יותר פריים ממה שהיא מצפה לקבל. בהזמנה מבוססת יעד אין עמדה כזו, ולכן המספר שסוחר מקליד אומר את מה שהוא אומר והערוץ מתכנן לספק אותו ולא להנחות אותו."

# ---------------------------------------------------------------------------
# Whether the goal fits inside the channel's expected rating
# ---------------------------------------------------------------------------

FITS = "fits"
TIGHT = "tight"
EXCEEDS_SUPPLY = "exceeds_supply"
UNKNOWN = "unknown"

FEASIBILITY_STATES = (
    {
        "value": FITS,
        "label_en": "Fits the supply",
        "label_he": "נכנס בהיצע",
        "meaning_en": "The goal needs less than half of the channel's expected rating on the days the flight has left.",
        "meaning_he": "היעד זקוק לפחות ממחצית הרייטינג הצפוי של הערוץ בימים שנותרו לטיסה.",
    },
    {
        "value": TIGHT,
        "label_en": "Tight against the supply",
        "label_he": "צמוד להיצע",
        "meaning_en": "The goal needs at least half of the channel's expected rating on the days the flight has left. It fits, and it leaves little room for anything else.",
        "meaning_he": "היעד זקוק למחצית לפחות מהרייטינג הצפוי של הערוץ בימים שנותרו לטיסה. הוא נכנס, ומשאיר מעט מקום לכל השאר.",
    },
    {
        "value": EXCEEDS_SUPPLY,
        "label_en": "Beyond the supply",
        "label_he": "מעבר להיצע",
        "meaning_en": "The goal needs more rating than the channel expects to have on the days the flight has left, so it cannot be delivered as booked.",
        "meaning_he": "היעד זקוק ליותר רייטינג ממה שהערוץ צופה שיהיה לו בימים שנותרו לטיסה, ולכן אי אפשר לספק אותו כפי שהוזמן.",
    },
    {
        "value": UNKNOWN,
        "label_en": "Not known",
        "label_he": "לא ידוע",
        "meaning_en": "The product cannot state this goal against the channel's rating, and says so rather than stating a figure it cannot derive.",
        "meaning_he": "המערכת אינה יכולה להעמיד את היעד הזה מול הרייטינג של הערוץ, ואומרת זאת במקום לנקוב בנתון שאינה יכולה לגזור.",
    },
)

FEASIBILITY_VALUES = tuple(entry["value"] for entry in FEASIBILITY_STATES)

EXCEEDS_PATH_EN = "Extend the flight, lower the goal, or spread it across more broadcast days."
EXCEEDS_PATH_HE = "האריכו את הטיסה, הקטינו את היעד, או פרסו אותו על יותר ימי שידור."

NOT_A_PROMISE_EN = "This is a supply verdict and not a promise of delivery. The weekly plan holds break counts and not per-campaign lines, so the product can say what the goal needs against what the channel has, and cannot say how much of it this order will receive."
NOT_A_PROMISE_HE = "זו קביעה על ההיצע ואינה הבטחת אספקה. התוכנית השבועית מחזיקה מספרי ברייקים ולא שורות לכל קמפיין, ולכן המערכת יכולה לומר למה היעד זקוק מול מה שיש לערוץ, ואינה יכולה לומר כמה מזה ההזמנה הזו תקבל."

SUPPLY_BASIS_EN = "The channel's expected rating is the planned segment rating the plan carries on those days, on the all-viewers base."
SUPPLY_BASIS_HE = "הרייטינג הצפוי של הערוץ הוא רייטינג המקטעים המתוכנן שהתוכנית נושאת באותם ימים, על בסיס כלל הצופים."

# ---------------------------------------------------------------------------
# What the remaining figure rests on
# ---------------------------------------------------------------------------

MEASURED = "measured"
BOOKED = "booked"

BASIS_MEASURED_EN = "Every broadcast day of this flight that has already run carries a per-spot source, so what is left to place is measured rather than assumed."
BASIS_MEASURED_HE = "לכל יום שידור של הטיסה הזו שכבר חלף יש מקור ברמת התשדיר, ולכן מה שנותר לשבץ נמדד ואינו מונח."

BASIS_CEILING_EN = "Broadcast days that have already run carry no per-spot source, so what is left to place is a ceiling and not a measured remainder."
BASIS_CEILING_HE = "ימי שידור שכבר חלפו אינם נושאים מקור ברמת התשדיר, ולכן מה שנותר לשבץ הוא תקרה ולא יתרה שנמדדה."

BASIS_BOOKED_EN = "The delivery ledger holds no broadcast day for this order, so what it has delivered is unknown and what is left to place is stated as the booked goal itself."
BASIS_BOOKED_HE = "ספר האספקה אינו מחזיק אף יום שידור עבור ההזמנה הזו, ולכן מה שסופק אינו ידוע ומה שנותר לשבץ נמסר כיעד שהוזמן עצמו."

# ---------------------------------------------------------------------------
# The seam into the placement engine
# ---------------------------------------------------------------------------

SEAM_EN = "The goal reaches the placement engine through one named seam and steers only where a break prefers to go. It never changes what a campaign is charged."
SEAM_HE = "היעד מגיע למנוע השיבוץ דרך תפר אחד בעל שם ומטה רק את המקום שאליו ברייק מעדיף ללכת. הוא לעולם אינו משנה את מה שקמפיין מחויב בו."

SEAM_INERT_EN = "No goal-based order has reached the placement engine, so the engine is placing exactly as it did before. A demo row is not a booking and never steers a real plan."
SEAM_INERT_HE = "אף הזמנה מבוססת יעד לא הגיעה למנוע השיבוץ, ולכן המנוע משבץ בדיוק כפי ששיבץ קודם. שורת הדגמה אינה הזמנה ולעולם אינה מטה תוכנית אמיתית."


def order_kind_entry(value: str) -> dict[str, Any]:
    """One order-kind record, or the not-an-order-yet record for anything unknown."""
    wanted = str(value or "").strip()
    for entry in ORDER_KINDS:
        if entry["value"] == wanted:
            return dict(entry)
    return dict(ORDER_KINDS[-1])


def feasibility_entry(value: str) -> dict[str, Any]:
    """One feasibility record, or the unknown record for anything unrecognised."""
    wanted = str(value or "").strip()
    for entry in FEASIBILITY_STATES:
        if entry["value"] == wanted:
            return dict(entry)
    return dict(FEASIBILITY_STATES[-1])


def basis_words(basis: str) -> dict[str, str]:
    """What the remaining figure rests on, in both languages.

    Returns an empty block for a basis this module holds no sentence for, so the
    caller falls through to the pacing board's own published refusal rather than
    paraphrasing it here.
    """
    table = {
        MEASURED: (BASIS_MEASURED_EN, BASIS_MEASURED_HE),
        "gap_in_elapsed": (BASIS_CEILING_EN, BASIS_CEILING_HE),
        "no_source": (BASIS_BOOKED_EN, BASIS_BOOKED_HE),
    }
    entry = table.get(str(basis or "").strip())
    if entry is None:
        return {}
    return {"basis_en": entry[0], "basis_he": entry[1]}


def seam_words(is_identity: bool) -> dict[str, str]:
    """What the seam is, and whether anything is currently travelling through it."""
    block = {"seam_en": SEAM_EN, "seam_he": SEAM_HE}
    if is_identity:
        block["inert_en"] = SEAM_INERT_EN
        block["inert_he"] = SEAM_INERT_HE
    return block


def vocabularies() -> dict[str, Any]:
    """Every closed list the goal-based order speaks, for a form and for a payload."""
    return {
        "order_kinds": [dict(entry) for entry in ORDER_KINDS],
        "feasibility_states": [dict(entry) for entry in FEASIBILITY_STATES],
        "no_spot_list_en": NO_SPOT_LIST_EN,
        "no_spot_list_he": NO_SPOT_LIST_HE,
        "means_what_it_says_en": MEANS_WHAT_IT_SAYS_EN,
        "means_what_it_says_he": MEANS_WHAT_IT_SAYS_HE,
        "not_a_promise_en": NOT_A_PROMISE_EN,
        "not_a_promise_he": NOT_A_PROMISE_HE,
        "supply_basis_en": SUPPLY_BASIS_EN,
        "supply_basis_he": SUPPLY_BASIS_HE,
        "complete_path_en": COMPLETE_PATH_EN,
        "complete_path_he": COMPLETE_PATH_HE,
    }
