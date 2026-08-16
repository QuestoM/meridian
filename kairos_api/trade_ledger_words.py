"""The words the credit ledger writes: levels, directions, units and reasons.

A helper of :mod:`kairos_api.trade_ledger`, declared under the section 8.2
naming rule and split out for the 450-line law. It holds every controlled value
a ledger entry may carry and the label and meaning a surface prints for one, in
both languages.

It imports nothing from the ledger, so the dependency runs one way: the ledger
reads its words, and the words know nothing about a frame, a lock or a file.
"""

from __future__ import annotations

# The three levels the trade holds credit at, at once. A campaign's credit, an
# advertiser's credit and an agency's credit are three separate balances; the
# domain document (docs/trade/domain.md section 8) records that an agency may
# spend its credit on a different campaign later, which is exactly why the
# balance cannot live on the deal that earned it.
CAMPAIGN = "campaign"
ADVERTISER = "advertiser"
AGENCY = "agency"
LEVELS = (CAMPAIGN, ADVERTISER, AGENCY)

# What an entry does to a balance. Two directions add and open a lot, two
# consume lots. There is no signed quantity anywhere: the direction carries the
# sign, so a negative number can never slip into a sum unremarked.
ACCRUE = "accrue"
UTILISE = "utilise"
EXPIRE = "expire"
ADJUST = "adjust"
DIRECTIONS = (ACCRUE, UTILISE, EXPIRE, ADJUST)

# The directions that open a lot of credit, and the ones that consume lots.
# An adjustment only ADDS. The correction that removes credit is written as a
# utilisation carrying ``manual_adjust``, so every balance-reducing movement
# lives under the one overdraft refusal and no path exists that takes a
# party's available credit below zero.
LOT_DIRECTIONS = frozenset({ACCRUE, ADJUST})
CONSUMING_DIRECTIONS = frozenset({UTILISE, EXPIRE})

# The unit is named on every entry and never assumed. Media value in shekels,
# airtime seconds and rating points are three different debts that happen to
# share a ledger; no arithmetic in this store ever adds across two of them.
ILS_MEDIA_VALUE = "ils_media_value"
SECONDS = "seconds"
RATING_POINTS = "rating_points"
UNITS = (ILS_MEDIA_VALUE, SECONDS, RATING_POINTS)

# Why the credit moved. Controlled, because a reason written as free text
# cannot be counted, compared or answered for.
POLICY_ACCRUAL = "policy_accrual"
ADDED_VALUE_GRANT = "added_value_grant"
SHORTFALL_CURE = "shortfall_cure"
PREEMPTION_CREDIT = "preemption_credit"
MANUAL_ADJUST = "manual_adjust"
EXPIRY = "expiry"
REASONS = (
    POLICY_ACCRUAL,
    ADDED_VALUE_GRANT,
    SHORTFALL_CURE,
    PREEMPTION_CREDIT,
    MANUAL_ADJUST,
    EXPIRY,
)

# Which reason a direction may carry. The commercial reasons sit on both sides
# of the ledger because they name the story of the credit, not its sign: a
# shortfall accrues a cure and the cure's delivery utilises it; a fixed
# added-value grant accrues and delivering the granted media utilises it. A
# policy accrual has no spending twin (spending it is the cure or the grant it
# funds), an expiry belongs only to the expire direction, and a manual
# adjustment is the only reason a hand may move credit by.
REASONS_FOR_DIRECTION: dict[str, frozenset[str]] = {
    ACCRUE: frozenset({POLICY_ACCRUAL, ADDED_VALUE_GRANT, SHORTFALL_CURE, PREEMPTION_CREDIT}),
    UTILISE: frozenset({SHORTFALL_CURE, ADDED_VALUE_GRANT, PREEMPTION_CREDIT, MANUAL_ADJUST}),
    ADJUST: frozenset({MANUAL_ADJUST}),
    EXPIRE: frozenset({EXPIRY}),
}

# A hand-written movement without a sentence saying why is a record that cannot
# be answered for, so the manual reason requires a note.
NOTE_REQUIRED = frozenset({MANUAL_ADJUST})


def reason_allowed(direction: str, reason: str) -> bool:
    return reason in REASONS_FOR_DIRECTION.get(direction, frozenset())


LEVEL_VOCABULARY = (
    {
        "value": CAMPAIGN,
        "label_en": "Campaign",
        "label_he": "קמפיין",
        "meaning_en": "Credit held against one campaign.",
        "meaning_he": "יתרת מייק גוד הרשומה מול קמפיין אחד.",
    },
    {
        "value": ADVERTISER,
        "label_en": "Advertiser",
        "label_he": "מפרסם",
        "meaning_en": "Credit the advertiser holds across all of its campaigns.",
        "meaning_he": "יתרה שהמפרסם מחזיק על פני כל הקמפיינים שלו.",
    },
    {
        "value": AGENCY,
        "label_en": "Agency",
        "label_he": "סוכנות",
        "meaning_en": "Credit the agency holds under its framework, spendable on a different campaign later.",
        "meaning_he": "יתרה שהסוכנות מחזיקה לפי הסכם המסגרת שלה, וניתן לנצל אותה בקמפיין אחר בהמשך.",
    },
)

DIRECTION_VOCABULARY = (
    {
        "value": ACCRUE,
        "label_en": "Accrual",
        "label_he": "צבירה",
        "meaning_en": "Credit enters the ledger and opens a lot with its own expiry.",
        "meaning_he": "זיכוי נכנס לספר ופותח מנה חדשה עם מועד פקיעה משלה.",
    },
    {
        "value": UTILISE,
        "label_en": "Utilisation",
        "label_he": "ניצול",
        "meaning_en": "Credit is spent, consuming the oldest lots still in force first.",
        "meaning_he": "זיכוי מנוצל, מהמנות הוותיקות שעודן בתוקף תחילה.",
    },
    {
        "value": EXPIRE,
        "label_en": "Expiry",
        "label_he": "פקיעה",
        "meaning_en": "Credit past its date is written off, naming the lots it dies from.",
        "meaning_he": "זיכוי שעבר את מועדו נמחק מהיתרה, תוך ציון המנות שמהן פקע.",
    },
    {
        "value": ADJUST,
        "label_en": "Adjustment",
        "label_he": "התאמה ידנית",
        "meaning_en": "A manual correction that adds credit. A correction that removes credit is a utilisation, so it cannot overdraw.",
        "meaning_he": "תיקון ידני שמוסיף זיכוי. תיקון שמפחית זיכוי נרשם כניצול, ולכן אינו יכול לחרוג מהיתרה.",
    },
)

UNIT_VOCABULARY = (
    {
        "value": ILS_MEDIA_VALUE,
        "label_en": "Media value (₪)",
        "label_he": "שווי מדיה בש״ח",
        "meaning_en": "Shekel value of media owed, not cash.",
        "meaning_he": "שווי מדיה בשקלים שחייבים ללקוח, לא מזומן.",
    },
    {
        "value": SECONDS,
        "label_en": "Seconds",
        "label_he": "שניות",
        "meaning_en": "Airtime seconds owed.",
        "meaning_he": "שניות שידור שחייבים ללקוח.",
    },
    {
        "value": RATING_POINTS,
        "label_en": "Rating points",
        "label_he": "נקודות רייטינג",
        "meaning_en": "Audience rating points owed.",
        "meaning_he": "נקודות רייטינג שחייבים ללקוח.",
    },
)

REASON_VOCABULARY = (
    {
        "value": POLICY_ACCRUAL,
        "label_en": "Policy accrual",
        "label_he": "צבירה לפי מדיניות",
        "meaning_en": "Accrued under the agreement's make-good accrual policy, at the rate the caller computed from it.",
        "meaning_he": "נצבר לפי מדיניות צבירת המייק גוד שבהסכם, בשיעור שחושב מתוכה.",
    },
    {
        "value": ADDED_VALUE_GRANT,
        "label_en": "Added-value media",
        "label_he": "מדיה נוספת",
        "meaning_en": "A fixed bonus-media grant, independent of any shortfall.",
        "meaning_he": "מענק מדיה נוספת קבוע, שאינו תלוי בחוסר כלשהו.",
    },
    {
        "value": SHORTFALL_CURE,
        "label_en": "Shortfall cure",
        "label_he": "השלמת חוסר",
        "meaning_en": "Credit owed for, or spent on, curing a measured shortfall.",
        "meaning_he": "זיכוי שנוצר בשל חוסר נמדד, או שנוצל כדי להשלים אותו.",
    },
    {
        "value": PREEMPTION_CREDIT,
        "label_en": "Pre-emption credit",
        "label_he": "זיכוי הקדמת שידור",
        "meaning_en": "A spot the control room pulled or moved; the remedy accrues or is delivered here.",
        "meaning_he": "תשדיר שחדר הבקרה הוריד או הזיז; הפיצוי נצבר או מסופק כאן.",
    },
    {
        "value": MANUAL_ADJUST,
        "label_en": "Manual adjustment",
        "label_he": "תיקון ידני",
        "meaning_en": "A person corrected the balance, and the note says why.",
        "meaning_he": "אדם תיקן את היתרה, וההערה אומרת מדוע.",
    },
    {
        "value": EXPIRY,
        "label_en": "Expiry",
        "label_he": "פקיעה",
        "meaning_en": "The credit passed its expiry date unspent.",
        "meaning_he": "הזיכוי עבר את מועד הפקיעה מבלי שנוצל.",
    },
)
