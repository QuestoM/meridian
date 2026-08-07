"""The words, the states and the published sentences the decision ledger uses.

A helper of :mod:`kairos_api.makegood_store`, declared under the section 8.2
naming rule and split out for the 450-line law. It holds every controlled value
this ledger stores, every label and meaning a surface prints for one, and the two
paragraphs that say what this product was never given the rule for.

It imports nothing from the store, so the dependency runs one way: the store
reads its words, and the words know nothing about a frame, a lock or a file.
"""

from __future__ import annotations

MAKE_GOOD = "make_good"
ACCEPTANCE = "acceptance"

RAISED = "raised"
OFFERED = "offered"
SETTLED = "settled"
DECLINED = "declined"
WITHDRAWN = "withdrawn"
ACCEPTED = "accepted"

# The entry state each kind is written in. Nothing else may be written directly.
ENTRY_STATE = {MAKE_GOOD: RAISED, ACCEPTANCE: ACCEPTED}

# Which state may follow which. A settled or withdrawn record is finished, so it
# has no successor: reopening one would rewrite a record somebody acted on. One
# table covers both kinds because no state is shared between their two paths
# except the ending, so a state names its kind without ambiguity.
TRANSITIONS: dict[str, frozenset[str]] = {
    RAISED: frozenset({OFFERED, WITHDRAWN}),
    OFFERED: frozenset({SETTLED, DECLINED, WITHDRAWN}),
    DECLINED: frozenset({OFFERED, WITHDRAWN}),
    ACCEPTED: frozenset({WITHDRAWN}),
    SETTLED: frozenset(),
    WITHDRAWN: frozenset(),
}

# The states that require an offer to exist before they can be entered, because
# settling or declining nothing is not a thing a person can mean.
NEEDS_OFFER = frozenset({SETTLED, DECLINED})

# The two transitions that close a record without any delivery being made, and
# that no later act can undo. Both take a controlled reason.
#
# Measured cause: a withdrawal fired on one click, took an optional free-text
# note and no reason at all, and could not be reversed, so a record somebody
# removed from the ledger was unauditable. The reference for this is Stripe,
# which requires a reason on every refund, requires a note when that reason is
# other, and confirms a cancellation before it fires.
OTHER = "other"
REASON_REQUIRED = frozenset({DECLINED, WITHDRAWN})

CLOSE_REASONS: dict[str, tuple[dict[str, str], ...]] = {
    DECLINED: (
        {
            "value": "offer_too_small",
            "label_en": "The offer was too small",
            "label_he": "ההצעה הייתה קטנה מדי",
        },
        {
            "value": "window_does_not_work",
            "label_en": "The window does not work for the client",
            "label_he": "החלון אינו מתאים ללקוח",
        },
        {
            "value": OTHER,
            "label_en": "Another reason, written below",
            "label_he": "סיבה אחרת, שנכתבת למטה",
        },
    ),
    WITHDRAWN: (
        {
            "value": "opened_in_error",
            "label_en": "The record should not have been opened",
            "label_he": "לא היה מקום לפתוח את הרשומה",
        },
        {
            "value": "no_longer_stands",
            "label_en": "The figure it was opened against no longer stands",
            "label_he": "הנתון שמולו נפתחה כבר אינו עומד",
        },
        {
            "value": OTHER,
            "label_en": "Another reason, written below",
            "label_he": "סיבה אחרת, שנכתבת למטה",
        },
    ),
}


def close_reasons(state: str) -> tuple[dict[str, str], ...]:
    """The reasons one closing transition may carry, or none when it needs none."""
    return CLOSE_REASONS.get(state, ())


def reason_allowed(state: str, reason: str) -> bool:
    return reason in {entry["value"] for entry in close_reasons(state)}

MEASURED_CLOSED = "measured_closed"
BOOKED_SHORT = "booked_short"
TO_DATE = "to_date"

STATE_VOCABULARY = (
    {
        "value": RAISED,
        "label_en": "Raised",
        "label_he": "נפתח",
        "meaning_en": "The shortfall is recorded with the figures it was measured from.",
        "meaning_he": "החוסר נרשם יחד עם הנתונים שממנו נמדד.",
    },
    {
        "value": OFFERED,
        "label_en": "Offered",
        "label_he": "הוצעה השלמה",
        "meaning_en": "Compensating delivery has been offered to the client.",
        "meaning_he": "הוצעה ללקוח השלמת שידור.",
    },
    {
        "value": SETTLED,
        "label_en": "Settled",
        "label_he": "נסגר",
        "meaning_en": "The client accepted the offer and the make-good is closed.",
        "meaning_he": "הלקוח קיבל את ההצעה והפיצוי נסגר.",
    },
    {
        "value": DECLINED,
        "label_en": "Declined",
        "label_he": "נדחה",
        "meaning_en": "The client refused the offer. A new offer may follow it.",
        "meaning_he": "הלקוח דחה את ההצעה. אפשר להציע הצעה חדשה אחריה.",
    },
    {
        "value": WITHDRAWN,
        "label_en": "Withdrawn",
        "label_he": "בוטל",
        "meaning_en": "The record should not have been opened, and the row records who withdrew it.",
        "meaning_he": "לא היה מקום לפתוח את הרשומה, והשורה רושמת מי ביטל אותה.",
    },
    {
        "value": ACCEPTED,
        "label_en": "Risk taken on",
        "label_he": "הסיכון התקבל",
        "meaning_en": "A person read the row and decided the risk stands as it is. The figures are the ones the board showed at that instant.",
        "meaning_he": "אדם קרא את השורה והחליט שהסיכון נשאר כפי שהוא. הנתונים הם אלה שהלוח הציג באותו רגע.",
    },
)

KIND_VOCABULARY = (
    {
        "value": MAKE_GOOD,
        "label_en": "Make-good",
        "label_he": "פיצוי שידור",
        "meaning_en": "Compensating delivery raised against a measured shortfall.",
        "meaning_he": "השלמת שידור שנפתחה מול חוסר נמדד.",
    },
    {
        "value": ACCEPTANCE,
        "label_en": "Risk taken on",
        "label_he": "קבלת הסיכון",
        "meaning_en": "A recorded decision to leave the risk on this campaign as it stands.",
        "meaning_he": "החלטה רשומה להשאיר את הסיכון בקמפיין הזה כפי שהוא.",
    },
)

DEFICIT_KIND_VOCABULARY = (
    {
        "value": MEASURED_CLOSED,
        "label_en": "Measured, flight closed",
        "label_he": "נמדד, הטיסה נסגרה",
        "meaning_en": "The flight has ended and every broadcast day of it carries a source, so the figure is final.",
        "meaning_he": "הטיסה הסתיימה ולכל יום שידור שלה יש מקור, ולכן הנתון סופי.",
    },
    {
        "value": BOOKED_SHORT,
        "label_en": "Booked short",
        "label_he": "ההזמנה חסרה",
        "meaning_en": "Every remaining day carries a source and the spots on them do not reach the goal.",
        "meaning_he": "לכל יום שנותר יש מקור והתשדירים שבהם אינם מגיעים ליעד.",
    },
    {
        "value": TO_DATE,
        "label_en": "Gap to date",
        "label_he": "פער עד כה",
        "meaning_en": "The gap against the pace reference at the counted day. It moves as the flight runs.",
        "meaning_he": "הפער מול ייחוס הקצב ביום שנספר. הוא משתנה ככל שהטיסה מתקדמת.",
    },
)

SIGN_OFF_EN = (
    "This product was not given the commercial rule for what a make-good may be offered against or who "
    "signs one off, so it records who acted and refuses to derive an entitlement."
)
SIGN_OFF_HE = (
    "המוצר לא קיבל את הכלל המסחרי לגבי מה מותר להציע כפיצוי או מי מאשר אותו, ולכן הוא רושם מי פעל "
    "ונמנע מלגזור זכאות."
)
SIGN_OFF_PATH_EN = "Supply the offer rule and the approver role, and this ledger will enforce them."
SIGN_OFF_PATH_HE = "ספקו את כלל ההצעה ואת תפקיד המאשר, וספר הפיצויים יאכוף אותם."

NO_INVENTORY_EN = (
    "An offer is a recorded value and a window. It does not reserve inventory: the spots are booked on "
    "the plan, and this row links to the campaign they belong to."
)
NO_INVENTORY_HE = (
    "הצעה היא ערך רשום וחלון תאריכים. היא אינה משריינת מלאי: התשדירים מוזמנים בתוכנית, והשורה הזו "
    "מקשרת לקמפיין שאליו הם שייכים."
)
