"""The trade-term taxonomy: machine twin of docs/trade/term-taxonomy.md.

One TermSpec per commercial term an agreement can carry. The extraction
pipeline classifies clauses against exactly this registry; the review surface
renders its Hebrew names; the rule compiler dispatches on term id. The
markdown catalogue is the human contract and tests/test_trade_taxonomy.py
pins the two to the same term list, so neither drifts silently.

Statuses are the honesty vocabulary of the catalogue:

- BINDS         representable AND changes behaviour through existing machinery
- REPRESENTABLE the model holds it faithfully; binding path wired this campaign
- TRACKED       stored, measured and surfaced; consequences stay human-driven
- RECORDED      stored and displayed with deadline tracking only
- NOT_APPLICABLE a known foreign structure with no Israeli evidence; classified
                 by name so a clause matching it is refused with a reason,
                 never silently dropped

Ranks are provenance: IL (Israeli primary source), TRADE (the owner/media
professional transcript), STD (standard practice not yet attested locally —
UI copy must not assert STD terms as local market fact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

FAMILIES: Mapping[str, str] = {
    "A": "זהות, היקף ומסמך",
    "B": "בסיס הכסף",
    "C": "הנחות, עמלות ותמריצים",
    "D": "התחייבויות המפרסם",
    "E": "התחייבויות הערוץ והשלמות",
    "F": "אילוצי שיבוץ",
    "G": "תהליך ומשפט",
    "H": "מדידה והתחשבנות",
    "NA": "לא רלוונטי לשוק הישראלי",
}

STATUSES = ("BINDS", "REPRESENTABLE", "TRACKED", "RECORDED", "NOT_APPLICABLE")
RANKS = ("IL", "TRADE", "STD")
BEHAVIOURS = (
    "prices",
    "constrains-hard",
    "constrains-soft",
    "obliges",
    "settles",
    "process",
    "meta",
)


@dataclass(frozen=True)
class TermSpec:
    """One commercial term the taxonomy recognises."""

    id: str
    family: str
    name_he: str
    name_en: str
    behaviours: tuple[str, ...]
    status: str
    rank: str
    # Hebrew phrases that signal the term in a clause. Grounding for the
    # classifier and highlighting for the reviewer; never a substitute for
    # model classification.
    cues: tuple[str, ...] = ()
    interacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r} for term {self.id}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r} for term {self.id}")
        if self.rank not in RANKS:
            raise ValueError(f"unknown rank {self.rank!r} for term {self.id}")
        for b in self.behaviours:
            if b not in BEHAVIOURS:
                raise ValueError(f"unknown behaviour {b!r} for term {self.id}")


def _t(*args, **kwargs) -> TermSpec:
    return TermSpec(*args, **kwargs)


_TERMS: tuple[TermSpec, ...] = (
    # ------------------------------------------------------------- Family A
    _t("agreement-parties", "A", "הצדדים להסכם", "Agreement parties",
       ("meta",), "REPRESENTABLE", "TRADE",
       cues=("בין", "לבין", "הצדדים", "המפרסם", "הסוכנות", "חברת המדיה", "הזכיין"),
       interacts=("agreement-level",)),
    _t("brand-scope", "A", "היקף מותגים", "Brand scope",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("מותג", "מותגי", "המוצרים", "קו המוצרים", "למעט מותג"),
       interacts=("budget-commitment",)),
    _t("channel-scope", "A", "היקף ערוצים", "Channel scope",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("בערוץ", "ערוצי", "נכסים דיגיטליים", "פלטפורמות")),
    _t("effective-window", "A", "תקופת תוקף", "Effective window",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("תקופת ההסכם", "בתוקף מיום", "ועד ליום", "יתחדש", "הודעה מוקדמת"),
       interacts=("term-effective-windows",)),
    _t("agreement-level", "A", "רמת ההסכם", "Agreement level",
       ("meta",), "REPRESENTABLE", "TRADE",
       cues=("הסכם מסגרת", "הסכם שנתי", "נספח קמפיין", "הזמנה"),
       interacts=("precedence-clause",)),
    _t("precedence-clause", "A", "סעיף עדיפות", "Precedence clause",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("יגבר", "עדיפות", "בכפוף ל", "על אף האמור", "גובר על"),
       interacts=("agreement-level", "amendment-layer")),
    _t("definitions", "A", "הגדרות", "Definitions",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("הגדרות", "לעניין הסכם זה", "משמעו", "כהגדרתו"),
       interacts=("cpp-daypart-table", "daypart-mix")),
    _t("amendment-layer", "A", "תיקון / נספח", "Amendment / appendix",
       ("meta",), "REPRESENTABLE", "TRADE",
       cues=("נספח", "תיקון להסכם", "תוספת להסכם", "מיום חתימתו יחול"),
       interacts=("precedence-clause", "term-effective-windows")),
    # ------------------------------------------------------------- Family B
    _t("cpp-daypart-table", "B", "טבלת CPP לפי רצועה", "CPP by daypart",
       ("prices", "settles"), "REPRESENTABLE", "IL",
       cues=("CPP", "עלות לנקודת רייטינג", "רצועה", "רצועת שידור", "בתי אב"),
       interacts=("volume-discount-ladder", "length-factor-table", "definitions")),
    _t("target-cpp", "B", "CPP לקהל יעד", "Target-audience CPP",
       ("prices", "settles"), "REPRESENTABLE", "IL",
       cues=("קהל יעד", "TRP", "נשים", "גברים", "בני"),
       interacts=("cpp-daypart-table", "trp-delivery-guarantee")),
    _t("length-factor-table", "B", "מקדמי אורך", "Length factor table",
       ("prices", "settles"), "REPRESENTABLE", "IL",
       cues=("מקדם אורך", "פרו רטה", "פרורטה", "ביחס ל-30", "שניות"),
       interacts=("cpp-daypart-table", "settlement-mechanics")),
    _t("ratecard-index", "B", "הצמדה למחירון", "Rate-card index",
       ("prices",), "REPRESENTABLE", "STD",
       cues=("מהמחירון", "אחוז מהמחירון", "מחירון התקף", "מחירון החברה"),
       interacts=("volume-discount-ladder",)),
    _t("fixed-spot-pricing", "B", "מחיר קבוע לספוט", "Fixed spot pricing",
       ("prices",), "REPRESENTABLE", "IL",
       cues=("מחיר קבוע", "לספוט", "לשידור בודד", "בתוכנית", "גמר"),
       interacts=("cpp-daypart-table",)),
    _t("sponsorship-terms", "B", "תנאי חסות", "Sponsorship terms",
       ("prices",), "REPRESENTABLE", "IL",
       cues=("חסות", "הודעת חסות", "בחסות", "מוגשת בחסות"),
       interacts=()),
    _t("gold-break-rates", "B", "מחירון ברייק זהב", "Gold-break rates",
       ("prices",), "REPRESENTABLE", "IL",
       cues=("ברייק זהב", "ברייק הזהב", "מקבץ זהב"),
       interacts=("gold-break-allocation", "position-entitlements")),
    _t("payment-indexation", "B", "הצמדה וריבית", "Indexation and interest",
       ("process",), "RECORDED", "STD",
       cues=("הצמדה", "מדד", "ריבית פיגורים", 'מע"מ'),
       interacts=("payment-terms",)),
    # ------------------------------------------------------------- Family C
    _t("volume-discount-ladder", "C", "מדרגות הנחת היקף", "Volume discount ladder",
       ("prices",), "REPRESENTABLE", "STD",
       cues=("מדרגות הנחה", "הנחת היקף", "הנחה בשיעור", "מעל תקציב", "רטרואקטיבית"),
       interacts=("budget-commitment", "agency-commission", "underspend-true-up")),
    _t("share-bonus", "C", "תמריץ נתח", "Share bonus",
       ("prices",), "TRACKED", "STD",
       cues=("נתח", "מסך תקציב", "מסך ההשקעה", "בונוס נתח"),
       interacts=("share-commitment", "added-value-media")),
    _t("seasonal-coefficients", "C", "מקדמי עונתיות", "Seasonal coefficients",
       ("prices",), "REPRESENTABLE", "IL",
       cues=("עונתיות", "חודשי", "ערב פסח", "חגים", "לא תחול הנחה בחודש"),
       interacts=("cpp-daypart-table", "volume-discount-ladder")),
    _t("agency-commission", "C", "עמלת סוכנות", "Agency commission",
       ("prices", "settles"), "BINDS", "TRADE",
       cues=("עמלת סוכנות", "עמלה בשיעור", "החזר", "עמלת מדיה"),
       interacts=("volume-discount-ladder", "settlement-mechanics")),
    _t("cash-discount", "C", "הנחת מזומן", "Cash discount",
       ("process",), "RECORDED", "STD",
       cues=("הנחת מזומן", "תשלום מוקדם", "בתוך", "ימים ממועד החשבונית"),
       interacts=("payment-terms",)),
    _t("success-deal", "C", "עסקת הצלחה", "Success deal",
       ("prices", "constrains-soft"), "TRACKED", "TRADE",
       cues=("עסקת הצלחה", "עסקאות הצלחה", "חלוקת הכנסות", "בונוס הצלחה"),
       interacts=("settlement-mechanics",)),
    _t("added-value-media", "C", "מדיה נוספת קבועה", "Added-value media",
       ("prices", "obliges"), "REPRESENTABLE", "TRADE",
       cues=("מדיה נוספת", "ערך מוסף", "בונוס מדיה", "תוספת מדיה בשיעור"),
       interacts=("makegood-accrual-policy",)),
    _t("new-business-incentive", "C", "תמריץ לקוח חדש", "New-business incentive",
       ("prices",), "REPRESENTABLE", "STD",
       cues=("מפרסם חדש", "לקוח חדש", "שנה ראשונה"),
       interacts=("effective-window",)),
    _t("package-bundle", "C", "חבילה משולבת", "Package bundle",
       ("prices",), "REPRESENTABLE", "IL",
       cues=("חבילה", "חבילה משולבת", "דיגיטל", "משולב טלוויזיה"),
       interacts=("channel-scope",)),
    # ------------------------------------------------------------- Family D
    _t("budget-commitment", "D", "התחייבות תקציב", "Budget commitment",
       ("obliges",), "REPRESENTABLE", "TRADE",
       cues=("מתחייב לתקציב", "תקציב שנתי בסך", "היקף התקשרות", "התחייבות כספית"),
       interacts=("volume-discount-ladder", "underspend-true-up", "brand-scope")),
    _t("share-commitment", "D", "התחייבות נתח", "Share commitment",
       ("obliges",), "TRACKED", "STD",
       cues=("נתח מתקציב", "מסך השקעות", "אחוז מסך", "הצהרת תקציב"),
       interacts=("share-bonus", "audit-rights")),
    _t("daypart-mix", "D", "תמהיל רצועות", "Daypart mix",
       ("obliges", "constrains-soft"), "REPRESENTABLE", "STD",
       cues=("תמהיל", "לכל היותר בפריים", "לפחות מחוץ לפריים", "פיזור רצועות"),
       interacts=("cpp-daypart-table", "definitions")),
    _t("flighting-obligation", "D", "התחייבות רציפות", "Flighting obligation",
       ("obliges",), "TRACKED", "STD",
       cues=("רציפות", "שבועות שידור", "ללא הפסקה", "קמפיינים בשנה"),
       interacts=("effective-window",)),
    _t("length-mix", "D", "תמהיל אורכים", "Length mix",
       ("obliges",), "TRACKED", "STD",
       cues=("תמהיל אורכים", "לפחות באורך 30", "אחוז תשדירי"),
       interacts=("length-factor-table",)),
    _t("cancellation-terms", "D", "תנאי ביטול", "Cancellation terms",
       ("process",), "RECORDED", "TRADE",
       cues=("ביטול", "דמי ביטול", "הודעה מראש של", "ימי עסקים לפני שידור"),
       interacts=("force-majeure",)),
    # ------------------------------------------------------------- Family E
    _t("trp-delivery-guarantee", "E", "התחייבות נקודות רייטינג", "TRP delivery guarantee",
       ("obliges", "settles"), "REPRESENTABLE", "IL",
       cues=("התחייבות לאספקת", "נקודות רייטינג", "TRP", "GRP", "יעד רייטינג"),
       interacts=("target-cpp", "shortfall-cure", "measurement-source")),
    _t("effective-cpp-cap", "E", "תקרת CPP אפקטיבי", "Effective-CPP cap",
       ("obliges", "settles"), "REPRESENTABLE", "STD",
       cues=("CPP אפקטיבי", "לא יעלה על", "עלות אפקטיבית לנקודה"),
       interacts=("trp-delivery-guarantee", "volume-discount-ladder")),
    _t("preferred-position-guarantee", "E", "התחייבות מיקומים מועדפים", "Preferred-position guarantee",
       ("obliges",), "BINDS", "TRADE",
       cues=("מיקומים מועדפים", "מיקום ראשון", "אחרון בברייק", "אחוז מיקומים"),
       interacts=("position-entitlements",)),
    _t("gold-break-allocation", "E", "הקצאת ברייקי זהב", "Gold-break allocation",
       ("obliges", "constrains-soft"), "REPRESENTABLE", "IL",
       cues=("ברייקי זהב", "הקצאה", "זכות סירוב ראשונה"),
       interacts=("gold-break-rates",)),
    _t("makegood-accrual-policy", "E", "מדיניות צבירת מייק גוד", "Make-good accrual policy",
       ("obliges", "settles"), "REPRESENTABLE", "TRADE",
       cues=("מייק גוד", "פיצוי", "צבירה", "זיכוי מדיה", "בונוסים"),
       interacts=("shortfall-cure", "added-value-media", "termination")),
    _t("shortfall-cure", "E", "מנגנון השלמה", "Shortfall cure",
       ("obliges", "settles"), "REPRESENTABLE", "TRADE",
       cues=("השלמה", "חוסר אספקה", "שידורי השלמה", "ללא חיוב", "יושלמו הנקודות"),
       interacts=("trp-delivery-guarantee", "makegood-accrual-policy")),
    _t("underspend-true-up", "E", "התחשבנות חוסר ניצול", "Under-spend true-up",
       ("obliges", "settles"), "REPRESENTABLE", "STD",
       cues=("אי עמידה בתקציב", "התחשבנות", "עדכון רטרואקטיבי", "החזר הנחה"),
       interacts=("budget-commitment", "volume-discount-ladder")),
    _t("overdelivery-treatment", "E", "טיפול בעודף אספקה", "Over-delivery treatment",
       ("settles",), "REPRESENTABLE", "STD",
       cues=("עודף אספקה", "אספקת יתר", "מעבר ליעד"),
       interacts=("trp-delivery-guarantee",)),
    _t("preemption-compensation", "E", "פיצוי על הקדמת שידור", "Pre-emption compensation",
       ("obliges",), "REPRESENTABLE", "TRADE",
       cues=("ירידה מלוח", "הורד מהשידור", "מהדורה מיוחדת", "פיצוי בגין אי שידור"),
       interacts=("makegood-accrual-policy", "force-majeure", "delivery-truth-source")),
    # ------------------------------------------------------------- Family F
    _t("competitive-separation", "F", "הפרדה תחרותית", "Competitive separation",
       ("constrains-hard",), "BINDS", "TRADE",
       cues=("הפרדה", "מתחרה", "מתחרים", "לא ישובץ באותו מקבץ"),
       interacts=("category-exclusivity",)),
    _t("category-exclusivity", "F", "בלעדיות קטגוריה", "Category exclusivity",
       ("constrains-hard", "prices"), "REPRESENTABLE", "STD",
       cues=("בלעדיות", "קטגוריה", "מפרסם יחיד", "בלעדי בתחום"),
       interacts=("competitive-separation", "shortfall-cure")),
    _t("content-adjacency-exclusion", "F", "הרחקה מתוכן", "Content adjacency exclusion",
       ("constrains-hard",), "BINDS", "STD",
       cues=("לא ישודר בסמוך", "הרחקה", "תוכן חדשותי קשה", "תוכניות ילדים"),
       interacts=("adjacency-purchase",)),
    _t("adjacency-purchase", "F", "רכישת סמיכות", "Adjacency purchase",
       ("constrains-soft", "prices"), "REPRESENTABLE", "IL",
       cues=("צמוד לחדשות", "סמוך למהדורה", "ברייק ראשון אחרי"),
       interacts=("content-adjacency-exclusion", "position-entitlements")),
    _t("programme-daypart-restrictions", "F", "הגבלות תוכניות ורצועות", "Programme/daypart restrictions",
       ("constrains-hard",), "BINDS", "TRADE",
       cues=("ישודר רק", "לא ישודר בתוכנית", "ברצועת", "בימים"),
       interacts=("definitions",)),
    _t("position-entitlements", "F", "זכויות מיקום בברייק", "Position entitlements",
       ("constrains-soft",), "BINDS", "TRADE",
       cues=("מיקום ראשון", "מיקום בברייק", "פותח", "סוגר", "טופ אנד טייל"),
       interacts=("preferred-position-guarantee", "creative-constraints")),
    _t("creative-constraints", "F", "אילוצי חומרים", "Creative constraints",
       ("constrains-hard",), "BINDS", "TRADE",
       cues=("חומר פרסום", "גרסה", "תוקף החומר", "מספר בית", "אישור שידור"),
       interacts=("position-entitlements",)),
    _t("spot-length-constraints", "F", "אילוצי אורך", "Spot-length constraints",
       ("constrains-hard",), "BINDS", "TRADE",
       cues=("באורך", "שניות בלבד", "אורך תשדיר"),
       interacts=("length-factor-table",)),
    _t("frequency-caps", "F", "תקרות תדירות", "Frequency caps",
       ("constrains-hard",), "BINDS", "STD",
       cues=("לכל היותר", "פעמים בשעה", "באותו מקבץ", "תדירות"),
       interacts=("competitive-separation",)),
    # ------------------------------------------------------------- Family G
    _t("payment-terms", "G", "תנאי תשלום", "Payment terms",
       ("process",), "RECORDED", "STD",
       cues=("שוטף", "תנאי תשלום", "ימים ממועד החשבונית"),
       interacts=("cash-discount", "payment-indexation")),
    _t("reporting-obligations", "G", "חובות דיווח", "Reporting obligations",
       ("process",), "RECORDED", "STD",
       cues=("דוח", "דיווח", "אחת לשבוע", "יעביר לסוכנות"),
       interacts=("delivery-truth-source",)),
    _t("audit-rights", "G", "זכויות ביקורת", "Audit rights",
       ("process",), "RECORDED", "STD",
       cues=("ביקורת", "רואה חשבון", "לעיין בספרים"),
       interacts=("share-commitment",)),
    _t("termination", "G", "סיום ההסכם", "Termination",
       ("process",), "RECORDED", "STD",
       cues=("סיום ההסכם", "הפרה יסודית", "ביטול ההסכם", "ישרדו את סיום"),
       interacts=("makegood-accrual-policy", "effective-window")),
    _t("force-majeure", "G", "כוח עליון", "Force majeure",
       ("process",), "RECORDED", "TRADE",
       cues=("כוח עליון", "מצב חירום", "מלחמה", "מבצע צבאי", "הנחיות פיקוד העורף"),
       interacts=("preemption-compensation", "cancellation-terms")),
    _t("confidentiality", "G", "סודיות", "Confidentiality",
       ("process",), "RECORDED", "IL",
       cues=("סודיות", "מידע סודי", "לא יגלה"),
       interacts=()),
    _t("credit-security", "G", "בטחונות ואשראי", "Credit and security",
       ("process",), "RECORDED", "STD",
       cues=("ערבות", "בטחונות", "מסגרת אשראי", "תשלום מראש"),
       interacts=("payment-terms",)),
    _t("dispute-resolution", "G", "יישוב מחלוקות", "Dispute resolution",
       ("process",), "RECORDED", "STD",
       cues=("סמכות שיפוט", "בוררות", "הדין החל", "מחלוקת"),
       interacts=("measurement-source",)),
    # ------------------------------------------------------------- Family H
    _t("settlement-mechanics", "H", "מנגנון התחשבנות", "Settlement mechanics",
       ("settles", "meta"), "REPRESENTABLE", "IL",
       cues=("התחשבנות", "רבע שעה", "רבעי שעה", "התאמות", "אחת לשבוע"),
       interacts=("cpp-daypart-table", "agency-commission", "shortfall-cure")),
    _t("measurement-source", "H", "מקור מדידה", "Measurement source",
       ("settles", "meta"), "REPRESENTABLE", "IL",
       cues=("ועדת המדרוג", "נתוני הצפייה", "מדרוג", "צפייה נדחית", "בתי אב"),
       interacts=("trp-delivery-guarantee", "target-cpp")),
    _t("delivery-truth-source", "H", "מקור אמת לשידור", "Delivery truth source",
       ("settles", "meta"), "REPRESENTABLE", "TRADE",
       cues=("שודר בפועל", "יומן שידור", "As Run", "דוח שידור בפועל"),
       interacts=("preemption-compensation", "reporting-obligations")),
    _t("term-effective-windows", "H", "חלונות תוקף לסעיפים", "Per-term effective windows",
       ("meta",), "REPRESENTABLE", "STD",
       cues=("בתקופה שמיום", "יחול רק בחודשים", "עד לתום הרבעון"),
       interacts=("effective-window", "amendment-layer")),
    # ------------------------------------------ Not applicable in this market
    _t("regional-feed-splits", "NA", "פיצול שידור אזורי", "Regional feed splits",
       ("meta",), "NOT_APPLICABLE", "STD",
       cues=("שידור אזורי", "פיצול אזורי")),
    _t("coop-invoicing", "NA", "חיוב משותף יצרן-קמעונאי", "Co-op invoicing",
       ("meta",), "NOT_APPLICABLE", "STD",
       cues=("חיוב משותף",)),
    _t("barter-inquiry", "NA", "עסקת ברטר / תשלום לפי פנייה", "Barter / per-inquiry",
       ("meta",), "NOT_APPLICABLE", "STD",
       cues=("ברטר", "תמורת מוצרים", "לפי פניות")),
)

TERMS: Mapping[str, TermSpec] = {spec.id: spec for spec in _TERMS}

if len(TERMS) != len(_TERMS):  # pragma: no cover - construction guard
    raise RuntimeError("duplicate term ids in the taxonomy registry")


# Clause classes that are commercially irrelevant BY DESIGN. A clause may be
# classified into one of these with a reason instead of a term; the list is
# closed so "irrelevant" can never become a silent dumping ground.
IRRELEVANT_CLASSES: Mapping[str, str] = {
    "signature-block": "בלוק חתימות",
    "notice-addresses": "כתובות למשלוח הודעות",
    "counterparts-execution": "עותקים וחתימה",
    "severability": "הפרדת סעיפים בטלים",
    "headings-interpretation": "כותרות לנוחות בלבד",
    "page-furniture": "כותרות עמוד, מספור וסימון",
    "table-of-contents": "תוכן עניינים",
    "preamble-recitals": "מבוא והואיל",
}


def get(term_id: str) -> TermSpec:
    """Return the spec for ``term_id`` or raise KeyError with the known ids."""
    try:
        return TERMS[term_id]
    except KeyError:
        raise KeyError(
            f"unknown trade term {term_id!r}; known ids: {sorted(TERMS)}"
        ) from None


def ids() -> tuple[str, ...]:
    return tuple(TERMS)


def by_family(family: str) -> tuple[TermSpec, ...]:
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    return tuple(spec for spec in _TERMS if spec.family == family)


def by_status(status: str) -> tuple[TermSpec, ...]:
    if status not in STATUSES:
        raise KeyError(f"unknown status {status!r}")
    return tuple(spec for spec in _TERMS if spec.status == status)


def classification_labels() -> tuple[str, ...]:
    """Every label the clause classifier may emit: terms + irrelevant classes.

    The pipeline adds its own reserved ``unmapped`` label for a clause the
    model understands to be commercial but cannot place; that label is NOT
    here because it must never look like a positive classification.
    """
    return tuple(TERMS) + tuple(IRRELEVANT_CLASSES)


def validate_interactions(specs: Iterable[TermSpec] = _TERMS) -> None:
    """Every ``interacts`` ref must name a real term. Raises on drift."""
    for spec in specs:
        for ref in spec.interacts:
            if ref not in TERMS:
                raise ValueError(
                    f"term {spec.id} interacts with unknown term {ref!r}"
                )


validate_interactions()
