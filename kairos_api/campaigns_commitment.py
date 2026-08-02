"""What a television campaign commits to, and the closed words it commits in.

Split out of :mod:`kairos_api.campaigns_api_store` so that store stays under the
project line limit, and kept separate on purpose: the store is persistence, this
is the commercial vocabulary an Israeli broadcast campaign is written in. Every
list here is sourced from ``docs/campaign-rate-card-research.md``, which is the
research done against real Israeli market sources, and every entry says where it
came from and whether this product can actually measure against it.

Three things a person books, and this module names all three.

**A budget in shekels.** The agreed spend, kept apart from any added-value or
make-good spend, because a bonus shekel and a paid shekel are not the same
shekel and a board that adds them cannot tell an operator what was invoiced.

**A rating-point goal against a named target audience.** Israeli television is
traded on CPP, cost per rating point, so the goal a campaign carries is a
rating-point count, and a rating point is meaningless without the audience it is
counted against. Research section 11 lists the audiences the market actually
buys. This product's own ratings are the all-viewers base, so exactly one of
those audiences is measurable here and the rest are reported as unavailable with
the reason, never as zero and never as a silently substituted base.

**A demo marker.** Nothing in this vocabulary invents a booking. When a row is
seeded rather than booked by an operator, it carries ``is_demo`` true and a note
saying so, and every surface reads that column.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Target audiences
# ---------------------------------------------------------------------------

# The all-viewers base is the one this product can count against, because the
# ratings on the traffic log and in the reference month are the general-audience
# TVR with no demographic split. The others are real market audiences and this
# product has no panel breakdown for them, so a goal stated against one of them
# reports its progress as unknown rather than against a base nobody asked for.
ALL_VIEWERS = "all_viewers"

TARGET_AUDIENCES = (
    {
        "value": ALL_VIEWERS,
        "label_en": "All viewers",
        "label_he": "כלל הצופים",
        "measurable": True,
        "reason_en": "The ratings this product holds are the general-audience TVR, which is this base.",
        "reason_he": "הרייטינג שהמערכת מחזיקה הוא TVR של כלל הצופים, שהוא הבסיס הזה.",
    },
    {
        "value": "adults_18_plus",
        "label_en": "Adults 18 plus",
        "label_he": "18 פלוס",
        "measurable": False,
        "reason_en": "No demographic panel breakdown is on disk, so progress against this audience is unknown.",
        "reason_he": "אין בדיסק פילוח פאנל דמוגרפי, ולכן ההתקדמות מול הקהל הזה אינה ידועה.",
    },
    {
        "value": "women_25_54",
        "label_en": "Women 25 to 54",
        "label_he": "נשים 25-54",
        "measurable": False,
        "reason_en": "No demographic panel breakdown is on disk, so progress against this audience is unknown.",
        "reason_he": "אין בדיסק פילוח פאנל דמוגרפי, ולכן ההתקדמות מול הקהל הזה אינה ידועה.",
    },
    {
        "value": "men_25_54",
        "label_en": "Men 25 to 54",
        "label_he": "גברים 25-54",
        "measurable": False,
        "reason_en": "No demographic panel breakdown is on disk, so progress against this audience is unknown.",
        "reason_he": "אין בדיסק פילוח פאנל דמוגרפי, ולכן ההתקדמות מול הקהל הזה אינה ידועה.",
    },
    {
        "value": "children_4_14",
        "label_en": "Children 4 to 14",
        "label_he": "ילדים 4-14",
        "measurable": False,
        "reason_en": "No demographic panel breakdown is on disk, so progress against this audience is unknown.",
        "reason_he": "אין בדיסק פילוח פאנל דמוגרפי, ולכן ההתקדמות מול הקהל הזה אינה ידועה.",
    },
)

AUDIENCE_VALUES = tuple(entry["value"] for entry in TARGET_AUDIENCES)
MEASURABLE_AUDIENCES = frozenset(
    entry["value"] for entry in TARGET_AUDIENCES if entry["measurable"]
)
AUDIENCE_SOURCE_EN = "Audience list from docs/campaign-rate-card-research.md section 11."
AUDIENCE_SOURCE_HE = "רשימת קהלי היעד מתוך docs/campaign-rate-card-research.md סעיף 11."

# ---------------------------------------------------------------------------
# How the campaign is priced, protected and paced
# ---------------------------------------------------------------------------

PRICE_MODELS = (
    {
        "value": "cpp",
        "label_en": "Cost per rating point",
        "label_he": "עלות לנקודת רייטינג",
        "note_en": "The standard Israeli trade. The 30 second spot is the base unit.",
        "note_he": "המסחר הישראלי הרגיל. תשדיר של 30 שניות הוא יחידת הבסיס.",
    },
    {
        "value": "flat",
        "label_en": "Flat price per spot",
        "label_he": "מחיר קבוע לתשדיר",
        "note_en": "Named spots at fixed prices, with no rating guarantee. Used by smaller advertisers.",
        "note_he": "תשדירים נקובים במחיר קבוע, בלי הבטחת רייטינג. נהוג אצל מפרסמים קטנים.",
    },
)

PRIORITIES = (
    {
        "value": "guaranteed",
        "label_en": "Guaranteed",
        "label_he": "מובטח",
        "what_to_do_en": "Nothing. It is protected in the plan and is not displaced.",
        "what_to_do_he": "דבר. הקמפיין מוגן בתוכנית ואינו נדחק.",
    },
    {
        "value": "preemptible",
        "label_en": "Preemptible",
        "label_he": "ניתן לדחיקה",
        "what_to_do_en": "Watch it. A higher priority booking can displace it.",
        "what_to_do_he": "עקבו אחריו. הזמנה בעדיפות גבוהה יותר יכולה לדחוק אותו.",
    },
)

PACING_MODES = (
    {"value": "even", "label_en": "Even", "label_he": "אחיד"},
    {"value": "front_loaded", "label_en": "Front loaded", "label_he": "מוטה להתחלה"},
)

PRICE_MODEL_VALUES = tuple(entry["value"] for entry in PRICE_MODELS)
PRIORITY_VALUES = tuple(entry["value"] for entry in PRIORITIES)
PACING_VALUES = tuple(entry["value"] for entry in PACING_MODES)

# ---------------------------------------------------------------------------
# The demo marker
# ---------------------------------------------------------------------------

DEMO_LABEL_EN = "Demo"
DEMO_LABEL_HE = "הדגמה"
DEMO_MEANING_EN = (
    "This row was written by the demo seed, not booked by an operator. Its identity, its creative "
    "and its delivery come from the real traffic log on disk; its budget, its rating goal and its "
    "flight dates are the seed's, because no signed insertion order exists for them."
)
DEMO_MEANING_HE = (
    "השורה הזו נכתבה על ידי זרע ההדגמה ולא הוזמנה על ידי מפעיל. הזהות, החומר והאספקה שלה מגיעים "
    "מיומן השידור האמיתי שבדיסק; התקציב, יעד הרייטינג ותאריכי הטיסה הם של הזרע, מפני שאין עבורם "
    "הזמנת רכש חתומה."
)
DEMO_REPLACE_EN = "Book the campaign through the clients flow to replace it with a real one."
DEMO_REPLACE_HE = "הזמינו את הקמפיין דרך מסלול הלקוחות כדי להחליף אותו בקמפיין אמיתי."

TRUE_WORDS = frozenset({"true", "yes", "1", "y"})


def is_demo(raw: Any) -> bool:
    """Whether a stored row is a demo row. Anything unrecognised is not."""
    return str(raw or "").strip().lower() in TRUE_WORDS


def demo_block(demo: bool) -> dict[str, Any]:
    """The demo marking a surface renders, in both languages, or the honest false."""
    if not demo:
        return {"is_demo": False}
    return {
        "is_demo": True,
        "label_en": DEMO_LABEL_EN,
        "label_he": DEMO_LABEL_HE,
        "meaning_en": DEMO_MEANING_EN,
        "meaning_he": DEMO_MEANING_HE,
        "replace_en": DEMO_REPLACE_EN,
        "replace_he": DEMO_REPLACE_HE,
    }


def audience_entry(value: str) -> Optional[dict[str, Any]]:
    """One audience record, or none when the value is not in the vocabulary."""
    wanted = str(value or "").strip()
    for entry in TARGET_AUDIENCES:
        if entry["value"] == wanted:
            return dict(entry)
    return None


def audience_is_measurable(value: str) -> bool:
    return str(value or "").strip() in MEASURABLE_AUDIENCES


def operator_channel() -> str:
    """The one channel every campaign this product books belongs to.

    Read from settings, never from a request body and never from a spot file's
    own channel column. A campaign is the operator's own inventory by
    definition, so no rival channel name can arrive on a booking, and a booking
    made while no channel is configured carries an empty channel rather than a
    guessed one.
    """
    try:
        from kairos_api import channel_scope

        return str(channel_scope.operator_channel() or "").strip()
    except Exception:  # noqa: BLE001 - an unconfigured channel is a state, not a crash
        return ""


def row_values(payload: Any) -> dict[str, str]:
    """The commitment half of a campaign row, validated, as strings for the store.

    Every field is optional and a blank one stays blank. Nothing here defaults a
    figure into existence: a campaign booked without a budget has no budget, and
    the store writes an empty cell rather than a zero somebody would read as a
    commitment to spend nothing.
    """
    from kairos_api.campaigns_api_words import validate_amount, validate_choice

    return {
        "channel": operator_channel(),
        "brand": str(getattr(payload, "brand", "") or "").strip(),
        "category": str(getattr(payload, "category", "") or "").strip(),
        "budget_ils": validate_amount(getattr(payload, "budget_ils", None), "budget_ils"),
        "bonus_ils": validate_amount(getattr(payload, "bonus_ils", None), "bonus_ils"),
        "rating_goal_points": validate_amount(
            getattr(payload, "rating_goal_points", None), "rating_goal_points"
        ),
        "rating_goal_audience": validate_choice(
            getattr(payload, "rating_goal_audience", ""), AUDIENCE_VALUES,
            "rating_goal_audience", allow_blank=True,
        ),
        "price_model": validate_choice(
            getattr(payload, "price_model", ""), PRICE_MODEL_VALUES, "price_model", allow_blank=True
        ),
        "priority": validate_choice(
            getattr(payload, "priority", ""), PRIORITY_VALUES, "priority", allow_blank=True
        ),
        "pacing_mode": validate_choice(
            getattr(payload, "pacing_mode", ""), PACING_VALUES, "pacing_mode", allow_blank=True
        ),
    }


def coerce_field(key: str, value: Any) -> Optional[str]:
    """One commitment field's validated string form, or none when it is not ours."""
    from kairos_api.campaigns_api_words import validate_amount, validate_choice

    amounts = {"budget_ils", "bonus_ils", "rating_goal_points"}
    choices = {
        "rating_goal_audience": AUDIENCE_VALUES,
        "price_model": PRICE_MODEL_VALUES,
        "priority": PRIORITY_VALUES,
        "pacing_mode": PACING_VALUES,
    }
    if key in amounts:
        return validate_amount(value, key)
    if key in choices:
        return validate_choice(value, choices[key], key, allow_blank=True)
    return None


def vocabularies() -> dict[str, Any]:
    """Every closed list this commitment speaks, for a form and for a payload."""
    return {
        "target_audiences": [dict(entry) for entry in TARGET_AUDIENCES],
        "target_audience_source_en": AUDIENCE_SOURCE_EN,
        "target_audience_source_he": AUDIENCE_SOURCE_HE,
        "price_models": [dict(entry) for entry in PRICE_MODELS],
        "priorities": [dict(entry) for entry in PRIORITIES],
        "pacing_modes": [dict(entry) for entry in PACING_MODES],
    }
