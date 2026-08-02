"""Why a spot is not in the money, in words a reader can act on.

Split out of :mod:`kairos_api.campaigns_read_money` to keep that module under
the project line limit, and separate for a second reason worth stating: this is
the only place in the Clients money layer that turns an engine artefact into a
sentence, so it is the only place that can get that translation wrong.

The engine writes a machine reason for a log. Measured on the shipped day, all
fifty six removed spots carry ``rule_id`` ``DEFAULT_ONE_PER_BREAK`` and
``max_per_break=1 reached for <advertiser> in break <clock>``, which is a
correct log line and an unreadable explanation for the person holding the
invoice. The rule behind it is a row in ``data/frequency_rules.csv`` carrying
the cap, the unit and what it counts over, so the sentence is built from the
rule itself.

The honest limit: when the id on the drop is not a rule in the file being read,
the cap is reported as unknown and no number is invented. That is the third
state, and it is why :func:`explain_drop` returns whether the limit was known.
"""

from __future__ import annotations

from typing import Any

# Noun sets, as (English singular, English plural, Hebrew singular phrase,
# Hebrew plural noun). A cap of one is the only cap the shipped rule file holds,
# so "1 spots" would be the sentence every reader of this board actually sees.
SPOT_WORDS = ("spot", "spots", "תשדיר אחד", "תשדירים")
POSITION_WORDS = ("position", "positions", "מיקום אחד", "מיקומים")
MINUTE_WORDS = ("minute", "minutes", "דקה אחת", "דקות")

# What the rule counts over, by its own scope vocabulary. A default or
# advertiser rule counts per client; the finer two count per campaign and per ad.
COUNTED_OVER = {
    "default": ("client", "לקוח"),
    "advertiser": ("client", "לקוח"),
    "campaign": ("campaign", "קמפיין"),
    "ad": ("ad", "תשדיר"),
}

UNIT_WORDS = {
    "minutes": MINUTE_WORDS,
    "positions": POSITION_WORDS,
}

# One sentence per limit type. ``cap`` arrives already carrying its noun, so the
# singular and the plural are chosen once rather than in five templates.
LIMIT_SENTENCES = {
    "max_per_break": (
        "A rule allows at most {cap} per {over} in one break, and {who} had already reached it in break {where}.",
        "כלל מתיר לכל היותר {cap} ל{over} בכל ברייק, והמכסה כבר מוצתה עבור ⁦{who}⁩ בברייק ⁦{where}⁩.",
    ),
    "max_per_day": (
        "A rule allows at most {cap} per {over} in one day, and {who} had already reached it.",
        "כלל מתיר לכל היותר {cap} ל{over} ביום אחד, והמכסה כבר מוצתה עבור ⁦{who}⁩.",
    ),
    "max_consecutive": (
        "A rule allows at most {cap} in a row per {over} in one break, and {who} would have exceeded it in break {where}.",
        "כלל מתיר לכל היותר {cap} ברצף ל{over} בברייק אחד, ו⁦{who}⁩ היה חורג ממנו בברייק ⁦{where}⁩.",
    ),
    "min_separation": (
        "A rule keeps two spots of the same {over} at least {cap} apart, and {who} was inside that gap.",
        "כלל שומר מרחק של {cap} לפחות בין שני תשדירים של אותו {over}, ו⁦{who}⁩ היה בתוך המרחק הזה.",
    ),
    "competitive_separation": (
        "A rule keeps competing advertisers at least {cap} apart, and {who} was inside that gap.",
        "כלל שומר מרחק של {cap} לפחות בין מפרסמים מתחרים, ו⁦{who}⁩ היה בתוך המרחק הזה.",
    ),
}

# Which noun the cap counts, by limit type. The two separation rules count in
# the unit the rule itself declares, so they resolve through UNIT_WORDS instead.
CAP_WORDS = {
    "max_per_break": SPOT_WORDS,
    "max_per_day": SPOT_WORDS,
    "max_consecutive": POSITION_WORDS,
}

RULE_DROP_EN = "An advertiser or agency rule forbids this spot, so it was never priced."
RULE_DROP_HE = "כלל מפרסם או סוכנות אוסר על התשדיר הזה, ולכן הוא מעולם לא תומחר."


def frequency_rules() -> dict[str, Any]:
    """Every authored frequency rule, keyed by id, or an empty index."""
    try:
        from kairos.optimize._frequency_rules import load_frequency_rules

        return {rule.rule_id: rule for rule in load_frequency_rules().rules}
    except Exception:  # noqa: BLE001 - an unreadable rule file is unknown, not a crash
        return {}


def number(value: Any) -> str:
    """A cap as a person writes it: 1 rather than 1.0, and 2.5 kept as 2.5."""
    amount = float(value or 0.0)
    return str(int(amount)) if amount == int(amount) else str(round(amount, 2))


def quantity(count: Any, words: tuple[str, str, str, str]) -> tuple[str, str]:
    """One count with its noun, singular or plural, in both languages."""
    en_one, en_many, he_one, he_many = words
    if float(count or 0.0) == 1.0:
        return f"1 {en_one}", he_one
    return f"{number(count)} {en_many}", f"⁦{number(count)}⁩ {he_many}"


def _cap_words(rule: Any, limit_type: str) -> tuple[str, str, str, str]:
    """The noun this rule's cap counts, from the limit type or the rule's unit."""
    if limit_type in CAP_WORDS:
        return CAP_WORDS[limit_type]
    unit = str(getattr(rule, "unit", "") or "")
    return UNIT_WORDS.get(unit, (unit, unit, f"⁦1⁩ {unit}", unit))


def explain_drop(drop: Any, rules: dict[str, Any]) -> tuple[str, str, bool]:
    """The sentence for one removed spot, and whether its limit was known."""
    rule_id = str(getattr(drop, "rule_id", "") or "")
    rule = rules.get(rule_id)
    limit_type = str(getattr(drop, "limit_type", "") or "")
    sentences = LIMIT_SENTENCES.get(limit_type)
    if rule is None or sentences is None:
        english = f"A frequency rule removed this spot. Rule {rule_id} is not in the rule file being read, so its limit is unknown."
        hebrew = f"כלל תדירות הסיר את התשדיר הזה. הכלל ⁦{rule_id}⁩ אינו בקובץ הכללים הנקרא, ולכן התקרה שלו אינה ידועה."
        return english, hebrew, False
    cap_en, cap_he = quantity(getattr(rule, "value", 0.0), _cap_words(rule, limit_type))
    over_en, over_he = COUNTED_OVER.get(str(getattr(rule, "scope", "")), ("client", "לקוח"))
    who = str(getattr(drop, "advertiser", "") or "")
    where = str(getattr(drop, "break_id", "") or "")
    return (
        sentences[0].format(cap=cap_en, over=over_en, who=who, where=where),
        sentences[1].format(cap=cap_he, over=over_he, who=who, where=where),
        True,
    )
