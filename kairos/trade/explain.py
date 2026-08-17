"""What a term will actually DO, in the operator's own language.

The review screen must tell a reviewer what each proposed rule does to
planning and pricing BEFORE approval. That sentence is generated here, from
the term's parameters and the COMPILER'S OWN VERDICT on it — never from the
model's prose, and never from a second re-implementation in the frontend.

Three properties the sentences must hold:

- **A term the compiler skipped says so, with the compiler's reason.** The
  reviewer sees "will not act automatically, because…" rather than a
  confident description of an effect that will never happen.
- **The mechanism is named**: blocks, warns, prices, steers placement,
  measured continuously, settled per period, or recorded only. A reviewer
  who cannot tell a hard block from a soft steer cannot review.
- **Nothing is invented.** A parameter the document did not give appears as
  a stated gap in the sentence itself.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from . import taxonomy
from .compile import CompiledArtifacts

# The mechanism vocabulary the sentences classify into, so a surface can badge
# them without parsing prose.
BLOCKS = "blocks"
WARNS = "warns"
PRICES = "prices"
STEERS = "steers"
MEASURES = "measures"
SETTLES = "settles"
RECORDS = "records"
INERT = "inert"

MECHANISM_HE = {
    BLOCKS: "חוסם שיבוץ",
    WARNS: "מתריע",
    PRICES: "משנה מחיר",
    STEERS: "מטה שיבוץ",
    MEASURES: "נמדד ברציפות",
    SETTLES: "נכנס להתחשבנות",
    RECORDS: "נרשם בלבד",
    INERT: "לא יפעל אוטומטית",
}


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ₪"
    except (TypeError, ValueError):
        return "סכום לא ידוע"


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "שיעור לא ידוע"
    return f"{number:g}%"


def _scope_phrase(scope: Mapping[str, Any]) -> str:
    parts: list[str] = []
    labels = {
        "advertisers": "מפרסמים", "brands": "מותגים", "campaigns": "קמפיינים",
        "programmes": "תוכניות", "genres": "ז'אנרים", "dayparts": "רצועות",
        "weekdays": "ימים", "positions": "מיקומים",
    }
    for key, label in labels.items():
        values = [str(v) for v in (scope.get(key) or [])]
        if values:
            shown = ", ".join(values[:3])
            more = f" ועוד {len(values) - 3}" if len(values) > 3 else ""
            parts.append(f"{label}: {shown}{more}")
    return " · ".join(parts)


def _period_he(period: Any) -> str:
    return {
        "year": "שנתי", "quarter": "רבעוני", "month": "חודשי",
        "campaign": "לכל קמפיין", "custom": "לתקופה שנקבעה",
    }.get(str(period), str(period or "לתקופה שלא צוינה"))


def _sentence(term_id: str, params: Mapping[str, Any],
              scope: Mapping[str, Any]) -> tuple[str, str]:
    """(mechanism, sentence) for a term whose compiler verdict is not a skip."""
    p = params

    if term_id == "programme-daypart-restrictions":
        hard = p.get("hard", True)
        mode = "רק בהיקף המסומן" if p.get("mode") == "allow" else "לא בהיקף המסומן"
        return ((BLOCKS if hard else STEERS),
                f"המנוע ישבץ {mode}" + (" — חסימה קשיחה" if hard else " — הטיה רכה"))
    if term_id == "content-adjacency-exclusion":
        radius = {"same_break": "באותו מקבץ", "adjacent_break": "במקבץ צמוד",
                  "same_programme": "באותה תוכנית"}.get(str(p.get("radius")), "")
        content = ", ".join(str(c) for c in (p.get("excluded_content") or []))
        return (BLOCKS if p.get("hard", True) else STEERS,
                f"תשדירי הלקוח לא ישובצו {radius} עם: {content}")
    if term_id == "competitive-separation":
        unit = {"same_break": "באותו מקבץ", "spots": "תשדירים",
                "minutes": "דקות"}.get(str(p.get("separation_unit")), "")
        quantity = p.get("separation_quantity")
        detail = (f"בהפרדה של {quantity:g} {unit}" if quantity and unit != "באותו מקבץ"
                  else "לא יופיעו יחד באותו מקבץ")
        return BLOCKS, f"מתחרי הקטגוריה {detail}"
    if term_id == "frequency-caps":
        unit = {"break": "במקבץ", "day": "ביממת שידור", "hour": "בשעה",
                "programme": "בתוכנית"}.get(str(p.get("unit")), "")
        return BLOCKS, f"לכל היותר {p.get('cap')} תשדירים {unit}"
    if term_id == "position-entitlements":
        positions = ", ".join(str(x) for x in (p.get("positions") or []))
        tail = " · כולל טופ אנד טייל" if p.get("top_and_tail") else ""
        return STEERS, f"עדיפות שיבוץ במיקומים: {positions}{tail}"
    if term_id == "adjacency-purchase":
        return STEERS, (f"עדיפות שיבוץ ב{p.get('target_content')} "
                        f"({p.get('break_relation')})")
    if term_id == "category-exclusivity":
        premium = p.get("premium_percent")
        money = f", בתוספת {_pct(premium)} על התעריף" if premium is not None else ""
        return PRICES, (f"בלעדיות בקטגוריית {p.get('category')} "
                        f"ב{p.get('exclusivity_scope')}{money}")
    if term_id == "ratecard-index":
        return PRICES, (f"תמחור לפי {_pct(p.get('index_percent'))} מהמחירון "
                        f"(גרסה {p.get('ratecard_version')})")
    if term_id == "success-deal":
        return STEERS, (f"עסקת הצלחה בשיעור {_pct(p.get('share_percent'))} — "
                        "עדיפות שיבוץ מוגברת, ההתחשבנות ידנית")
    if term_id == "volume-discount-ladder":
        tiers = p.get("tiers") or []
        mechanics = {"retroactive": "רטרואקטיבי — חציית מדרגה מתמחרת מחדש את כל התקופה",
                     "marginal": "שולי — כל מדרגה מתמחרת את הפלח שלה",
                     "unstated": "המנגנון לא נקבע במסמך ולכן לא יחושב"}.get(
            str(p.get("mechanics")), "")
        top = max((float(t.get("discount_percent") or 0) for t in tiers), default=0)
        return SETTLES, (f"{len(tiers)} מדרגות הנחה עד {_pct(top)}, חישוב {_period_he(p.get('period'))}; "
                         f"{mechanics}")
    if term_id == "agency-commission":
        base = {"gross": "על הברוטו", "net_of_discount": "על הנטו לאחר הנחות",
                "unstated": "על בסיס שלא צוין"}.get(str(p.get("base")), "")
        form = {"invoice_deduction": "ניכוי מהחשבונית",
                "periodic_rebate": "החזר תקופתי"}.get(str(p.get("form")), "")
        return SETTLES, f"עמלת סוכנות {_pct(p.get('percent'))} {base} · {form}"
    if term_id == "cpp-daypart-table":
        rows = p.get("rows") or []
        return SETTLES, (f"טבלת CPP ל-{len(rows)} רצועות, קהל: "
                         f"{p.get('audience') or 'לא צוין'} — בסיס ההתחשבנות מול הלקוח")
    if term_id == "length-factor-table":
        return SETTLES, f"{len(p.get('rows') or [])} מקדמי אורך מול בסיס 30 שניות"
    if term_id == "fixed-spot-pricing":
        return SETTLES, f"{len(p.get('rows') or [])} מחירים קבועים, נסלקים ככתבם"
    if term_id == "gold-break-rates":
        surcharge = p.get("surcharge_percent")
        fixed = len(p.get("fixed_prices") or [])
        bits = []
        if surcharge is not None:
            bits.append(f"תוספת {_pct(surcharge)}")
        if fixed:
            bits.append(f"{fixed} מחירים קבועים")
        return SETTLES, "ברייק זהב: " + (" · ".join(bits) or "תנאים ללא ערך מספרי")
    if term_id == "seasonal-coefficients":
        rows = p.get("rows") or []
        blackouts = sum(1 for r in rows if r.get("discount_blackout"))
        return SETTLES, (f"{len(rows)} תקופות עונתיות" +
                         (f", מהן {blackouts} ללא זכאות להנחה" if blackouts else ""))
    if term_id == "budget-commitment":
        amount = (p.get("amount") or {}).get("amount")
        return MEASURES, (f"התחייבות תקציב {_money(amount)} ({_period_he(p.get('period'))}) — "
                          "העמידה נמדדת ברציפות מול הפעילות בפועל")
    if term_id == "trp-delivery-guarantee":
        points = p.get("points")
        tolerance = p.get("tolerance_percent")
        target = (f"{points:g} נקודות" if points is not None
                  else "כמות שנקבעת בכל הזמנה")
        tail = f", סטייה מותרת {_pct(tolerance)}" if tolerance is not None else ""
        return MEASURES, (f"התחייבות אספקה: {target} מול קהל "
                          f"{p.get('audience') or 'שלא צוין'}{tail}")
    if term_id == "effective-cpp-cap":
        return MEASURES, (f"תקרת CPP אפקטיבי {_money(p.get('cap'))} לנקודה, "
                          f"קהל {p.get('audience') or 'שלא צוין'}")
    if term_id == "preferred-position-guarantee":
        method = {"agency": "שיטת הסוכנות", "channel": "שיטת הערוץ",
                  "unstated": "שיטה שלא צוינה"}.get(str(p.get("counting_method")), "")
        positions = ", ".join(str(x) for x in (p.get("preferred_positions") or []))
        return MEASURES, (f"{_pct(p.get('target_percent'))} מההופעות במיקומים "
                          f"{positions} — נמדד ב{method}")
    if term_id == "makegood-accrual-policy":
        accruals = p.get("accruals") or []
        levels = ", ".join(sorted({str(a.get("level")) for a in accruals}))
        return MEASURES, (f"צבירת מייק גוד ברמות: {levels}; "
                          f"פקיעה: {p.get('expiry') or 'לא צוינה'}")
    if term_id == "shortfall-cure":
        form = {"bonus_spots": "שידורי בונוס", "credit": "זיכוי כספי",
                "carry_forward": "העברה לתקופה הבאה", "mixed": "משולב"}.get(
            str(p.get("cure_form")), "")
        return MEASURES, (f"במקרה חוסר: {form}, בתוך {p.get('cure_window')}; "
                          f"איכות ההשלמה: {p.get('quality_rule')}")
    if term_id == "underspend-true-up":
        return MEASURES, f"אי-עמידה בתקציב: {p.get('trigger_note')}"
    if term_id == "daypart-mix":
        rows = p.get("rows") or []
        return MEASURES, (f"מסגרת תמהיל ל-{len(rows)} רצועות "
                          f"(בסיס: {p.get('basis')})")
    if term_id == "effective-window" or term_id == "term-effective-windows":
        return RECORDS, "חלון תוקף — קובע מתי תנאים אחרים פועלים"
    if term_id == "precedence-clause":
        return RECORDS, (f"עדיפות: {p.get('winner')} גובר על {p.get('loser')} — "
                         "משמש להכרעת סתירות")
    if term_id == "definitions":
        entries = p.get("entries") or []
        return RECORDS, f"{len(entries)} הגדרות שקובעות את משמעות המונחים בהסכם"

    spec = taxonomy.get(term_id)
    return RECORDS, f"{spec.name_he} — נרשם ומוצג"


def explain_instance(instance: Mapping[str, Any],
                     artifacts: Optional[CompiledArtifacts] = None) -> dict[str, Any]:
    """What this proposed term will do, with the compiler's verdict on it."""
    term_id = str(instance.get("term_id"))
    spec = taxonomy.get(term_id)
    params = instance.get("params") or {}
    scope = instance.get("scope") or {}
    instance_id = str(instance.get("instance_id", ""))

    skip_reasons = [
        s["reason_he"] for s in (artifacts.skipped if artifacts else [])
        if s.get("instance_id") == instance_id
    ]
    bound_rules = [
        r["rule_id"] for r in (
            (artifacts.conditions + artifacts.frequency_rules) if artifacts else []
        )
        if f":{instance_id}" in str(r.get("rule_id", ""))
    ]
    settlement = [
        t for t in ((artifacts.settlement.get("terms", []) if artifacts else []))
        if t.get("instance_id") == instance_id
    ]

    mechanism, sentence = _sentence(term_id, params, scope)
    missing = list(instance.get("missing") or [])
    if missing:
        sentence += f" · חסרים במסמך: {', '.join(missing)}"

    if artifacts is not None and not bound_rules and not settlement and skip_reasons:
        mechanism = INERT
    scope_text = _scope_phrase(scope)
    return {
        "instance_id": instance_id,
        "term_id": term_id,
        "term_name_he": spec.name_he,
        "family": spec.family,
        "family_he": taxonomy.FAMILIES[spec.family],
        "mechanism": mechanism,
        "mechanism_he": MECHANISM_HE[mechanism],
        "sentence_he": sentence,
        "scope_he": scope_text,
        "will_not_act_reasons": skip_reasons,
        "bound_rule_ids": bound_rules,
        "settlement_kinds": [t.get("kind") for t in settlement],
        "incomplete": bool(missing),
        "rank": spec.rank,
    }


def explain_termset(termset: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    """Every term of a proposal, explained, with the compiler run once."""
    from .compile import compile_termset

    artifacts = compile_termset(termset, head)
    explained = [explain_instance(i, artifacts) for i in termset.get("instances", [])]
    by_mechanism: dict[str, int] = {}
    for item in explained:
        by_mechanism[item["mechanism"]] = by_mechanism.get(item["mechanism"], 0) + 1
    return {
        "terms": explained,
        "by_mechanism": by_mechanism,
        "mechanism_labels": dict(MECHANISM_HE),
        "compiled": artifacts.summary(),
    }
