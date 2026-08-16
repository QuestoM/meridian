"""The compiler: an approved termset becomes the product's own rule primitives.

Design commitment 1 of the engine (docs/trade/engine-design.md): the agreement
is the source; the existing machinery is the muscle. Placement terms compile to
advertiser/agency conditions and frequency rules; period-arithmetic money
(ladders, commissions, cash discounts, seasonal blackouts, length factors)
compiles to a per-agreement SETTLEMENT SPEC consumed at statement time —
a retroactive ladder is not a per-spot multiplier and pretending otherwise
would misprice every spot until year end.

Honesty rules:

- **Every artifact carries its source.** rule_id encodes agreement/version/
  instance (``TRD:<agreement>:<version>:<instance>[:n]``), so the moment a rule
  acts, the surface can name the clause that put it there.
- **What cannot bind is SKIPPED WITH A REASON, never silently.** A category
  separation with no member list cannot block anyone; it lands in
  ``skipped`` and the review surface shows it before approval, not after.
- **The compiler is pure.** It emits artifacts; the API layer writes them,
  snapshots the touched stores first, and proves byte-identity when an
  agreement layer is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import taxonomy

RULE_PREFIX = "TRD"

# Steering magnitudes for soft placement effects. PRESSURE is a placement-only
# lever — it raises or lowers a slot's apparent value in ranking and is NEVER
# charged, so these constants move no revenue. The agreements state the RIGHT
# (preferred positions, bought adjacency, success-deal priority) but not an
# engine magnitude, so these are the product's own operational defaults: named
# here, carried into every compiled row's notes, and tunable in one place.
STEER_POSITION_ENTITLEMENT = 25
STEER_ADJACENCY_PURCHASE = 40
STEER_SUCCESS_DEAL = 30
STEER_SOFT_AVOID = -50


@dataclass
class CompiledArtifacts:
    """Everything an approved termset turns into."""

    agreement_id: str
    version_id: str
    conditions: list[dict[str, Any]] = field(default_factory=list)
    frequency_rules: list[dict[str, Any]] = field(default_factory=list)
    settlement: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "conditions": len(self.conditions),
            "frequency_rules": len(self.frequency_rules),
            "settlement_terms": len(self.settlement.get("terms", [])),
            "skipped": len(self.skipped),
        }


def _rule_id(agreement_id: str, version_id: str, instance_id: str,
             n: Optional[int] = None) -> str:
    base = f"{RULE_PREFIX}:{agreement_id}:{version_id}:{instance_id}"
    return base if n is None else f"{base}:{n}"


def parse_rule_id(rule_id: str) -> Optional[dict[str, str]]:
    """The attribution a surface reads back off a compiled rule id."""
    parts = str(rule_id or "").split(":")
    if len(parts) < 4 or parts[0] != RULE_PREFIX:
        return None
    return {"agreement_id": parts[1], "version_id": parts[2], "instance_id": parts[3]}


def _tokens(values: Any) -> str:
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    return ",".join(items) if items else "ANY"


def _advertiser_for(instance: Mapping[str, Any], head: Mapping[str, Any]) -> Optional[str]:
    scope_advertisers = [str(a) for a in instance.get("scope", {}).get("advertisers", [])]
    if len(scope_advertisers) == 1:
        return scope_advertisers[0]
    if scope_advertisers:
        return None  # multi-advertiser rows are emitted per advertiser by callers
    counterparty = head.get("counterparty", {}) or {}
    return str(counterparty.get("advertiser")) or None if counterparty.get("advertiser") else None


def _agency_for(head: Mapping[str, Any]) -> Optional[str]:
    counterparty = head.get("counterparty", {}) or {}
    return str(counterparty.get("agency")) if counterparty.get("agency") else None


def _note(instance: Mapping[str, Any], sentence: str) -> str:
    cite = ""
    citations = instance.get("citations") or []
    if citations:
        first = citations[0]
        cite = f" [{first.get('document_id')}/{first.get('clause_id')}]"
    return f"{sentence}{cite}"


def _condition_row(*, advertiser_id: str, rule_id: str, effect: str, value: Any,
                   mode: str, instance: Mapping[str, Any], sentence: str) -> dict[str, Any]:
    scope = instance.get("scope", {}) or {}
    return {
        "advertiser_id": advertiser_id,
        "rule_id": rule_id,
        "scope_positions": _tokens(scope.get("positions")),
        "scope_genres": _tokens(scope.get("genres")),
        "scope_dayparts": _tokens(scope.get("dayparts")),
        "scope_programmes": _tokens(scope.get("programmes")),
        "scope_campaigns": _tokens(scope.get("campaigns")),
        "scope_weekdays": _tokens(scope.get("weekdays")),
        "effect": effect,
        "value": value,
        "mode": mode,
        "notes": _note(instance, sentence),
    }


class _Compiler:
    def __init__(self, termset: Mapping[str, Any], head: Mapping[str, Any]):
        self.termset = termset
        self.head = head
        self.out = CompiledArtifacts(
            agreement_id=str(termset.get("agreement_id", "")),
            version_id=str(termset.get("version_id", "")),
        )
        # An agency framework binds every advertiser it represents; the list
        # rides the agreement-parties instance (typically from an appendix).
        self.represented: list[str] = []
        for inst in termset.get("instances", []):
            if inst.get("term_id") == "agreement-parties":
                self.represented = [
                    str(a) for a in inst.get("params", {}).get(
                        "represented_advertisers", [])
                ]
                break

    # ------------------------------------------------------------- helpers
    def _skip(self, instance: Mapping[str, Any], reason_he: str) -> None:
        self.out.skipped.append({
            "instance_id": instance.get("instance_id"),
            "term_id": instance.get("term_id"),
            "reason_he": reason_he,
        })

    def _rid(self, instance: Mapping[str, Any], n: Optional[int] = None) -> str:
        return _rule_id(self.out.agreement_id, self.out.version_id,
                        str(instance.get("instance_id")), n)

    def _advertisers(self, instance: Mapping[str, Any]) -> tuple[list[str], Optional[str]]:
        """(advertisers to emit for, or a skip reason when none can be named).

        A brand-scoped term without an advertiser is NOT fanned out to every
        represented advertiser and NOT fuzzy-matched by name: brand-to-
        advertiser binding is a human mapping the review screen owns, and a
        wrong guess here would price the wrong client's spots.
        """
        scope = instance.get("scope", {}) or {}
        scoped = [str(a) for a in scope.get("advertisers", [])]
        if scoped:
            return scoped, None
        if scope.get("brands"):
            return [], (
                "מונח ברמת מותג ללא שיוך מפרסם; יש לשייך את המותג למפרסם "
                "בעת הסקירה כדי שהכלל ייכנס לתוקף"
            )
        one = _advertiser_for(instance, self.head)
        if one:
            return [one], None
        if self.represented:
            return list(self.represented), None
        return [], None

    def _emit_condition(self, instance: Mapping[str, Any], *, effect: str,
                        value: Any = 1.0, mode: str = "multiplier",
                        sentence: str) -> bool:
        scope = instance.get("scope", {}) or {}
        if scope.get("campaigns"):
            # The engine's Condition dataclass is campaign-ready; the
            # conditions STORE has no scope_campaigns column yet. Writing the
            # row without the narrowing would widen a campaign rule to every
            # campaign — worse than not binding at all.
            self._skip(instance, (
                "הכלל תחום לקמפיין מסוים ומחסן התנאים אינו שומר עדיין תיחום "
                "קמפיין; הכלל לא יופעל אוטומטית עד הרחבת המחסן"
            ))
            return False
        advertisers, skip_reason = self._advertisers(instance)
        if skip_reason:
            self._skip(instance, skip_reason)
            return False
        if not advertisers:
            agency = _agency_for(self.head)
            if agency:
                row = _condition_row(
                    advertiser_id=agency, rule_id=self._rid(instance),
                    effect=effect, value=value, mode=mode,
                    instance=instance, sentence=sentence,
                )
                row["agency_id"] = row.pop("advertiser_id")
                row["_store"] = "agency_conditions"
                self.out.conditions.append(row)
                return True
            self._skip(instance, "אין מפרסם או סוכנות שאליהם ניתן לקשור את הכלל")
            return False
        for i, advertiser in enumerate(advertisers):
            row = _condition_row(
                advertiser_id=advertiser,
                rule_id=self._rid(instance, i if len(advertisers) > 1 else None),
                effect=effect, value=value, mode=mode,
                instance=instance, sentence=sentence,
            )
            row["_store"] = "advertiser_conditions"
            self.out.conditions.append(row)
        return True

    def _settlement_term(self, instance: Mapping[str, Any], kind: str,
                         payload: Mapping[str, Any]) -> None:
        self.out.settlement.setdefault("terms", []).append({
            "kind": kind,
            "instance_id": instance.get("instance_id"),
            "term_id": instance.get("term_id"),
            "scope": instance.get("scope", {}),
            "window": instance.get("window", {}),
            **payload,
        })

    # ------------------------------------------------------------ per term
    def compile(self) -> CompiledArtifacts:
        for instance in self.termset.get("instances", []):
            term_id = str(instance.get("term_id"))
            handler = getattr(self, "_t_" + term_id.replace("-", "_"), None)
            if handler is not None:
                handler(instance)
                continue
            spec = taxonomy.get(term_id)
            behaviours = set(spec.behaviours)
            if behaviours & {"prices", "constrains-hard", "constrains-soft"}:
                self._skip(instance, (
                    f"'{spec.name_he}' מתומחר/מגביל אך אין לו עדיין מהדר; "
                    "נרשם ומוצג, אינו פועל אוטומטית"
                ))
            # obliges terms are materialised by the obligations engine;
            # settles/process/meta terms are stored and surfaced — neither is a
            # compiler concern, and neither is silent.
        return self.out

    # constraints ----------------------------------------------------------
    def _t_programme_daypart_restrictions(self, instance: Mapping[str, Any]) -> None:
        mode = str(instance.get("params", {}).get("mode", "forbid"))
        hard = bool(instance.get("params", {}).get("hard", True))
        effect = "require" if mode == "allow" else "forbid"
        if not hard and effect == "forbid":
            self._emit_condition(instance, effect="pressure", value=STEER_SOFT_AVOID, mode="percent",
                                 sentence="הגבלה רכה מההסכם: הימנעות מההיקף המסומן (עוצמת הטיה תפעולית)")
            return
        self._emit_condition(
            instance, effect=effect,
            sentence=("שיבוץ רק בהיקף המותר על פי ההסכם" if effect == "require"
                      else "ההסכם אוסר שיבוץ בהיקף המסומן"),
        )

    def _t_content_adjacency_exclusion(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        excluded = [str(x) for x in params.get("excluded_content", [])]
        scope = dict(instance.get("scope", {}) or {})
        if not (scope.get("genres") or scope.get("programmes")):
            genres = [g for g in excluded if g]
            if not genres:
                self._skip(instance, "אין תוכן מוגדר להרחקה שניתן למפות לז'אנר או לתוכנית")
                return
            scope["genres"] = genres
        patched = {**instance, "scope": scope}
        self._emit_condition(
            patched,
            effect="forbid" if params.get("hard", True) else "pressure",
            value=1.0 if params.get("hard", True) else STEER_SOFT_AVOID,
            mode="multiplier" if params.get("hard", True) else "percent",
            sentence="הרחקת תוכן מההסכם: לא ישובץ בסמיכות לתכנים המנויים",
        )

    def _t_competitive_separation(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        rivals = [str(r) for r in params.get("rivals", [])]
        holder = _advertiser_for(instance, self.head)
        members = sorted({*(rivals or []), *([holder] if holder else [])})
        if len(members) < 2:
            self._skip(instance, (
                "הפרדה תחרותית ללא רשימת מתחרים מפורשת אינה ניתנת לאכיפה; "
                "יש למפות את חברי הקטגוריה בעת הסקירה"
            ))
            return
        unit = str(params.get("separation_unit", "same_break"))
        quantity = params.get("separation_quantity") or 0
        if unit == "same_break":
            limit_type, value, unit_out = "max_per_break", 1, ""
        else:
            limit_type = "min_separation"
            value = quantity
            unit_out = "minutes" if unit == "minutes" else "positions"
        self.out.frequency_rules.append({
            "rule_id": self._rid(instance),
            "limit_type": limit_type,
            "scope": "competing_group",
            "advertiser_id": "",
            "campaign": "",
            "ad": "",
            "pair_lead": "",
            "pair_closer": "",
            "competing_group": str(params.get("category") or "קטגוריה מוסכמת"),
            "members": ",".join(members),
            "value": value,
            "value_max": "",
            "unit": unit_out,
            "enabled": True,
            "notes": _note(instance, "הפרדה תחרותית מכוח ההסכם"),
        })

    def _t_frequency_caps(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        unit = str(params.get("unit", "break"))
        cap = int(params.get("cap", 1))
        advertisers, skip_reason = self._advertisers(instance)
        if skip_reason or not advertisers:
            self._skip(instance, skip_reason or "תקרת תדירות ללא מפרסם שניתן לקשור אליה")
            return
        # The frequency engine's vocabulary today: per-break, per-day,
        # consecutive, min-separation. An hourly or per-programme cap has no
        # enforcing primitive yet and is skipped BY NAME rather than mapped to
        # a different unit that would enforce something the contract did not say.
        limit_map = {"break": ("max_per_break", ""), "day": ("max_per_day", "")}
        if unit not in limit_map:
            self._skip(instance, (
                f"תקרת תדירות ליחידה '{unit}' אינה נתמכת עדיין במנוע התדירות "
                "(נתמך: לברייק, ליום); נרשמה למעקב ולא תיאכף אוטומטית"
            ))
            return
        limit_type, unit_out = limit_map[unit]
        for i, advertiser in enumerate(advertisers):
            self.out.frequency_rules.append({
                "rule_id": self._rid(instance, i if len(advertisers) > 1 else None),
                "limit_type": limit_type,
                "scope": "advertiser",
                "advertiser_id": advertiser,
                "campaign": "", "ad": "", "pair_lead": "", "pair_closer": "",
                "competing_group": "", "members": "",
                "value": cap, "value_max": "", "unit": unit_out,
                "enabled": True,
                "notes": _note(instance, "תקרת תדירות מכוח ההסכם"),
            })

    def _t_position_entitlements(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        positions = [str(p) for p in params.get("positions", [])]
        scope = dict(instance.get("scope", {}) or {})
        if positions and not scope.get("positions"):
            scope["positions"] = positions
        patched = {**instance, "scope": scope}
        self._emit_condition(
            patched, effect="pressure", value=STEER_POSITION_ENTITLEMENT, mode="percent",
            sentence="זכות מיקומים מההסכם: העדפת שיבוץ במיקומים המנויים (עוצמת הטיה תפעולית)",
        )
        if params.get("top_and_tail"):
            self._settlement_term(instance, "top_and_tail_right", {
                "note_he": (
                    "זכות טופ אנד טייל נקבעה בהסכם; כלל הזוג נכנס לתוקף ברמת "
                    "הקמפיין עם רישום החומרים (פותח/סוגר)"
                ),
            })

    def _t_category_exclusivity(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        premium = params.get("premium_percent")
        if premium is not None:
            self._emit_condition(
                instance, effect="premium", value=1 + float(premium) / 100.0,
                mode="multiplier",
                sentence=f"תוספת בלעדיות {premium}% מכוח ההסכם",
            )
        self._skip(instance, (
            "חסימת מתחרי הקטגוריה דורשת רשימת חברים; יש להשלים קבוצה מתחרה "
            "בעת הסקירה כדי שהבלעדיות תיאכף אוטומטית"
        ))

    def _t_adjacency_purchase(self, instance: Mapping[str, Any]) -> None:
        self._emit_condition(
            instance, effect="pressure", value=STEER_ADJACENCY_PURCHASE, mode="percent",
            sentence="רכישת סמיכות מההסכם: העדפת שיבוץ בתוכן היעד (עוצמת הטיה תפעולית)",
        )

    def _t_success_deal(self, instance: Mapping[str, Any]) -> None:
        self._emit_condition(
            instance, effect="pressure", value=STEER_SUCCESS_DEAL, mode="percent",
            sentence="עסקת הצלחה: עדיפות שיבוץ מוגברת מכוח ההסכם (עוצמת הטיה תפעולית)",
        )

    def _t_creative_constraints(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "creative_constraints", {
            "note_he": "אילוצי חומרים נאכפים בשכבת הנכסים והבקרה הטכנית",
            "params": instance.get("params", {}),
        })

    def _t_spot_length_constraints(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "spot_lengths", {
            "allowed_lengths_seconds": instance.get("params", {}).get(
                "allowed_lengths_seconds", []),
            "note_he": "אורכי תשדיר מותרים; נבדק מול ספר התשדירים היומי",
        })

    # money ----------------------------------------------------------------
    def _t_ratecard_index(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        index = params.get("index_percent")
        if index is None:
            self._skip(instance, "הצמדה למחירון ללא אחוז אינה ניתנת ליישום")
            return
        self._emit_condition(
            instance, effect="premium", value=float(index) / 100.0,
            mode="multiplier",
            sentence=f"תמחור לפי {index}% מהמחירון, מכוח ההסכם",
        )

    def _t_volume_discount_ladder(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "discount_ladder", {
            "tiers": instance.get("params", {}).get("tiers", []),
            "basis": instance.get("params", {}).get("basis"),
            "mechanics": instance.get("params", {}).get("mechanics"),
            "period": instance.get("params", {}).get("period"),
            "note_he": (
                "מדרגות הנחה הן חשבון תקופתי (רטרואקטיבי/שולי) ומיושמות "
                "בהתחשבנות, לא כמכפיל פר-ספוט"
            ),
        })

    def _t_agency_commission(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "agency_commission", {
            "percent": instance.get("params", {}).get("percent"),
            "base": instance.get("params", {}).get("base"),
            "form": instance.get("params", {}).get("form"),
            "note_he": "עמלת סוכנות מיושמת בשכבת הנטו של ההתחשבנות",
        })

    def _t_cash_discount(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "cash_discount", {
            "percent": instance.get("params", {}).get("percent"),
            "qualifying_terms": instance.get("params", {}).get("qualifying_terms"),
            "note_he": "הנחת מזומן שייכת לסליקת תשלומים, לא לתמחור שיבוץ",
        })

    def _t_seasonal_coefficients(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "seasonal_coefficients", {
            "rows": instance.get("params", {}).get("rows", []),
            "note_he": (
                "מקדמי עונתיות והחרגות הנחה מיושמים בהתחשבנות התקופתית; "
                "מקדמי תאריך מנועיים נשארים בשכבת התמחור המופעלת של הערוץ"
            ),
        })

    def _t_length_factor_table(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "length_factors", {
            "rows": instance.get("params", {}).get("rows", []),
            "note_he": "מקדמי אורך פר-הסכם מיושמים בהתחשבנות מול 30 שניות",
        })

    def _t_cpp_daypart_table(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "cpp_table", {
            "rows": instance.get("params", {}).get("rows", []),
            "audience": instance.get("params", {}).get("audience"),
            "note_he": (
                "טבלת CPP היא בסיס ההתחשבנות מול הלקוח; המנוע ממשיך לתמחר "
                "מלאי במחירון הערוץ, וההפרש נמדד כ-CPP אפקטיבי"
            ),
        })

    def _t_fixed_spot_pricing(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "fixed_spots", {
            "rows": instance.get("params", {}).get("rows", []),
            "note_he": "מחירים קבועים נסלקים ככתבם; אינם עוברים דרך מכפילי המנוע",
        })

    def _t_gold_break_rates(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "gold_rates", {
            "surcharge_percent": instance.get("params", {}).get("surcharge_percent"),
            "fixed_prices": instance.get("params", {}).get("fixed_prices", []),
            "note_he": "תעריפי ברייק זהב פר-הסכם מיושמים בהתחשבנות",
        })

    def _t_sponsorship_terms(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "sponsorship", {
            "params": instance.get("params", {}),
            "note_he": "חסות נסלקת במחיר קבוע פר שידור; מלאי נפרד מספוטים",
        })

    def _t_target_cpp(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "target_cpp", {
            "audience": instance.get("params", {}).get("audience"),
            "cpp": instance.get("params", {}).get("cpp"),
            "note_he": (
                "CPP לקהל יעד הוא בסיס התחשבנות; מדידתו ממתינה למטבע "
                "קהל-יעד סלוק (ראו שער ההתחשבנות ברייטינג)"
            ),
        })

    def _t_package_bundle(self, instance: Mapping[str, Any]) -> None:
        params = instance.get("params", {})
        outside = [c for c in params.get("components", []) if not c.get("in_product")]
        self._settlement_term(instance, "bundle", {
            "components": params.get("components", []),
            "bundle_price": params.get("bundle_price"),
            "note_he": "רכיבי חבילה מחוץ למוצר נרשמים ואינם מתומחרים כאן",
        })
        if outside:
            self._skip(instance, (
                "רכיבי חבילה שאינם טלוויזיה ("
                + ", ".join(str(c.get("component")) for c in outside)
                + ") מחוץ להיקף המוצר; נרשמו בלבד"
            ))

    def _t_added_value_media(self, instance: Mapping[str, Any]) -> None:
        # The grant itself is an obligation/ledger act; nothing per-spot.
        return

    def _t_new_business_incentive(self, instance: Mapping[str, Any]) -> None:
        self._settlement_term(instance, "new_business_incentive", {
            "params": instance.get("params", {}),
            "note_he": "תמריץ לקוח חדש מיושם בהתחשבנות לתקופתו",
        })


def compile_termset(termset: Mapping[str, Any], head: Mapping[str, Any]) -> CompiledArtifacts:
    """Pure compilation of one approved termset. See module docstring."""
    return _Compiler(termset, head).compile()
