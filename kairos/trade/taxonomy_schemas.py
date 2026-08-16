"""Strict parameter schemas per trade term.

Every term in taxonomy.TERMS has exactly one schema here; the pipeline
validates extracted parameters against it and records missing REQUIRED fields
as incompleteness (surfaced at review) rather than discarding the instance —
a half-extracted term shown honestly beats a dropped one.

Two universal blocks live on the instance ENVELOPE, not in these schemas:

- ``scope``  — what the term applies to (advertisers, brands, campaigns,
  channels, programmes, genres, dayparts, weekdays, positions, lengths).
- ``window`` — the term's own effective window when it differs from the
  agreement's.

Values are extracted VERBATIM in the document's own vocabulary (Hebrew daypart
names, audience phrasing, position words); canonicalisation to product
vocabularies happens at review/compile time where a human can see the mapping.
Numbers use the humane conventions of the documents themselves: percents as
0–100, money in ILS unless the instance says otherwise.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import taxonomy

# --------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------

MONEY: dict[str, Any] = {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string", "default": "ILS"},
        "basis": {
            "type": "string",
            "enum": ["gross", "net_of_commission", "ratecard", "unstated"],
        },
    },
    "required": ["amount", "basis"],
    "additionalProperties": False,
}

PERCENT: dict[str, Any] = {"type": "number", "minimum": 0, "maximum": 1000}

DATE: dict[str, Any] = {"type": "string", "format": "date"}

WINDOW: dict[str, Any] = {
    "type": "object",
    "properties": {"from": DATE, "to": DATE},
    "additionalProperties": False,
}

SCOPE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "advertisers": {"type": "array", "items": {"type": "string"}},
        "brands": {"type": "array", "items": {"type": "string"}},
        "campaigns": {"type": "array", "items": {"type": "string"}},
        "channels": {"type": "array", "items": {"type": "string"}},
        "programmes": {"type": "array", "items": {"type": "string"}},
        "genres": {"type": "array", "items": {"type": "string"}},
        "dayparts": {"type": "array", "items": {"type": "string"}},
        "weekdays": {"type": "array", "items": {"type": "string"}},
        "positions": {"type": "array", "items": {"type": "string"}},
        "lengths_seconds": {"type": "array", "items": {"type": "number"}},
    },
    "additionalProperties": False,
}

AUDIENCE: dict[str, Any] = {"type": "string", "minLength": 1}

_MEASUREMENT_WINDOW_ENUM = ["campaign", "month", "quarter", "year", "custom"]


def _obj(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


def _rows(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": dict(item), "minItems": 1}


_NOTE = {"type": "string"}

# A minimal schema for process/legal terms whose commercial content is the
# clause text plus tracked dates: what it says, when it bites.
_PROCESS = _obj(
    {
        "summary": {"type": "string", "minLength": 1},
        "deadlines": _rows(
            _obj({"label": {"type": "string"}, "on": DATE}, ["label"])
        ) | {"minItems": 0},
        "details": _NOTE,
    },
    ["summary"],
)

# --------------------------------------------------------------------------
# Per-term schemas
# --------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- A
    "agreement-parties": _obj(
        {
            "counterparty_type": {
                "type": "string",
                "enum": ["agency", "advertiser", "advertiser_via_agency"],
            },
            "agency": {"type": "string"},
            "advertiser": {"type": "string"},
            "direct_client": {"type": "boolean"},
            "signatories": {"type": "array", "items": {"type": "string"}},
            # Agency frameworks name the advertisers they cover, typically in
            # an appendix; additions may need channel approval.
            "represented_advertisers": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        ["counterparty_type"],
    ),
    "brand-scope": _obj(
        {
            "included_brands": {"type": "array", "items": {"type": "string"}},
            "excluded_brands": {"type": "array", "items": {"type": "string"}},
        },
        [],
    ),
    "channel-scope": _obj(
        {
            "channels": {"type": "array", "items": {"type": "string"}},
            "non_tv_assets": {"type": "array", "items": {"type": "string"}},
        },
        ["channels"],
    ),
    "effective-window": _obj(
        {
            "starts_on": DATE,
            "ends_on": DATE,
            "auto_renewal": {"type": "boolean"},
            "renewal_notice_days": {"type": "integer", "minimum": 0},
        },
        ["starts_on", "ends_on"],
    ),
    "agreement-level": _obj(
        {
            "level": {
                "type": "string",
                "enum": ["agency_framework", "advertiser", "campaign"],
            },
            "parent_agreement": {"type": "string"},
        },
        ["level"],
    ),
    "precedence-clause": _obj(
        {
            "winner": {"type": "string", "minLength": 1},
            "loser": {"type": "string", "minLength": 1},
            "scope_note": _NOTE,
            "verbatim": {"type": "string", "minLength": 1},
        },
        ["winner", "loser", "verbatim"],
    ),
    "definitions": _obj(
        {
            "entries": _rows(
                _obj(
                    {
                        "term": {"type": "string", "minLength": 1},
                        "definition": {"type": "string", "minLength": 1},
                        "daypart_bounds": _obj(
                            {"start": {"type": "string"}, "end": {"type": "string"}},
                            ["start", "end"],
                        ),
                    },
                    ["term", "definition"],
                )
            )
        },
        ["entries"],
    ),
    "amendment-layer": _obj(
        {
            "base_agreement": {"type": "string"},
            "effective_on": DATE,
            "modifies": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string", "minLength": 1},
        },
        ["effective_on", "summary"],
    ),
    # ---------------------------------------------------------------- B
    "cpp-daypart-table": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "daypart": {"type": "string", "minLength": 1},
                        "cpp": {"type": "number", "exclusiveMinimum": 0},
                        "audience": AUDIENCE,
                    },
                    ["daypart", "cpp"],
                )
            ),
            "audience": AUDIENCE,
            "vintage": {"type": "string"},
            "base_length_seconds": {"type": "number", "default": 30},
        },
        ["rows", "audience"],
    ),
    "target-cpp": _obj(
        {
            "audience": AUDIENCE,
            "cpp": {"type": "number", "exclusiveMinimum": 0},
            "vintage": {"type": "string"},
        },
        ["audience", "cpp"],
    ),
    "length-factor-table": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "length_seconds": {"type": "number", "exclusiveMinimum": 0},
                        "factor": {"type": "number", "exclusiveMinimum": 0},
                    },
                    ["length_seconds", "factor"],
                )
            ),
            "rounding_rule": _NOTE,
        },
        ["rows"],
    ),
    "ratecard-index": _obj(
        {
            "index_percent": PERCENT,
            "ratecard_version": {"type": "string", "minLength": 1},
        },
        ["index_percent", "ratecard_version"],
    ),
    "fixed-spot-pricing": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "programme": {"type": "string"},
                        "slot_note": {"type": "string"},
                        "length_seconds": {"type": "number"},
                        "price": MONEY,
                    },
                    ["price"],
                )
            )
        },
        ["rows"],
    ),
    "sponsorship-terms": _obj(
        {
            "programme": {"type": "string", "minLength": 1},
            "airings": {"type": "integer", "minimum": 1},
            "period": {"type": "string"},
            "price_per_airing": MONEY,
            "notice_length_seconds": {"type": "number"},
        },
        ["programme", "price_per_airing"],
    ),
    "gold-break-rates": _obj(
        {
            "surcharge_percent": PERCENT,
            "fixed_prices": _rows(
                _obj(
                    {"scope_note": {"type": "string"}, "price": MONEY}, ["price"]
                )
            )
            | {"minItems": 0},
        },
        [],
    ),
    "payment-indexation": _PROCESS,
    # ---------------------------------------------------------------- C
    "volume-discount-ladder": _obj(
        {
            "tiers": _rows(
                _obj(
                    {
                        "threshold": {"type": "number", "minimum": 0},
                        "discount_percent": PERCENT,
                    },
                    ["threshold", "discount_percent"],
                )
            ),
            "basis": {
                "type": "string",
                "enum": ["ratecard_gross", "net_of_commission", "committed", "actual", "unstated"],
            },
            "mechanics": {
                "type": "string",
                "enum": ["retroactive", "marginal", "unstated"],
            },
            "period": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
        },
        ["tiers", "basis", "mechanics", "period"],
    ),
    "share-bonus": _obj(
        {
            "share_threshold_percent": PERCENT,
            "award_discount_percent": PERCENT,
            "award_media_percent": PERCENT,
            "denominator_source": _NOTE,
            "period": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
        },
        ["share_threshold_percent", "period"],
    ),
    "seasonal-coefficients": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "period_label": {"type": "string", "minLength": 1},
                        "coefficient": {"type": "number", "exclusiveMinimum": 0},
                        "discount_blackout": {"type": "boolean"},
                    },
                    ["period_label"],
                )
            )
        },
        ["rows"],
    ),
    "agency-commission": _obj(
        {
            "percent": PERCENT,
            "base": {
                "type": "string",
                "enum": ["gross", "net_of_discount", "unstated"],
            },
            "form": {
                "type": "string",
                "enum": ["invoice_deduction", "periodic_rebate", "unstated"],
            },
            "payment_cycle": {"type": "string"},
        },
        ["percent", "base", "form"],
    ),
    "cash-discount": _obj(
        {
            "percent": PERCENT,
            "qualifying_terms": {"type": "string", "minLength": 1},
        },
        ["percent", "qualifying_terms"],
    ),
    "success-deal": _obj(
        {
            "share_percent": PERCENT,
            "measurement_basis": {"type": "string", "minLength": 1},
            "settlement_cycle": {"type": "string"},
            "blend_note": _NOTE,
        },
        ["share_percent", "measurement_basis"],
    ),
    "added-value-media": _obj(
        {
            "percent": PERCENT,
            "basis": {"type": "string", "minLength": 1},
            "delivery_window": {"type": "string"},
            "quality_note": _NOTE,
        },
        ["percent", "basis"],
    ),
    "new-business-incentive": _obj(
        {
            "award_note": {"type": "string", "minLength": 1},
            "qualification": {"type": "string"},
        },
        ["award_note"],
    ),
    "package-bundle": _obj(
        {
            "components": _rows(
                _obj(
                    {
                        "component": {"type": "string", "minLength": 1},
                        "in_product": {"type": "boolean"},
                    },
                    ["component", "in_product"],
                )
            ),
            "bundle_price": MONEY,
            "bundle_discount_percent": PERCENT,
        },
        ["components"],
    ),
    # ---------------------------------------------------------------- D
    "budget-commitment": _obj(
        {
            "amount": MONEY,
            "period": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
            "splits": _rows(
                _obj(
                    {
                        "label": {"type": "string", "minLength": 1},
                        "amount": {"type": "number", "minimum": 0},
                    },
                    ["label", "amount"],
                )
            )
            | {"minItems": 0},
            "tolerance_percent": PERCENT,
        },
        ["amount", "period"],
    ),
    "share-commitment": _obj(
        {
            "share_percent": PERCENT,
            "period": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
            "denominator_source": {"type": "string", "minLength": 1},
            "declaration_cadence": {"type": "string"},
        },
        ["share_percent", "period", "denominator_source"],
    ),
    "daypart-mix": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "daypart": {"type": "string", "minLength": 1},
                        "min_percent": PERCENT,
                        "max_percent": PERCENT,
                    },
                    ["daypart"],
                )
            ),
            "basis": {
                "type": "string",
                "enum": ["money", "rating_points", "spots"],
            },
        },
        ["rows", "basis"],
    ),
    "flighting-obligation": _obj(
        {
            "rules": _rows(
                _obj({"rule": {"type": "string", "minLength": 1}}, ["rule"])
            )
        },
        ["rules"],
    ),
    "length-mix": _obj(
        {
            "rows": _rows(
                _obj(
                    {
                        "length_seconds": {"type": "number"},
                        "min_percent": PERCENT,
                        "max_percent": PERCENT,
                    },
                    ["length_seconds"],
                )
            ),
            "basis": {"type": "string", "enum": ["money", "spots", "seconds"]},
        },
        ["rows", "basis"],
    ),
    "cancellation-terms": _obj(
        {
            "windows": _rows(
                _obj(
                    {
                        "days_before_air": {"type": "integer", "minimum": 0},
                        "fee_percent": PERCENT,
                        "allowed": {"type": "boolean"},
                    },
                    ["days_before_air"],
                )
            ),
            "notes": _NOTE,
        },
        ["windows"],
    ),
    # ---------------------------------------------------------------- E
    "trp-delivery-guarantee": _obj(
        {
            "points": {"type": "number", "exclusiveMinimum": 0},
            "audience": AUDIENCE,
            "vintage": {"type": "string"},
            "window": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
            "daypart_split": _rows(
                _obj(
                    {
                        "daypart": {"type": "string"},
                        "points": {"type": "number", "minimum": 0},
                    },
                    ["daypart", "points"],
                )
            )
            | {"minItems": 0},
            "tolerance_percent": PERCENT,
            "checkpoints": {"type": "string"},
        },
        ["points", "audience", "window"],
    ),
    "effective-cpp-cap": _obj(
        {
            "cap": {"type": "number", "exclusiveMinimum": 0},
            "audience": AUDIENCE,
            "window": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
            "spend_basis": {
                "type": "string",
                "enum": ["gross", "net_of_commission", "unstated"],
            },
            "true_up_form": {"type": "string"},
        },
        ["cap", "audience", "window"],
    ),
    "preferred-position-guarantee": _obj(
        {
            "preferred_positions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "target_percent": PERCENT,
            "counting_method": {
                "type": "string",
                "enum": ["agency", "channel", "unstated"],
            },
            "window": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
        },
        ["preferred_positions", "target_percent", "counting_method"],
    ),
    "gold-break-allocation": _obj(
        {
            "count": {"type": "integer", "minimum": 0},
            "period": {"type": "string", "enum": _MEASUREMENT_WINDOW_ENUM},
            "first_refusal": {"type": "boolean"},
        },
        ["count", "period"],
    ),
    "makegood-accrual-policy": _obj(
        {
            "accruals": _rows(
                _obj(
                    {
                        "trigger": {"type": "string", "minLength": 1},
                        "rate_percent": PERCENT,
                        "level": {
                            "type": "string",
                            "enum": ["campaign", "advertiser", "agency"],
                        },
                    },
                    ["trigger", "level"],
                )
            ),
            "utilisation": {"type": "string", "minLength": 1},
            "expiry": {"type": "string"},
            "quality_note": _NOTE,
        },
        ["accruals", "utilisation"],
    ),
    "shortfall-cure": _obj(
        {
            "cure_form": {
                "type": "string",
                "enum": ["bonus_spots", "credit", "carry_forward", "mixed"],
            },
            "quality_rule": {"type": "string", "minLength": 1},
            "cure_window": {"type": "string", "minLength": 1},
            "valuation_basis": {"type": "string"},
            "recursion_rule": {"type": "string"},
        },
        ["cure_form", "quality_rule", "cure_window"],
    ),
    "underspend-true-up": _obj(
        {
            "trigger_note": {"type": "string", "minLength": 1},
            "re_rating_rule": {"type": "string"},
            "fee_percent": PERCENT,
            "waiver_conditions": _NOTE,
        },
        ["trigger_note"],
    ),
    "overdelivery-treatment": _obj(
        {
            "treatment": {
                "type": "string",
                "enum": ["banked", "charged", "absorbed", "unstated"],
            },
            "banking_cap": {"type": "string"},
        },
        ["treatment"],
    ),
    "preemption-compensation": _obj(
        {
            "qualifying_events": {"type": "string", "minLength": 1},
            "remedy_form": {"type": "string", "minLength": 1},
            "window": {"type": "string"},
            "quality_rule": {"type": "string"},
        },
        ["qualifying_events", "remedy_form"],
    ),
    # ---------------------------------------------------------------- F
    "competitive-separation": _obj(
        {
            "rivals": {"type": "array", "items": {"type": "string"}},
            "category": {"type": "string"},
            "separation_unit": {
                "type": "string",
                "enum": ["same_break", "spots", "minutes"],
            },
            "separation_quantity": {"type": "number", "minimum": 0},
            "hard": {"type": "boolean"},
        },
        ["separation_unit", "hard"],
    ),
    "category-exclusivity": _obj(
        {
            "category": {"type": "string", "minLength": 1},
            "exclusivity_scope": {"type": "string", "minLength": 1},
            "carve_outs": {"type": "array", "items": {"type": "string"}},
            # Exclusivity is commonly priced; the surcharge rides the clause.
            "premium_percent": PERCENT,
        },
        ["category", "exclusivity_scope"],
    ),
    "content-adjacency-exclusion": _obj(
        {
            "excluded_content": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "radius": {
                "type": "string",
                "enum": ["same_break", "adjacent_break", "same_programme"],
            },
            "hard": {"type": "boolean"},
        },
        ["excluded_content", "radius", "hard"],
    ),
    "adjacency-purchase": _obj(
        {
            "target_content": {"type": "string", "minLength": 1},
            "break_relation": {"type": "string", "minLength": 1},
            "premium_note": _NOTE,
        },
        ["target_content", "break_relation"],
    ),
    "programme-daypart-restrictions": _obj(
        {
            "mode": {"type": "string", "enum": ["allow", "forbid"]},
            "hard": {"type": "boolean"},
        },
        ["mode", "hard"],
    ),
    "position-entitlements": _obj(
        {
            "positions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "per_break_max": {"type": "integer", "minimum": 1},
            "top_and_tail": {"type": "boolean"},
        },
        ["positions"],
    ),
    "creative-constraints": _obj(
        {
            "rules": _rows(
                _obj(
                    {
                        "creative": {"type": "string"},
                        "valid_until": DATE,
                        "rotation_percent": PERCENT,
                        "note": _NOTE,
                    },
                    [],
                )
            ),
            "qc_gate": {"type": "boolean"},
        },
        ["rules"],
    ),
    "spot-length-constraints": _obj(
        {
            "allowed_lengths_seconds": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
            },
            "notes": _NOTE,
        },
        ["allowed_lengths_seconds"],
    ),
    "frequency-caps": _obj(
        {
            "unit": {
                "type": "string",
                "enum": ["break", "programme", "hour", "day"],
            },
            "cap": {"type": "integer", "minimum": 1},
        },
        ["unit", "cap"],
    ),
    # ---------------------------------------------------------------- G
    "payment-terms": _obj(
        {
            "terms": {"type": "string", "minLength": 1},
            "billing_cycle": {"type": "string"},
        },
        ["terms"],
    ),
    "reporting-obligations": _obj(
        {
            "reports": _rows(
                _obj(
                    {
                        "report": {"type": "string", "minLength": 1},
                        "cadence": {"type": "string", "minLength": 1},
                    },
                    ["report", "cadence"],
                )
            )
        },
        ["reports"],
    ),
    "audit-rights": _PROCESS,
    "termination": _obj(
        {
            "summary": {"type": "string", "minLength": 1},
            "notice_days": {"type": "integer", "minimum": 0},
            "survival": {"type": "array", "items": {"type": "string"}},
        },
        ["summary"],
    ),
    "force-majeure": _obj(
        {
            "qualifying_events": {"type": "string", "minLength": 1},
            "relief": {"type": "string", "minLength": 1},
        },
        ["qualifying_events", "relief"],
    ),
    "confidentiality": _PROCESS,
    "credit-security": _PROCESS,
    "dispute-resolution": _PROCESS,
    # ---------------------------------------------------------------- H
    "settlement-mechanics": _obj(
        {
            "grain": {"type": "string", "minLength": 1},
            "cadence": {"type": "string"},
            "application_order": {"type": "string"},
            "rounding": {"type": "string"},
        },
        ["grain"],
    ),
    "measurement-source": _obj(
        {
            "source": {"type": "string", "minLength": 1},
            "audience_basis": {"type": "string", "minLength": 1},
            "vintage": {"type": "string", "minLength": 1},
            "final_rule": {"type": "string"},
        },
        ["source", "audience_basis", "vintage"],
    ),
    "delivery-truth-source": _obj(
        {
            "source_order": {"type": "string", "minLength": 1},
            "discrepancy_rule": {"type": "string"},
        },
        ["source_order"],
    ),
    "term-effective-windows": _obj(
        {
            "applies_to": {"type": "string", "minLength": 1},
            "window_note": {"type": "string", "minLength": 1},
        },
        ["applies_to", "window_note"],
    ),
    # ---------------------------------------------------------------- NA
    "regional-feed-splits": _PROCESS,
    "coop-invoicing": _PROCESS,
    "barter-inquiry": _PROCESS,
}


def schema_for(term_id: str) -> dict[str, Any]:
    try:
        return SCHEMAS[term_id]
    except KeyError:
        raise KeyError(f"no schema for term {term_id!r}") from None


def validate_registry_alignment() -> None:
    """Schemas and taxonomy must cover exactly the same term ids."""
    schema_ids = set(SCHEMAS)
    term_ids = set(taxonomy.TERMS)
    if schema_ids != term_ids:
        missing = sorted(term_ids - schema_ids)
        extra = sorted(schema_ids - term_ids)
        raise RuntimeError(
            f"taxonomy/schema drift: missing schemas {missing}, orphans {extra}"
        )


validate_registry_alignment()
