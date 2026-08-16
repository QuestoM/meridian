"""The model-facing extraction stages, as pure functions over a ``call`` seam.

Each stage takes a callable with the StageCaller.call signature and returns
plain data; nothing here imports an SDK, so the whole pipeline is testable
with fakes and the live runner is the only place a provider exists.

Prompt discipline, per the AI-layer contract: bounded context (a clause, its
neighbours, the definitions and referenced clauses — never the whole
document), a forced tool schema on every call, refusal as a first-class
answer (``unmapped`` for classification, ``missing`` fields for
parameterisation), and verbatim quotes checked HERE — a citation the source
does not contain is dropped and the drop is recorded, because a fabricated
quote poisons the one thing the review screen exists to guarantee.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Optional

from . import taxonomy, taxonomy_schemas
from .documents import Clause, UNMAPPED

CallFn = Callable[..., dict[str, Any]]

CLASSIFY_BATCH = 10

# Families whose parameterisation is genuine interpretation work: conditional
# obligations, cures, precedence/definitions/amendment meta, ladder mechanics,
# measurement bases. Routed to the reasoning tier; everything else to mid.
HARD_TERMS = frozenset(
    [spec.id for spec in taxonomy.TERMS.values() if spec.family in ("E", "H")]
    + ["volume-discount-ladder", "precedence-clause", "definitions",
       "amendment-layer", "share-commitment", "success-deal"]
)


def _catalogue_text() -> str:
    lines = ["המונחים המסחריים המוכרים (term_id — שם — רמזים):"]
    for family, family_he in taxonomy.FAMILIES.items():
        specs = taxonomy.by_family(family)
        if not specs:
            continue
        lines.append(f"\n[{family}] {family_he}:")
        for spec in specs:
            cues = ", ".join(spec.cues[:4])
            lines.append(f"- {spec.id} — {spec.name_he}" + (f" (רמזים: {cues})" if cues else ""))
    lines.append("\nסיווגי אי-רלוונטיות מסחרית (irrelevant:<class>):")
    for key, label in taxonomy.IRRELEVANT_CLASSES.items():
        lines.append(f"- irrelevant:{key} — {label}")
    return "\n".join(lines)


CLASSIFY_SYSTEM = (
    "אתה מסווג סעיפים של הסכמי סחר לפרסום בטלוויזיה הישראלית עבור מנוע כללים. "
    "לכל סעיף קבע את כל התוויות המתאימות מתוך הקטלוג בלבד. כללים: "
    "(1) כל סעיף מקבל לפחות תווית אחת. "
    "(2) סעיף הנושא כמה מונחים מקבל כמה תוויות. "
    "(3) 'unmapped' רק לסעיף בעל תוכן מסחרי שאינו מתאים לאף מונח — וזו תשובה "
    "לגיטימית וחשובה; לעולם אל תמתח מונח כדי להימנע ממנה, וצרף בה note המתאר "
    "מה הסעיף עושה. "
    "(4) irrelevant:<class> רק לטקסט משפטי-טקסי חסר תוכן מסחרי, עם note קצר. "
    "(5) אל תמציא תוויות שאינן בקטלוג.\n\n"
) + _catalogue_text()

CLASSIFY_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "note": {"type": "string"},
                },
                "required": ["clause_id", "labels"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


def _valid_labels() -> frozenset[str]:
    return frozenset(
        list(taxonomy.TERMS)
        + [f"irrelevant:{key}" for key in taxonomy.IRRELEVANT_CLASSES]
        + [UNMAPPED]
    )


def classify_clauses(clauses: list[Clause], call: CallFn) -> dict[str, dict[str, Any]]:
    """clause_id -> {labels: [...], note: str}. Every clause answered.

    A clause the model skipped or labelled outside the catalogue is retried in
    a smaller batch once; a clause still unanswered lands as ``unmapped`` with
    the failure named — never silently dropped.
    """
    valid = _valid_labels()
    out: dict[str, dict[str, Any]] = {}

    def _run_batch(batch: list[Clause]) -> list[Clause]:
        prompt_lines = ["סווג את הסעיפים הבאים:"]
        for clause in batch:
            heading = f" (תחת: {clause.heading})" if clause.heading else ""
            prompt_lines.append(f"\n<clause id=\"{clause.clause_id}\"{heading}>\n"
                                f"{clause.text}\n</clause>")
        result = call(
            stage="classify", tier="small",
            system=CLASSIFY_SYSTEM,
            content="\n".join(prompt_lines),
            tool_name="record_classifications",
            tool_schema=CLASSIFY_TOOL_SCHEMA,
        )
        answered = set()
        for entry in result.get("classifications", []):
            cid = str(entry.get("clause_id", ""))
            labels = [str(l) for l in entry.get("labels", [])]
            kept = [l for l in labels if l in valid]
            if cid in {c.clause_id for c in batch} and kept:
                out[cid] = {"labels": kept, "note": str(entry.get("note", ""))}
                answered.add(cid)
        return [c for c in batch if c.clause_id not in answered]

    remaining: list[Clause] = []
    for start in range(0, len(clauses), CLASSIFY_BATCH):
        remaining.extend(_run_batch(clauses[start:start + CLASSIFY_BATCH]))
    still: list[Clause] = []
    for clause in remaining:
        still.extend(_run_batch([clause]))
    for clause in still:
        out[clause.clause_id] = {
            "labels": [UNMAPPED],
            "note": "הסיווג לא הוחזר על ידי המודל; מסומן לבדיקה אנושית",
        }
    return out


# ---------------------------------------------------------------- references

_REF_APPENDIX = re.compile(r"נספח\s+([א-ת])'?(?:\s+סעיף\s+(\d+))?")
_REF_SECTION = re.compile(r"(?:סעיף|בסעיף|לסעיף)\s+(\d+(?:\.\d+)+)")
_HEBREW_LETTER_ORDER = "אבגדהוזחטיכלמנסעפצקרשת"


def referenced_clause_ids(text: str, known_ids: Iterable[str]) -> list[str]:
    """Deterministic cross-reference resolution: the clause ids a clause
    points at, restricted to ids that actually exist in the document."""
    known = set(known_ids)
    found: list[str] = []
    for match in _REF_APPENDIX.finditer(text):
        letter, section = match.group(1), match.group(2)
        index = _HEBREW_LETTER_ORDER.find(letter)
        if index < 0:
            continue
        prefix = f"app{chr(ord('A') + index)}"
        if section:
            candidate = f"{prefix}-{section}"
            if candidate in known:
                found.append(candidate)
        else:
            found.extend(sorted(k for k in known if k.startswith(prefix + "-")))
    for match in _REF_SECTION.finditer(text):
        candidate = match.group(1)
        if candidate in known:
            found.append(candidate)
    seen: set[str] = set()
    return [f for f in found if not (f in seen or seen.add(f))]


# ------------------------------------------------------------- parameterise

PARAMETERISE_SYSTEM = (
    "אתה מחלץ פרמטרים מדויקים של מונח מסחרי מתוך סעיף בהסכם סחר לטלוויזיה "
    "הישראלית. כללים מחייבים: "
    "(1) חלץ אך ורק מה שכתוב — אל תשלים ערך שאינו בטקסט; שדה נדרש שחסר "
    "בטקסט נרשם ב-missing, וזו תוצאה נכונה. "
    "(2) quotes: לכל חילוץ צרף ציטוטים מילוליים מדויקים מתוך הסעיף (העתקה "
    "תו-בתו, כולל ניקוד ופיסוק כפי שהם), התומכים בערכים שחילצת. "
    "(3) scope: ציין על מה המונח חל (מפרסמים, מותגים, תוכניות, ז'אנרים, "
    "רצועות, ימים, מיקומים) — במילות המסמך עצמו. "
    "(4) window: אם למונח חלון תוקף משלו (מתאריך/עד תאריך), ציין ISO. "
    "(5) אחוזים כמספרים (12 ולא 0.12); סכומים בשקלים כמספרים; תאריכים "
    "בפורמט YYYY-MM-DD. "
    "(6) confidence: high רק כשהטקסט חד-משמעי; low כשנדרשה פרשנות."
)


def _wrap_schema(term_id: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "params": taxonomy_schemas.schema_for(term_id),
            "scope": taxonomy_schemas.SCOPE,
            "window": taxonomy_schemas.WINDOW,
            "quotes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "missing": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["params", "quotes", "confidence"],
        "additionalProperties": False,
    }


def parameterise(
    clause: Clause,
    term_id: str,
    call: CallFn,
    *,
    neighbours: Optional[list[Clause]] = None,
    referenced: Optional[list[Clause]] = None,
    definitions_note: str = "",
) -> dict[str, Any]:
    """One (clause × term) extraction under the term's strict schema.

    Returns {params, scope, window, quotes(list of verified verbatim strings),
    dropped_quotes, confidence, missing, notes}. Quotes are verified against
    the clause AND any referenced clauses; when every quote fails verification
    the confidence is forced to low and the failure is named in notes.
    """
    spec = taxonomy.get(term_id)
    tier = "reason" if term_id in HARD_TERMS else "mid"
    sections = [
        f"המונח לחילוץ: {term_id} — {spec.name_he} ({spec.name_en}).",
    ]
    if definitions_note:
        sections.append(f"הגדרות שנקבעו במסמך:\n{definitions_note}")
    if neighbours:
        for n in neighbours:
            sections.append(f"<שכן id=\"{n.clause_id}\">\n{n.text}\n</שכן>")
    sections.append(f"<הסעיף id=\"{clause.clause_id}\">\n{clause.text}\n</הסעיף>")
    for ref in referenced or []:
        sections.append(f"<מוזכר id=\"{ref.clause_id}\">\n{ref.text}\n</מוזכר>")
    result = call(
        stage="parameterise", tier=tier,
        system=PARAMETERISE_SYSTEM,
        content="\n\n".join(sections),
        tool_name="record_term",
        tool_schema=_wrap_schema(term_id),
    )
    quote_sources = [clause] + list(referenced or [])
    verified: list[dict[str, str]] = []
    dropped: list[str] = []
    for quote in result.get("quotes", []):
        text = str(quote)
        home = next((c for c in quote_sources if text in c.text), None)
        if home is not None:
            verified.append({"clause_id": home.clause_id, "quote": text})
        else:
            dropped.append(text)
    confidence = str(result.get("confidence", "low"))
    notes = str(result.get("notes", ""))
    if not verified:
        confidence = "low"
        notes = (notes + " | " if notes else "") + (
            "אף ציטוט שהוחזר אינו מופיע מילולית במקור; נדרש אימות אנושי"
        )
    params = dict(result.get("params", {}))
    missing = sorted(set(
        [str(m) for m in result.get("missing", [])]
        + [f for f in taxonomy_schemas.schema_for(term_id).get("required", [])
           if f not in params]
    ))
    return {
        "params": params,
        "scope": dict(result.get("scope", {})),
        "window": dict(result.get("window", {})),
        "quotes": verified,
        "dropped_quotes": dropped,
        "confidence": confidence,
        "missing": missing,
        "notes": notes,
    }


# -------------------------------------------------------------- transcription

TRANSCRIBE_SYSTEM = (
    "אתה מתמלל עמוד סרוק של הסכם מסחרי בעברית, נאמן למקור תו-בתו: שמור על "
    "מספור הסעיפים, על סדר השורות ועל שמות באנגלית כפי שהם. טבלה תומלל "
    "כשורות מופרדות בקו אנכי (|). קטע בלתי-קריא יסומן [לא קריא] — לעולם אל "
    "תנחש תוכן. הערה בכתב יד בשוליים תתומלל בשורה נפרדת שתחילתה "
    "[הערת שוליים]."
)

TRANSCRIBE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "illegible_regions": {"type": "integer", "minimum": 0},
        "has_margin_notes": {"type": "boolean"},
    },
    "required": ["text"],
    "additionalProperties": False,
}


def transcribe_page(image_content_block: dict[str, Any], page_number: int,
                    call: CallFn) -> dict[str, Any]:
    return call(
        stage="transcribe", tier="reason",
        system=TRANSCRIBE_SYSTEM,
        content=[
            {"type": "text", "text": f"עמוד {page_number}. תמלל במלואו."},
            image_content_block,
        ],
        tool_name="record_page",
        tool_schema=TRANSCRIBE_TOOL_SCHEMA,
    )
