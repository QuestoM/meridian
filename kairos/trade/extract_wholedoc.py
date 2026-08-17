"""The second reader: one pass over the whole agreement, in one call.

The pipeline reads clause by clause. That is what makes the completeness
guarantee mechanical - every clause the segmenter found is handed to a model
and comes back with a disposition - but it is also its blind spot: a clause is
judged with its neighbours and its named cross-references, and nothing else.
An appendix that quietly redefines "רצועת פריים" twenty pages later, a
commitment whose real basis is stated in the recitals, a discount ladder whose
period is only knowable from the term of the agreement - those are visible to a
reader holding the entire document at once.

So this module is a SECOND reader with the opposite shape: the whole document,
every clause with its id, the taxonomy, one call on the reasoning tier. It
proposes instances anchored to the same clause ids the segmenter produced, so
the two readings can be laid side by side.

WHAT THIS READER IS NOT ALLOWED TO DO. It does not decide which clauses exist,
and it cannot retire one. The clause list is the segmenter's, produced without
a model, and the coverage ledger is computed from it. A reader that could also
define the denominator could report perfect coverage of a document it had only
half read - which is exactly the failure the completeness guarantee exists to
make impossible. This reader may only say what terms it sees and where.

Its output is not the answer either. Where the two readers agree, the agreement
is the evidence; where they disagree, kairos.trade.arbitrate rules on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from . import taxonomy, taxonomy_schemas
from .documents import Clause

CallFn = Callable[..., dict[str, Any]]

WHOLEDOC_SYSTEM = """אתה קורא הסכם מסחרי חתום בשוק הטלוויזיה הישראלי, במלואו ובבת אחת.

המשימה: לזהות כל מונח מסחרי בהסכם ולעגן אותו בסעיף שממנו הוא נובע.

היתרון שלך על פני קריאה סעיף-אחר-סעיף הוא שאתה רואה את כל המסמך יחד. השתמש בו:
- הגדרה שנקבעה בסעיף אחד וחלה על סעיף רחוק ממנו.
- נספח שמפרט, מצמצם או מבטל תנאי שנכתב בגוף ההסכם.
- תנאי שהבסיס שלו (ברוטו/נטו, קהל היעד, התקופה) נאמר במקום אחר במסמך.
- סתירה בין שני סעיפים: דווח על שניהם כשני מונחים נפרדים. אל תכריע ביניהם ואל
  תשמיט אחד מהם.

חוקים:
- כל מונח חייב clause_id מתוך רשימת הסעיפים שקיבלת, ומובאה מילולית שמופיעה
  באותו סעיף בדיוק כפי שהיא כתובה. אל תתקן ניסוח, אל תתרגם, אל תקצר.
- params חייב להתאים לסכמה של אותו מונח. שדה שהמסמך אינו נותן - רשום ב-missing
  ואל תמציא ערך.
- אם סעיף אינו מסחרי (כותרת, חתימות, מבוא, כתובות), פשוט אל תפיק ממנו מונח.
- עדיף מונח אחד מדויק על שלושה משוערים."""


def _term_catalogue() -> str:
    """The taxonomy as the reader sees it: id, both names, one line of behaviour."""
    lines = []
    for term_id, spec in sorted(taxonomy.TERMS.items()):
        lines.append(f"{term_id} | {spec.name_he} | {spec.name_en} | {', '.join(spec.behaviours)}")
    return "\n".join(lines)


def _wholedoc_schema() -> dict[str, Any]:
    """Instances only, each anchored to a clause id and a verbatim quote.

    params is deliberately loose HERE (a free object) and tightened per term
    afterwards against taxonomy_schemas: one call cannot carry sixty-four
    mutually exclusive schemas, and a union of them would let the model pick
    the loosest. Validating per term after the fact keeps the strict-schema
    contract without pretending one schema fits every term.
    """
    return {
        "type": "object",
        "properties": {
            "instances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clause_id": {"type": "string", "minLength": 1},
                        "term_id": {"type": "string", "minLength": 1},
                        "params": {"type": "object"},
                        "scope": taxonomy_schemas.SCOPE,
                        "window": taxonomy_schemas.WINDOW,
                        "quote": {"type": "string", "minLength": 1},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "missing": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                    },
                    "required": ["clause_id", "term_id", "params", "quote", "confidence"],
                    "additionalProperties": False,
                },
            },
            "document_notes": {"type": "string"},
        },
        "required": ["instances"],
        "additionalProperties": False,
    }


def read_whole_document(
    clauses: list[Clause],
    call: CallFn,
    *,
    page_images: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """One reading of the entire agreement. Returns {instances, notes, dropped}.

    Every returned instance is checked here, in code, before it is anyone's
    evidence: the clause id must be one the segmenter produced, the term id must
    be in the taxonomy, and the quote must appear verbatim in that clause. A
    reading that fails any of the three is DROPPED and counted, never repaired
    into something plausible - a second reader that invents its agreements is
    worse than no second reader.
    """
    by_id = {c.clause_id: c for c in clauses}
    body = "\n\n".join(
        f'<סעיף id="{c.clause_id}">\n{c.text}\n</סעיף>' for c in clauses
    )
    content: Any = (
        "רשימת המונחים בטקסונומיה (מזהה | שם | שם באנגלית | התנהגות):\n"
        f"{_term_catalogue()}\n\n"
        "ההסכם המלא, סעיף אחר סעיף:\n\n"
        f"{body}"
    )
    if page_images:
        content = list(page_images) + [{"type": "text", "text": content}]

    result = call(
        stage="wholedoc", tier="reason",
        system=WHOLEDOC_SYSTEM,
        content=content,
        tool_name="record_document",
        tool_schema=_wholedoc_schema(),
    )

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for raw in result.get("instances", []):
        if not isinstance(raw, dict):
            dropped.append({"reason": "not an object", "detail": str(raw)[:120]})
            continue
        clause_id = str(raw.get("clause_id", ""))
        term_id = str(raw.get("term_id", ""))
        quote = str(raw.get("quote", ""))
        clause = by_id.get(clause_id)
        if clause is None:
            dropped.append({"reason": "clause id not in the document", "detail": clause_id})
            continue
        if term_id not in taxonomy.TERMS:
            dropped.append({"reason": "term id not in the taxonomy", "detail": term_id})
            continue
        if quote not in clause.text:
            dropped.append({"reason": "quote is not verbatim in its clause",
                            "detail": f"{clause_id}: {quote[:60]}"})
            continue
        kept.append({
            "clause_id": clause_id,
            "term_id": term_id,
            "params": raw.get("params") or {},
            "scope": raw.get("scope") or {},
            "window": raw.get("window") or {},
            "quote": quote,
            "confidence": str(raw.get("confidence", "medium")),
            "missing": [str(m) for m in (raw.get("missing") or [])],
            "notes": str(raw.get("notes", "")),
        })
    return {
        "instances": kept,
        "dropped": dropped,
        "notes": str(result.get("document_notes", "")),
    }


def instances_from_records(records: list[dict[str, Any]], clauses: list[Clause],
                           *, document_id: str) -> list[Any]:
    """The validated records as TermInstances, so one scorer can read either reading.

    TermInstance refuses an instance with no citation, so this is also the last
    check that the second reader anchored everything it claimed.
    """
    from .documents import Citation, TermInstance

    by_id = {c.clause_id: c for c in clauses}
    out = []
    for index, record in enumerate(records, start=1):
        clause = by_id[record["clause_id"]]
        page = clause.pages[0] if getattr(clause, "pages", None) else 1
        out.append(TermInstance(
            instance_id=f"whole-{index:03d}-{record['term_id']}",
            term_id=record["term_id"],
            params=record["params"],
            citations=[Citation(document_id=document_id, page=page,
                                clause_id=record["clause_id"], quote=record["quote"])],
            confidence=record["confidence"],
            scope=record["scope"],
            window=record["window"],
            missing=record["missing"],
            notes=record["notes"],
        ))
    return out


def vision_blocks(image_paths: list[Any]) -> list[Any]:
    """Page images as provider content blocks, for a document declared scanned."""
    from .extract_provider import image_block

    return [image_block(Path(p).read_bytes()) for p in image_paths]


def align(pipeline_instances: list[Any], whole_instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Lay the two readings side by side, keyed by (clause, term).

    Three outcomes per key, and the names matter because the arbiter is only
    asked about the last one:

    - ``agreed``     both readers produced the pair. The agreement is evidence.
    - ``pipeline_only`` / ``whole_only``   one reader saw a term the other did
      not. Not automatically an error in either direction: the clause reader
      misses document-wide context, the whole reader misses depth.
    - ``params_differ``  both produced the pair and their parameters are not
      identical. This is where money actually diverges.
    """
    def key(clause_id: str, term_id: str) -> tuple[str, str]:
        return (str(clause_id), str(term_id))

    pipeline_by_key: dict[tuple[str, str], Any] = {}
    for inst in pipeline_instances:
        cites = getattr(inst, "citations", []) or []
        clause_id = cites[0].clause_id if cites else ""
        pipeline_by_key[key(clause_id, inst.term_id)] = inst

    whole_by_key = {key(i["clause_id"], i["term_id"]): i for i in whole_instances}

    agreed, params_differ, pipeline_only, whole_only = [], [], [], []
    for k, inst in pipeline_by_key.items():
        other = whole_by_key.get(k)
        if other is None:
            pipeline_only.append(k)
        elif json.dumps(inst.params, sort_keys=True, ensure_ascii=False) == \
                json.dumps(other["params"], sort_keys=True, ensure_ascii=False):
            agreed.append(k)
        else:
            params_differ.append(k)
    for k in whole_by_key:
        if k not in pipeline_by_key:
            whole_only.append(k)

    return {
        "agreed": sorted(agreed),
        "params_differ": sorted(params_differ),
        "pipeline_only": sorted(pipeline_only),
        "whole_only": sorted(whole_only),
        "pipeline_by_key": pipeline_by_key,
        "whole_by_key": whole_by_key,
    }
