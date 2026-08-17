"""The arbiter: a third reading that holds everything and decides.

Two readers disagree about a signed contract. The cheap answer is to flag the
disagreement and make the reviewer settle it - but a reviewer handed forty
flags has been given the machine's problem to solve, and the flags themselves
become the noise that hides the two disagreements that actually move money.

So the third model is a JUDGE, not a third opinion. It receives the whole
document, the taxonomy definition of every contested term, and both readers'
candidates side by side, and it returns ONE ruling per contested pair: which
reading governs, or its own corrected reading, with the reason and the verbatim
quote it rests on. That ruling is the proposal a person then reviews.

WHAT THE JUDGE CANNOT DO, and why the guarantees survive it:

- It cannot change the clause list. Coverage is computed from the segmenter's
  clauses, which no model produced, so "every clause is accounted for" stays a
  mechanical fact rather than a model's claim.
- It cannot invent evidence. Every ruling's quote is checked in code against
  the clause it names; a ruling whose quote is not verbatim keeps the ruling
  but is forced to low confidence and says so, exactly as the clause pipeline
  does with a failed quote.
- It cannot approve anything. Its output is the proposed rule set. The hard
  rule is unchanged: the AI proposes, a person commits.

What the judge buys is that the human sees ONE ruled proposal with its
reasoning attached, instead of a pile of unresolved contradictions - and that
where the two readers agreed, nobody spent a model call re-deciding it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from . import taxonomy, taxonomy_schemas
from .documents import Citation, Clause, TermInstance

CallFn = Callable[..., dict[str, Any]]

# THE JUDGE'S PROMPT IS AN EXPERIMENTAL VARIABLE, so both versions live here
# and the measurement says which one produced which number.
#
# v1-primed was the first attempt and it carries a defect worth keeping on the
# record: it TELLS the judge that the clause reader "is precise on the clause's
# own details" and the whole-document reader "may be less precise" - an
# unearned prior planted on exactly the dimension in dispute - and it tells it
# to use the drop-the-term verdict "sparingly", which is a thumb on the scale
# towards keeping whatever was proposed. The first measurement suggested the
# opposite of the prior was true.
#
# v2-neutral describes each reader's SHAPE (one saw a clause, one saw the
# document) and says nothing about reliability, splits the two kinds of dispute
# apart - which parameters are right, versus whether the term is there at all -
# and removes the stigma from dropping a term nobody should have proposed.
#
# Select with KAIROS_TRADE_ARBITER_PROMPT; the default is what ships.
ARBITER_PROMPTS = {
    "v1-primed": """אתה השופט. שני קוראים קראו את אותו הסכם מסחרי חתום ואינם מסכימים.

קורא א' קרא סעיף-אחר-סעיף: הוא מדייק בפרטי הסעיף עצמו, אך רואה רק את הסעיף,
שכניו וההפניות המפורשות שלו.
קורא ב' קרא את המסמך כולו בבת אחת: הוא רואה הגדרות, נספחים והקשר רחוק, אך עלול
להיות פחות מדויק בפרטי הסעיף.

לפניך המסמך המלא, הגדרות המונחים, וכל מחלוקת בנפרד. הכרע בכל מחלוקת.

לכל מחלוקת החזר בדיוק אחת מההכרעות:
- "a"        קריאת קורא א' נכונה כפי שהיא.
- "b"        קריאת קורא ב' נכונה כפי שהיא.
- "revised"  שתיהן שגויות או חלקיות, ואתה כותב את הפרמטרים הנכונים בעצמך.
- "neither"  אין כאן מונח מסחרי כלל, ושתי הקריאות שגויות. השתמש בזה בקמצנות.

חוקים מחייבים:
- הכרעה חייבת מובאה מילולית מתוך הסעיף הנקוב, בדיוק כפי שהיא כתובה במסמך.
- params חייב להתאים לסכמת המונח. שדה שהמסמך אינו נותן נרשם ב-missing ולא מומצא.
- ההנמקה בעברית, משפט אחד, ואומרת מה במסמך הכריע - לא איזה קורא נשמע בטוח יותר.
- הכרעה על סתירה בין שני סעיפים בהסכם אינה עניינך: אם המסמך עצמו סותר את עצמו,
  שתי הקריאות נכונות במקומן, ומנגנון התקדימות יכריע ביניהן אחר כך.""",
    "v2-neutral": """אתה השופט. שני קוראים קראו את אותו הסכם מסחרי חתום ואינם מסכימים.

קורא א' ראה כל סעיף בנפרד, עם שכניו וההפניות המפורשות שלו.
קורא ב' ראה את המסמך כולו בבת אחת.

זה כל ההבדל ביניהם, והוא הבדל בשדה הראייה בלבד. אל תניח שאחד מהם מדויק יותר
מהשני, לא בפרטים ולא בהקשר: הכרע לפי מה שכתוב במסמך שלפניך, ולא לפי איזה קורא
נשמע בטוח יותר או ראה יותר.

לכל מחלוקת החזר בדיוק אחת מההכרעות:
- "a"        קריאת קורא א' נכונה כפי שהיא.
- "b"        קריאת קורא ב' נכונה כפי שהיא.
- "revised"  אף אחת אינה נכונה במלואה, ואתה כותב את הפרמטרים בעצמך.
- "neither"  אין בסעיף הזה מונח מסחרי כזה, ושתי הקריאות שגויות.

שני סוגי מחלוקת, ושתי שאלות שונות:

1. שני הקוראים ראו את אותו מונח באותו סעיף וחלוקים על הפרמטרים. השאלה היא
   שדה-שדה: מה בדיוק אומר הסעיף על כל שדה. אל תבחר צד בגלל שרוב השדות שלו
   נכונים - אם לכל אחד יש שדה נכון, ההכרעה היא "revised" עם הצירוף הנכון.

2. רק אחד מהקוראים ראה כאן מונח. השאלה אינה מי צודק אלא האם הסעיף באמת קובע את
   המונח הזה. סעיף שמזכיר מספר אינו בהכרח קובע התחייבות; סעיף שמפנה למקום אחר
   אינו בהכרח קובע בעצמו. אם הסעיף אינו קובע את המונח - "neither", וזו הכרעה
   נכונה ורגילה, לא ויתור.

חוקים מחייבים:
- הכרעה חייבת מובאה מילולית מתוך הסעיף הנקוב, בדיוק כפי שהיא כתובה במסמך.
- params חייב להתאים לסכמת המונח. שדה שהמסמך אינו נותן נרשם ב-missing ולא מומצא.
  קריאה שממציאה ערך לשדה שהסעיף שותק עליו גרועה מקריאה שמשאירה אותו חסר.
- ההנמקה בעברית, משפט אחד, ואומרת מה במסמך הכריע.
- סתירה בין שני סעיפים בהסכם אינה עניינך: אם המסמך סותר את עצמו, שתי הקריאות
  נכונות במקומן, ומנגנון התקדימות יכריע ביניהן אחר כך.""",
}

ARBITER_PROMPT_VERSION = (
    os.environ.get("KAIROS_TRADE_ARBITER_PROMPT", "").strip() or "v2-neutral"
)
ARBITER_SYSTEM = ARBITER_PROMPTS[ARBITER_PROMPT_VERSION]



def _contest_block(kind: str, clause: Clause, term_id: str,
                   a_side: Optional[dict[str, Any]],
                   b_side: Optional[dict[str, Any]]) -> str:
    spec = taxonomy.get(term_id)
    parts = [
        f'<מחלוקת סוג="{kind}" סעיף="{clause.clause_id}" מונח="{term_id}">',
        f"המונח: {spec.name_he} ({spec.name_en}) — {', '.join(spec.behaviours)}",
        f"סכמת המונח: {json.dumps(taxonomy_schemas.schema_for(term_id), ensure_ascii=False)}",
        f"<טקסט_הסעיף>\n{clause.text}\n</טקסט_הסעיף>",
    ]
    parts.append("קורא א': " + (json.dumps(a_side, ensure_ascii=False) if a_side else "לא ראה כאן את המונח הזה"))
    parts.append("קורא ב': " + (json.dumps(b_side, ensure_ascii=False) if b_side else "לא ראה כאן את המונח הזה"))
    parts.append("</מחלוקת>")
    return "\n".join(parts)


def _ruling_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rulings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clause_id": {"type": "string", "minLength": 1},
                        "term_id": {"type": "string", "minLength": 1},
                        "verdict": {"type": "string", "enum": ["a", "b", "revised", "neither"]},
                        "params": {"type": "object"},
                        "scope": taxonomy_schemas.SCOPE,
                        "window": taxonomy_schemas.WINDOW,
                        "quote": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "missing": {"type": "array", "items": {"type": "string"}},
                        "reason_he": {"type": "string", "minLength": 1},
                    },
                    "required": ["clause_id", "term_id", "verdict", "reason_he"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rulings"],
        "additionalProperties": False,
    }


def arbitrate(
    clauses: list[Clause],
    alignment: dict[str, Any],
    call: CallFn,
    *,
    document_id: str = "",
) -> dict[str, Any]:
    """Rule on every contested pair. Returns {instances, rulings, agreed_count}.

    One call per document, not one per disagreement: the judge's whole value is
    that it holds the document while it decides, and re-sending the document
    forty times would be the same reading paid for forty times.
    """
    by_id = {c.clause_id: c for c in clauses}
    pipeline_by_key = alignment["pipeline_by_key"]
    whole_by_key = alignment["whole_by_key"]

    contests: list[tuple[str, str, str]] = []  # (kind, clause_id, term_id)
    for clause_id, term_id in alignment["params_differ"]:
        contests.append(("שתי קריאות שונות", clause_id, term_id))
    for clause_id, term_id in alignment["pipeline_only"]:
        contests.append(("רק קורא א' ראה", clause_id, term_id))
    for clause_id, term_id in alignment["whole_only"]:
        contests.append(("רק קורא ב' ראה", clause_id, term_id))

    agreed_instances = [pipeline_by_key[k] for k in alignment["agreed"]]
    if not contests:
        return {"instances": list(agreed_instances), "rulings": [],
                "agreed_count": len(agreed_instances), "called": False}

    def _side_a(k: tuple[str, str]) -> Optional[dict[str, Any]]:
        inst = pipeline_by_key.get(k)
        if inst is None:
            return None
        return {"params": inst.params, "scope": inst.scope, "window": inst.window,
                "confidence": inst.confidence, "missing": inst.missing}

    blocks = []
    for kind, clause_id, term_id in contests:
        clause = by_id.get(clause_id)
        if clause is None:
            continue
        k = (clause_id, term_id)
        blocks.append(_contest_block(kind, clause, term_id, _side_a(k), whole_by_key.get(k)))

    document = "\n\n".join(
        f'<סעיף id="{c.clause_id}">\n{c.text}\n</סעיף>' for c in clauses
    )
    content = (
        "ההסכם המלא:\n\n" + document
        + "\n\nהמחלוקות להכרעה:\n\n" + "\n\n".join(blocks)
    )
    result = call(
        stage="arbitrate", tier="reason",
        system=ARBITER_SYSTEM,
        content=content,
        tool_name="record_rulings",
        tool_schema=_ruling_schema(),
    )

    instances: list[TermInstance] = list(agreed_instances)
    rulings: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for raw in result.get("rulings", []):
        if not isinstance(raw, dict):
            continue
        clause_id = str(raw.get("clause_id", ""))
        term_id = str(raw.get("term_id", ""))
        key = (clause_id, term_id)
        if key in seen_keys or term_id not in taxonomy.TERMS or clause_id not in by_id:
            continue
        seen_keys.add(key)
        verdict = str(raw.get("verdict", ""))
        record = {
            "clause_id": clause_id, "term_id": term_id, "verdict": verdict,
            "reason_he": str(raw.get("reason_he", "")),
        }
        if verdict == "neither":
            record["outcome"] = "dropped"
            rulings.append(record)
            continue

        a_inst = pipeline_by_key.get(key)
        b_side = whole_by_key.get(key)
        if verdict == "a" and a_inst is not None:
            instances.append(a_inst)
            record["outcome"] = "pipeline reading kept"
            rulings.append(record)
            continue

        if verdict == "b" and b_side is not None:
            params, scope, window = b_side["params"], b_side["scope"], b_side["window"]
            quote, confidence = b_side["quote"], b_side["confidence"]
            missing = b_side["missing"]
        else:  # revised, or a verdict whose named side is missing
            params = raw.get("params") or (b_side or {}).get("params") or {}
            scope = raw.get("scope") or {}
            window = raw.get("window") or {}
            quote = str(raw.get("quote", "")) or (b_side or {}).get("quote", "")
            confidence = str(raw.get("confidence", "medium"))
            missing = [str(m) for m in (raw.get("missing") or [])]

        # The judge's evidence is checked like anyone else's.
        note = ""
        if quote not in by_id[clause_id].text:
            confidence = "low"
            note = ("ההכרעה לא נשענה על מובאה מילולית מן הסעיף; "
                    "הובאה בהנמקה בלבד ולכן דורגה בביטחון נמוך")
            quote = by_id[clause_id].text[:120]
        instance_id = f"arb-{clause_id}-{term_id}".replace(" ", "")
        instances.append(TermInstance(
            instance_id=instance_id,
            term_id=term_id,
            params=params,
            citations=[Citation(document_id=document_id, page=by_id[clause_id].pages[0]
                                if getattr(by_id[clause_id], "pages", None) else 1,
                                clause_id=clause_id, quote=quote)],
            confidence=confidence,
            scope=scope,
            window=window,
            missing=missing,
            notes=" · ".join(x for x in [f"הוכרע בבוררות: {record['reason_he']}", note] if x),
        ))
        record["outcome"] = "whole-document reading kept" if verdict == "b" else "arbiter's own reading"
        rulings.append(record)

    return {"instances": instances, "rulings": rulings,
            "agreed_count": len(agreed_instances), "called": True}
