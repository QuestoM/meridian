"""One term, stated twice, shown once — without losing either place it was said.

A clause that points at another clause makes the extractor read that other
clause again. Measured on the corpus: clause 7.1 says "as defined in 2.2", the
parameterise stage is handed 2.2 alongside 7.1 so it can verify quotes against
both, and it returns the same CPP a second time — anchored to 2.2, which already
has its own instance. Two cards reach the reviewer with identical numbers and
nothing to tell them apart, and the review surface is flat, so they land side by
side. Fifty-three clauses in the shipped corpus carry more than one instance.

There was a defence for this and it is aimed at the wrong moment.
:mod:`kairos.trade.extract_run` merges a pointer into its referent BEFORE
extraction, and only when the referring clause matches one of six Hebrew
phrasings and the classifier gave both clauses the same term. All three
conditions have to hold. The evidence that two readings are the same — identical
parameters, overlapping citations — does not exist until AFTER extraction, which
is where this module works.

:mod:`kairos.trade.precedence` states in its own docstring that identical
parameter duplicates "merge downstream". Until now nothing downstream merged.

WHAT IS AND IS NOT THE SAME READING
-----------------------------------
Collapsing too eagerly is worse than not collapsing: a reviewer who is shown one
card where the document said two different things approves a term that is not in
the agreement. So the bar is two-tiered, and the tier depends on whether the two
instances rest on the SAME EVIDENCE.

* **Sharing a cited clause.** They are two readings of one text. They merge when
  nothing they both state disagrees, and one phrasing may be fuller than the
  other — "גברים בני 18-44" and "גברים 18-44" are the same audience read twice.
* **No shared clause.** Two separate places in the document said something. They
  merge only on exact agreement: every shared parameter equal after
  normalisation, no containment allowed, and at least one of them substantive.

Both tiers require the same term, the same scope and the same window. NUMBERS
MUST BE EXACTLY EQUAL in both tiers — a CPP of 2,400 and a CPP of 2,412 are not
a duplicate, they are a contradiction, and contradictions belong to
:mod:`kairos.trade.precedence`, which cannot see one this module has erased.

WHAT A MERGE KEEPS
------------------
Everything. The citations union, so the reviewer sees every clause the term was
stated in — which is more than either card carried alone. The richer parameter
wherever one instance said something and the other left it blank. The higher
confidence, the narrower ``missing``, and both sets of notes. A clause whose
instance was folded away stays mapped, now to the survivor, so nothing becomes
unmapped and no clause loses its disposition.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from kairos.trade import taxonomy_schemas
from kairos.trade.documents import Citation, ClauseDisposition, TermInstance
from kairos.trade.standing import EMPTY_LEAVES

CONFIDENCE_ORDER = ("low", "medium", "high")

# Words whose presence flips a phrase rather than extending it. A reading that
# differs from another only by one of these is its opposite, not its fuller
# version, and merging the two is not recoverable.
NEGATIONS = frozenset({
    "לא", "אינו", "אינה", "אינם", "אין", "ללא", "למעט", "פרט", "מלבד", "בלבד",
    "שאינו", "שאינם", "מבלי", "אסור", "excluding", "except", "not", "without",
    "only", "no",
})


def _substantive(value: Any) -> bool:
    """Does this leaf actually say something?

    Shares its definition of emptiness with :mod:`kairos.trade.standing`, so
    "this instance says nothing" means one thing in this engine and not two.
    """
    if isinstance(value, bool):
        return value is not False
    if isinstance(value, str):
        return value.strip() not in ("", "<UNKNOWN>", "unstated")
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return value not in EMPTY_LEAVES


def leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """A params tree flattened to dotted paths, for leaf-by-leaf comparison.

    A list of plain values stays WHOLE rather than becoming one leaf per index.
    Dayparts, lengths and programme names are sets the document happened to
    write in an order; comparing them position by position makes the same two
    dayparts listed the other way round look like a disagreement, which would
    keep two identical readings apart for no reason. Lists of objects — rate
    rows, discount ladders — keep their positions, because there the order is
    part of what was agreed.
    """
    out: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            out.update(leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        if any(isinstance(item, (Mapping, list, tuple)) for item in value):
            for index, item in enumerate(value):
                out.update(leaves(item, f"{prefix}[{index}]"))
        else:
            out[prefix] = list(value)
    else:
        out[prefix] = value
    return out


def _text(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split())


def _leaves_agree(left: Any, right: Any, *, loose: bool) -> bool:
    """Do two readings of the same field say the same thing?

    ``loose`` allows one phrasing to be a fuller version of the other, which is
    only ever granted to instances resting on the same clause. Numbers are exact
    in both modes: a tolerance here would silently merge two different prices
    into one, and the difference between two prices is the whole product.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return sorted(_text(v) for v in left) == sorted(_text(v) for v in right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, (int, float)) != isinstance(right, (int, float)):
        return False
    one, other = _text(left), _text(right)
    if one == other:
        return True
    if not loose or not one or not other:
        return False
    if one in other or other in one:
        return True
    # Hebrew phrasings of one fact differ by inserted words as often as by
    # truncation — "גברים בני 18-44" and "גברים 18-44" are the same audience,
    # and neither contains the other. So one phrasing may be the other's words
    # plus more, UNLESS the extra words REVERSE it. "כולל מע\"מ" and "לא כולל
    # מע\"מ" would pass a word-subset test and mean opposite things, and this
    # merge is irreversible.
    short, long = sorted((set(one.split()), set(other.split())), key=len)
    if len(short) < 2 or not short <= long:
        return False
    return not (long - short) & NEGATIONS


def _envelope_agrees(left: Mapping[str, Any], right: Mapping[str, Any],
                     *, loose: bool) -> bool:
    """Scope and window decide when a term applies, so they have to match.

    They follow the same two tiers as the parameters, and for the same reason.
    Two readings of ONE clause may differ in how much of its envelope each
    picked up — measured, one reading of clause 2.2 caught the three programmes
    it names and the other, reached through a reference in 7.1, did not. That is
    one envelope read twice, and the fuller reading wins.

    Between readings of DIFFERENT clauses it is the opposite: an envelope stated
    in one place and absent in another is not the same envelope. One applies to
    three named programmes and the other to everything the agreement covers, and
    merging those would quietly widen a term to slots it was never sold for.
    """
    left_leaves = {k: v for k, v in leaves(dict(left)).items() if _substantive(v)}
    right_leaves = {k: v for k, v in leaves(dict(right)).items() if _substantive(v)}
    shared = set(left_leaves) & set(right_leaves)
    if any(not _leaves_agree(left_leaves[k], right_leaves[k], loose=loose) for k in shared):
        return False
    return loose or set(left_leaves) == set(right_leaves)


def _clauses(instance: TermInstance) -> set[str]:
    return {c.clause_id for c in instance.citations}


def same_reading(left: TermInstance, right: TermInstance) -> Optional[str]:
    """Why these two are one term stated twice, or None when they are not."""
    if left.term_id != right.term_id:
        return None

    shared_clauses = _clauses(left) & _clauses(right)
    loose = bool(shared_clauses)

    if not _envelope_agrees(left.scope, right.scope, loose=loose):
        return None
    if not _envelope_agrees(left.window, right.window, loose=loose):
        return None

    left_leaves = leaves(dict(left.params))
    right_leaves = leaves(dict(right.params))
    shared = set(left_leaves) & set(right_leaves)
    stated = [
        key for key in shared
        if _substantive(left_leaves[key]) and _substantive(right_leaves[key])
    ]
    for key in stated:
        if not _leaves_agree(left_leaves[key], right_leaves[key], loose=loose):
            return None

    if loose:
        # Two readings of one clause. Even when neither states a parameter — an
        # interpretive pair — the clause itself is the shared evidence.
        return f"both read clause {sorted(shared_clauses)[0]}"
    if not stated:
        # Nothing said, nothing shared. Two empty instances of the same term in
        # different clauses are two different silences, not one.
        return None
    return "identical in every parameter both state, in the same scope and window"


def _richer(left: Any, right: Any) -> Any:
    """The reading that says more, when both say the same thing."""
    if not _substantive(left):
        return right
    if not _substantive(right):
        return left
    if isinstance(left, str) and isinstance(right, str):
        return left if len(_text(left)) >= len(_text(right)) else right
    return left


def _merge_params(left: Any, right: Any) -> Any:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        out = dict(left)
        for key, value in right.items():
            out[key] = _merge_params(left[key], value) if key in left else value
        return out
    if isinstance(left, list) and isinstance(right, list):
        return left if len(left) >= len(right) else right
    return _richer(left, right)


def _merge_citations(left: Sequence[Citation], right: Sequence[Citation]) -> list[Citation]:
    """Every place the term was stated, each place named once."""
    out: list[Citation] = []
    seen: set[tuple[str, str, int]] = set()
    for citation in list(left) + list(right):
        key = (citation.clause_id, _text(citation.quote), citation.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out


def _merge_notes(left: str, right: str) -> str:
    parts: list[str] = []
    for note in (left or "", right or ""):
        for piece in str(note).split(" | "):
            text = piece.strip()
            if text and text not in parts:
                parts.append(text)
    return " | ".join(parts)


def merge(left: TermInstance, right: TermInstance) -> TermInstance:
    """One instance carrying both readings. The survivor keeps the first id."""
    params = _merge_params(dict(left.params), dict(right.params))
    stated = {key for key, value in leaves(params).items() if _substantive(value)}
    required = taxonomy_schemas.schema_for(left.term_id).get("required", [])
    missing = sorted(
        field for field in required
        if not any(key == field or key.startswith(f"{field}.") or
                   key.startswith(f"{field}[") for key in stated)
    )
    return TermInstance(
        instance_id=left.instance_id,
        term_id=left.term_id,
        params=params,
        citations=_merge_citations(left.citations, right.citations),
        confidence=max((left.confidence, right.confidence),
                       key=lambda c: CONFIDENCE_ORDER.index(c)
                       if c in CONFIDENCE_ORDER else 0),
        scope=_merge_params(dict(left.scope), dict(right.scope)),
        window=_merge_params(dict(left.window), dict(right.window)),
        missing=missing,
        notes=_merge_notes(left.notes, right.notes),
    )


def collapse(
    instances: Iterable[TermInstance],
    dispositions: Iterable[ClauseDisposition] = (),
) -> tuple[list[TermInstance], list[ClauseDisposition], dict[str, Any]]:
    """Fold duplicate readings together, and say exactly what was folded.

    Order is preserved and the earliest instance survives, so a collapsed
    extraction reads in the same order as the document. Returns the instances,
    the dispositions rewritten to point at survivors, and a report — because a
    merge that nobody can audit is indistinguishable from a loss.
    """
    kept: list[TermInstance] = []
    folded_into: dict[str, str] = {}
    merges: list[dict[str, Any]] = []

    for instance in instances:
        for position, survivor in enumerate(kept):
            why = same_reading(survivor, instance)
            if why is None:
                continue
            before = _clauses(survivor)
            kept[position] = merge(survivor, instance)
            folded_into[instance.instance_id] = survivor.instance_id
            merges.append({
                "kept": survivor.instance_id,
                "folded": instance.instance_id,
                "term_id": instance.term_id,
                "why": why,
                "clauses_gained": sorted(_clauses(instance) - before),
            })
            break
        else:
            kept.append(instance)

    rewritten: list[ClauseDisposition] = []
    for disposition in dispositions:
        ids: list[str] = []
        for iid in disposition.instance_ids:
            survivor = folded_into.get(iid, iid)
            if survivor not in ids:
                ids.append(survivor)
        rewritten.append(ClauseDisposition(
            clause_id=disposition.clause_id,
            disposition=disposition.disposition,
            instance_ids=tuple(ids),
            irrelevant_class=disposition.irrelevant_class,
            reason=disposition.reason,
        ))

    report = {
        "instances_before": len(kept) + len(merges),
        "instances_after": len(kept),
        "merged": merges,
    }
    return kept, rewritten, report
