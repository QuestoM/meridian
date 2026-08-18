"""Deterministic conflict detection and precedence resolution.

Contracts contradict themselves; the product must be deterministic about
which term wins and WHY, and must refuse to coin-flip money. The algebra,
strongest rule first (docs/trade/engine-design.md §2):

1. explicit precedence edges (precedence clauses, amendment supremacy,
   agreement supersession), applied over document regions or whole documents;
2. the later effective date of the introducing document (amendments change
   the base — that is what they are for);
3. the more specific agreement level (campaign > advertiser > agency framework);
4. the more specific scope (more constrained dimensions wins);
5. for CONSTRAINTS only: the more restrictive side wins (safety tiebreak);
6. for money: no silent tiebreak — the conflict stays OPEN for a human.

Every resolution carries a Hebrew explanation the review surface shows
verbatim, and an open conflict blocks approval through the review gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from . import taxonomy

# Scope dimensions, mirrored from the condition engine's semantics:
# an empty set means ANY; two sets intersect when either is ANY or they share
# a member.
_SCOPE_DIMS = (
    "advertisers", "brands", "campaigns", "channels", "programmes",
    "genres", "dayparts", "weekdays", "positions", "lengths_seconds",
)

LEVEL_RANK = {"agency_framework": 0, "advertiser": 1, "campaign": 2}

# Behaviour split for rule 5/6: a constraints conflict may fall back to the
# safer side; a money conflict may not.
_MONEY_BEHAVIOURS = {"prices", "settles", "obliges"}


@dataclass(frozen=True)
class Provenance:
    """Where an instance came from, for precedence purposes."""

    agreement_id: str
    level: str
    document_id: str
    region: str  # body | appendix | amendment
    effective_date: str  # ISO date the introducing document takes effect

    def __post_init__(self) -> None:
        if self.level not in LEVEL_RANK:
            raise ValueError(f"unknown agreement level {self.level!r}")
        if self.region not in ("body", "appendix", "amendment"):
            raise ValueError(f"unknown document region {self.region!r}")


@dataclass(frozen=True)
class Edge:
    """One explicit precedence edge: winner-selector beats loser-selector.

    A selector matches an instance's provenance by any subset of
    {agreement_id, document_id, region}. The source is the clause that
    grants the precedence, quoted in every explanation that uses it.
    """

    winner: Mapping[str, str]
    loser: Mapping[str, str]
    source: str  # human sentence, e.g. 'סעיף 10.1: יגברו הוראות הנספחים'

    def _matches(self, selector: Mapping[str, str], prov: Provenance) -> bool:
        for key, value in selector.items():
            if getattr(prov, key, None) != value:
                return False
        return bool(selector)

    def decides(self, a: Provenance, b: Provenance) -> Optional[str]:
        """'a', 'b', or None when this edge does not separate the two."""
        if self._matches(self.winner, a) and self._matches(self.loser, b):
            return "a"
        if self._matches(self.winner, b) and self._matches(self.loser, a):
            return "b"
        return None


@dataclass
class Candidate:
    """One approved term instance entering resolution."""

    instance_id: str
    term_id: str
    params: dict[str, Any]
    scope: dict[str, Any]
    window: dict[str, Any]
    provenance: Provenance

    def behaviours(self) -> tuple[str, ...]:
        return taxonomy.get(self.term_id).behaviours


@dataclass
class Conflict:
    """A detected contradiction and what the algebra did about it."""

    conflict_id: str
    term_id: str
    instance_ids: tuple[str, str]
    contested: str
    resolution: str  # resolved_by_rule | open
    winner: Optional[str]
    rule: Optional[str]
    explanation_he: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "term_id": self.term_id,
            "instances": list(self.instance_ids),
            "contested": self.contested,
            "resolution": self.resolution,
            "winner": self.winner,
            "rule": self.rule,
            "explanation_he": self.explanation_he,
        }


def _tokens(scope: Mapping[str, Any], dim: str) -> frozenset:
    values = scope.get(dim) or ()
    return frozenset(str(v) for v in values)


def scopes_intersect(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    for dim in _SCOPE_DIMS:
        left, right = _tokens(a, dim), _tokens(b, dim)
        if left and right and not (left & right):
            return False
    return True


def windows_intersect(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    a_from, a_to = str(a.get("from") or ""), str(a.get("to") or "9999-12-31")
    b_from, b_to = str(b.get("from") or ""), str(b.get("to") or "9999-12-31")
    return a_from <= b_to and b_from <= a_to


def scope_specificity(scope: Mapping[str, Any]) -> int:
    return sum(1 for dim in _SCOPE_DIMS if _tokens(scope, dim))


def _params_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a == b


def _restrictiveness(candidate: Candidate) -> Optional[float]:
    """A comparable 'how restrictive' score for constraint terms, higher =
    stricter. Only defined where a safety direction genuinely exists."""
    params = candidate.params
    if candidate.term_id == "programme-daypart-restrictions":
        return 1.0 if params.get("mode") == "forbid" else 0.0
    if candidate.term_id == "competitive-separation":
        unit_rank = {"same_break": 1.0, "spots": 2.0, "minutes": 3.0}
        base = unit_rank.get(str(params.get("separation_unit")), 0.0)
        return base + float(params.get("separation_quantity") or 0)
    if candidate.term_id == "frequency-caps":
        cap = params.get("cap")
        return -float(cap) if cap is not None else None  # lower cap = stricter
    if candidate.term_id == "content-adjacency-exclusion":
        radius_rank = {"same_break": 1.0, "adjacent_break": 2.0, "same_programme": 3.0}
        return radius_rank.get(str(params.get("radius")), 0.0)
    return None


def _conflict_id(term_id: str, a: str, b: str) -> str:
    digest = hashlib.sha256(f"{term_id}|{min(a, b)}|{max(a, b)}".encode()).hexdigest()
    return f"cf-{digest[:10]}"


def _decide(a: Candidate, b: Candidate, edges: Iterable[Edge]) -> tuple[str, Optional[str], Optional[str], str]:
    """(resolution, winner_instance_id, rule, explanation_he)."""
    # 1. explicit edges
    for edge in edges:
        side = edge.decides(a.provenance, b.provenance)
        if side is not None:
            winner = a if side == "a" else b
            return (
                "resolved_by_rule", winner.instance_id, "explicit_precedence",
                f"הוכרע לפי סעיף עדיפות מפורש: {edge.source}",
            )
    # 2. later effective date of the introducing document
    if a.provenance.effective_date != b.provenance.effective_date:
        winner = a if a.provenance.effective_date > b.provenance.effective_date else b
        return (
            "resolved_by_rule", winner.instance_id, "later_document",
            "הוכרע לטובת המסמך המאוחר: תיקון או נספח מאוחר גובר על הנוסח שקדם לו "
            f"({winner.provenance.effective_date})",
        )
    # 3. agreement level
    a_rank, b_rank = LEVEL_RANK[a.provenance.level], LEVEL_RANK[b.provenance.level]
    if a_rank != b_rank:
        winner = a if a_rank > b_rank else b
        names = {"campaign": "הסכם קמפיין", "advertiser": "הסכם מפרסם",
                 "agency_framework": "הסכם מסגרת סוכנות"}
        return (
            "resolved_by_rule", winner.instance_id, "specific_level",
            f"הוכרע לפי רמת ההסכם: {names[winner.provenance.level]} גובר על "
            f"{names[(a if winner is b else b).provenance.level]}",
        )
    # 4. scope specificity
    a_spec, b_spec = scope_specificity(a.scope), scope_specificity(b.scope)
    if a_spec != b_spec:
        winner = a if a_spec > b_spec else b
        return (
            "resolved_by_rule", winner.instance_id, "specific_scope",
            "הוכרע לפי היקף תחולה: הוראה ספציפית גוברת על הוראה כללית",
        )
    # 5. constraints: safer side
    behaviours = set(a.behaviours())
    if not (behaviours & _MONEY_BEHAVIOURS):
        ra, rb = _restrictiveness(a), _restrictiveness(b)
        if ra is not None and rb is not None and ra != rb:
            winner = a if ra > rb else b
            return (
                "resolved_by_rule", winner.instance_id, "safer_constraint",
                "הוכרע לצד המחמיר: באילוצי שיבוץ סתירה נפתרת לטובת המגבלה "
                "המחמירה יותר",
            )
    # 6. open
    return (
        "open", None, None,
        "סתירה שאין בה הכרעה דטרמיניסטית — נדרשת הכרעה אנושית לפני אישור",
    )


def detect_and_resolve(
    candidates: list[Candidate],
    edges: Optional[list[Edge]] = None,
) -> list[Conflict]:
    """Find every contradiction among the candidates and run the algebra.

    Two instances of the SAME term conflict when their scopes and windows can
    describe the same moment and their parameters differ. Identical-parameter
    duplicates are not conflicts: :func:`kairos.trade.collapse.collapse` folds
    those into one instance before this runs. That sentence used to read "they
    merge downstream" — and nothing downstream merged, so the reviewer saw both
    for as long as this module has existed. Cross-term contradictions ride the
    same machinery through the restrictiveness map where one exists.
    """
    edges = list(edges or [])
    conflicts: list[Conflict] = []
    by_term: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_term.setdefault(candidate.term_id, []).append(candidate)

    for term_id, group in sorted(by_term.items()):
        if len(group) < 2:
            continue
        spec = taxonomy.get(term_id)
        if spec.family == "NA":
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if not scopes_intersect(a.scope, b.scope):
                    continue
                if not windows_intersect(a.window, b.window):
                    continue
                if _params_equal(a.params, b.params):
                    continue
                resolution, winner, rule, explanation = _decide(a, b, edges)
                conflicts.append(
                    Conflict(
                        conflict_id=_conflict_id(term_id, a.instance_id, b.instance_id),
                        term_id=term_id,
                        instance_ids=(a.instance_id, b.instance_id),
                        contested=(
                            f"שני מופעים של '{spec.name_he}' חלים על אותו היקף "
                            "עם פרמטרים שונים"
                        ),
                        resolution=resolution,
                        winner=winner,
                        rule=rule,
                        explanation_he=explanation,
                    )
                )
    return conflicts


def edges_from_precedence_instances(
    instances: Iterable[Mapping[str, Any]],
    *,
    agreement_id: str,
    document_id: str,
) -> list[Edge]:
    """Interpret approved precedence-clause instances into edges.

    v1 recognises the two shapes the market actually writes: appendix-over-body
    (or the reverse) inside one document, and this-document-over-another
    (supersession / amendment supremacy). Anything else stays uninterpreted —
    an uninterpreted precedence clause resolves nothing, silently breaks
    nothing, and remains visible in review.
    """
    out: list[Edge] = []
    for inst in instances:
        if inst.get("term_id") != "precedence-clause":
            continue
        params = inst.get("params", {})
        winner_text = str(params.get("winner", ""))
        loser_text = str(params.get("loser", ""))
        source = str(params.get("verbatim") or winner_text)
        appendix_words = ("נספח", "הנספחים")
        body_words = ("גוף ההסכם", "ההסכם עצמו")
        if any(w in winner_text for w in appendix_words) and any(
            w in loser_text for w in body_words
        ):
            out.append(Edge(
                winner={"document_id": document_id, "region": "appendix"},
                loser={"document_id": document_id, "region": "body"},
                source=source,
            ))
        elif any(w in winner_text for w in body_words) and any(
            w in loser_text for w in appendix_words
        ):
            out.append(Edge(
                winner={"document_id": document_id, "region": "body"},
                loser={"document_id": document_id, "region": "appendix"},
                source=source,
            ))
        elif "הסכם זה" in winner_text or "תיקון זה" in winner_text:
            out.append(Edge(
                winner={"agreement_id": agreement_id},
                loser={},  # matches nothing by itself; kept for the record
                source=source,
            ))
    return [e for e in out if e.winner and e.loser]


def region_of_clause(clause_id: str) -> str:
    """The segmentation convention: appendix clause ids start with 'app';
    amendments arrive as their own documents and carry region at ingest."""
    return "appendix" if str(clause_id).startswith("app") else "body"
