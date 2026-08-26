"""Does this counterparty already exist? — resolution before creation.

A document (an agreement, a daily file) names an agency or an advertiser by a
string. The string is not an identity: "או.אם.די", "OMD", "OMD ישראל בע\"מ" and
a typo of any of them are one commercial party. Creating a second record for a
party we already carry silently splinters its rules, its rebate and its history
across two ids that never reconcile — the most expensive kind of duplicate,
because every downstream number looks right on its own.

So before Kai proposes creating a party, it asks this module: is there already
a record this name means? The answer has two layers, and the second only runs
when the first leaves doubt:

1. **Deterministic signals**, free and always computed: normalized-name
   equality (Hebrew-aware, company suffixes stripped), an alias/display-name
   hit, a VAT-id match (the strongest single signal a company has), a fuzzy
   ratio and a token overlap. These alone settle the easy cases — an exact
   normalized match or a shared VAT id is a match without a model.

2. **Model adjudication**, only for the near-but-not-certain candidates and
   only when a credential is configured: one forced-tool call (the same
   structured-output shape the extractor uses, on the same subscription auth)
   that judges each candidate SAME / DIFFERENT / UNCERTAIN with a confidence
   and a one-line reason that cites the evidence. The model never invents a
   candidate; it only rules on the ones the signals already surfaced.

The verdict is deliberately conservative: ``exact`` needs a deterministic
certainty (normalized-exact or VAT), ``probable`` needs the model to affirm a
strong candidate, everything softer is ``possible`` (worth showing, not worth
auto-acting), and ``none`` means create freely. With no model the verdict is
capped at ``possible`` — a fuzzy match is a lead, never a fact, and the module
says so rather than pretending certainty it did not earn.

Nothing here writes. It reports, with evidence, so the operator (or Kai's own
next proposal) decides between using an existing record and creating a new one.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Company-form suffixes that are noise for identity: two records that differ
# only by "בע\"מ" / "Ltd" are the same party. Matched after punctuation is
# stripped, so quote styles in בע"מ / בע״מ / בעמ all collapse together.
_SUFFIXES = (
    "בעמ", "בע\"מ", "בע״מ", "בעם",
    "ltd", "limited", "inc", "incorporated", "llc", "co", "company",
    "ישראל", "israel",
)

# Verdict tiers, strongest first.
EXACT = "exact"
PROBABLE = "probable"
POSSIBLE = "possible"
NONE = "none"

# A candidate below this fuzzy ratio, with no other signal, is not worth the
# operator's attention or the model's token — it is a different party.
_FUZZY_FLOOR = 0.62
# At or above this, a fuzzy-only candidate is close enough to ADJUDICATE (send
# to the model) but never close enough to call a match on the ratio alone.
_FUZZY_ADJUDICATE = 0.74


def _strip_final_letters(text: str) -> str:
    """Fold the five Hebrew final forms to their medial forms, so a name that
    ends a token in one record and not the other still matches."""
    return text.translate(str.maketrans("ךםןףץ", "כמנפצ"))


def normalize_name(raw: str) -> str:
    """One party name reduced to its identity core: lowercased, Latin/Hebrew
    punctuation and quotes removed, whitespace collapsed, company suffixes and
    the bare country word dropped, Hebrew finals folded. Empty in, empty out."""
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    # Quotes and the Hebrew geresh/gershayim are DELETED, not spaced: בע"מ is one
    # suffix token, not two, and an acronym written או"אם keeps its letters
    # adjacent. Everything else non-alphanumeric (dots, dashes) becomes a space.
    text = re.sub(r"[\"'׳״`]+", "", text)
    text = re.sub(r"[^0-9a-z֐-׿ ]+", " ", text)
    text = _strip_final_letters(text)
    tokens = [t for t in text.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def _tokens(normalized: str) -> frozenset[str]:
    return frozenset(t for t in normalized.split() if t)


def _token_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", str(raw or ""))


@dataclass
class Candidate:
    """One existing record scored against the queried name."""

    entity_id: str
    name: str
    signals: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    # Filled by the model pass when it runs; None means the signals stood alone.
    verdict: Optional[str] = None  # "same" | "different" | "uncertain"
    confidence: Optional[float] = None
    reason: str = ""

    def public(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "entity_id": self.entity_id,
            "name": self.name,
            "signals": self.signals,
            "score": round(self.score, 3),
        }
        if self.verdict is not None:
            body["model_verdict"] = self.verdict
            body["model_confidence"] = self.confidence
            body["model_reason"] = self.reason
        return body


@dataclass
class Roster:
    """The existing parties of one kind, with every identity string each one
    answers to. ``vat`` is digits-only for a clean equality test."""

    kind: str  # "agency" | "advertiser"
    records: list[dict[str, Any]]  # {entity_id, name, aliases: [str], vat}


def agency_roster() -> Roster:
    from kairos_api import agencies

    frame = agencies._load_frame()
    records = []
    for _, row in frame.iterrows():
        aliases = [str(row.get("name", "")), str(row.get("display_name", "")),
                   *str(row.get("aliases", "")).split("|")]
        records.append({
            "entity_id": str(row.get("agency_id", "")).strip(),
            "name": str(row.get("name", "")).strip() or str(row.get("agency_id", "")).strip(),
            "aliases": [a.strip() for a in aliases if a.strip()],
            "vat": _digits(row.get("vat_id", "")),
        })
    return Roster(kind="agency", records=records)


def advertiser_roster() -> Roster:
    from kairos_api import advertisers

    frame = advertisers._load_frame()
    observed = advertisers._observed_names()
    records = []
    seen_ids = set()
    for _, row in frame.iterrows():
        entity_id = str(row.get("advertiser_id", "")).strip()
        seen_ids.add(entity_id)
        aliases = [str(row.get("name", "")), str(row.get("display_name", "")), entity_id]
        records.append({
            "entity_id": entity_id,
            "name": str(row.get("display_name", "")).strip() or str(row.get("name", "")).strip() or entity_id,
            "aliases": [a.strip() for a in aliases if a.strip()],
            "vat": _digits(row.get("vat_id", "")),
        })
    # Advertisers seen on air but never given a rule row are still real parties
    # a duplicate could collide with; carry them by name (their id is the name).
    for name in observed:
        if name not in seen_ids:
            records.append({"entity_id": name, "name": name, "aliases": [name], "vat": ""})
    return Roster(kind="advertiser", records=records)


def _roster_for(kind: str) -> Roster:
    if kind == "agency":
        return agency_roster()
    if kind == "advertiser":
        return advertiser_roster()
    raise ValueError("kind must be 'agency' or 'advertiser'")


def _score_record(query_norm: str, query_tokens: frozenset[str], query_vat: str,
                   record: dict[str, Any]) -> Candidate:
    """The deterministic signals of one existing record against the query."""
    best_fuzzy = 0.0
    best_overlap = 0.0
    exact = False
    alias_hit = False
    for alias in record["aliases"]:
        alias_norm = normalize_name(alias)
        if not alias_norm:
            continue
        if alias_norm == query_norm:
            exact = True
        # A raw (un-normalized) alias equality still counts as an alias hit even
        # when normalization would have caught it; kept explicit for the trace.
        if alias.strip().lower() == query_norm or alias_norm == query_norm:
            alias_hit = alias_hit or (alias.strip() != "")
        best_fuzzy = max(best_fuzzy, difflib.SequenceMatcher(None, query_norm, alias_norm).ratio())
        best_overlap = max(best_overlap, _token_overlap(query_tokens, _tokens(alias_norm)))
    vat_match = bool(query_vat) and query_vat == record.get("vat", "")
    signals = {
        "normalized_exact": exact,
        "alias_hit": alias_hit and not exact,
        "vat_match": vat_match,
        "fuzzy_ratio": round(best_fuzzy, 3),
        "token_overlap": round(best_overlap, 3),
    }
    # Score orders candidates; the verdict (below) is what actually decides.
    score = max(
        1.0 if exact else 0.0,
        1.0 if vat_match else 0.0,
        0.90 * best_fuzzy + 0.10 * best_overlap,
    )
    return Candidate(entity_id=record["entity_id"], name=record["name"], signals=signals, score=score)


def _deterministic_candidates(kind: str, name: str, vat_id: str,
                              extra_aliases: list[str]) -> list[Candidate]:
    roster = _roster_for(kind)
    query_norm = normalize_name(name)
    # Fold any evidence aliases (e.g. a second spelling from the document) into
    # the query token space so a match on either spelling surfaces.
    query_tokens = _tokens(query_norm)
    for alias in extra_aliases:
        query_tokens = query_tokens | _tokens(normalize_name(alias))
    query_vat = _digits(vat_id)
    scored = [_score_record(query_norm, query_tokens, query_vat, rec) for rec in roster.records]
    # Keep exact/vat always; keep fuzzy only above the floor.
    kept = [
        c for c in scored
        if c.signals["normalized_exact"] or c.signals["vat_match"]
        or c.signals["fuzzy_ratio"] >= _FUZZY_FLOOR
    ]
    kept.sort(key=lambda c: c.score, reverse=True)
    return kept


def _needs_model(candidates: list[Candidate]) -> bool:
    """Adjudicate only the genuinely ambiguous: a strong fuzzy candidate that is
    NOT already a deterministic certainty. Exact/VAT are settled; weak fuzzies
    are not worth a token."""
    for c in candidates:
        if c.signals["normalized_exact"] or c.signals["vat_match"]:
            continue
        if c.signals["fuzzy_ratio"] >= _FUZZY_ADJUDICATE or c.signals["token_overlap"] >= 0.5:
            return True
    return False


_ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "rulings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["same", "different", "uncertain"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["entity_id", "verdict", "confidence", "reason"],
            },
        },
    },
    "required": ["rulings"],
}


def _adjudicate(kind: str, name: str, evidence: str, candidates: list[Candidate]) -> bool:
    """Ask the model to rule on the ambiguous candidates, in place. Returns
    True if a model ran, False if none was configured (honest degradation)."""
    from kairos_api import assistant_auth

    auth = assistant_auth.resolve_auth()
    if auth is None:
        return False
    from kairos.trade.extract_provider import system_prefix

    subjects = [
        {"entity_id": c.entity_id, "name": c.name, "signals": c.signals}
        for c in candidates
    ]
    kind_he = "סוכנות" if kind == "agency" else "מפרסם"
    system = (
        "You resolve whether a commercial party named in a document is the SAME "
        f"as an existing {kind} on file, for an Israeli TV ad-sales operation. "
        "Two records are the SAME party when a trader would bill them as one: "
        "spelling variants, transliteration (Hebrew/Latin), company-form "
        "suffixes and typos do not make a different party; a genuinely different "
        "company with a similar name does. Rule on EACH candidate you are given. "
        "Never invent a candidate. When unsure, say 'uncertain' rather than "
        "guess. Answer only through the tool."
    )
    prompt = (
        f"The document names this {kind} ({kind_he}): {name!r}.\n"
        f"Evidence from the document: {evidence or '(none supplied)'}.\n\n"
        f"Existing candidates (with the deterministic signals already computed):\n"
        f"{subjects}\n\n"
        "For each candidate entity_id, rule same/different/uncertain with a "
        "confidence 0..1 and a one-line reason that cites the evidence or the "
        "signal you relied on."
    )
    client = assistant_auth.build_client(auth, timeout=40.0, max_retries=2)
    response = client.messages.create(
        model=_model_name(),
        max_tokens=1200,
        system=[*system_prefix(auth.mode), {"type": "text", "text": system}],
        messages=[{"role": "user", "content": prompt}],
        tools=[{"name": "rule_on_candidates",
                "description": "Structured ruling on each candidate.",
                "input_schema": _ADJUDICATION_SCHEMA}],
        tool_choice={"type": "tool", "name": "rule_on_candidates"},
    )
    rulings = _tool_input(response).get("rulings", [])
    by_id = {c.entity_id: c for c in candidates}
    for ruling in rulings:
        target = by_id.get(str(ruling.get("entity_id", "")))
        if target is None:
            continue
        target.verdict = str(ruling.get("verdict", "")) or None
        try:
            target.confidence = float(ruling.get("confidence"))
        except (TypeError, ValueError):
            target.confidence = None
        target.reason = str(ruling.get("reason", ""))[:300]
    return True


def _model_name() -> str:
    from kairos_api import assistant

    return assistant._model_name()


def _tool_input(response: Any) -> dict[str, Any]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "tool_use":
            return dict(getattr(block, "input", {}) or {})
    return {}


def _verdict(candidates: list[Candidate], model_ran: bool) -> tuple[str, Optional[Candidate]]:
    """The single verdict for the query, and the candidate it rests on."""
    if not candidates:
        return NONE, None
    top = candidates[0]
    if top.signals["normalized_exact"] or top.signals["vat_match"]:
        return EXACT, top
    # A model affirmation of a strong candidate is PROBABLE.
    for c in candidates:
        if c.verdict == "same" and (c.confidence or 0) >= 0.7:
            return PROBABLE, c
    # Anything the signals surfaced but nothing confirmed is worth showing.
    if top.score >= _FUZZY_FLOOR or top.verdict in {"same", "uncertain"}:
        # Without a model a fuzzy lead is capped at POSSIBLE by construction;
        # with a model, a non-affirmed candidate is POSSIBLE too.
        return POSSIBLE, top
    return NONE, None


_ACTION = {
    EXACT: "use_existing",
    PROBABLE: "use_existing",
    POSSIBLE: "ask",
    NONE: "create_new",
}


def resolve_counterparty(kind: str, name: str, *, vat_id: str = "",
                         aliases: Optional[list[str]] = None, evidence: str = "",
                         limit: int = 5) -> dict[str, Any]:
    """Is there already a record this name means? A verdict with its evidence.

    ``kind``: 'agency' or 'advertiser'. ``vat_id`` and ``aliases`` are extra
    identity the document supplied (a VAT id is the strongest single signal).
    ``evidence`` is a short free-text quote the model may cite. Never writes;
    returns candidates, a verdict (exact/probable/possible/none) and a
    recommended action (use_existing/ask/create_new).
    """
    if not str(name or "").strip():
        raise ValueError("name is required to resolve a counterparty")
    candidates = _deterministic_candidates(kind, name, vat_id, aliases or [])
    model_ran = False
    if _needs_model(candidates):
        try:
            model_ran = _adjudicate(kind, name, evidence, candidates)
        except Exception:  # noqa: BLE001 - a model failure degrades to signals, never crashes the read
            model_ran = False
    verdict, chosen = _verdict(candidates, model_ran)
    return {
        "kind": kind,
        "query": {"name": name, "normalized": normalize_name(name),
                  "vat_id": _digits(vat_id) or None, "aliases": aliases or []},
        "verdict": verdict,
        "recommended_action": _ACTION[verdict],
        "match": chosen.public() if chosen is not None else None,
        "candidates": [c.public() for c in candidates[:limit]],
        "model_used": model_ran,
        "basis": (
            "Deterministic identity signals (normalized name, alias, VAT id, "
            "fuzzy ratio, token overlap); ambiguous candidates adjudicated by "
            "the model with structured output when a credential is configured. "
            "A fuzzy-only lead is capped at 'possible' - never auto-acted."
        ),
    }
