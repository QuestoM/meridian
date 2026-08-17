# The trade engine — design

2026-08-16. The domain contract is `docs/trade/domain.md`; the term catalogue
is `docs/trade/term-taxonomy.md` (+ `kairos/trade/taxonomy.py`, test-pinned).
This file fixes the architecture: entities, lifecycle, precedence, binding,
obligations, extraction, and surfaces. Section 10 records the seam-level
wiring decisions against the measured map of the existing system.

## 0. The four design commitments

1. **The agreement is the source; existing machinery is the muscle.** An
   approved agreement COMPILES into the product's existing rule primitives
   (advertiser/agency conditions, predicate constraints, frequency/separation
   rules, pricing entries) rather than into a parallel rule universe. Binding
   then comes through paths that already move money and block placements —
   and every compiled artifact carries `source: {agreement_id, term_id,
   instance_id}` so the moment a rule acts, the surface can say WHICH clause
   of WHICH agreement acted.
2. **Terms that compile ≠ terms that track.** `prices`/`constrains` terms
   compile to planning primitives. `obliges` terms materialise as living
   OBLIGATIONS in a new tracker (standing, projection, alarm, cure), feeding
   the make-good ledger. `settles` terms parameterise settlement.
   `process`/`meta` terms are stored, displayed, deadline-tracked. Nothing is
   dropped; each behaviour class has a defined landing.
3. **Nothing binds without a person.** Extraction produces a PROPOSAL.
   Approval is a human act that (a) requires the completeness gate green,
   (b) snapshots the reviewed state as an immutable agreement version through
   the same version machinery the rest of the product uses, and (c) runs the
   compiler. Un-approval (supersede/expire) decompiles cleanly.
4. **Determinism about conflicts.** Precedence is a total, explainable order;
   where the documents leave genuine ambiguity, the system holds an OPEN
   CONFLICT object that blocks approval of the affected terms — it never
   coin-flips.

## 1. Entities and stores

New package `kairos_api/trade_*` (API + stores) over `kairos/trade/*` (pure
logic). Storage follows the repository's store idioms (atomic writes,
versioned logical names, audit attribution).

- **Agreement** — id, counterparty (agency/advertiser refs), level
  (agency_framework / advertiser / campaign), title, status
  (`draft → in_review → approved → superseded/expired`), effective window,
  parent ref (amendment layering), source documents list.
- **AgreementDocument** — one uploaded file: original bytes on disk under
  `data/agreements/<agreement_id>/`, sha256, ingest route (digital/scanned),
  page count, language. Documents are immutable once attached.
- **DocumentExtraction** — the pipeline output (`kairos/trade/documents.py`):
  clauses, term instances (each with citations, confidence, missing-fields),
  clause dispositions, coverage. Stored as JSON per document, versioned by
  pipeline run.
- **ReviewState** — per proposed instance: `proposed → seen → edited |
  confirmed | rejected`, reviewer attribution, edit history (the reviewer's
  values never overwrite the extraction's — both are kept and the diff is
  visible). Reviewer-added instances carry `origin: reviewer` and a citation
  they point at (or an explicit "not in document" marker, which is allowed
  but loud).
- **AgreementVersion** — the immutable snapshot approval creates: the full
  reviewed term set, coverage state, precedence decisions, compiler output
  manifest, approver, timestamp, note. Restores and diffs work like every
  other versioned object in the product.
- **Obligation** — materialised from approved `obliges` instances:
  `{obligation_id, source, kind, basis, window, target, tolerance,
  checkpoints, consequence_ref, status}` plus computed
  `{standing, projection, alarm_level, evaluated_at}`.
- **MakeGoodLedger** — the three-level accrual/utilisation ledger:
  append-only entries `{entry_id, level: campaign|advertiser|agency,
  party_ref, direction: accrue|utilise|expire, quantity (money or seconds or
  points — unit named), reason, source, created_by, created_at}` with
  balances derived, never stored.
- **ConflictRecord** — two term instances + the contested scope + resolution
  (`resolved_by_rule | resolved_by_human | open`), the winning instance, and
  the explanation string shown at review and at act-time.

## 2. Precedence — the deterministic algebra

Applied per SCOPE INTERSECTION (two terms conflict only where their scopes
and windows overlap and their effects are incompatible):

1. **Explicit precedence clauses** (A6) between the two sources, if any.
2. **Later effective date** of the term's introducing document (amendment
   over base) — amendments exist to change the base.
3. **More specific agreement level**: campaign > advertiser > agency
   framework.
4. **More specific scope** within one agreement: programme > genre >
   daypart > channel-wide; named-brand > all-brands.
5. **Safer direction as a tiebreak for CONSTRAINTS only**: the more
   restrictive constraint wins (forbidding is safer than allowing).
6. **Money never tiebreaks silently.** Two pricing terms still tied after
   1–4 raise an OPEN CONFLICT: a human picks at review, and the pick is
   recorded on the version.

The resolver's output is stable, total for constraints, and every resolution
carries its explanation ("סעיף 4.2 בנספח ב' גובר על סעיף 7 בהסכם הבסיס —
נספח מאוחר קובע"). Conflicts that reach planning (should not happen —
approval blocks on open conflicts) fail closed: constraint = restrictive,
price = refuse to price with a named error.

## 3. The obligation engine

For each approved obliging instance, a **standing computation** per kind,
scheduled on data changes (plan writes, settlement runs, rating arrivals)
and on demand:

- basis resolvers per kind: billed spend (D1), delivered points vs committed
  curve (E1), effective CPP (E2), preferred-position rate with its counting
  method (E3), mix percentages (D3/D5), calendar coverage (D4), accrual
  balances (E5).
- **projection**: end-of-window estimate from pace to date + the plan's
  remaining committed activity + (for rating kinds) the audience model's
  expected TVR with its stated uncertainty. Projection provenance is always
  shown (measured-to-date vs modelled-forward split).
- **alarm ladder**: `on_track / watch / at_risk / breached`, with thresholds
  from the term's tolerance and checkpoints; alarms surface in the existing
  notification/attention surfaces and on the agreement page — EARLY, while
  buying/planning can still respond.
- **consequence engagement**: a breach (or an approaching one) opens a CURE
  worksheet from the term's cure parameters (E6): required quantity, allowed
  inventory quality, window; accepted cures write ledger entries and
  (optionally, human-approved) planning steers for bonus placements.

## 4. The extraction pipeline

`kairos/trade/extract_*` + `kairos_api/trade_extract_api.py`. Stages, each
with a stored artifact so reruns are partial and debuggable:

1. **Ingest** — accept PDF (digital or scanned) / images; hash, page count,
   route detection (text layer present? density heuristic); page text via
   PDF text extraction for digital, page IMAGES for scanned.
2. **Normalise** — RTL cleanup (bidi control chars, hyphenation), page
   headers/footers stripped (repeating-line detection), tables detected and
   kept as structured blocks.
3. **Segment** — clause boundaries from the document's own numbering
   (סעיף x.y, נספח א), headings, and layout; every character of the document
   belongs to exactly one clause (completeness starts here). Model-assisted
   when the numbering is broken; the segmenter's output is human-visible.
4. **Classify** — each clause → taxonomy term(s) | irrelevant-class |
   unmapped, with the term registry + Hebrew cues in the prompt context.
   Small-model stage, batched.
5. **Parameterise** — per (clause × term): extract params under the term's
   strict schema (tool-use JSON schema enforcement, retry on validation
   failure); confidence; verbatim quote per extracted value. Reasoning-model
   stage for the hard families (conditional/outcome terms, precedence,
   definitions), small model for simple ones — routed per term.
6. **Resolve cross-references** — definitions applied, "כמפורט בנספח"
   followed, amendment targets bound, scope inheritance (agreement-level
   scope flows to instances without their own).
7. **Assemble** — instances merged (a CPP table split over pages),
   duplicates coalesced with citations kept, conflicts detected (§2), the
   proposal built and validated (`DocumentExtraction.validate()`).

**Model routing** (the AI-layer contract): segmentation assist + clause
classification + simple params → small fast model; interpretation-heavy
parameterisation, cross-reference resolution, conflict analysis → the top
reasoning model; vision (scanned pages) → a vision-capable tier. Every call:
strict schema, bounded context (clause + its neighbours + the relevant
taxonomy slice — never the whole document), measured tokens/latency/cost
persisted on the extraction run, graceful partial results (a failed clause
becomes `unmapped` with the failure named, never a dropped clause).

### 4b. The second reader and the arbiter

`kairos/trade/extract_wholedoc.py` + `kairos/trade/arbitrate.py`.

The pipeline above reads clause by clause, and that shape is what makes the
completeness guarantee mechanical. It is also its blind spot: a clause is
judged with its neighbours and its named cross-references and nothing else,
so a definition set twenty pages away, or a basis stated only in the recitals,
is invisible to it.

So there is a SECOND reader with the opposite shape - the whole agreement, all
sixty-four term schemas, one call on the reasoning tier - and where the two
readings disagree, a THIRD call rules on it holding the document, the taxonomy
definition of each contested term, and both candidates side by side. One call
per document, not one per disagreement.

Three constraints keep the guarantees intact while a model decides:

1. **Neither the second reader nor the arbiter may change the clause list.**
   Coverage is computed from the segmenter's clauses, which no model produced.
   A reader that could also define the denominator could report perfect
   coverage of a document it had half read.
2. **Evidence is checked in code, not trusted.** A reading whose clause id does
   not exist, whose term is not in the taxonomy, or whose quote is not verbatim
   in the clause it names is DROPPED and counted. A ruling whose quote fails
   the same check survives as a ruling but is forced to low confidence and says
   so.
3. **The ruling is a proposal.** The hard rule is unchanged: a person approves.
   What the arbiter buys is that the person meets one ruled proposal with its
   reasoning attached instead of a pile of unresolved contradictions.

Two failure modes were measured rather than imagined, and both are pinned by
test in `tests/test_trade_arbitration.py`:

- A whole-document stage sharing the clause stages' 4000-token output ceiling
  is truncated mid-answer and returns ZERO terms with ZERO errors - the worst
  shape a failure can take. Stages now carry their own ceilings, and the notes
  field that consumed the budget is capped.
- A reader judged on schemas it was never shown scores its correct readings as
  parameter misses. All sixty-four schemas ride in its catalogue.

The arbiter's prompt is itself an experimental variable (`ARBITER_PROMPTS`,
selected by `KAIROS_TRADE_ARBITER_PROMPT`) because the first version told the
judge which reader to trust on the dimension in dispute. Both versions are
measured against each other in `docs/trade/arbitration-accuracy.md`.

## 5. Review and approval

The review surface (Commercial → הסכמים) shows document beside proposal:

- source pane: page-faithful document (PDF frame / page images), clause
  highlighting synced to selection.
- proposal pane: clause list with dispositions, instances grouped by family,
  per-instance: Hebrew term name, parameters in plain language, confidence,
  citation chips (click → source pane jumps), completeness gaps, conflicts.
- actions: confirm / edit (schema-validated form) / reject / add-missed
  (reviewer instance) / mark-irrelevant (closed reason list) / resolve
  conflict. Every action attributed and reversible until approval.
- the coverage header: clauses total / mapped / irrelevant / unmapped /
  unseen, and the approval button stays disabled with the reason named until
  unmapped = 0-or-disposed and unseen = 0. **The gate is server-enforced**,
  not a disabled button alone.
- plain-language effect: per instance, what it will DO ("יחסום שיבוץ מול
  מתחרי הקטגוריה בכל מקבץ", "יוסיף 4% הנחה מדורגת מעל 8 מ' ₪") — generated
  from the compiled artifact's own semantics, not from the model's prose.
- approval: creates the AgreementVersion, runs the compiler, and from that
  moment planning and pricing act under the new rules — visibly attributed.

## 6. Simulation

"מה היה קורה" mode: apply a PROPOSED (pre-approval) agreement version to
historical or currently-planned activity and show: money delta (pricing terms
re-applied to the period's activity), constraint violations that would have
existed (which placements would have been blocked/warned), obligation
standings as they would have evolved, and the make-good exposure. Runs on the
scenario machinery's isolated paths — never mutates live state — and its
report is the commercial director's pre-signature evaluation instrument.

## 7. Accuracy measurement

`tests/trade_corpus/` holds ground-truth agreements (§ the corpus doc).
The measurement harness runs the real pipeline against each corpus document
and scores, per term family and overall:

- **clause coverage**: % of clauses with a correct disposition class;
- **term recall/precision**: expected instances found (by term id + scope
  match) / found instances that exist in ground truth;
- **parameter accuracy**: exact-match for enums/numbers (with unit
  normalisation), fuzzy for free text, per-field;
- **citation fidelity**: cited quote actually appears in the cited clause.

Numbers are published in `docs/trade/extraction-accuracy.md`, per corpus
document and aggregated, dated, with the pipeline/model version — measured,
never asserted. Provider-dependent tests are gated behind the same env flags
the assistant suites use; the corpus + harness run in CI without a provider
by replaying stored pipeline outputs (recorded-run fixtures), and the live
measurement is an explicit operator/dev act.

## 8. Surfaces

- **Commercial → הסכמים**: agreement list (status, counterparty, window,
  coverage, standing summary) → agreement page (documents, versions, terms
  by family, obligations board, ledger, simulation) → review screen (per
  document) → approval.
- **Attribution at act-time**: planning boards show, on a blocked/warned/
  priced element, the responsible rule with its agreement + clause link
  (extending the existing rule-attribution seams).
- **Obligations everywhere they matter**: the delivery/pacing views gain the
  agreement dimension; version comparison (Pillar 2) shows per-version
  obligation impact; Today surfaces at-risk obligations.

## 9. What is deliberately NOT built

- No auto-approval path of any kind, including "high confidence".
- No OCR-text-layer trust for scanned docs: scanned pages go to vision
  models; a text layer in a scanned PDF is used only as a hint.
- No cross-agreement global optimisation of cures (the ledger records; the
  operator decides utilisation).
- No invented defaults for missing parameters — incompleteness is a review
  state, not a fill-in-the-blank.

## 10. Seam wiring (measured against the live tree)

Filled after the code-map pass; each row names the exact target the compiler
writes and the guard that proves identity when the agreement layer is off.
See `docs/trade/seam-map.md`.
