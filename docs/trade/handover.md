# Paper to plan — handover for the Channel 13 demonstration

Written for the person giving the demonstration. Everything here is built,
committed, and verified on this machine; every number cited is measured, and
where a capability is off or honest-but-thin, that is stated rather than
smoothed over. The deeper design record is `docs/trade/domain.md` (the trade
domain), `docs/trade/term-taxonomy.md` (all 64 terms and what each does), and
`docs/trade/engine-design.md` (the extraction pipeline stage by stage).

## What exists, in one paragraph

A signed Hebrew trade agreement (PDF, digital or scanned) is uploaded to an
agreement record. The system reads it — ingest, segment, classify, extract
under strict per-term schemas, resolve cross-references — and proposes a
complete mapping: every clause either becomes structured commercial terms with
verbatim citations, or is classified irrelevant with a stated reason, or is
flagged loudly as understood-but-unmapped. Approval is impossible while any
clause is unseen, any proposal undecided, any unmapped clause unacknowledged,
or any conflict open — the server enforces the gate, not the button. On
approval the terms compile into live rules under the `TRD:` namespace, bind
into the same constraint and settlement stores the optimizer already reads,
and the agreement's commitments start being measured against real As-Run
delivery: delivered vs committed, pace vs elapsed time, projected shortfall,
and the make-good ledger. A simulation mode answers "what would this agreement
do to the money" against measured activity without writing anything. Daily
plans carry competing versions with an N-way comparison whose profitability
delta exposes its reasoning; the rating forecast is a first-class stage with
honest uncertainty and a published backtest.

## The demonstration, step by step

Have two terminals: `uvicorn kairos_api.server:app` from the repo root, and
`npm run preview` in `tv-break-dashboard` (or the deployed equivalent). The
demo store already holds six agreements in every lifecycle state.

1. **The agreements shelf** — מסחרי → הסכמי סחר. Six cards: draft, three in
   review with their blocker counts, and two approved. Point at the one card
   whose window has no end date: the law is that every agreement carries an
   end date; "forever" is recorded as 2099 and the card says so in words
   (ללא מועד סיום) instead of pretending a date.

2. **Review, the heart of Pillar 1** — the three in-review agreements each
   hold a different reason approval is impossible, and together they ARE the
   completeness guarantee:
   - הסכם מסגרת שנתי — אופק מדיה 2026: the deep review. Source document beside
     the proposed terms; walk one clause — verbatim quote, extracted
     parameters, confidence, the plain-language "what this rule will do". Try
     to approve: the server refuses and names every blocker (49 clauses
     unseen, 41 proposals undecided).
   - הסכם מפרסם — נובה פארם 2026: carries an **unmapped clause** — a term the
     pipeline understood but the taxonomy cannot represent. It is flagged
     loudly and blocks approval until a person acknowledges it with a reason.
     Say the sentence from the brief: being honest about a term we could not
     map is a success; quietly ignoring it is the one unacceptable outcome.
   - הסכם מסגרת — קבוצת ריטייל 2026: holds **three open conflicts** — clauses
     that fight over the same commercial fact. The precedence engine explains
     the decision order; a person resolves each, and the resolution is
     recorded with its reasoning. Conflicts block approval like anything else
     unresolved.
   The hard rule to say out loud: **the AI proposes, a person commits** —
   nothing extracted ever binds without this approval.

3. **An approved agreement that measures** — open הסכם מסגרת שנתי — סנו
   2025—2026. This counterparty is a real advertiser in the engine's measured
   world, so its commitments track live: budget commitment ₪480,000 gross with
   delivered-so-far from the As-Run book (engine-priced, labelled "not an
   invoice"), the TRP guarantee counted in the same all-viewers currency the
   delivery data actually carries, pace vs elapsed, both honestly at_risk,
   days with no source excluded from the count rather than counted as zero.
   The make-good policy routes to the three-level accrual ledger. Open the
   simulation panel: on measured gross the headline names the discount tier
   reached (none yet — below the first rung), the commission taken, the net
   left, and lists what it refuses to simulate rather than guessing.

4. **Rules that bind** — from the same record, show the live rules the
   approval wrote (`TRD:` ids), and that withdrawing the agreement removes
   them (supersession is part of the proof script). The optimizer reads these
   from the same stores it always read — no parallel rulebook.

5. **Daily plan versions, Pillar 2** — שידור → גרסאות היום. Three named
   people's competing versions of 01/11/2024 are already on the table (דנה
   לוי's untouched engine day, יואב כהן's gold-break variant, מיכל אברהם's
   two-break extension — all priced by the real engine at creation). Select
   all three and press השוואה: each side arrives with a one-line decision
   headline, the revenue delta AND its reasoning (which dayparts, which
   buckets, how much), revenue net of retention, the effect on contractual
   standing stated honestly ("unknown: no delivery row for this day" on this
   dataset), the guardrail verdict, and inventory consequences. Adopting a
   version demands a written reason, and the rejected alternatives stay in
   history.

6. **The forecast stage, Pillar 3** — שידור → תחזית רייטינג. Every programme
   of the day with its expected rating, the honest range, the drivers behind
   the number (including the families held out, by name), and the pre-model
   historical mean beside it. The measured verdict is PRINTED at the top of
   the table: the model beats the historical mean in log space (0.683 vs
   0.707) and loses in arithmetic rating points (1.188 vs 0.898), which is
   exactly why activation is gated and the number ships with its uncertainty.

7. **Live extraction, if there is time** — upload the סנו PDF to a fresh
   agreement and let the pipeline run while talking; it takes minutes, not
   seconds, and the progress is visible. Come back to a full proposal with
   citations. (If the room is impatient, the six seeded agreements ARE the
   result of this pipeline's corpus.)

## The numbers we publish

From `docs/trade/extraction-accuracy.md`, produced by
`scripts/trade_extraction_accuracy.py` running the real pipeline against
ground truth authored independently of it — nothing asserted by hand:

- Full-corpus run (eight documents, 2026-08-17): **every clause the pipeline
  read carries a disposition — the completeness guarantee holding
  mechanically**; disposition class 94.4%; term recall 93.8% (152/162); term
  precision 70.7%; parameter accuracy 65.1% (396/608 leaves); **zero
  citation-fidelity failures** (every quote the pipeline cites really is in
  the document); planted conflicts detected 5/6. The eighth document is the
  live demo's own סנו agreement: recall 88.9%, parameters 71.9%. Cost: ~0.7M
  input + ~0.2M output tokens and ~50 minutes for the whole corpus,
  per-document rows in the report.
- The weakest families are named, not hidden: G (process/legal) 17.0% and
  H (measurement/settlement) 21.7% parameter accuracy — these are the terms
  a reviewer must lean on the source pane for; the review gate is what makes
  that safe. Strongest: C (discounts/commissions) 81.5%, D (advertiser
  commitments) 76.9% — the money-bearing families.
- Segmentation round-trip on the corpus: clause recovery within ±10% on all
  eight documents, including the scanned document (vision route) and the
  reversed-bare-head form pdftotext produces for flat "1." numbering (caught
  by this corpus, fixed in `kairos/trade/segment.py`, pinned by test).
- Forecast backtest (walk-forward, 2,867 observations over five folds): the
  two objectives disagree, and both are published. Log-space RMSE 0.683 vs
  0.707 — the model wins, four folds of five. Arithmetic MAE 1.188 vs 0.898 —
  the pre-model historical mean wins, every fold, bias −0.249. The cause is
  named in the model's own module header: the retransformation shortfall
  (exp of a log-space level estimates the geometric centre, below the
  arithmetic mean of a right-skewed rating distribution). Interval coverage
  0.927 against a nominal 0.80 — the band is conservative, the right
  direction to err. This is why `audience_model_activation` stays OFF.

## How the AI layer is routed, and why

- **Tiers**: classification and page transcription run on the small model
  (Haiku); parameter extraction under strict schemas on the mid tier;
  cross-reference resolution and conflict reasoning reserve the top tier.
  Rationale: routing to the smallest model that does the job well, keeping
  heavy reasoning for genuine reasoning. Cost and latency per document are
  measured and published in the accuracy report.
- **Strict schemas everywhere**: every term has a JSON schema; anything that
  fails validation becomes an incomplete instance that NAMES its failure
  instead of a silent guess. A quote that is not verbatim in the source forces
  confidence to low.
- **Deliberate context**: each stage receives only its needs (a batch of
  clauses, one page image, one instance plus its cross-references), never the
  whole document by default.
- **Failure is loud**: rate limits back off and retry; a dead batch degrades
  to unmapped-flagged clauses, never to invented terms. The OAuth identity
  gate (Sonnet/Opus behind the Claude Code system block on Max) is handled in
  `kairos/trade/extract_provider.py`.
- **Draft state for money**: extraction writes proposals; only the human gate
  writes rules.

## Honest limits to say before they ask

- Parameter accuracy is the weakest number (64.3% overall; family G —
  process/legal — is the worst at 16.0%). The completeness guarantee is what
  makes this safe: wrong or thin parameters arrive as reviewable proposals
  with the source beside them, never as silently applied rules.
- Delivery standing is engine-priced from the As-Run book and says so; it is
  not an invoice reconciliation.
- The demo dataset carries one measured broadcast week (April 2025), so
  delivered-vs-committed on the סנו agreement is honestly far behind pace —
  that is the alarm machinery working, not a bug.
- Calendar-driven audience families and several pricing layers ship OFF by
  default with the measured reason recorded; nothing pretends.

## Where things live

| Piece | Path |
|---|---|
| Term taxonomy (human + machine) | `docs/trade/term-taxonomy.md`, `kairos/trade/taxonomy.py` |
| Extraction pipeline | `kairos/trade/extract_*.py`, `ingest.py`, `segment.py` |
| Completeness + review gate | `kairos/trade/documents.py`, `kairos_api/trade_review.py` |
| Precedence + obligations + compile | `kairos/trade/precedence.py`, `obligations*.py`, `compile.py` |
| Binding + API | `kairos_api/trade_bind.py`, `kairos_api/trade_api.py` |
| Simulation + explanations | `kairos/trade/simulate.py`, `explain.py` |
| UI | `tv-break-dashboard/src/trade/` |
| Corpus + accuracy harness | `tests/trade_corpus/`, `scripts/trade_extraction_accuracy.py` |
| End-to-end proof | `scripts/trade_end_to_end_proof.py` |
| Evidence harness | `tv-break-dashboard/scripts/capture-trade-evidence.mjs` |
