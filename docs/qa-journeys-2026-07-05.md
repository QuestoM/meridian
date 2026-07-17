# Kairos QA: customer journeys, rule conformance, API bug hunt (2026-07-05)

Scope: the committed plan (output/weekly_break_schedule.csv, 8,704 rows, 4
channels, 2024-11-01 to 2024-11-30), the saved settings
(data/kairos_settings.json), the FastAPI surface (59 routes, exercised
in-process with TestClient), and the dashboard code paths. No product code was
changed; all verification is tests plus this report. The repository was under
live development during the audit: an uncommitted auth layer
(kairos_api/auth.py, middleware in server.py) landed mid-session and is noted
where it affects results.

## 1. The operator's five journeys, as the product implements them today

1. Morning check. Overview page (TVBreakDashboard.jsx:2529, useKairosData
   :1035-1123) fans out to /api/overview, /api/schedule, /api/break-operations
   and friends. Tiles are the committed plan's own sums (verified equal to the
   CSV: 8,991 breaks, 1,078,920 ad seconds, 215.34M ILS) and nulls render as
   "-" (finiteNumber guard :610-616), never fabricated zeros. The staleness
   banner (ScheduleStalenessBanner.jsx:24) is driven by
   overview.schedule_freshness, computed uncached per request
   (server.py:1905-1910) from the sidecar stamp
   (kairos/export/schedule_freshness.py). Verified lifecycle: no stamp reads
   unknown, stamp reads fresh, any input-group change flips stale naming the
   group, restamp (what a recompute does) clears it. Current live state is
   honestly stale (changed: settings, constraints) because
   data/kairos_constraints.csv appeared after the last stamp; the operator
   should recompute.
2. Decision loop. Recommendations come from /api/overview
   (server.py:750-826), owned channel only. Approve posts
   /api/break-decisions (server.py:2054), which resolves into a REAL override
   row (source=recommendation, rec_id, status=active, semantic anchor trio)
   in data/manual_overrides.csv via the same store as POST /api/overrides.
   Reject persists status=dismissed and provably never becomes an engine
   constraint. The override edit flips schedule freshness to stale
   (overrides group). Recompute-this-day (POST /api/jobs/recompute with a
   scope, recompute_api.py:102) runs the incremental engine: verified that a
   forbid override drives its segment to 0 breaks and 0 revenue while every
   other committed row survives byte-identically. The activity feed is
   client-side only (ActivityFeed.jsx, localStorage kairos.activity); there is
   no server-side feed endpoint.
3. Manual decision loop. ScheduleEditor caches /api/schedule/segments (owned
   channel only, with anchors); clicking opens ScheduleInspector which reads
   GET /api/schedule/segment/{id}. Verified field-for-field agreement with the
   CSV row (identity, plan, economics, retention, anchor) on a 9-row sample,
   plus the competitor boundary (any non-owned segment returns 404). Effect
   preview GET /api/overrides/effect isolates one candidate decision and
   writes nothing (store bytes unchanged); its baseline reproduces the
   committed day today (39 breaks, revenue within 1 agora) BUT it ignores the
   saved settings by construction, see BUG-4. Save posts /api/overrides;
   download GET /api/export/schedule.csv streams the saved plan (verified
   equal shape, ids, break counts, revenue sum).
4. Pricing. PricingManager reads GET /api/pricing: base, program and day
   layers live; show, position and ad_type ship activation-OFF with the
   promo=0 hazard disclosed as a structured warning. PUT /api/pricing
   deep-merges overrides into KairosSettings.pricing_overrides after
   validation (422 on a negative rate, nothing persisted). The price-slot
   tester multiplies out exactly (final_cpp equals base times the product of
   its own reported layers) and a saved base edit moves the tester by exactly
   the edit ratio; reset restores the shipped card. The same
   pricing_from_settings seam feeds the optimizer and export, so a saved edit
   is genuinely live at the next recompute.
5. Uploads and settings. GET /api/uploads/status reports honest tri-state
   in_use: programmes, spots and dayparts CSV uploads read in_use=false with
   the real reason while the reference xlsx files shadow them; the daily Wally
   file reads in_use=true; the rate card discloses that no engine code reads
   it. Settings: PUT /api/settings persists and echoes the full object; the
   floor maps into engine guardrails via guardrails_from_settings; and
   objective_mode is wired from the saved settings through the recompute body
   (recompute_api.py:52) into build_weekly_schedule and the optimizer, where
   an unknown mode raises (verified).

## 2. Rule conformance results (the committed plan vs the saved settings)

Method: segments rebuilt through the exporter's own path, joined to the CSV by
segment_id (120 of 120 channel-days matched exactly; program_type and
predicted_retention recompute to the cent, proving the plan matches today's
inputs), break geometry reconstructed with the engine's own
_segment_break_objects, then the engine's guardrail checks were run. Standing
gate: tests/test_guardrail_conformance.py (12 tests, all green, 5 seconds).

| Rule | Limit (settings) | Observed (worst case in plan) | Verdict |
|---|---|---|---|
| (a) breaks per broadcast hour | 4 | 4 | PASS (binding) |
| (b) ad seconds per hour | 720s | 480s | PASS |
| (c) protected programme hour cap | 480s | 480s over 875 protected hours | PASS (binding) |
| (d) daily ad load per channel-day | 9,600s | 9,600s | PASS (binding) |
| (e) retention floor (segments with breaks) | 0.72 | 0.83 min; 0-break rows honestly read 1.0 and are exempt per engine semantics | PASS |
| spacing (engine bonus rule) | 420s | 420.0s min gap | PASS (binding) |
| (f) gold breaks per day | 3 | 0 | PASS |
| (g) num_breaks bounds | 0..4 | 0..4; total_break_time = k x length exactly | PASS |
| (h) segment_id identity | unique, date\|channel\|index | 8,704 unique, all well-formed, date and channel parts agree with row | PASS |
| (i) is_gold provenance | only via gold override / pin / segment flag | all False with empty stores, exactly as the engine allows | PASS |

The plan sits exactly at four limits (hourly count, protected hour, daily
load, spacing), which is what a correct optimizer chasing revenue inside a
safe envelope should produce.

## 3. Ranked bug list

Each bug is pinned by a deliberately FAILING test in
tests/test_qa_known_bugs.py (run: pytest tests/test_qa_known_bugs.py). The
tests flip green when fixed and then serve as the regression gate.

1. BUG-1 (HIGH, honesty of the compliance promise). /api/compliance evaluates
   about 0.4 percent of the plan. The verdict chain runs through
   _build_break_operations, which truncates the EPG to the first 12
   programmes per channel (kairos_api/server.py:469), re-synthesizes break
   times, and caps counts at min(5, duration//18) (server.py:500). Result:
   the endpoint reports a maximum observed daily load of 24 minutes while the
   committed plan really peaks at 160.0 minutes (exactly at the cap) on
   several channel-days; days 2 to 30 are never checked, so a violation there
   would still read compliant. Repro: test_bug1. Suggested fix: evaluate
   guardrails on breaks reconstructed from the full weekly CSV (the
   conformance suite demonstrates the reconstruction) or persist and evaluate
   the optimizer's own placements.
2. BUG-2 (HIGH, fabrication). The break-operations board synthesizes is_gold
   from settings heuristics (prime hour and first break in programme,
   server.py:549-555) instead of reading the plan's is_gold. The committed
   plan has zero gold breaks, yet on a prime-time EPG slice the board marks 9
   breaks gold, 4 on one channel-day, above the cap of 3, and the compliance
   builder counts those fabricated golds against the gold guardrail
   (server.py:1162-1165). Today BUG-1's truncation hides it (the first 12
   programmes are early morning); the two bugs cancel until the EPG window
   moves. Repro: test_bug2. Suggested fix: read is_gold from the joined plan
   row and delete the synthesis.
3. BUG-3 (MEDIUM, settings switch does not gate the engine).
   gold_breaks_enabled and sponsorships_enabled are consulted only by the
   display synthesis (server.py:550-551) and the gold report
   (insights_api.py). guardrails_from_settings (kairos/service.py:128-153)
   drops both flags, so with sponsorships disabled an active gold override
   still emits is_gold=True placements (optimizer.py:394). Repro: test_bug3.
   Suggested fix: gate gold overrides and pins where settings become engine
   inputs and report the rejection via rejected_overrides.
4. BUG-4 (MEDIUM, preview policy drift). The override effect preview, the
   number the operator reads before saving a decision, never reads the saved
   settings: kairos_api/overrides.py:223-226 builds segments with the bare
   YAML pricing and classifier, and :314-319 calls optimize_breaks with
   default Guardrails() and the default revenue weight, while the committed
   plan is built from the saved settings (kairos/export/schedule.py:228-244).
   Today the saved guardrails equal the engine defaults, so the preview
   measurably reproduces the committed day; the moment the operator tightens
   any rule the preview quotes numbers from a policy world the recompute will
   not produce (demonstrated: with a saved 1-break-per-hour cap the preview
   still reports 39 baseline breaks, above the 24 that could possibly comply).
   Repro: test_bug4. Suggested fix: route the preview through
   kairos.optimize.day_core._optimize_one_day with the same settings seams the
   plan uses.
5. BUG-5 (LOW, constant placeholder rendered as insight). Every plan row
   persists position="middle" hardcoded (kairos/export/incremental.py:432),
   and break_type derives from the constant 120s length, so all five
   recommendations render the identical title "Review middle medium break"
   (server.py:783-806 keys title and candidate grouping on these constant
   fields). Observed in the live payload. Suggested fix: compute the real
   position or drop it from recommendation copy and grouping.
6. BUG-6 (LOW, latent legacy paths). The no-breaks fallback branch of
   compliance compares AVERAGE retention to the floor (server.py:1321) though
   the rule is per-segment; and _summarize_schedule / _infer_hourly_break_counts
   fill a missing num_breaks with 1 (server.py:323, :1082), fabricating a
   break on malformed rows. Both only bite on degraded CSVs; worth cleaning
   while touching BUG-1.

Observations, not bugs: the live freshness banner currently reads stale
(settings, constraints) and is correct; data/kairos_constraints.csv sits
untracked and header-only, so a recompute will not move revenue; the activity
feed is device-local (localStorage), so decisions do not appear on another
operator's machine, a product choice worth confirming; /api/auth/* is
session-gated by design in the new uncommitted auth layer, /api/health stays
public and /api/auth/me honestly reports auth_disabled with no seeded user
store.

## 4. Standing tests added (all under tests/, no product code touched)

- test_guardrail_conformance.py: 12 tests, the rule-by-rule conformance gate
  on the committed CSV against the saved settings, mirroring engine
  semantics. All pass; a future failure is a real plan-conformance bug.
- test_journey_flows.py: 9 tests covering the freshness lifecycle, the
  decision loop (approve persists an anchored active override, reject stays
  dismissed and never constrains, override edits flip staleness, incremental
  recompute applies the override and preserves all other rows
  byte-identically), the settings echo and guardrail mapping, and
  objective_mode wiring into the engine. All writes are redirected to
  temporary copies.
- test_journey_inspector_pricing.py: 11 tests covering inspector-vs-CSV
  agreement, the competitor boundary, the effect preview delta and
  no-write guarantee, export-equals-saved-plan, the pricing state contract,
  the tester's multiply-out law, edit-moves-price and reset, invalid-edit
  rejection, and honest upload in_use semantics.
- test_api_surface_qa.py: 11 tests, the standing API gate: every
  parameterless GET route answers 200 with parseable JSON, no NaN or Infinity
  tokens, no mojibake, Hebrew round-trips intact, overview equals the plan,
  the compliance payload carries the full seven-rule set, recommendations
  bind only to the owned channel, and the empty-schedule summary stays null
  (auth routes scoped out as session-gated by design).
- test_qa_known_bugs.py: 4 deliberately failing tests pinning BUG-1 through
  BUG-4 with full evidence in each docstring.

Run: /Users/home/.venvs/meridian/bin/python -m pytest
tests/test_guardrail_conformance.py tests/test_journey_flows.py
tests/test_journey_inspector_pricing.py tests/test_api_surface_qa.py
(43 pass, about 13 seconds). tests/test_qa_known_bugs.py fails 4 of 4 by
design until the fixes land.
