# Training versus runs

Discovery artifact for the experience gauntlet. Read-only investigation, no product code
touched. Every claim carries its evidence: a file and line, an endpoint plus the response
actually received, a measured count, or a screenshot path. Claims that are not directly
evidenced are labelled INFERRED with the thing they were inferred from.

Investigated on 2026-07-31 against the repository at `/Users/home/Code/questo/meridian`
(HEAD `5a80a709`) and the running instance at `http://127.0.0.1:8010` with authentication
disabled (`GET /api/auth/me` returned `{"auth_disabled": true}`).

Screenshots are untracked session artifacts under `.playwright-mcp/`, which is gitignored
at `.gitignore:84`:

- `.playwright-mcp/tvr-01-overview-staleness.png`
- `.playwright-mcp/tvr-02-data-model-parameters.png`
- `.playwright-mcp/tvr-03-calendar-model-panel.png`
- `.playwright-mcp/tvr-04-optimizer-two-freshness.png`
- `.playwright-mcp/tvr-05-drift-card.png`

## The headline

Two activities live in this system. The product has a complete, working, operational
surface for one of them and no surface at all for the other, yet it shows the other's
internals on four operator pages and uses the same two words for both.

- Of the 113 HTTP operations the live app publishes (`GET /openapi.json`), **zero** are
  TRAINING. 14 are RUN, 50 are CONFIGURATION, 49 are READ. Full assignment in section 1.2.
- No endpoint, button, job or assistant tool can start a model build. There is no
  `subprocess` call anywhere in `kairos_api/` except the Claude Code keychain read at
  `kairos_api/assistant_auth.py:99`, and no API module imports `kairos.model.train` or
  either compute script (verified by grep across `kairos_api/*.py`).
- Training is therefore a shell command run by whoever has the repo:
  `PYTHONUTF8=1 python scripts/compute_measured_coefficients.py` and
  `scripts/compute_audience_model.py` (both docstrings say "Run from the repo root").
- A training run leaves no trace in any of the product's three audit systems. It is not in
  the activity log (only mutating `/api` requests are recorded,
  `kairos_api/activity_log.py:3-9`), not in the version timeline (9 logical operator-state
  files, no model artifacts, `kairos_api/version_store.py:46-47`), and not in the run log
  (`RunRecord` is one optimization run, `kairos/observability/run_log.py:59-73`). The
  scripts write the artifact and print to stdout, nothing else
  (`scripts/compute_measured_coefficients.py:691-707`, `scripts/compute_audience_model.py:64-78`).
- Meanwhile the broadcaster's screens carry the model's insides. `GET /api/impact` (the
  coefficients artifact metadata, including the drift monitor) is fetched on **every**
  dashboard load for every user at `TVBreakDashboard.jsx:1295`, alongside `/api/parameters`
  at line 1296.
- And the same word does both jobs. `TVBreakDashboard.jsx:4184` tells the operator to
  "recompute the coefficients" (a training act, no button exists). `ScheduleStalenessBanner.jsx`
  gives them a button labelled "Recompute now" (a run). Both render on the Data page, about
  ten lines of screen apart. Screenshot `.playwright-mcp/tvr-02-data-model-parameters.png`.

The rebuild's job is not to hide training. It is to give training its own home, its own
verbs, its own audience and its own audit trail, and to leave the broadcaster with a plan
that never changes underneath them without a sentence naming who changed the model and when.

## The evidence base

| Surface | What it yielded | Where |
| --- | --- | --- |
| Endpoint census | 90 paths, 113 operations, 0 of them training | live `GET /openapi.json` |
| Route decorators | 113 decorators across 26 router modules | grep `@router.(get\|post\|put\|patch\|delete)` in `kairos_api/` |
| Training entry points | 2 artifact-producing scripts, 1 posterior trainer, 15 measurement or investigation scripts | `scripts/` docstrings |
| Coefficients artifact | 47 metadata keys, 36 cells, 2,532 measured breaks, 6 gate verdicts | `models/tv_break_coefficients.json` |
| Audience artifact | 8 factor-family gates, 3 source fingerprints, `activation_default: false` | `models/audience_model.json`; live `GET /api/model/audience` |
| Freshness systems | 2 independent ones, 12 schedule input groups, 3 coefficient source fingerprints | `kairos/export/schedule_freshness.py:83-95`; `kairos/model/freshness.py:38-42` |
| Live plan state | `{"status": "stale", "computed_at": "2026-07-28T08:38:38.170135+00:00", "changed": ["settings", "coefficients"]}` | live `GET /api/overview` |
| Live coefficient state | `{"status": "fresh", "computed_at": "2026-07-29T00:12:13.100043+00:00"}` | live `POST /api/optimal-plan` |
| Affiliation wall | 3 of 56 mutating operations gated unconditionally, 1 conditionally | `kairos_api/events_api.py:378,399,426`; `kairos_api/pricing_api.py:234,241` |
| Wall behaviour, measured | a channel-affiliated operator session passes the middleware on every path tested including `POST /api/jobs/recompute` and `PUT /api/settings` | in-process probe against an isolated `KAIROS_AUTH_DIR`, section 3.3 |
| Wall behaviour, historical | `chan1 POST /api/events 403`, `chan1 PUT /api/pricing 403`, `comp1 POST /api/events 201` | Settings page activity log, 2026-07-29 10:17 |
| Assistant surface | 31 read tools, 8 propose tools, 0 training tools | `kairos_api.assistant_tools.READ_TOOL_NAMES`, `PROPOSE_TOOL_NAMES` |
| Vocabulary | "recompute" 159 times in the UI and 124 in the backend; "rebuild" 9 and 85; "training" 17 and 166 | grep counts, section 5.1 |

## 1. Every action a person can trigger, classified

### 1.1 The four classes as used here

- **TRAINING** decides what the model may believe: it fits coefficients, runs a held-out
  gate, decides whether a factor is real, and writes a model artifact. Its output is a
  model version. Rare, research, company staff.
- **RUN** computes or publishes a plan, a forecast, a scenario or an export from the
  current model plus the current configuration. Its output is a plan version or a
  transient preview. Constant, operational, broadcaster.
- **CONFIGURATION** changes a stored input that a RUN reads: settings, guardrails, the
  rate card, constraints, overrides, advertisers, agencies, calendar events, uploaded
  source data, accounts, saved versions. It changes nothing on screen until a RUN happens.
- **READ** returns stored or derived state without producing a new plan or changing state.

### 1.2 HTTP API: all 113 operations

Counted from the live `GET /openapi.json`: 90 paths, 113 operations. Every operation
appears in exactly one list below (14 + 50 + 49 = 113, verified programmatically).

**TRAINING (0 of 113).** There is no training operation in the API. Proof: no
`subprocess`, `Popen`, `os.system` or `runpy` call in any `kairos_api` module except
`kairos_api/assistant_auth.py:27,99`; no import of `kairos.model.train`,
`scripts.compute_measured_coefficients` or `scripts.compute_audience_model` from any API
module. `kairos/model/train.py:1-18` is an env-gated skeleton whose only in-repo caller is
`scripts/train_impact_model.py:29`.

**RUN (14).** Each of these produces a plan, a preview or a published file.

| Operation | What it does | Where |
| --- | --- | --- |
| `POST /api/recompute-schedule` | rebuilds `output/weekly_break_schedule.csv` from saved settings, synchronous | `kairos_api/recompute_api.py:78-86` |
| `POST /api/jobs/recompute` | same body as a background job, optional `{channel, day}` scope | `kairos_api/recompute_api.py:114-122` |
| `GET /api/jobs/{job_id}` | job status for the above, running / done / failed | `kairos_api/recompute_api.py:158` |
| `POST /api/optimal-plan` | a real optimal break plan for one channel-day, transient | `kairos_api/scenario_api.py:283-288` |
| `POST /api/scenario` | a real optimization for the scenario levers, transient | `kairos_api/scenario_api.py:230-235` |
| `POST /api/scenario-compare` | two scenarios computed and diffed | `kairos_api/insights_api.py` router mount |
| `GET /api/optimizer-plan` | memoized full optimization of the saved decision | `kairos_api/scenario_api.py:182-196` |
| `POST /api/optimizer-plan` | the same optimization on request-supplied levers | `kairos_api/scenario_api.py:199-201` |
| `GET /api/optimizer/net-comparison` | computes the saved objective against a net-focused plan | `kairos_api/scenario_api.py:331-344` |
| `GET /api/constraints/effect` | runs `_optimize_one_day` twice, with and without constraints | `kairos_api/constraints.py:333-351` |
| `GET /api/overrides/effect` | the same preview for overrides | `kairos_api/overrides.py` `@router.get("/effect")` |
| `GET /api/yield-per-second` | rebuilds plan segments and re-prices retention cost three ways | `kairos_api/insights_api.py:100-141` |
| `GET /api/export/schedule.csv` | publishes the weekly plan as CSV | `kairos_api/exporters.py` |
| `GET /api/export/spots.csv` | publishes the daily spot ledger as CSV | `kairos_api/exporters.py` |

**CONFIGURATION (50).** Grouped by store.

- Settings and rate card (3): `PUT /api/settings`, `PUT /api/pricing`,
  `POST /api/pricing/price-slot` (writes nothing but is the rate-card editing tool;
  classed here because it exists only to support a configuration decision).
- Placement rules (6): `POST|PUT|DELETE /api/constraints[/{id}]`,
  `POST|PUT|DELETE /api/overrides[/{id}]`.
- Commercial parties (14): `POST|PUT|DELETE /api/advertisers[/{id}]` and its
  `/conditions` triple; `POST|PUT /api/agencies[/{id}]`,
  `POST /api/agencies/{id}/deactivate`, `POST|DELETE /api/agencies/{id}/advertisers[...]`,
  `POST|PUT|DELETE /api/agencies/{id}/conditions[/{rule_id}]`.
- Calendar (3): `POST /api/events`, `PUT /api/events/{id}`, `DELETE /api/events/{id}`.
- Source data (1): `POST /api/uploads/{kind}`, where kind is one of programmes, spots,
  dayparts, daily, advertiser_rules, rate_card, campaign_flights
  (`kairos_api/uploads.py:159-171`).
- Plan decisions (1): `POST /api/break-decisions`, which persists a real override rather
  than a log entry (`kairos_api/dashboard_api.py:1824-1827`).
- Versions (3): `POST /api/versions/snapshot`, `PATCH /api/versions/{id}`,
  `POST /api/versions/{id}/restore`.
- Accounts and session (7): `POST /api/auth/login|logout|change-password`,
  `POST|DELETE /api/auth/users[/{username}]`,
  `PUT /api/auth/users/{username}/affiliation`,
  `POST /api/auth/users/{username}/reset-password`.
- Assistant state (12): `POST /api/assistant/ask`, `POST /api/assistant/ask/stream`,
  the four conversation operations, `POST .../proposals/{batch}/apply` and `/reject`,
  `POST /api/assistant/restore/{restore_id}`, `DELETE /api/assistant/thread`,
  `POST /api/assistant/upload`, `DELETE /api/assistant/uploads/{id}`. The ask endpoints
  are classed CONFIGURATION rather than READ because they write conversation memory and
  can capture pending proposal items.

**READ (49).** All remaining GETs. Four of them read training artifacts and are called
out here because they are the only training visibility the product has:

| Operation | Training content it returns | Where |
| --- | --- | --- |
| `GET /api/impact` | the whole coefficients metadata block (47 keys) plus `drift` | `kairos_api/catalog_api.py:550-562`; verified live |
| `GET /api/model/audience` | `available`, `computed_at`, `activation`, 8 gate verdicts, base summary | `kairos_api/audience_api.py:59-89`; verified live |
| `GET /api/parameters` | `coefficient_freshness`, `first_break_active`, `first_break_multiplier` | verified live, top-level keys |
| `GET /api/events` | `model_context` with `training_window`, `measurement`, `wartime_disclosure`, `training_gate` | verified live |

### 1.3 UI controls

The navigation has 17 entries, read live from the DOM: Overview, Optimizer, Schedule,
Inventory, Break Library, Campaigns, Forecasts, Events calendar, Reports, Data,
Advertisers, Agencies, Pricing, Overrides, Kai AI assistant, Restore changes, Settings.

**TRAINING controls: none.** No button, toggle, menu item or keyboard path in 20,172 lines
of `tv-break-dashboard/src` starts a model build. Verified by the absence of any UI call
to a training endpoint (there is none) and by the API path census of the frontend
(`grep -o "API_BASE}/api/..."`, 38 distinct paths, none training).

**RUN controls (5 distinct, 3 different labels for the same action):**

| Control | Endpoint | Where |
| --- | --- | --- |
| "Run Optimization" (Optimizer and Overview headers) | `POST /api/scenario` preview | `TVBreakDashboard.jsx:2470` |
| "Apply to weekly schedule" | `PUT /api/settings` then the full recompute | `TVBreakDashboard.jsx:2482`, handler at `:1990-2013` |
| "Recompute now" (staleness banner, every page) | `POST /api/jobs/recompute` with polling | `ScheduleStalenessBanner.jsx`, handler at `TVBreakDashboard.jsx:1923-1988` |
| "Recompute weekly schedule" (Settings, twice: engine focus and constraint builder) | same handler | `TVBreakDashboard.jsx:5470`, `:5616` |
| "Export" (schedule and spots) | `GET /api/export/*.csv` | Break Library and Reports toolbars |

**CONFIGURATION controls:** the Settings page (channel, optimizer balance, engine focus,
profile, guardrails, protected content, commercial policy, campaign pacing, constraint
builder), the Pricing page (base CPP, premium layers, layer activation switches including
the events layer), the Advertisers and Agencies pages, the Overrides console, the Events
calendar (create, edit, deactivate, import a holiday year), the Data page Upload tab, the
Restore changes page, and the admin accounts dialog.

**READ surfaces that are training surfaces:** section 2, findings F4 and F5.

### 1.4 Scripts and command-line entry points

| Script | Class | Evidence |
| --- | --- | --- |
| `scripts/compute_measured_coefficients.py` | TRAINING | "Compute the measured per-break retention coefficients and write the JSON"; 6 gate override flags (`--series`, `--counterprogramming`, `--placebo-correction`, `--interval-calibration`, `--moderated-variances`, `--output`), each with an env-var twin |
| `scripts/compute_audience_model.py` | TRAINING | "Rebuild the audience (expected TVR) model artifact ... re-measures every family gate on five temporal folds" |
| `scripts/train_impact_model.py` | TRAINING | "Prepare the real TV data and train the Meridian impact posterior"; MCMC flags `--n-chains --n-adapt --n-burnin --n-keep` |
| `meridian_tv_break/cli/train_tv_break.bat` | TRAINING | runs `data_transform` then `train_model` |
| `scripts/classify_unclassified.py` | TRAINING | resolves unclassified programme titles with an LLM; the output cache is a schedule input group, `schedule_freshness.py:245-252` |
| `scripts/validate_classifier.py`, `scripts/audit_scale_readiness.py`, `scripts/analyze_afterwindow_bias.py`, `scripts/measure_counterprogramming.py`, `scripts/measure_spotlevel_clip.py`, `scripts/first_break_before_after.py`, `scripts/investigate_first_break*.py` (3), `scripts/estimate_candidate_revenue_movement.py`, `scripts/probe_frontier_gap.py` | TRAINING, research only | one-off measurement scripts, docstrings say they only measure and print |
| `scripts/validation/` (13 modules: coverage_holdout, parameter_recovery, determinism_seeds, risk_lambda_efficacy, run_placebo, run_leave_one_out, run_selection_bias, decision_sensitivity, model_value, run_clean_baseline, run_inference, plus 2 libs) | TRAINING, validation | model-validation harness |
| `scripts/export_schedule.py` | RUN | "Generate the real optimized weekly break schedule the dashboard reads" |
| `run_optimization.py` (repo root) | RUN | top-level optimization driver |
| `scripts/init_auth.py` | CONFIGURATION | seeds the first admin account |
| `meridian_tv_break/cli/query_tv_break.bat` | RUN | runs `query_optimizer` |
| `Start TV Dashboard.bat` (repo root) | RUN, launcher | starts uvicorn on 8000 and the dashboard on 3000 |

The delivered tree therefore puts `train_tv_break.bat` and `Start TV Dashboard.bat` in the
same checkout. A broadcaster with the folder can double-click either.

### 1.5 Assistant tools and proposals

31 read tools and 8 propose tools, enumerated in-process from
`kairos_api.assistant_tools.READ_TOOL_NAMES` and `PROPOSE_TOOL_NAMES`.

- **TRAINING tools: 0.** No propose tool touches a model artifact. The system prompt makes
  this explicit: "Training is MEASURED, never asserted ... no one may fake a retention
  coefficient meanwhile" (`kairos_api/assistant.py:172`).
- **RUN proposals: 1.** `propose_recompute` with scope `'full'` or `{"days": [...]}`
  (`kairos_api/assistant_tools.py:265-272`).
- **CONFIGURATION proposals: 7.** `propose_settings_change` (20 allowed fields,
  `assistant_tools.py:32-54`), `propose_constraint`, `propose_override`,
  `propose_pricing_change`, `propose_advertiser_change`, `propose_event_change`,
  `propose_agency_change`.
- **READ tools that return training content: 3.** `get_audience_model` (gate verdicts),
  `get_audience_stability` (the level-drift monitor, `assistant_read_tools.py:197`),
  `get_event_pipeline` (the tri-state `event_layer_gate`). A fourth, `get_schedule_freshness`,
  reports the plan-side verdict whose changed-group list can name `coefficients`.

## 2. Where the two are confusable today

Fourteen findings. Each is a thing the rebuild has to close.

### F1. One word, both activities, ten lines apart on the same page

`TVBreakDashboard.jsx:4184` (Data page, drift card): "The plan's coefficients assume a
steady audience level. A drift above the threshold means the weekly level moves more than
the measurement's own precision, so **recompute the coefficients** when new data lands."
Hebrew: "ולכן מומלץ לחשב את המקדמים מחדש כשנקלטים נתונים חדשים."

`ScheduleStalenessBanner.jsx` (same page, above it): a button labelled **"Recompute now"**,
Hebrew "הריצו חישוב מחדש", which calls `POST /api/jobs/recompute` and rebuilds the weekly
plan. The two acts share the verb in both languages, and only one of them has a button.
Screenshots `.playwright-mcp/tvr-02-data-model-parameters.png` and `.playwright-mcp/tvr-05-drift-card.png`.

### F2. "Rebuild" also means both

Training: `CalendarAudienceModel.jsx:153`, "Each factor family ships only after passing its
measured held-out gate on a **rebuild**", Hebrew "בבנייה מחדש".
Run: `TVBreakDashboard.jsx:2472`, "Saves these levers and **rebuilds** the whole weekly
schedule", Hebrew "ובונה מחדש את כל הלוח השבועי"; and the toast at `:2010-2011`, "Saved
these levers and rebuilding the whole weekly schedule", "ההגדרות נשמרו והלוח השבועי כולו
נבנה מחדש".

### F3. The staleness banner fuses a training event and a configuration event into one sentence with one button

Measured live on 2026-07-31:
`GET /api/overview` returned
`"schedule_freshness": {"status": "stale", "computed_at": "2026-07-28T08:38:38.170135+00:00", "changed": ["settings", "coefficients"]}`.

The rendered banner reads: "settings and the model's learned values (coefficients) changed
since this schedule was computed on 7/28/2026, 11:38:38 AM. Recompute to refresh the
schedule, reports, and break plan." (captured from the live DOM).

Someone retrained the model and someone changed a setting. The operator sees one amber
strip, one verb, one button, and no way to tell that one of those two events came from
outside their organisation. The twelve input groups at `schedule_freshness.py:83-95` mix
four categories with no separation: operator configuration (settings, constraints,
overrides, advertiser rules, campaign flights, special events), source data (data,
inventory), and model artifacts (coefficients, the impact model, program classifications,
the audience model).

### F4. The Data page carries a model-health dashboard

The Data page's third tab is literally titled "Model and parameters"
(`TVBreakDashboard.jsx:3947`) and its page body says "see what the model learned"
(`:3956-3957`). Captured live it renders: Model explainability with 12 coefficient rows and
their credible intervals, an empirical-Bayes pooling note ("tau^2 = 1.73e-04 sits far below
within-cell variance 0.065, so the per-cell effects collapse toward a single pooled value"),
and an Audience level stability card with a five-week drift table and a
**"Needs attention"** chip.

The chip is a training verdict (`level_drift.binding: true` in the artifact) rendered to a
person who has no training action available. Screenshot `.playwright-mcp/tvr-05-drift-card.png`.

### F5. The Events calendar carries a second, larger model-health dashboard

Captured live from the Events calendar page, panel "What the model relies on today":
retention measurement mode ("Global baseline, calendar-blind"), monthly seasonality verdict
("Measured held-out improvement 0.0%, stays off"), week-to-week level drift with a
five-row table and a "Binding" chip, training window ("2024-11-01 .. 2024-11-30 (30 days,
2,532 measured breaks)"), coefficients computed at, a wartime disclosure paragraph, and an
"Audience model (expected rating)" table of 8 factor families with Active or Off verdicts.
Below it, 63 events each annotated "The training data did not see this condition".
Screenshot `.playwright-mcp/tvr-03-calendar-model-panel.png`.

This is the startup's model-health view, in full, on the page a broadcaster opens to add a
holiday.

### F6. Two freshness systems, opposite verdicts, same screen

On the Optimizer page, captured live, the summary block is followed by a green region
labelled **"Model measurements current, Measured Jul 29, 2026"**, and directly above it the
amber **"Saved schedule is out of date ... coefficients changed"**. Both are true (the
coefficients are fresh against their source files; the plan predates them) and together they
read as a contradiction. Screenshot `.playwright-mcp/tvr-04-optimizer-two-freshness.png`.

The two systems are genuinely different: `kairos/model/freshness.py` compares three source
fingerprints against the coefficients artifact; `kairos/export/schedule_freshness.py`
compares twelve input groups against the saved plan. Nothing on screen says they are
different questions.

### F7. Four or five clocks, none of them labelled by activity

Measured live, all visible in one session:

| Stamp | Value on 2026-07-31 | Meaning |
| --- | --- | --- |
| Header "Updated 11:38 AM" | max mtime of the plan CSV and 4 data files (`dashboard_api.py:1462-1472`) | neither a run nor a training time |
| Banner "computed on 7/28/2026, 11:38:38 AM" | plan sidecar `computed_at` | last RUN |
| "Measured Jul 29, 2026" chip | `coefficient_freshness.computed_at = 2026-07-29T00:12:13` | last coefficient TRAINING |
| Calendar "Coefficients computed at 29/07/2026, 03:12:13" | same value, local time | last coefficient TRAINING |
| Calendar "Audience model computed at 29/07/2026, 09:41:29" | `models/audience_model.json` `computed_at` | last audience TRAINING |

Three of these are training times, one is a run time, one is a file mtime, and the word
"Updated" is attached to the one that means least.

### F8. A number's basis depends on a switch with no control and no display

`audience_model_activation` (`kairos_api/core.py:135-143`) decides whether forward-dated
segments take `baseline_tvr` from the trained model or from the historical mean. It is a
`KairosSettings` field, so `PUT /api/settings` can set it, but:

- it is absent from `ALLOWED_SETTINGS_FIELDS`, so the assistant cannot propose it
  (`assistant_tools.py:32-54`);
- it appears in **no** `.jsx` or `.js` file (grep across `tv-break-dashboard/src`), so there
  is no control anywhere in the dashboard;
- the honest basis note the backend computes for it,
  `summary.audience_model = {"state": "off", "computed_at": "2026-07-29T06:41:29.045699+00:00"}`
  (measured live on `GET /api/overview`), and the same note on `forecasts.by_day_basis`, is
  **never rendered**: the only frontend reference to the string `audience_model` is the
  staleness label map at `ScheduleStalenessBanner.jsx:47`.

So the flag that decides where every forward rating comes from can only be flipped by
someone with `curl`, and the disclosure the backend already produces never reaches a screen.

### F9. A training gate silently changes the run, self-activating, with no announcement

`kairos/service.py:102-125` reads `first_break_multiplier` out of the coefficients metadata
on every optimization and takes `max(assumption, measured)`. The current artifact carries
`first_break_multiplier: 1.0, first_break_active: false, first_break_reason: "p=0.2034 not < 0.01; multiplier left at 1.0 (off)"`.
A future rebuild that clears p < 0.01 will raise the retention cost of every first break in
every plan, with no operator action and no notification. The value is exposed on
`GET /api/parameters` as `first_break_multiplier` and `first_break_active`, and it appears
in the Data page's parameter ledger as "Retention assumption -3pp", which is a different
number entirely (see F10).

### F10. The parameter ledger shows the assumption, not the model

`GET /api/parameters` returns `assumptions.retention_impact_per_break: -0.03` and
`assumptions.revenue_weight: 0.5`, while `settings.revenue_weight` is 60 and the measured
coefficients on the same page range from -0.046 to -0.053. The Data page renders "Retention
assumption -3pp, Default used when a segment has no measurements" beside 12 measured cells
around -5pp. `kairos/optimize/pricing.py:151` labels the field "per-break drop, until
Meridian is trained". A reader has to know which number the optimizer actually used
(`impact_source: "measured"`, verified live on `POST /api/optimal-plan`).

### F11. The source-file list shows the unused training artifact and hides the used one

`GET /api/files` returned 8 rows, the last being `models/tv_break_posterior.pkl`.
`models/tv_break_coefficients.json` is not in the list. But
`kairos/model/impact.py:289-304` resolves in this order: measured coefficients JSON first,
Meridian posterior second, declared assumption third. The JSON exists, so the posterior is
never read; the live plan confirms `impact_source: "measured"`. The Data page's Source files
tab therefore lists a 1.2 MB training artifact dated 2026-07-01 that changes nothing and
omits the 21 KB artifact that drives every retention number.

The posterior is nonetheless a schedule freshness input group
(`schedule_freshness.py:212-219`), so replacing it would mark the plan stale without
changing a single figure.

### F12. Training changes are outside every undo the product offers

`version_store.py:46-47` lists the nine logical files a version captures: settings,
constraints, overrides, advertisers, conditions, events, agencies, agency_links,
agency_conditions. No model artifact is in that list. The Restore changes page can put back
a rate card from three weeks ago and cannot put back yesterday's coefficients. The only
history of previous model versions is `models/candidates/`, five hand-kept JSON files dated
2026-07-05 and 2026-07-06, which nothing in the product reads.

### F13. Uploading data is an operator action whose real consequence is a training input

`POST /api/uploads/{kind}` accepts programmes, spots and dayparts. Those three files are
exactly the coefficients artifact's `source_fingerprints`
(`data/reference/Spots.xlsx`, `Programmes.xlsx`, `Dayparts.xlsx`, read live from the
artifact). Uploading a new one flips coefficient freshness from `fresh` to `stale`, and the
only remedy is a training run the uploader cannot start. The upload surface itself warns
about a different thing entirely: it says the uploaded CSV is shadowed while the reference
xlsx exists (`kairos_api/uploads.py:73-80`).

### F14. The assistant has no grounding for the retention model

`kairos_api/assistant_keywords.py:40-92` defines 12 keyword-triggered grounding sections.
There is a section for `audience_model` (triggered by "מודל קהל", "audience model",
"expected rating"). There is **no** trigger for the retention coefficient model: not
"מקדמים", not "coefficients", not "אימון", not "training", not "drift". An operator who
asks in Hebrew why the plan moved gets no coefficient context attached, though the
`get_audience_stability` read tool exists and would answer it.

### F15. The events pricing layer is walled server-side and open client-side

`PUT /api/pricing` refuses a channel-affiliated session that touches
`pricing_activation.events` (`kairos_api/pricing_api.py:234,241`) with the Hebrew detail
"הפעלת תמחור אירועים שמורה לצוות החברה". But `GET /api/pricing` carries no `can_edit`
field (verified live, top-level keys are currency, units, base, layers, activation, events,
has_overrides, note), so `PricingEventsLayer.jsx` renders the toggle enabled for everyone
and the refusal only arrives as a 403 after the click. The events surface does it the other
way: `GET /api/events` returns `can_edit` (verified live, `true` on this instance) and
`CalendarEvents.jsx:99,247` hides the controls and shows "Event editing is available to the
company team only." Two surfaces, one rule, two behaviours.

## 3. The permission reality

### 3.1 What exists

Two orthogonal dimensions on an account:

- `ROLES = ("admin", "operator", "viewer")`, `kairos_api/auth_store.py:36`.
- `AFFILIATIONS = ("company", "channel")`, `kairos_api/auth_store.py:39`. A record without
  the field reads company, `auth_store.py:263-282`, so every legacy account keeps full
  access.

The deployed store has one account: `admin`, role `admin`, no `affiliation` key, created
2026-07-05T18:32:23+00:00 (`data/auth/users.json`). It therefore reads as company. The
running instance has enforcement bypassed entirely: `GET /api/auth/me` returned
`{"auth_disabled": true}`, which per `kairos_api/auth.py:79-80` means `KAIROS_AUTH_DISABLED`
is set on the 8010 process.

### 3.2 What the role wall does, exactly

`kairos_api/auth.py:88-111`, the single enforcement rule:

- non-`/api/` paths are always open (the SPA shell);
- with the store unseeded or the bypass set, everything is open;
- `/api/auth/login` and `/api/health` are public;
- no session yields 401;
- `/api/auth/users*` requires `admin`;
- any POST, PUT, PATCH or DELETE outside `/api/auth/` requires `admin` or `operator`, so a
  viewer is read-only.

That is the whole role wall. It contains no notion of training, of the model, or of
affiliation.

### 3.3 What the affiliation wall does, exactly

`require_company_editor` is called from exactly five places in the codebase (grep across
`kairos_api/`):

| Call site | Operation | Condition |
| --- | --- | --- |
| `events_api.py:378` | `POST /api/events` | always |
| `events_api.py:399` | `PUT /api/events/{id}` | always |
| `events_api.py:426` | `DELETE /api/events/{id}` | always |
| `pricing_api.py:234` | `PUT /api/pricing` | only when the payload carries `pricing_activation.events` |
| `pricing_api.py:241` | `PUT /api/pricing` | only when `reset` would clear a live events activation |

So **3 of 56 mutating operations are gated unconditionally, and 1 more conditionally**.
Two further gates exist in the assistant lane: `events_access.py:71-74` blocks the propose
side for `propose_event_change`, `propose_agency_change`, `propose_agency_link_change`,
`propose_agency_condition_change` and any pricing proposal touching
`pricing_activation.events` (wired at `assistant_tools.py:414-420`), and
`events_access.assistant_apply_block` mirrors it on apply (wired at
`assistant_actions.py:451-455`).

The wall is real and it works. Historical proof from the Settings page activity log,
2026-07-29 10:17, read live: `chan1 POST /api/events 403`, `chan1 PUT /api/events/82b410c191fb 403`,
`chan1 DELETE /api/events/82b410c191fb 403`, `chan1 PUT /api/pricing 403` twice, against
`comp1 POST /api/events 201` and `comp1 PUT /api/pricing 200`.

### 3.4 Measured: what a channel-affiliated operator can reach today

Probe run in-process against an isolated `KAIROS_AUTH_DIR` under the scratchpad (no product
file touched): created `chanop` (operator, channel) and `coop` (operator, company), created
sessions, then called `auth.enforce_request` with a stub request for 25 paths.
`auth.auth_active()` returned `True`; `is_company_user('chanop')` returned `False`;
`is_company_user('coop')` returned `True`.

Result: the middleware returned **PASS** for every path except `POST /api/auth/users`
(403, admin only). Every one of these is reachable by a channel-affiliated operator:

`POST /api/jobs/recompute`, `POST /api/recompute-schedule`, `PUT /api/settings`,
`PUT /api/pricing` (any key except the events activation), `POST /api/uploads/spots`,
`POST /api/optimal-plan`, `POST /api/scenario`, `POST /api/constraints`,
`POST /api/overrides`, `POST /api/break-decisions`, `POST /api/advertisers`,
`POST /api/agencies`, `POST /api/assistant/ask`,
`POST /api/assistant/proposals/{batch}/apply`, `POST /api/versions/{id}/restore`,
and every GET including `GET /api/impact`, `GET /api/model/audience`, `GET /api/parameters`,
`GET /api/files`, `GET /api/reports`, `GET /api/overview`.

The same probe confirmed that `events_access.requester_is_company` correctly returns
`False` for that session on every path. The guard knows. Only three routes ask it.

### 3.5 Every training surface currently reachable by a channel-affiliated account

All of them. Specifically:

| Surface | Reachable | Evidence |
| --- | --- | --- |
| `GET /api/impact`, the full 47-key coefficients metadata including every gate reason and the drift monitor | yes | no guard on the route; fetched on every dashboard load at `TVBreakDashboard.jsx:1295` |
| `GET /api/model/audience`, 8 gate verdicts with held-out deltas and reasons | yes | `kairos_api/insights_api.py:638-640`, no guard |
| `GET /api/parameters`, `coefficient_freshness`, `first_break_active`, `first_break_multiplier` | yes | verified live, no guard |
| `GET /api/events` `model_context`, training window, wartime disclosure, `training_gate` | yes | `events_api.py:373`, the read side is deliberately open |
| Data page, "Model and parameters" tab | yes | no gating in `TVBreakDashboard.jsx:3944-4064` |
| Events calendar, "What the model relies on today" panel | yes | `CalendarEventsModel.jsx`, no `canEdit` on the model panel |
| Assistant read tools `get_audience_model`, `get_audience_stability`, `get_event_pipeline` | yes | "Read tools stay open to every authenticated account", `assistant_event_pipeline.py:18` |
| `PUT /api/settings` carrying `audience_model_activation` | yes | not in `ALLOWED_SETTINGS_FIELDS` for the assistant, but the raw endpoint takes the whole `KairosSettings` model and has no affiliation guard |

The last row is the sharp one. A channel-affiliated operator cannot create a holiday, and
can turn the trained audience model on or off for every forward-dated rating in the system
with one `PUT`.

Two further gaps worth naming:

- The dashboard never reads its own session affiliation. `GET /api/auth/me` returns
  `affiliation` (`auth.py:183`), and the only frontend uses of the word are the admin
  accounts dialog (`TVBreakDashboard.jsx:5887-6189`) and the `affiliationLabel` helper
  (`Login.jsx:93-102`). No screen adapts to who is looking; only `/api/events`'s `can_edit`
  changes anything.
- The affiliation labels are bare: `{company: {en: 'Company', he: 'חברה'}, channel: {en: 'Channel', he: 'ערוץ'}}`
  (`Login.jsx:94-97`). Nothing anywhere explains what the distinction buys or costs.

## 4. What each side needs to see

### 4.1 The broadcaster: what a run dashboard must answer

Six questions, in this order. Everything needed to answer them already exists in the
payloads; most of it is not on screen, or is on screen next to something that contradicts it.

1. **Is today's plan current, and if not, what made it old?** Available:
   `overview.schedule_freshness` with `status`, `computed_at` and `changed` group labels.
   Missing: the split. "coefficients", "the impact model", "program classifications" and
   "the audience model" are model changes that came from outside the broadcaster; the other
   eight groups are their own edits or their own data. One banner cannot carry both. The
   rebuild needs two states: "your changes are not in the plan yet" (self-service, one
   button) and "the model changed under you" (informational, names who and when, and offers
   the same button with a different sentence).
2. **What will change if I run it?** Available: `GET /api/constraints/effect` and
   `GET /api/overrides/effect` compute exactly this for one channel-day
   (`constraints.py:339-350`). `GET /api/optimizer/net-comparison` computes the objective
   delta. Missing: nothing joins them to the recompute button.
3. **What did the last run produce, and how does it compare to the one before?**
   Available: `output/run_log.jsonl`, 488 records, each carrying `run_id`, `created_at`,
   `engine_version`, `input_checksums`, `guardrails`, `assumptions` and a `summary` of
   `total_breaks`, `projected_revenue`, `average_retention`, `compliant`
   (`kairos/observability/run_log.py:59-73`). Missing: no endpoint serves it and no screen
   shows it. `get_run_log_summary` exists as an assistant read tool and has no UI twin.
4. **What are these numbers actually the projection of?** Available and good:
   `summary.basis` carries `scope_channel`, `n_dates`, `date_from`, `date_to`,
   `n_channels_total`; `yield-per-second.basis` carries the literal formula; the Optimizer
   page already prints "the projection for the planning week 1 Nov 2024 - 7 Nov 2024
   (7 days), taken from the saved plan for your channel (רשת 13)". Missing: the audience
   model basis note, which is computed and never rendered (F8).
5. **Am I compliant, and where am I at risk?** Available: `GET /api/compliance` with
   `checks`, `violations`, `status`; `overview.recommendations`; `GET /api/make-good-alerts`.
   This is the best-served question in the product.
6. **What is booked, what is unsold, and what is a make-good exposure?** Available:
   `GET /api/inventory` (summary, by_daypart, by_hour), `GET /api/campaigns`,
   `GET /api/gold-breaks`, `GET /api/make-good-alerts` with an honest
   `data_available` plus `reason`.

What the run dashboard must **not** contain: gate verdicts, held-out deltas, tau^2, pooling
notes, drift tables, training windows, or any per-cell coefficient. It needs exactly one
model fact: which model version this plan was computed with, as a date and a name, plus a
link to whoever owns it.

### 4.2 The startup: what a model-health dashboard must answer

Ground truth for each of the six questions below, read from the artifacts on disk today.

**1. Per-gate verdicts and reasons.** The artifacts already carry every one, with the
reason written in full sentences. That is the whole dashboard, and none of it has a home.

Retention model (`models/tv_break_coefficients.json`, `metadata`):

| Gate | Verdict | Reason as stored |
| --- | --- | --- |
| series | off | "series RMSE (fold mean 0.26239) does not beat genre RMSE (fold mean 0.24200) by the required 2% margin (mean improvement -8.5% over 5 temporal folds, fold sd 3.4pp)" |
| first break | off | "p=0.2034 not < 0.01; multiplier left at 1.0 (off)"; n_first 476, n_later 816 |
| counter-programming | off | "does not beat the no-covariate RMSE ... by the required 2% (mean improvement -0.1% over 5 temporal folds, fold sd 0.4pp)" |
| detrend seasonality | off | "month_minute RMSE (1.70793) does not beat the global RMSE (1.70793) ... (actual improvement 0.0%)" |
| event layer | off | "no event on/off contrast in the measured window: all 2532 measured breaks lie inside an active calendar event ... the gate re-measures automatically once history with both conditions exists" |
| placebo correction | **on** | "active by default: content-only baseline applied and each genre's measured no-break drift (pooled +0.01422 log over 6141 matched pseudo-breaks) subtracted" |
| interval calibration | **on** | "seeded parametric-bootstrap mixture quantiles (B=2000, measured width factor 1.15x at 95%)" |
| moderated variances | off | `moderated_variances: false`, prior_df 5.85 measured either way |

Audience model (`models/audience_model.json`, `gates`), verified live on
`GET /api/model/audience`: weekday_slot **on** (+25.55%), series **on** (+16.08%),
competitor_lineup **on** (+2.16%), calendar_religious_blackout off (+0.06%, below the 2%
bar), and four off with `held_out_delta_pct: null` because there was nothing to contrast:
calendar_hanukkah ("no Hanukkah days in the measured window"),
calendar_school_and_chol_hamoed, season ("fewer than two season cells with at least 10
observations (1 of 1 cells qualify)"), operator_events ("every one of the 3459 observations
falls on operator-event days").

The dashboard must distinguish three off-states that today all render as a grey "Off" chip
(`CalendarAudienceModel.jsx:30-33`): **tested and lost** (a real held-out delta below the
bar), **untestable** (no contrast in the window, `held_out_delta_pct: null`), and
**not yet measured** (`verdict: unknown`, the tri-state at `events_access.py:88-98`). Those
are three completely different pieces of news for a researcher.

**2. Data coverage and contrast.** Available: `total_breaks_measured: 2532`,
`channels: 36` (which is actually the cell count, `len(coefficients)`, mislabelled at
`scripts/compute_measured_coefficients.py:604`), `negative_cells: 36`,
`before_after_window_minutes: 3`, `source_data: "data/reference"`, and the training window
`2024-11-01 .. 2024-11-30 (30 days)` served on `GET /api/events`. The wartime disclosure is
the sharpest coverage fact the product owns: "The whole 30 day training window was measured
under wartime conditions; the ceasefire took effect only on 2024-11-27, leaving a
post-ceasefire tail of 132 of 2532 measured breaks."

Missing: a per-factor coverage table. Today each family's absence of contrast is buried in
prose inside a gate reason. A researcher needs cells, n per cell, and the contrast ratio,
per factor, as data.

**3. Drift.** Available and complete: `level_drift` with `status: "measured"`,
`n_breaks: 2532`, `n_weeks: 5`, `window_days: 30`, five `weekly_levels` rows each with n,
mean_log_effect and se, `drift_per_week: 0.0202`, `drift_se: 0.0117`,
`slope_per_week: 0.00654`, `pooled_half_width_95: 0.0094`, `binding_threshold: 0.0188`,
`binding: true`, and the criterion in words. This is a good measurement rendered to the
wrong audience (F4). Missing: drift over more than one artifact. There is one snapshot and
no series, because each rebuild overwrites the file.

**4. What a rebuild would change.** Partially available and entirely offline.
`models/candidates/` holds five alternative artifacts (afterwindow, calibrated, competitor,
placebo_corrected, spotclip) and `scripts/estimate_candidate_revenue_movement.py` computes
"Revenue movement if a candidate coefficients artifact were adopted". Nothing surfaces it.
The model-health dashboard needs this as a first-class view: for each candidate, the gate
deltas, the coefficient deltas, and the money the adopted plan would move.

**5. What is blocked on data that does not exist yet.** Available as reasons, needs to be a
list. Today: the event layer (needs days outside an active event), Hanukkah (needs Hanukkah
days), school holidays and Chol HaMoed (needs those days), season (needs a second season
cell with 10+ observations), operator events (needs ordinary days), and the series layer for
retention (needs enough per-title history to beat genre by 2%). Every one of these unblocks
by calendar time plus ingestion, not by code. The dashboard should say, per blocked factor,
what condition would unblock it and roughly when.

**6. Provenance and reproducibility.** Available: `source_fingerprints` on both artifacts
(3 files each; the retention set is Spots, Programmes and Dayparts; the audience set is
Spots, `data/calendar_events.csv` and `kairos/config/israel_calendar.csv`),
`computed_at` on both, `interval_seed: 20260706`, `bootstrap_B: 2000`, `placebo seed: 42`,
`method: measured_detrended_pooled`, `pooling_method: empirical_bayes`,
`between_cell_variance_tau2`, `learned_pseudo_count: 376.43`. Missing: who ran it, on what
machine, with which flags. The six gate-override flags (`--series`, `--counterprogramming`,
`--placebo-correction`, `--interval-calibration`, `--moderated-variances`) and their env-var
twins are not recorded in the artifact, so a forced gate is indistinguishable from a
self-activated one after the fact, except by reading the reason string.

The model-health dashboard also needs one thing the run dashboard needs from it: a stable
**model version identity**. Today the only identifier is a `computed_at` timestamp on a file
that is overwritten in place.

## 5. The vocabulary problem

### 5.1 Every word the product currently uses

Counts are `grep -roi` over `tv-break-dashboard/src` (`*.jsx`, `*.js`) for the UI column and
over `kairos_api`, `kairos`, `scripts` (`*.py`) for the backend column.

| Word | UI | Backend | Means TRAINING | Means RUN | Means both |
| --- | --- | --- | --- | --- | --- |
| recompute | 159 | 124 | `TVBreakDashboard.jsx:4184` | `ScheduleStalenessBanner.jsx`, `recompute_api.py` | **yes** |
| rebuild | 9 | 85 | `CalendarAudienceModel.jsx:153`, `audience_api.py:78` | `TVBreakDashboard.jsx:2472`, `recompute_api.py:36` | **yes** |
| compute | 207 | 355 | `computed_at` on both artifacts | `computed_at` on the plan sidecar | **yes** |
| training | 17 | 166 | yes | no | no |
| train | 0 | 96 | yes | no | no |
| retrain | 0 | 3 | yes | no | no |
| measure / measured | 105 / 57 | 673 / 424 | yes | no | no |
| fit | 7 | 53 | yes | no | no |
| gate | 28 | 219 | yes | no | no |
| optimize / optimization | 120 | 840 | no | yes | no |
| run | 72 | 226 | "training run" in docs | "Run Optimization", run log | **yes** |
| scenario | 78 | 163 | no | yes | no |
| simulate | 2 | 28 | no | yes | no |
| apply | 121 | 152 | no | yes (and proposals) | no |
| refresh | 231 | 3 | no | UI refetch, not a run | no |
| export | 290 | 150 | no | yes | no |
| publish | 1 | 12 | no | yes | no |

Hebrew, same method:

| Word | UI | Backend | Meaning today |
| --- | --- | --- | --- |
| חישוב מחדש | 25 | 2 | RUN (the banner, the buttons) |
| חשבו מחדש | 5 | 0 | RUN (imperative) |
| לחשב את המקדמים מחדש | 1 | 0 | **TRAINING**, `TVBreakDashboard.jsx:4184` |
| בנייה מחדש | 1 | 0 | **TRAINING**, `CalendarAudienceModel.jsx:153` |
| נבנה מחדש / בונה מחדש | 4 | 2 | RUN, `TVBreakDashboard.jsx:2011,2472` |
| אימון | 11 | 2 | TRAINING only |
| מדידה / נמדד | 9 / 23 | 4 / 0 | TRAINING only |
| מקדמים / מקדמי | 7 / 11 | 0 | TRAINING artifact |
| אופטימיזציה | 10 | 0 | RUN |
| הרצת | 1 | 0 | RUN ("הרצת אופטימיזציה") |
| רענון | 11 | 0 | UI refetch |
| שער / הכרעת | 4 / 2 | 1 / 0 | TRAINING gate |

Two words are load-bearing for both activities in both languages: **recompute / חישוב
מחדש** and **rebuild / בנייה מחדש**. A third, **compute / computed_at**, stamps both.

### 5.2 The proposal

One word per activity, in each language, used everywhere and never for anything else. Both
proposed words are already in the product with exactly the right meaning and no collision,
so this is a narrowing, not an invention.

| Class | English canonical | Hebrew canonical | Verb | Output artifact | Never say |
| --- | --- | --- | --- | --- | --- |
| TRAINING | **training** (a *training*; *train the model*) | **אימון** (*אימון*; *לאמן את המודל*) | train / לאמן | **model version** / **גרסת מודל** | rebuild, recompute, build, בנייה מחדש, חישוב מחדש |
| RUN | **run** (a *run*; *run the plan*) | **הרצה** (*הרצה*; *להריץ את התוכנית*) | run / להריץ | **plan version** / **גרסת תוכנית** | rebuild, recompute, בנייה מחדש, חישוב מחדש |
| CONFIGURATION | **setting** / **change** | **הגדרה** / **שינוי** | save / לשמור | the saved store | apply (reserve for proposals) |
| READ | no verb | no verb | view / לצפות | none | n/a |

Why these two:

- **אימון** appears 11 times in the UI and 2 in the backend and in every case means model
  training ("חלון האימון", "האימון הדו-שנתי", "נתוני האימון"). It has zero run usages. It
  is already the right word; it just needs to become the only word.
- **הרצה / להריץ** appears once in the UI, in "הרצת אופטימיזציה", the Run Optimization
  button, and never for training. English "run" already names the run log, the run record
  and the engine version stamped on it.
- **חישוב מחדש** and **rebuild** are retired from both activities. They are the collision.
  Retiring rather than reassigning avoids relabelling one act while the other keeps a word
  the operator has learned means something else.

Consequential renames this implies:

| Today | Becomes |
| --- | --- |
| Button "Recompute now" / "הריצו חישוב מחדש" | "Run the plan" / "הריצו את התוכנית" |
| Button "Recompute weekly schedule" / "חישוב מחדש של הלוח השבועי" | "Run the weekly plan" / "הרצת הלוח השבועי" |
| Button "Apply to weekly schedule" | "Save and run" / "שמרו והריצו" |
| Button "Run Optimization" | "Preview" / "תצוגה מקדימה" (it saves nothing, `TVBreakDashboard.jsx:1990-1993`) |
| Note "recompute the coefficients when new data lands" | "the model needs retraining when new data lands" / "המודל דורש אימון מחדש כשנקלטים נתונים חדשים", with a named owner, not a button |
| "Model measurements current, Measured Jul 29" | "Model version 2026-07-29, current" / "גרסת מודל 2026-07-29, עדכנית" |
| Staleness label `coefficients` -> "the model's learned values (coefficients)" | "a newer model version exists" / "קיימת גרסת מודל חדשה יותר" |
| Staleness label `the audience model` | "a newer audience model version exists" / "קיימת גרסה חדשה של מודל הקהל" |
| `POST /api/recompute-schedule`, `POST /api/jobs/recompute` | keep the paths (frozen contract), rename every label and message around them |

### 5.3 One naming trap to avoid

Do not name the training activity "rebuild" and the run activity "recompute", which is the
split the code half-implements today. They are near-synonyms in English and effectively
identical in Hebrew (בנייה מחדש and חישוב מחדש both read as "doing it again"), so the split
carries no signal to a reader who has not been told the convention. The pair has to differ
in kind, not in degree: **אימון** is a different verb about a different object (the model),
**הרצה** is a different verb about a different object (the plan).
