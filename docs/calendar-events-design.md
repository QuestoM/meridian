# Calendar and events: design

Owner ask (2026-07-28, named the most important front): the model should relate
to day-of-week, dates, holidays, and operator-managed special events (wars in
Israel, with start, end, intensity), with a dedicated UI tab that both manages
events and shows honestly what the model already conditions on and how event
impact is decided.

Ground truth this design is built on (recon 2026-07-28, all measured):

- The shipped retention measurement is calendar-blind: no weekday, date, holiday
  or event enters the detrend, pooling or placebo steps. The month-of-day
  seasonal baseline exists but measured 0.0% held-out improvement and stays off.
- Pricing IS weekday-aware today: a live program-class by day premium (Thu 1.05,
  Fri 1.15, Sat 1.20) applied to real calendar dates at scoring time. These are
  rate-card assertions, not measured from audience history.
- The training history is exactly 30 days, 2024-11-01 to 2024-11-30. Each
  weekday occurs 4 to 5 times; 14 of 36 cells lack full weekday coverage
  (median 5 breaks per cell-weekday). A global weekday retention contrast is
  measurable and gateable; per-cell weekday coefficients are not.
- The window contains ZERO Jewish or Israeli holidays, and sits almost entirely
  inside wartime: the Israel-Hezbollah ceasefire took effect 2024-11-27, leaving
  a post-ceasefire tail of about 3.5 days (132 of 2532 measured breaks). A
  holiday retention coefficient or a war-intensity retention multiplier claimed
  from this data would be fabrication.
- No holiday library exists in the venv; no events store, API or tab exists.
- The full coefficient rebuild is measured at ~19 seconds, so re-measuring under
  event annotations is cheap the day richer history lands.

## The honesty line that governs everything here

Retention effects must be MEASURED; pricing premiums may be OPERATOR-ASSERTED
(they already are: Friday 1.15 is a rate-card assertion). Therefore:

- v1 ships event data, disclosure, and operator-owned pricing hooks.
- Event RETENTION coefficients ship only in v2, measured, behind the same
  held-out gate the series and competitor layers use, once history that actually
  contains contrast (holidays, war on/off) exists.

## v1 (build now)

1. Events store: `data/calendar_events.csv` with columns `event_id, name,
   type (holiday|war|special|sport|other), start_date, end_date (inclusive,
   empty = open-ended), intensity (1..5, operator-judged), notes, active`.
   Atomic writes, module lock, version snapshots (`snapshot_manual_edit`,
   a new 'events' logical file), CRUD router `kairos_api/events_api.py`.
2. Bundled holiday table: a static, checked-in Israeli holiday list for
   2024-2027 (`kairos/config/israel_holidays.csv`: date, name, kind
   national|religious, is_school_holiday). Deterministic, offline, editable.
   Rendered read-only beside operator events, importable into the store per
   year with one click (so operators can attach intensity or disable rows).
3. Calendar tab in the dashboard nav, three panels:
   a. Events management: list, add, edit, deactivate; war-type events highlight
      their open-ended state; every date-picker uses the shared DateField.
   b. What the model conditions on TODAY, built from real metadata and config,
      no invention: the weekday pricing premiums table (labeled as rate-card
      assertions); measurement mode global with the seasonal baseline measured
      at 0.0% and off; level-drift weekly levels with the binding flag;
      computed_at and window; and one new disclosure line stating that the
      whole 30-day training window was measured under wartime conditions with
      the ceasefire only on 2024-11-27. This panel is where the operator sees
      how impact is decided: measured coefficients only, gates and verdicts
      named.
   c. Overlap view: each stored event intersected with (i) the coefficient
      training window and (ii) the current plan dates, flagging affected days
      on both, so the operator sees which plan days sit inside an event and
      whether the model's training data even saw that condition.
4. Event awareness on plan surfaces: the Overview basis note and the schedule
   canvas mark days covered by an active event (name badge), display-only.
5. Optional pricing hook, owner-gated OFF: an event-date pricing layer in the
   `price_slot` stack (the existing layer machinery with activation flags and
   source labels), letting an operator assert, for example, a multiplier for
   days inside a named event. Ships off; activating moves real money and is an
   owner decision, exactly like the position and ad-type layers.

## v2 (only with richer history)

A measured event layer: join events to the per-break effects frame on
`break_start.date()` before pooling, fit event-window contrasts (holiday,
war-intensity, weekday) as gated additive layers copying `series_gate.py`
(five temporal folds, +2% held-out RMSE bar, self-activating each rebuild).
The seam is a one-line date merge; the rebuild costs ~19s. Until the gate
passes, the tab displays the verdict honestly instead of activating.

## Non-goals

Live news feeds or automatic event detection; fabricated event multipliers on
retention; per-cell weekday coefficients on 30 days of data.
