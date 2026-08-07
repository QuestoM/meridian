# Does the system hold the current month, 2026-08

Audited 2026-08-07, read only, against file contents rather than filenames.

## The answer

**No, and there was nothing to capture.** No broadcast data for August 2026
exists anywhere. The newest observed airing in the product is 2025-04-27,
fifteen months before today. The only thing touching 2026-08 is five real
campaign booking windows with no delivery rows behind them.

On the question that actually matters, whether the product is HONEST about not
having it, **it passes on every time-anchored surface but one.**

## Date coverage

The engine reads `data/reference/*.xlsx`; the same-named CSVs are shadowed and
the Sources page says so.

| Input | Rows | Coverage |
|---|---|---|
| Spots.xlsx | 50,386 | 2024-11-01 to 2024-11-30, 30 days |
| Programmes.xlsx | 8,704 | 2024-11-01 to 2024-11-30, 30 days |
| Dayparts.xlsx | 43,200 | 2024-11-01 to 2024-11-30, 30 days |
| daily_input Wally CSV | 175 | 2025-04-27 only |
| campaigns.csv | 56 | 51 demo at 2025-04, **5 real at 2026-08** |
| campaign_delivery.csv | 368 | 2025-04-27 to 05-03, only the first day has a source |
| campaign_flights.csv | **0** | header only, and this is what pacing reads |
| calendar_events.csv | 63 | 2023-10 to 2027-12, **zero overlap with 2026-08** |

## What the surfaces say about today, and they are mostly right

Overview names the gap outright: the window is the first seven days of the saved
plan, not a calendar week, and the plan holds no date inside the current week.
The day route answers `available:false` with a null projection rather than a
zero. Campaign delivery for the five real August rows reads unknown, because a
guard deliberately refuses a confident zero. Make-good names the missing input
by filename. No surface computes a month-to-date figure at all, so there is no
such number to get wrong.

## Real defects

1. **FIXED 2026-08-07.** The compliance report labelled its period with the
   licence date rather than the data's period, so checks computed over November
   2024 announced themselves as `2026-06-14`, a date that is neither the data
   nor today. Its two siblings built from the same file already did it right.
   The licence date now has its own label.
2. **Pacing's idea of today is hardcoded 54 days stale.** `core.py` defaults
   `effective_date` to `2026-06-14` and `pacing_reference_date` is empty in
   settings, so the reference today is June while real today is August. Real
   today falls inside a live campaign flight; the reference date does not.
   Latent rather than live, because flights is empty so pacing is identity at
   1.0 everywhere. It arms the moment anyone uploads flights.
3. **Two disconnected campaign stores.** The board reads `campaigns.csv`, where
   the August bookings live. Pacing reads `campaign_flights.csv`, which is
   empty. Booking a campaign on the board does not reach the optimizer, and
   nothing on screen says the two are different files.
4. **Demo rows read as forthcoming.** Fifty-one demo campaigns show active and
   "scheduled, not yet aired" for windows that closed 2025-05-03. The board
   stamps its as-of once at the top, so this is weak disclosure rather than
   fabrication, but it is the closest thing to stale-as-current here. Status is
   an operator lifecycle flag never reconciled against dates.
5. **Orphaned sample data.** `Programmes - today.csv` is internally
   contradictory, but nothing reads it.

## The Israeli week law is correct for this month

Tested against the real formula rather than reasoned about. August 2026 opens on
a Saturday, which is a weekend day here and the classic breaker of a naive week
calculation.

- 2026-08-01, Saturday, resolves to the week 2026-07-26 to 2026-08-01. Correct.
- 2026-08-07, today, resolves to 2026-08-02 to 2026-08-08. Correct.
- 2026-08-31, Monday, resolves to 2026-08-30 to 2026-09-05. Month crossing held.

Weekend is Friday and Saturday in both the API and the model. No Monday-first
idiom exists anywhere. The rendered header is Sunday-first.
