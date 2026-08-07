# Date parsing, swept 2026-08-07

Read-only sweep of every date parse in `kairos/`, `kairos_api/` and `scripts/`.
Written down so nobody spends another hour suspecting this.

## Verdict

**No day-first problem anywhere.** Zero at-risk sites. Every parse is safe either
by format (ISO, or an explicit `%d/%m/%Y`) or by data (an internal column that
was already parsed upstream).

## Why it was suspected

`data/Spots.csv` holds 30 raw values from `01/11/2024` to `30/11/2024`. Parsed
month-first that collapses to 12 wrong days, one per month, all on the 11th,
with no error raised. Silent corruption, not a crash, which is the worst shape a
data bug can take.

The product parses it correctly. The 12-day reading came from a naive throwaway
script, not from the engine. Worth recording, because the false alarm is more
likely to recur than the bug.

## The one site that looks wrong and is not

`kairos/data/loaders.py:264` uses `dayfirst=False` while lines 151, 152, 184 and
221 use `dayfirst=True`. That is deliberate: this product reads two different
formats.

- The Israeli reference workbook is `DD/MM/YYYY`.
- The daily traffic CSV is `M/D/YYYY`. Measured on the only file on disk, whose
  `תאריך` column holds `4/27/2025`.

Both conventions are documented in the module's own docstring and repeated
inline at the parse site. For the real file the flag cannot change the answer,
because every day component exceeds 12. For a genuinely ambiguous row such as
`4/5/2025` it would, and `count_ambiguous_daily_dates` at lines 233 to 251 exists
precisely to surface that class of row to the operator at upload time rather than
guess.

So a mixed convention in one module is correct here. It reflects two real input
formats, not an inconsistency.

## Scope

Every `pd.to_datetime`, `datetime.strptime`, `fromisoformat` and hand-rolled
split across the three trees. Each classified as safe by format, safe by data, or
at risk, with the caller read rather than assumed.
