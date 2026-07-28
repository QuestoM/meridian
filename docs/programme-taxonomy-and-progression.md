# Programme taxonomy and season progression: design

Owner ask (2026-07-28): category, subcategory, programme, season and episode
levels; first-episode and finale flags; and a numeric season-progress feature
(episode 2 of 10 encoded as the 10-20 percent range, as a number, never a
string) that both training and forward scoring condition on.

Ground truth this design is built on (recon 2026-07-28, all measured on the
real files):

- The EPG carries NO taxonomy, season or episode columns; all signal is in the
  title string. 'עונה' (season) appears on 15.1% of rows with a 100%-precision
  integer parse; 'פרק' (episode) appears on ZERO of 3,562 rows and zero of the
  125 forward rows. The rerun marker ש.ח covers 43.3% of rows.
- No source anywhere carries episodes-per-season, so the denominator of
  'episode 2 of 10' is unknowable from the data.
- History is exactly 30 days (2024-11-01..30), not two years. Within it, a
  per-series first-run airing ordinal is computable (median 5 first-run
  airings per series), but it is window-truncated: a mid-season episode looks
  like airing 1-5.
- The direct empirical precedent already shipped: the series training layer
  (per-title conditioning with empirical-Bayes pooling and a self-activating
  five-fold held-out gate) measures MINUS 8.5% held-out skill on this window
  and correctly stays off. Finer title-derived features on 30 days are noise.
- A 16-genre classifier with an editable YAML taxonomy plus an AI fallback
  already classifies 100% of titles, and `canonicalize_series` already maps
  417 titles to 299 series keys. The retention cells however key on the
  4-value POSITIONAL pricing class, so genre never reaches the cells today.

## Honest verdict, stated plainly

The full ask is not supported by the data on disk. What each level needs:

| Level | Coverage today | Blocker |
|---|---|---|
| Category (16 genres) | 100% | none, already built |
| Subcategory | n/a | a taxonomy edit in the YAML, no data blocker |
| Programme / series key | 100% | none, already built |
| Season number | 15.1% of rows | title convention, partial by nature |
| Episode number | 0% | absent from every source |
| Episode k of N (percent) | 0% | needs an external per-series episode-count table |
| First-episode / finale flags | 0% | same external table; window-edge proxies would mislabel |

## What we build now (all honest at current data)

1. Extraction layer (`title_features.py` extension): per title, emit
   `category` (classifier), `series_key` (canonicalizer), `season_number`
   (int or null, the measured 100%-precision parse), `is_rerun` (ש.ח), and
   `airing_ordinal` (count of prior first-run airings of the series inside the
   loaded history, null for reruns). All numeric or null, never strings.
2. Surface it: the schedule inspector, break library and canvas show category,
   series, season and rerun honestly ('עונה 7', 'שידור חוזר'), with '-' where
   the title carries nothing. The columns also join the weekly CSV export so
   traffic sees them.
3. Progression encoding, prepared but not activated: a single float
   `season_progress` in [0,1] defined as (episode_ordinal - 0.5) / total
   episodes, computed ONLY when an external episode table exists; until then
   the field is null and the UI says why. No string encodings anywhere.
4. Training readiness, gated: the measurement rows gain the extraction columns
   so the day richer history lands, a progression layer can be fit as a gated
   additive layer copying `series_gate.py` (five temporal folds, +2% held-out
   RMSE bar, self-activating). Expectation set honestly: on the current window
   the gate would keep it OFF, as the finer series layer already measures -8.5%.

## External prerequisites (owner supplies, then v2 activates)

- A per-series episode-count table (series_key, season, total_episodes,
  premiere_date, finale_date) from the broadcaster's content system, enabling
  season_progress, first-episode and finale flags at real coverage.
- Multi-month history, without which any progression coefficient is noise by
  the measured precedent.

## Non-goals

Scraping external websites for episode data; treating the stale
`programme_type` legacy column as a taxonomy; per-title cells in the pooled
model (the EB pooling would crush them, and the gate already proves it).
