# Extraction accuracy, measured

Run 2026-08-17T12:53:33+00:00 against the corpus in `tests/trade_corpus/agreements`.
Models: small=claude-haiku-4-5-20251001, mid=claude-sonnet-5, reason=claude-opus-5.

Every number here is produced by `scripts/trade_extraction_accuracy.py`
running the real pipeline against ground truth authored independently of
it. Nothing on this page is asserted by hand.

## Aggregate

| measure | value |
|---|---|
| documents | 8 |
| clauses accounted for | 198/197 (100.5%) |
| disposition class correct | 94.4% |
| term recall | 152/162 (93.8%) |
| term precision | 152/215 (70.7%) |
| parameter accuracy | 396/608 (65.1%) |
| citation fidelity failures | 0 |
| planted conflicts detected | 5/6 |

## Per document

| document | clauses | class | recall | precision | params | conflicts |
|---|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 12/12 | 91.7% | 100.0% | 36.8% | 48.3% | 0/0 |
| `heb-annual-framework-2026` | 50/50 | 98.0% | 95.1% | 72.2% | 63.3% | 1/1 |
| `heb-contradictory-2026` | 26/26 | 96.2% | 100.0% | 78.6% | 71.8% | 3/4 |
| `heb-direct-advertiser-2026` | 32/32 | 96.9% | 93.3% | 82.3% | 58.4% | 1/1 |
| `heb-edge-stress-2026` | 28/28 | 92.9% | 91.3% | 80.8% | 71.4% | 0/0 |
| `heb-sano-annual-2025` | 13/12 | 91.7% | 88.9% | 53.3% | 71.9% | 0/0 |
| `heb-scanned-smallbiz-2026` | 14/14 | 78.6% | 75.0% | 75.0% | 68.8% | 0/0 |
| `heb-sponsorship-bundle-2026` | 23/23 | 95.7% | 100.0% | 66.7% | 66.2% | 0/0 |

## Parameter accuracy by term family

| family | instances | leaves | accuracy |
|---|---|---|---|
| A — זהות, היקף ומסמך | 28 | 141 | 74.5% |
| B — בסיס הכסף | 15 | 109 | 64.2% |
| C — הנחות, עמלות ותמריצים | 18 | 92 | 81.5% |
| D — התחייבויות המפרסם | 13 | 65 | 76.9% |
| E — התחייבויות הערוץ והשלמות | 20 | 73 | 61.6% |
| F — אילוצי שיבוץ | 17 | 52 | 71.2% |
| G — תהליך ומשפט | 31 | 53 | 17.0% |
| H — מדידה והתחשבנות | 10 | 23 | 21.7% |

## What the pipeline missed

- `heb-annual-framework-2026` — frequency-caps (gt-freq-day)
- `heb-annual-framework-2026` — precedence-clause (gt-supersession)
- `heb-direct-advertiser-2026` — competitive-separation (dt-separation-soft)
- `heb-direct-advertiser-2026` — frequency-caps (dt-freq-day)
- `heb-edge-stress-2026` — programme-daypart-restrictions (es-restrict-kids)
- `heb-edge-stress-2026` — barter-inquiry (es-barter)
- `heb-sano-annual-2025` — agreement-parties (sn-parties)
- `heb-scanned-smallbiz-2026` — agreement-parties (sc-parties)
- `heb-scanned-smallbiz-2026` — term-effective-windows (sc-margin-window)
- `heb-scanned-smallbiz-2026` — frequency-caps (sc-freq-day)

## Cost and latency

| document | seconds | calls | input tokens | output tokens |
|---|---|---|---|---|
| `heb-amendment-q4-2026` | 266.6 | 31 | 101,608 | 16,574 |
| `heb-annual-framework-2026` | 815.3 | 59 | 204,008 | 51,751 |
| `heb-contradictory-2026` | 345.7 | 31 | 79,924 | 21,403 |
| `heb-direct-advertiser-2026` | 417.0 | 38 | 102,598 | 27,469 |
| `heb-edge-stress-2026` | 371.1 | 29 | 81,161 | 22,619 |
| `heb-sano-annual-2025` | 163.0 | 17 | 41,489 | 9,756 |
| `heb-scanned-smallbiz-2026` | 144.1 | 18 | 46,261 | 8,705 |
| `heb-sponsorship-bundle-2026` | 283.2 | 30 | 77,268 | 20,415 |
