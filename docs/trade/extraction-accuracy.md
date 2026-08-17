# Extraction accuracy, measured

Run 2026-08-17T11:47:43+00:00 against the corpus in `tests/trade_corpus/agreements`.
Models: small=claude-haiku-4-5-20251001, mid=claude-sonnet-5, reason=claude-opus-5.

Every number here is produced by `scripts/trade_extraction_accuracy.py`
running the real pipeline against ground truth authored independently of
it. Nothing on this page is asserted by hand.

## Aggregate

| measure | value |
|---|---|
| documents | 7 |
| clauses accounted for | 185/185 (100.0%) |
| disposition class correct | 94.1% |
| term recall | 142/153 (92.8%) |
| term precision | 142/201 (70.6%) |
| parameter accuracy | 369/574 (64.3%) |
| citation fidelity failures | 0 |
| planted conflicts detected | 5/6 |

## Per document

| document | clauses | class | recall | precision | params | conflicts |
|---|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 12/12 | 91.7% | 100.0% | 50.0% | 44.8% | 0/0 |
| `heb-annual-framework-2026` | 50/50 | 96.0% | 92.7% | 70.4% | 63.6% | 1/1 |
| `heb-contradictory-2026` | 26/26 | 96.2% | 100.0% | 84.6% | 71.8% | 3/4 |
| `heb-direct-advertiser-2026` | 32/32 | 96.9% | 90.0% | 75.0% | 58.0% | 1/1 |
| `heb-edge-stress-2026` | 28/28 | 92.9% | 91.3% | 70.0% | 71.4% | 0/0 |
| `heb-scanned-smallbiz-2026` | 14/14 | 78.6% | 75.0% | 75.0% | 65.6% | 0/0 |
| `heb-sponsorship-bundle-2026` | 23/23 | 95.7% | 100.0% | 62.1% | 64.7% | 0/0 |

## Parameter accuracy by term family

| family | instances | leaves | accuracy |
|---|---|---|---|
| A — זהות, היקף ומסמך | 26 | 138 | 71.0% |
| B — בסיס הכסף | 15 | 109 | 67.9% |
| C — הנחות, עמלות ותמריצים | 16 | 80 | 80.0% |
| D — התחייבויות המפרסם | 12 | 60 | 78.3% |
| E — התחייבויות הערוץ והשלמות | 18 | 62 | 61.3% |
| F — אילוצי שיבוץ | 17 | 52 | 67.3% |
| G — תהליך ומשפט | 28 | 50 | 16.0% |
| H — מדידה והתחשבנות | 10 | 23 | 21.7% |

## What the pipeline missed

- `heb-annual-framework-2026` — frequency-caps (gt-freq-day)
- `heb-annual-framework-2026` — precedence-clause (gt-supersession)
- `heb-annual-framework-2026` — dispute-resolution (gt-jurisdiction)
- `heb-direct-advertiser-2026` — agreement-level (dt-level)
- `heb-direct-advertiser-2026` — competitive-separation (dt-separation-soft)
- `heb-direct-advertiser-2026` — frequency-caps (dt-freq-day)
- `heb-edge-stress-2026` — programme-daypart-restrictions (es-restrict-kids)
- `heb-edge-stress-2026` — barter-inquiry (es-barter)
- `heb-scanned-smallbiz-2026` — agreement-parties (sc-parties)
- `heb-scanned-smallbiz-2026` — term-effective-windows (sc-margin-window)
- `heb-scanned-smallbiz-2026` — frequency-caps (sc-freq-day)

## Cost and latency

| document | seconds | calls | input tokens | output tokens |
|---|---|---|---|---|
| `heb-amendment-q4-2026` | 179.9 | 16 | 40,418 | 11,402 |
| `heb-annual-framework-2026` | 828.3 | 59 | 204,277 | 53,233 |
| `heb-contradictory-2026` | 316.9 | 29 | 75,798 | 19,036 |
| `heb-direct-advertiser-2026` | 431.6 | 40 | 107,325 | 28,861 |
| `heb-edge-stress-2026` | 411.3 | 33 | 91,291 | 25,951 |
| `heb-scanned-smallbiz-2026` | 150.6 | 18 | 46,261 | 8,730 |
| `heb-sponsorship-bundle-2026` | 313.0 | 32 | 81,764 | 22,316 |
