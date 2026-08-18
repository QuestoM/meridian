# Three readings of the same corpus, measured

Produced by `scripts/trade_arbitration_bench.py` against the corpus in
`tests/trade_corpus/agreements` and the same independently authored ground
truth as `extraction-accuracy.md`. Every number is a real provider run.

Arbiter prompt: `v2-neutral`.

| reading | what it is | recall | precision | parameters | seconds |
|---|---|---|---|---|---|
| pipeline | A — clause by clause (shipped) | 93.2% (151/162) | 68.0% (151/222) | 64.5% (389/603) | 3107 |
| whole | B — one call, whole document | 93.8% (152/162) | 81.3% (152/187) | 70.7% (402/569) | 663 |
| arbitrated | C — A and B, disagreements ruled by a third call | 94.4% (153/162) | 71.2% (153/215) | 69.6% (424/609) | 1221 |

## Which reading wins what

- **recall — how much of the truth was found**: C at 94.4%, ahead of B by 0.6 points.
- **precision — how much of what was found is real**: B at 81.3%, ahead of C by 10.1 points.
- **parameter accuracy — how right the values are**: B at 70.7%, ahead of C by 1.0 points.
- **wall clock over the whole corpus**: B at 663s, ahead of C by 558s.

Read them together rather than one at a time: a reading that finds more
terms and is wrong about more of them has not improved anything a
reviewer has to sit through.


## Where the two readers disagreed

| document | agreed | different parameters | only A saw | only B saw | rulings |
|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 1 | 8 | 8 | 3 | 19 |
| `heb-annual-framework-2026` | 7 | 32 | 14 | 8 | 54 |
| `heb-contradictory-2026` | 6 | 17 | 4 | 1 | 22 |
| `heb-direct-advertiser-2026` | 3 | 24 | 9 | 2 | 35 |
| `heb-edge-stress-2026` | 2 | 18 | 8 | 5 | 31 |
| `heb-sano-annual-2025` | 4 | 6 | 1 | 2 | 9 |
| `heb-scanned-smallbiz-2026` | 4 | 6 | 3 | 3 | 12 |
| `heb-sponsorship-bundle-2026` | 7 | 13 | 16 | 2 | 31 |

## How the arbiter ruled

| verdict | meaning | count |
|---|---|---|
| `b` | the whole-document reader governs | 74 |
| `revised` | neither; the arbiter wrote the term itself | 57 |
| `a` | the clause reader governs | 50 |
| `neither` | no commercial term here at all | 32 |

## Per document

| document | A recall | B recall | C recall | A params | B params | C params |
|---|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 100.0% | 100.0% | 100.0% | 44.8% | 58.6% | 48.3% |
| `heb-annual-framework-2026` | 95.1% | 92.7% | 95.1% | 61.0% | 68.5% | 66.7% |
| `heb-contradictory-2026` | 100.0% | 100.0% | 100.0% | 73.2% | 71.8% | 73.2% |
| `heb-direct-advertiser-2026` | 93.3% | 93.3% | 93.3% | 58.4% | 65.2% | 67.3% |
| `heb-edge-stress-2026` | 91.3% | 91.3% | 95.7% | 70.4% | 72.4% | 75.8% |
| `heb-sano-annual-2025` | 77.8% | 88.9% | 88.9% | 81.5% | 87.5% | 84.4% |
| `heb-scanned-smallbiz-2026` | 75.0% | 83.3% | 75.0% | 62.5% | 70.6% | 68.8% |
| `heb-sponsorship-bundle-2026` | 100.0% | 100.0% | 100.0% | 67.7% | 76.5% | 70.6% |
