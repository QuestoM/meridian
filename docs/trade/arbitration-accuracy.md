# Three readings of the same corpus, measured

Produced by `scripts/trade_arbitration_bench.py` against the corpus in
`tests/trade_corpus/agreements` and the same independently authored ground
truth as `extraction-accuracy.md`. Every number is a real provider run.

| reading | what it is | recall | precision | parameters | seconds |
|---|---|---|---|---|---|
| pipeline | A — clause by clause (shipped) | 100.0% (7/7) | 41.2% (7/17) | 44.8% (13/29) | 238 |
| whole | B — one call, whole document | 100.0% (7/7) | 58.3% (7/12) | 58.6% (17/29) | 43 |
| arbitrated | C — A and B, disagreements ruled by a third call | 100.0% (7/7) | 38.9% (7/18) | 55.2% (16/29) | 95 |

## Where the two readers disagreed

| document | agreed | different parameters | only A saw | only B saw | rulings |
|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 1 | 8 | 8 | 3 | 19 |

## How the arbiter ruled

| verdict | meaning | count |
|---|---|---|
| `revised` | neither; the arbiter wrote the term itself | 8 |
| `b` | the whole-document reader governs | 7 |
| `a` | the clause reader governs | 2 |
| `neither` | no commercial term here at all | 2 |

## Per document

| document | A recall | B recall | C recall | A params | B params | C params |
|---|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 100.0% | 100.0% | 100.0% | 44.8% | 58.6% | 55.2% |
