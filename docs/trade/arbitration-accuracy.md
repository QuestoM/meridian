# Three readings of the same corpus, measured

Produced by `scripts/trade_arbitration_bench.py` against the corpus in
`tests/trade_corpus/agreements` and the same independently authored ground
truth as `extraction-accuracy.md`. Every number is a real provider run.

| reading | what it is | recall | precision | parameters | seconds |
|---|---|---|---|---|---|
| pipeline | A — clause by clause (shipped) | 100.0% (7/7) | 38.9% (7/18) | 44.8% (13/29) | 255 |
| whole | B — one call, whole document | 0.0% (0/7) | — (0/0) | — (0/0) | 59 |
| arbitrated | C — A and B, disagreements ruled by a third call | 0.0% (0/7) | — (0/0) | — (0/0) | 64 |

## Where the two readers disagreed

| document | agreed | different parameters | only A saw | only B saw | rulings |
|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 0 | 0 | 18 | 0 | 0 |

## How the arbiter ruled

| verdict | meaning | count |
|---|---|---|

## Per document

| document | A recall | B recall | C recall | A params | B params | C params |
|---|---|---|---|---|---|---|
| `heb-amendment-q4-2026` | 100.0% | 0.0% | 0.0% | 44.8% | — | — |
