# <PIECE> dossier

Written before the wave, read by every round of this piece. The gate at
`scripts/gauntlet/wave_preflight.py` refuses to launch a wave whose pieces do not
have one of these, and refuses again when the line counts below have drifted from
the repository. So this file cannot quietly rot into decoration.

Every section here exists because a builder round was once spent rediscovering
it. Fill all of them. A half-written dossier passes a human skim and fails a
builder.

---

## Job stories and their done conditions

Quote them in full from `job-stories.md`. Do not reference them by number: a
builder that has to go and look them up is a builder spending context on
navigation.

## Baseline numbers

From `discovery/06-baseline.md`, for these stories only. The stopwatch figure,
the click count, whatever the bars measure. A critic that has to re-measure the
baseline every round is a full sweep per round per piece.

## File inventory

Every path this piece may write, with its CURRENT line count and what it holds.
The count is checked by the gate, so a stale one fails loudly rather than
misleading quietly. Mark anything near the 450 cap out loud.

| path | lines | note |
|---|---|---|
| `TODO/path.py` | 0 lines | TODO |

## The API surface this piece owns

Routes, payload shapes, what each field means, and which of them another piece
reads. Ownership is absolute: every path not listed here is frozen.

## Reference product, and what to compare

Which product, which screen, and the specific thing to compare. "Compare with the
reference" is not an instruction; "compare the column order and the empty state"
is.

## Trade facts that bind this piece

Quoted from `media-domain-from-the-trade.md`, not referenced. The trade outranks
the code, and a fact a builder did not read cannot bind it.

## What is already built

From the contract and the state file. This is the section that stopped three
builders in one wave from spending a full context to discover the work was
already on disk.

## Exact commands

The interpreter by absolute path, the test invocation, the build, the port this
piece runs on. Not the name of the interpreter. Four of five attempts in one wave
died on `ModuleNotFoundError` because a brief said "python" and meant
`~/.venvs/meridian/bin/python`.

## Recon is not the job

Leave this line in. If you are thirty calls in with no edit, stop reading and
start writing.
