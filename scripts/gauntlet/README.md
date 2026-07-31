# verify_wave.py

Proves, or fails to prove, that the working tree is behaviourally identical to a
reference commit. Built for closing a wave whose whole claim is that nothing
changed, because that claim is cheap to assert and tedious to prove.

```
~/.venvs/meridian/bin/python scripts/gauntlet/verify_wave.py --reference 5a80a709
```

## What it checks

| check | what it proves | cost |
|---|---|---|
| `api` | The OpenAPI surface is unchanged: paths, methods, response models, parameters, named route by route | seconds |
| `bodies` | Every argument-free GET returns the same bytes. This is the real bar for a route split; a matching schema is necessary and not sufficient | minutes |
| `engine` | The golden weekly schedule reproduces the same CSV and aggregate hash. Invoked by name, because it carries no test prefix and a plain pytest run does not collect it | minutes |
| `suite` | The test suite, with `--suite-both` to run the reference too so failures that pre-date the wave are told apart from failures it caused | minutes |
| `moved` | Nothing under `data/`, `models/` or `config/` moved except what the build order declares, cross-referenced against the workbench state | seconds |
| `frontend` | All seventeen routes render the same text, both builds served against one API so the frontend is the only variable | minutes |

`--only api,bodies` runs a subset. Everything not requested is reported as not
checked, never as passed.

## Exit codes

`0` everything requested ran and passed. `1` something failed. `2` nothing
failed but something could not run, so the proof is incomplete. A gap is not a
pass, which is why `2` is not `0`; pass `--allow-unchecked` if you want it to be.

## What it will not do

It does not touch the shared tree or the index. The reference is materialised
with `git archive`, which writes nothing under `.git` and takes no lock a builder
could block on. The working tree is copied before anything runs against it,
because the suite writes into `data/` and running it where builders are working
would be the exact mutation this harness exists to rule out. Everything lands in
a temp directory that is removed on the way out unless you pass `--keep`.

## Reading a failure

Two distinctions do most of the work. Under `moved`, files under the app's own
runtime stores are a record of somebody using the product, not a change to it,
so they are reported separately and never counted against the wave. Under
`bodies`, a payload differing only in time-like fields is separated from one
differing in substance; neither is normalised away, because silently ignoring a
field is how a real difference hides.
