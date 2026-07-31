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

Scored on what you asked for. `0` everything requested ran and passed. `1`
something failed. `2` a requested check tried and could not finish, so the proof
has a hole in it; pass `--allow-unchecked` to accept that.

Declining to run a check and a check failing to run are different things and are
scored differently. Checks you did not request never move the exit code, but the
verdict line still names them, so a partial run cannot read as a full gate.

`--route-deadline` bounds each GET in the `bodies` check, and the probe writes its
results as it goes, so a hang still yields every route measured before it rather
than nothing.

Two routes are excluded by name and reported as unproven: `/api/constraints/effect`
and `/api/overrides/effect`. Called without arguments they have no upper bound, and
the deadline cannot stop them, because it is a signal and they spend their time
inside numpy where the interpreter never gets a chance to raise it. Naming them as
unproven is honest; letting them hang the run, or quietly counting them as
agreeing, would not be. Proving those two needs a parameterised call, which is
work this harness does not yet do.

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
