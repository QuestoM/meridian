# The gauntlet workbench

`workbench.html` is the owner's window into the Meridian experience gauntlet: where the run is, what is
blocked and on whom, how each job story stands against its measured baseline, and what was deferred and why.

**Opening it.** Double-click `workbench.html`. Opened from disk the browser will not let it read files beside
it, so it renders the copy of the state embedded in the page and says so, in amber, at the top, with that
copy's date. Publishing with `--embed` below keeps that copy current, so the amber banner reports a fresh
state rather than a stale one. For the live file instead, either pick `state.json` with the button in that
banner, or run `python3 -m http.server 8021` here and open `http://127.0.0.1:8021/workbench.html`.

**Publishing a round.** Never hand-edit `state.json` and never edit the html. Pipe one round record to
`update_state.py`. This is the whole publish step:

```
echo '{"piece_id":"W0-1","piece_status":"in_critique","changed_he":"...","evidence":[{"path":"tests/test_w0_1_route_identity.py"}],"verdict":"passed","verdict_he":"עבר","next_action_he":"...","measurements":[{"label_he":"מסלולים זהים","value":25,"unit":""}]}' | python3 docs/ux-gauntlet/update_state.py --embed
```

It appends the round, updates that piece's status and round history, refreshes `meta.updated_at`, and with
`--embed` regenerates the page's inlined copy from the state, which is a derived refresh and not an edit.
Drop `--embed` only if you deliberately want the disk-opened page to keep showing the older snapshot. It sets
only the keys you send, so a partial record clears nothing, and re-sending the same `round` merges into it
rather than duplicating. Add `--dry-run` to validate a record and write nothing.

## How a claim enters this page

The script enforces the format. These six hold up the content, and a claim that skips them is worth less
than no claim, because it looks like evidence.

1. **Measure it, do not argue it.** A conclusion reached by reasoning over facts is an argument. Run the
   thing, count the rows, time the call, read the rendered geometry. Report what came back.
2. **Prefer a discriminator that cannot be a mistake.** A capability someone mentioned can be a mistake in
   the mentioning. A recorded file length of 6,236 that matches one commit exactly and misses another by 397
   lines cannot be. When two explanations are open, look for the measurement that only one of them survives.
3. **Run the counter-check, and say it came back empty.** Do not stop at the first confirming fact. Go
   looking for evidence of the opposite conclusion, and report the result either way. "No file was deleted
   across the range, so no artifact could exist under the old commit and not the new one" is what makes a
   verdict safe. An investigation that only confirms has not been tested.
4. **Distrust your eye, trust the instrument.** A screenshot that looks wrong is a hypothesis. Measure before
   you fix, or you will correct something that was already right.
5. **An unknown stays unknown.** Null with a stated reason, rendered as an honest empty state. Never a zero,
   never a placeholder, never a number carried over from a nearby thing that happened to be measured.
6. **Attribute against a commit, never against the working tree.** The working tree is not any wave's output.
   It is the sum of every wave that has touched it plus whatever is in flight this second, so a finding
   measured there has no owner until you diff it against a commit. Extract the commit with `git archive` and
   run the same thing twice. This is rule 3 applied to attribution rather than to evidence: a failure that is
   real, reproducible and correctly diagnosed can still be pinned on the wrong wave, and blaming the wrong
   wave is worse than reporting nothing, because it sends someone to fix code that is already correct.

**The one rule the script enforces.** It refuses a record that claims anything without an `evidence` entry
carrying a `path`, a measurement whose value is null without a `note_he` saying why, or a piece id the state
does not know.
