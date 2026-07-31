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

**The one rule.** It refuses a record that claims anything without an `evidence` entry carrying a `path`, a
measurement whose value is null without a `note_he` saying why, or a piece id the state does not know. An
unknown value stays null with its reason and the page draws it as an honest empty state, never as a zero.
