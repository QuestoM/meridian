# The gauntlet workbench

`workbench.html` is the owner's window into the Meridian experience gauntlet: where the run is, what is
blocked and on whom, how each job story stands against its measured baseline, and what was deferred and why.

**Opening it.** Double-click `workbench.html`. Opened from disk the browser will not let it read files beside
it, so it renders the state embedded at generation time and says so, in amber, at the top, with that copy's
date. For the live file, either pick `state.json` with the button in that banner, or run
`python3 -m http.server 8021` here and open `http://127.0.0.1:8021/workbench.html`.

**How an agent updates it.** Never hand-edit `state.json` and never touch the html. Pipe one round record to
`update_state.py`, which validates it and writes atomically:

```
echo '{"piece_id":"W0-1","piece_status":"in_critique","changed_he":"...","evidence":[{"path":"tests/test_w0_1_route_identity.py"}],"verdict":"passed","verdict_he":"עבר","next_action_he":"...","measurements":[{"label_he":"מסלולים זהים","value":25,"unit":""}]}' | python3 docs/ux-gauntlet/update_state.py
```

It appends the round, updates that piece's status and round history, and refreshes `meta.updated_at`. It sets
only the keys you send, so a partial record clears nothing, and re-sending the same `round` merges into it
rather than duplicating. Add `--dry-run` to validate without writing, or `--embed` to also refresh the copy
inlined in the html so a page opened from disk is not a stale snapshot.

**The one rule.** It refuses a record that claims anything without an `evidence` entry carrying a `path`, a
measurement whose value is null without a `note_he` saying why, or a piece id the state does not know. An
unknown value stays null with its reason and the page draws it as an honest empty state, never as a zero.
