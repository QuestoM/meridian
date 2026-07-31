# The gauntlet workbench

`workbench.html` is the owner's window into the Meridian experience gauntlet: where the run is, what is
blocked and on whom, how each job story stands against its measured baseline, and what was deferred and why.

**Opening it.** Double-click `workbench.html`. Opened from disk the browser will not let it read files beside
it, so it renders the state embedded at generation time and says so, in amber, at the top, with that copy's
date. For the live file, either pick `state.json` with the button in that banner, or run
`python3 -m http.server 8021` here and open `http://127.0.0.1:8021/workbench.html`.

**How an agent updates it.** Edit `state.json`. Never the html, which is a renderer with no content of its
own. Append a round to `rounds` carrying its `piece_id`, move that piece's `status`, and write a job story's
`current` only once a critic measured it in a browser against the running app.

**The one rule.** Nothing enters `state.json` that was not measured. Every value carries its source; an
unknown value stays `null` with a stated reason and the page draws it as an honest empty state. Never a
zero, never a placeholder.
