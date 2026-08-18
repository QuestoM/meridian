# The rivals' schedules, pulled instead of typed

Every competitor input this engine has ever had arrived as a file somebody put on
disk by hand — which in practice means most weeks had none, because nobody
re-types a rival's schedule. This feed replaces that with a daily pull of the
WHOLE competitive lineup, into the one contract the engine already reads.

Run it before the daily simulation:

```bash
python -m kairos.model.keshet_feed --all
```

Measured on a real morning: 704 broadcasts across three rivals over thirteen
days, in one file, from one command.

It prints one line per channel and exits 0 only if every one refreshed. Every
line names its channel, because "the competitor schedule was not refreshed"
could be about any of four:

```
כאן 11: נמשך לוח כאן 11 לראשונה: 222 שידורים על פני 9 ימים.
עכשיו 14: לוח עכשיו 14 רוענן: אין שינוי מאז הפעם הקודמת.
קשת 12: לוח קשת 12 רוענן: 2 הוזזו, 1 הוכרזו.
קשת 12: לוח קשת 12 לא רוענן. הלוח שבידינו בן 26.0 שעות.
```

`--channel "כאן 11"` pulls one. `--days N` sets the window. `--json` gives the
whole result, including the per-programme diff.

## What it writes, and what reads it

`data/reference/CompetitorProgrammes.csv` — the file
`kairos.model.future_epg` already looks for. Nothing downstream changed: same
columns, same loader, same honest-absence behaviour when it is missing. The file
is **not** in git: it is third-party data, it is rewritten daily, and its absence
costs nothing but freshness.

**One file, every rival.** The contract has always carried a Channel column and
the loader has always read every channel out of it, so the place the optimizer
needs already existed. A refresh replaces only the rows of the channel it pulled
and carries the others through untouched — pulling one rival is never a deletion
of another — and the diff is computed within the channel, or a rival that did
not move would report as rebuilt whenever a different one was refreshed.

**Each channel's age is its own**, stamped in
`CompetitorProgrammes.freshness.json` beside the file. The file's modified time
cannot answer "how old is this rival's schedule": refreshing one channel touches
the file, so a channel nobody has pulled for a week would read as one minute
old. A missing stamp degrades to unknown, never to fresh.

## Where each rival comes from

| Channel | Source | Auth | Window |
|---|---|---|---|
| קשת 12 | Kway (`api.kway.co.il`) | signed-in session, renewed below | 13 days, one call |
| כאן 11 | FreeTV (`web.freetv.tv`) | none | 9 days, one call per day |
| עכשיו 14 | FreeTV | none | 9 days, one call per day |
| רשת 13 | FreeTV | none | skipped while it is the operator's own channel |

A channel with no source is refused **by name**. An empty schedule for a channel
that is broadcasting is the most expensive lie a plan can be told, so the
registry in `keshet_feed.SOURCES` is one table and a missing rival is a missing
row in it, not a different feature.

### Why FreeTV and not each broadcaster's own site

All three were fetched and compared. The differences are all in the one field
that matters:

- **`c14.co.il`** gives fifteen days in one call, and its end time is a **bare
  clock** whose date is only the key of the object around it. A programme running
  to 00:20 ends before it starts unless the converter rolls the day.
- **`13tv.co.il`** gives a title and a start and **no end and no duration at
  all**, so a length can only be inferred from the next programme's start —
  which silently swallows anything the editorial grid leaves out.
- Both refuse a browser user-agent and answer a command-line one: the opposite
  of what anyone would guess, and the opposite of each other.

FreeTV's `since` and `till` are fully dated ISO instants, so the duration is a
subtraction and the midnight case cannot arise. One converter is correct for
every channel.

**The channel number is checked, not trusted.** FreeTV calls them "ערוץ 11",
"ערוץ 12" — not the names this engine's history uses — so the mapping is written
down and verified against the publication before every pull. If a live id stops
carrying the title it carried when it was mapped, the pull refuses. A reused id
would otherwise file one rival's whole evening under another rival's name, and
the rows would be well formed, the file would load, and nothing downstream could
tell.

**Two flags are read from the feed and one is a trap.** The field called `live`
is not a flag at all but an object naming the channel, and a non-empty dict is
truthy — reading it generously marked every broadcast as live, reruns included.
The real flag is `liveBroadcast`. `repeat` is trustworthy and corroborated: on a
real window, 40 of 40 titles carrying "(ש.ח.)" have it set and 0 of 42 without
it do.

### Keshet stays on the licensed source, and here is the alternative

Measured: `mako.co.il/AjaxPage?jspName=EPGResponse.jsp` answers **200 with no
account at all** and returns the same 300 programmes in the same shape — the
existing converter reads it unchanged, 300 in and 300 out. So the credential
below is not technically required for Keshet.

It is kept anyway: the subscription is paid for, it carries fields the free feed
does not, and swapping a licensed data path for the competitor's own website is
a commercial decision rather than a refactoring. The alternative is written down
so the choice can be made rather than discovered.

## The session, which is the hard part

The publication answers 401 to anyone not signed in, and the session it issues
lives about twenty-one hours. A daily pull would therefore fail daily, which is
how a feature like this quietly stops working a week after it ships.

What was measured, and what it rules out:

- **There is no token to keep.** `localStorage` holds table column widths.
  The only cookie a page script can read is `XSRF-TOKEN`; the session itself is
  `sfp_access`, HttpOnly.
- **The login cannot be driven by URL.** `/api/auth/google?env=prod` answers
  `{"success":false,"message":"Code challenge is missing."}` — the flow is Google
  OAuth **with PKCE**, so the verifier belongs to the application. Hand-rolling
  the handshake would be a second, weaker copy of somebody else's protocol.

So the renewal lets the application perform its own handshake, in a browser
profile where **Google** is still signed in — a session that outlives Kway's by
months. One long-lived thing renews one short-lived thing on demand.

The session is kept in the **macOS login keychain**, not a dotfile. The profile
holding the Google login lives at `~/.kairos/kway-profile`, deliberately outside
this repository. No password is stored anywhere.

Measured, end to end from an empty keychain: **renewed in 7 seconds, headless**,
then reused in 0.4 seconds while it stays accepted. It runs with no window, so
it does not need a screen or a logged-in desktop.

| Variable | Value |
|---|---|
| `KAIROS_KWAY_ACCOUNT` | which account to continue as when Google asks |
| `KAIROS_KWAY_PROFILE` | where the signed-in browser profile lives |
| `KAIROS_CHROME` | path to the browser |

### The bug that cost two days, so it is not repeated

The renewal worked once and then stopped, with no change on either side. It was
Google's **account chooser** — no password, no code, just "which account?" — and
nothing in the code clicked it, so the flow sat on that page until the budget
expired and reported a timeout that blamed the network.

Two wrong answers were tried first and are recorded here because both were
plausible: a stale profile lock, and "Google refuses headless browsers." The
second was written into a docstring before it was ever measured, and it is
false.

What is true is stranger. At the same URL, in the same profile, minutes apart,
Google serves **two different pages** under that name. With a window the accounts
are scripted `[data-identifier]` tiles. Without one it serves its no-JavaScript
variant — the form carries `bgresponse=js_disabled` — where each account is a
submit `<button name="chooser[select]">` wrapped in an `<li>` that contains
nothing else. Wrapper and button therefore carry **identical text**, so ranking
candidates by text length ties, the stable sort hands the win to the wrapper, and
clicking the wrapper does nothing at all. Silently. For ninety seconds.

The fix: find the account by its **attribute** (`data-identifier` or
`data-email`), never by its text, and climb to the nearest genuinely clickable
ancestor. `tests/test_kway_session.py` pins both page shapes as static fixtures —
no network, no real account — and the fixture keeps the `<li>` wrapper on
purpose, because without it the broken selector passes.

### Signing in again ends the session you already had

Measured, twice, deliberately: renew once and the token works; renew again a
moment later and **both the old cookie and the new one answer 401 "Session
ended."** The server ends the standing session when a new sign-in arrives, and
for a moment afterwards the browser's cookie jar still holds the dead value — so
a harvest taken on the page's word walks away with a token that has just been
killed, reports success, and fails at the first real call.

Two rules follow, and both are now enforced:

1. **Never sign in while a session is alive.** The renewal loads the dashboard
   first and takes the profile's standing session when the server still accepts
   it. The cheapest renewal is the one that does not happen — measured, that path
   costs 4 seconds instead of 11.
2. **A session is only taken once the server accepts it from outside the
   browser.** The jar is read, tried over plain HTTPS with exactly the request
   the caller will make, and read again until it works or the settle window
   closes. "The page is signed in" and "these cookies work" are different facts.

This is why a run can report `renewed` and still be wrong if either rule is
dropped, and why `tests/test_kway_session.py` pins both.

## When a person is genuinely needed

Three states, and each names the one step:

- **Google asks for a password or a second factor** — a new device, a policy
  change, a revoked grant.
- **Google asks to grant the application access.** Choosing among accounts
  already signed in is mechanical; granting access is a decision, and this does
  not make it on anybody's behalf.
- **The named account is not signed in to the profile**, or several are and none
  was named.

In all three the previous schedule stays exactly where it is, with its age, and
the run is told it is stale. The answer is always the same one command:

```bash
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --user-data-dir=~/.kairos/kway-profile https://app.kway.co.il/dashboard
```

## The channel's name is load-bearing

The competitor schedule joins to history **by channel name**. A name that is
close but not equal — a stray space, an invisible direction mark, `קשת` without
its number — produces a file that loads cleanly, validates cleanly, and
contributes exactly zero, because the audience lookup finds no history under it.
`future_epg` gives an unmatched rival 0.0 strength, correctly; nothing
downstream can tell that channel from a mistyped one.

So the name is resolved against the engine's own registry
(`kairos.data.loaders.CHANNELS`) before anything is pulled, spacing and direction
marks are treated as presentation, and a name matching **more than one** channel
is refused rather than resolved to the first. Filing the rival's schedule under
the **operator's own** channel is refused too: the counter-programming features
drop the operator's channel from the rival list, so that file would leave no
rivals at all and every adjustment would silently become zero.

## What never happens

A failed pull never looks like a successful one. No session, network down, shape
changed, publication empty — each ends as `refreshed: False` with a reason and
the previous schedule untouched. An empty pull in particular is treated as a
failure, because "the rival airs nothing" is a claim no publication has ever
actually made.
