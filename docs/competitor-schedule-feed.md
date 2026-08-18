# The rival's schedule, pulled instead of typed

Every competitor input this engine has ever had arrived as a file somebody put on
disk by hand — which in practice means most weeks had none, because nobody
re-types a rival's schedule. This feed replaces that with a daily pull, and
attaches it to the contract the engine already reads.

Run it before the daily simulation:

```bash
python -m kairos.model.keshet_feed --operator-channel "רשת 13"
```

It prints one line and exits 0 on a refresh, 1 on anything else:

```
נמשך לוח המתחרים לראשונה: 300 שידורים על פני 13 ימים.
לוח המתחרים רוענן: אין שינוי מאז הפעם הקודמת.
לוח המתחרים רוענן: 2 הוזזו, 1 הוכרזו.
לוח המתחרים לא רוענן. הלוח שבידינו בן 26.0 שעות.
```

`--json` gives the whole result, including the per-programme diff.

## What it writes, and what reads it

`data/reference/CompetitorProgrammes.csv` — the file
`kairos.model.future_epg` already looks for. Nothing downstream changed: same
columns, same loader, same honest-absence behaviour when it is missing. The file
is **not** in git: it is a licensed third party's data, it is rewritten daily,
and its absence costs nothing but freshness.

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
