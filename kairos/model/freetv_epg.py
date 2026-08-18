"""Every rival's schedule from one publication, without a credential.

The Keshet feeder that came first needed a signed-in session, a browser profile
and a PKCE handshake, and it works. It also solved one channel. The optimizer
reads the whole competitive lineup out of one file, so three quarters of that
file stayed empty while the machinery for the last quarter grew.

This is the other four-fifths, and it turned out to be simpler than the one.
FreeTV publishes a live-channel list and a programme feed for each of them, over
plain HTTPS, with no account. Measured on all four channels this engine has
history for — כאן 11, קשת 12, רשת 13, עכשיו 14 — one shape, one converter, and
every record dated at both ends.

WHY THIS ONE AND NOT THE CHANNELS' OWN SITES
--------------------------------------------
Each broadcaster publishes its own grid and each publishes it differently, and
the differences are all in the one field that matters. Measured:

* ``c14.co.il`` gives fifteen days in a single call, and its end time is a BARE
  CLOCK whose date is only the key of the object around it. A programme running
  to 00:20 ends before it starts unless the converter knows to roll the day.
* ``13tv.co.il`` gives a title and a start and NO end and NO duration at all, so
  a length can only be inferred from the next programme's start — which silently
  swallows anything the editorial grid leaves out.
* Both refuse a browser user-agent and answer a command-line one, which is the
  opposite of what a person would guess and the opposite of each other.

FreeTV's ``since`` and ``till`` are fully dated ISO instants. The duration is a
subtraction, the midnight case cannot arise, and one converter serves every
channel. Where a channel's own site is richer this loses a little; what it wins
is that the same code is correct for all of them.

WHAT IS PAID FOR THAT
---------------------
One request per day per channel — a two-day span answers 400
``LIVE_PROGRAMME_INVALID_TIMESPAN``, measured, so the loop is not an oversight.
The horizon runs at least nine days out, thinning as it goes, which is more
than a weekly plan needs.

THE CHANNEL NUMBER IS CHECKED, NOT TRUSTED
-------------------------------------------
FreeTV names its channels "ערוץ 11", "ערוץ 12" — not the names this engine's
history uses, and not names that can be mapped by pattern without deciding that
"ערוץ 13" means רשת 13. That decision is written down below and VERIFIED against
the publication before every pull: the live id must still carry the title it
carried when the mapping was written, or the pull refuses. A channel id that
gets reused for a different station would otherwise file a rival's whole
schedule under another rival's name, and nothing downstream could tell.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from kairos.data.title_features import series_join_key

LIVES_URL = "https://web.freetv.tv/api/products/lives?platform=BROWSER&lang=HEB"
PROGRAMMES_URL = (
    "https://web.freetv.tv/api/products/lives/programmes"
    "?liveId%5B%5D={live_id}&since={since}&till={till}&lang=HEB&platform=BROWSER"
)

# A default user-agent is refused by two of the publications this module was
# measured against, and this one does not care. Naming the caller is the polite
# thing to do when reading somebody else's feed on a schedule.
USER_AGENT = "kairos-competitor-feed/1.0 (+broadcast planning)"

# Every clock a broadcaster publishes is local, and Israel is UTC+2 in winter
# and UTC+3 in summer. This feed's instants are UTC, so the conversion is real
# and a fixed offset would be wrong for half of every year.
BROADCAST_TZ = ZoneInfo("Asia/Jerusalem")

# This engine's channel name -> (FreeTV live id, the title FreeTV publishes).
# The title is not decoration: it is what makes the id checkable.
CHANNELS: dict[str, tuple[int, str]] = {
    "כאן 11": (3370462, "ערוץ 11"),
    "קשת 12": (3340020, "ערוץ 12"),
    "רשת 13": (3328044, "ערוץ 13"),
    "עכשיו 14": (3457869, "ערוץ 14"),
}


class FreeTvError(RuntimeError):
    """The publication could not be read, or did not say what was expected."""


def _get(url: str, timeout: float = 45.0) -> Any:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read()[:160].decode("utf-8", "replace")
        raise FreeTvError(f"{url.split('?')[0]} answered {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise FreeTvError(f"the publication could not be reached ({exc.reason})") from exc


def live_channels() -> dict[int, str]:
    """Every live channel the publication currently lists, id -> title."""
    payload = _get(LIVES_URL)
    items = payload if isinstance(payload, list) else (
        payload.get("data") or payload.get("items") or [])
    if not items:
        raise FreeTvError("the publication listed no live channels at all")
    return {int(item["id"]): str(item.get("title") or "") for item in items if item.get("id")}


def verify_channel(channel: str, listed: Optional[dict[int, str]] = None) -> int:
    """The live id for one of this engine's channels, checked against the source.

    Refuses rather than guesses. An id whose title has changed is the one
    failure that cannot be caught downstream: the rows would be well-formed,
    the file would load, and a rival's evening would be filed under a different
    rival's name for as long as nobody noticed.
    """
    if channel not in CHANNELS:
        raise FreeTvError(
            f"no FreeTV channel is mapped for {channel}. Mapped: {', '.join(CHANNELS)}")
    live_id, expected = CHANNELS[channel]
    titles = live_channels() if listed is None else listed
    found = titles.get(live_id)
    if found is None:
        raise FreeTvError(
            f"FreeTV no longer lists channel {live_id}, which was {channel} ({expected})")
    if found.strip() != expected:
        raise FreeTvError(
            f"FreeTV channel {live_id} is now '{found.strip()}' and was '{expected}' "
            f"when it was mapped to {channel}. Refusing to file its schedule under "
            f"a name it may no longer be."
        )
    return live_id


def _stamp(when: datetime) -> str:
    """The publication's own date format, URL-encoded, in broadcast local time."""
    local = when.astimezone(BROADCAST_TZ)
    offset = local.strftime("%z")
    return f"{local:%Y-%m-%dT%H:%M}%2B{offset[1:]}" if offset.startswith("+") else \
           f"{local:%Y-%m-%dT%H:%M}-{offset[1:]}"


def fetch_day(live_id: int, day: date) -> list[dict[str, Any]]:
    """One broadcast day for one channel. One request, because two answer 400."""
    start = datetime(day.year, day.month, day.day, tzinfo=BROADCAST_TZ)
    payload = _get(PROGRAMMES_URL.format(
        live_id=live_id, since=_stamp(start), till=_stamp(start + timedelta(days=1))))
    if not isinstance(payload, list):
        raise FreeTvError(f"the programme feed for {live_id} was not a list of programmes")
    return payload


def fetch(channel: str, *, days: int = 8, start: Optional[date] = None,
          verify: bool = True) -> list[dict[str, Any]]:
    """A window of days for one channel, as the publication returns them.

    A day that fails is not silently skipped: the whole pull raises, because a
    schedule with a hole in it is worse than no schedule — the hole reads as
    "the rival broadcasts nothing that day", which is a claim no publication
    made.
    """
    live_id = verify_channel(channel) if verify else CHANNELS[channel][0]
    first = start or datetime.now(BROADCAST_TZ).date()
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for offset in range(max(1, days)):
        for record in fetch_day(live_id, first + timedelta(days=offset)):
            # A programme straddling midnight is returned by both days it
            # touches. It is one broadcast and belongs in the file once.
            key = (record.get("id"), record.get("since"))
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
    return out


# ------------------------------------------------------------ the conversion

def _instant(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(BROADCAST_TZ)


def _flag(record: dict[str, Any], name: str) -> bool:
    """One named boolean, and only a boolean.

    Written narrowly on purpose. A generous version of this — "any of these
    fields that looks true" — marked every broadcast as live, because the field
    called ``live`` is not a flag at all but an OBJECT naming the channel
    (``{"type_": "LIVE", "id": 3370462}``), and a non-empty dict is truthy. The
    real flag is ``liveBroadcast``, measured at 30 true of 82.
    """
    value = record.get(name)
    return value is True or value == 1


def to_contract_rows(records: Iterable[dict[str, Any]], *, channel: str
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The publication's records as the competitor contract this engine reads.

    Same columns and same conventions as :mod:`kairos.model.keshet_epg`, so a
    row pulled from here and a row pulled from Keshet's own publication are
    indistinguishable downstream — which is the whole point of one file.

    A record that cannot be placed in time is COUNTED and reported, never
    dropped in silence: the difference between "the rival aired nothing then"
    and "we could not read one line" is the difference between a decision and a
    guess.
    """
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        since = _instant(record.get("since"))
        till = _instant(record.get("till"))
        title = str(record.get("title") or "").strip()
        if since is None:
            dropped.append({"title": title, "reason": "unreadable start clock"})
            continue
        if till is None or till <= since:
            dropped.append({"title": title,
                            "reason": "the end is missing or not after the start"})
            continue
        rows.append({
            "Channel": channel,
            "Title": title,
            "Date": f"{since:%d/%m/%Y}",
            "Start time": f"{since:%H:%M:%S}",
            "End time": f"{till:%H:%M:%S}",
            "Duration": int((till - since).total_seconds()),
            # liveBroadcast, not "live" — see _flag. And ``repeat`` is
            # corroborated by the broadcaster's own title marker: measured on a
            # real window, 40 of 40 titles carrying "(ש.ח.)" have repeat true
            # and 0 of 42 without it do.
            "Live": _flag(record, "liveBroadcast"),
            "Rerun": _flag(record, "repeat"),
            "ProgramCode": str(record.get("id") or ""),
            "HouseNumber": "",
            "Description": str(record.get("description") or record.get("lead") or "").strip(),
            # The identity a future programme is found by. Written into the file
            # rather than recomputed by each reader, so the join is auditable and
            # cannot drift between the three places that need it.
            "SeriesKey": series_join_key(title),
        })

    rows.sort(key=lambda r: (r["Date"].split("/")[::-1], r["Start time"]))
    days = sorted({r["Date"] for r in rows})
    return rows, {
        "records_in": len(rows) + len(dropped),
        "dropped": dropped,
        "days": days,
        "window_start": days[0] if days else "",
        "window_end": days[-1] if days else "",
        "channel": channel,
        "source": "freetv",
    }


def pull_rows(channel: str, *, days: int = 8, start: Optional[date] = None
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch and convert in one call, for a caller that only wants the rows."""
    return to_contract_rows(fetch(channel, days=days, start=start), channel=channel)
