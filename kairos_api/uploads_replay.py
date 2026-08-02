"""The stored validation report: kept as codes, rendered when it is read.

Split out of ``uploads_validate.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule the package already follows.

**A refusal was written once and replayed for ever, with the channel name inside
it frozen at the moment it was written.** Measured on the shipped surface with
the operator channel ``עכשיו 14``: a dayparts refusal stored earlier, while
``רשת 13`` was the configured channel, came back through
``GET /api/uploads/status`` three times, in ``errors[0]``, in
``findings[0].message`` and in ``findings[0].message_he``, and the source card
printed it inside its own last-checked block as this account's "own channel".
The boundary was applied where the sentence was written and never where it was
read, so the account reading it was shown a name it does not own.

The fix is the store format. A report goes to disk as the finding's own code,
the copy table's key and the measured fields the sentence was rendered from,
never as the sentence, and every read renders it again from those against the
channel this account owns NOW. The numbers in a rendered sentence stay the ones
measured on the file; only the channel-dependent part is re-derived, by the same
function that derived it the first time, so the write and the read cannot drift.

Two things cannot be re-rendered, and both are handled rather than assumed away.
A violation the frozen :mod:`kairos.data.contracts` raised carries its own
English detail and no code this module could render from, so that sentence is
quoted and then swept: one that names a channel this account may not read is
withheld with a sentence saying so and what to do about it, rather than dropped,
because a refusal that quietly loses a reason is worse than one that says a
reason is being withheld. A report written before this format existed is the
same case, and the two on disk today are exactly it.

The sweep runs over every string a read returns, authored or quoted, so a fifth
way of carrying a name into one of these payloads is withheld and not printed.
"""

from __future__ import annotations

from typing import Any

from kairos.data.loaders import CHANNELS
from kairos_api import uploads_channels, uploads_messages

__all__ = ["boundary", "channel_fields", "channel_key", "rendered", "to_store"]

# The facts of a report that are the file's own and carry no sentence at all.
FACTS = ("dataset", "filename", "checked_at", "accepted", "is_valid", "rows_loaded")

# The row numbers a finding names travel with it unrendered: they are positions
# in the operator's own file, and no boundary applies to a number.
ROWS = ("rows", "rows_total")

# The marker that says this record holds codes rather than sentences. A report
# without it was written before this format and is quoted and swept instead.
FORMAT_KEY = "renders_at_read"

# What a reader is told in place of a sentence that names a channel this account
# may not read. It names what is missing and the one act that supplies it, which
# is the door this destination is built around.
WITHHELD = {
    "en": "This reason was recorded while another operator channel was configured and it names a channel this account may not read, so it is withheld. Check the file again to read the reason under this account.",
    "he": "הסיבה הזו נרשמה כשהיה מוגדר ערוץ מפעיל אחר, והיא נושאת שם של ערוץ שאסור להציג לחשבון הזה, ולכן היא אינה מוצגת. בדקו את הקובץ שוב כדי לקרוא את הסיבה תחת החשבון הזה.",
}


def boundary(names: Any, owned: str) -> dict[str, Any]:
    """The raw material of the channel refusal, already inside the boundary.

    The names a reader may see and the count of the ones they may not, which is
    what goes to disk: the sentence built from them is not stored, because the
    account that reads it next may own a different channel.
    """
    shown, withheld = uploads_channels.withhold(names, owned)
    return {"names": shown, "withheld": int(withheld)}


def channel_key(owned: str) -> str:
    """Which of the two channel refusals this account gets, from its own channel.

    With no channel configured there is no name to check a header against, so
    the refusal says where to set one instead of naming anybody.
    """
    return "no_recognized_channel_columns" if owned else "no_recognized_channel_columns_unset"


def channel_fields(bound: dict[str, Any], owned: str) -> dict[str, Any]:
    """The fields of the channel refusal, resolved against one owner's boundary.

    Called at the moment the refusal is written and again at every read. The
    withheld count is the stored one plus whatever this account may not read on
    top of it, so the arithmetic stays the file's own however often it is read.
    """
    shown, more = uploads_channels.withhold(bound.get("names") or [], owned)
    withheld = int(bound.get("withheld") or 0) + int(more)
    listed = ", ".join(shown)
    found = (listed, listed) if shown else uploads_messages.say("no_columns_found")
    clause = uploads_messages.say("withheld_columns", withheld=withheld) if withheld else ("", "")
    return {"count": len(CHANNELS), "owned": owned, "found": found, "clause": clause}


def to_store(payload: Any) -> Any:
    """The report as it goes to disk: codes, keys and measured fields.

    The flat ``errors`` and ``warnings`` lists are not stored either. They are
    the findings restated one severity at a time, and a rendered sentence is
    exactly what may not be kept, so they are rebuilt on the way out.
    """
    if not isinstance(payload, dict):
        return payload
    stored = {key: payload.get(key) for key in FACTS}
    stored["findings"] = [_to_store_finding(finding) for finding in payload.get("findings") or []]
    stored[FORMAT_KEY] = True
    return stored


def rendered(stored: Any, owned: str) -> Any:
    """A stored report as sentences again, against the channel owned right now."""
    if not isinstance(stored, dict):
        return stored
    findings = [_rendered_finding(finding, owned) for finding in stored.get("findings") or []]
    payload: dict[str, Any] = {key: stored.get(key) for key in FACTS}
    if stored.get(FORMAT_KEY):
        payload["errors"] = _flat(findings, "error", owned)
        payload["warnings"] = _flat(findings, "warning", owned)
    else:
        payload["errors"] = _swept(stored.get("errors"), owned)
        payload["warnings"] = _swept(stored.get("warnings"), owned)
    payload["findings"] = findings
    return payload


def _to_store_finding(finding: Any) -> dict[str, Any]:
    """One finding, with its sentence replaced by what the sentence was made of."""
    if not isinstance(finding, dict):
        return {}
    record = {key: finding.get(key) for key in ("column", "code", "severity") if key in finding}
    for key in ROWS:
        if key in finding:
            record[key] = finding[key]
    if finding.get("boundary") is not None:
        # Both the key and the fields of this one come from the boundary as it
        # stands at the read, so neither is stored: what is stored is the
        # material they are derived from.
        record["boundary"] = finding["boundary"]
    elif finding.get("key"):
        record["key"] = finding["key"]
        record["fields"] = {name: _stored_value(value) for name, value in (finding.get("fields") or {}).items()}
    else:
        # The frozen contracts' own sentence, quoted: the counts and column names
        # inside it are theirs to compute and no code here could re-author them.
        record["message"] = finding.get("message")
        if finding.get("message_he"):
            record["message_he"] = finding["message_he"]
    return record


def _rendered_finding(finding: Any, owned: str) -> dict[str, Any]:
    """One finding as the two sentences a surface prints, both inside the boundary."""
    if not isinstance(finding, dict):
        return {}
    record = {key: finding.get(key) for key in ("column", "code", "severity") if key in finding}
    if finding.get("boundary") is not None:
        english, hebrew = uploads_messages.say(channel_key(owned), **channel_fields(finding["boundary"], owned))
    elif finding.get("key"):
        english, hebrew = uploads_messages.say(finding["key"], **{name: _live_value(value) for name, value in (finding.get("fields") or {}).items()})
    else:
        # A sentence nothing here can re-author: the frozen contracts' own, or
        # one stored before this format existed. Quoted, both halves, and swept.
        english, hebrew = str(finding.get("message") or ""), str(finding.get("message_he") or "")
    english, hebrew = _inside_the_boundary(english, hebrew, owned)
    record["message"] = english
    if hebrew:
        record["message_he"] = hebrew
    for key in ROWS:
        if key in finding:
            record[key] = finding[key]
    return record


def _stored_value(value: Any) -> Any:
    """A field on its way to disk, with a two-language fragment named as one.

    A fragment assembled before its sentence travels as a pair, and a pair
    written as a list would read back as a list of two strings with nothing
    saying which is which. On disk it is a record with the two language keys.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return {"en": str(value[0]), "he": str(value[1])}
    return value


def _live_value(value: Any) -> Any:
    """The same field on its way back, as the pair the copy table renders from."""
    if isinstance(value, dict) and set(value) == {"en", "he"}:
        return (value["en"], value["he"])
    return value


def _names_a_rival(text: Any, owned: str) -> bool:
    """Whether a sentence carries a channel this account may not read."""
    body = str(text or "")
    return any(rival in body for rival in uploads_channels.rivals(owned))


def _inside_the_boundary(english: str, hebrew: str, owned: str) -> tuple[str, str]:
    """The pair as it may be printed: the sentence, or the notice replacing it."""
    if _names_a_rival(english, owned) or _names_a_rival(hebrew, owned):
        return WITHHELD["en"], WITHHELD["he"]
    return english, hebrew


def _swept(lines: Any, owned: str) -> list[str]:
    """A stored flat list with every line that names a rival channel replaced."""
    return [
        WITHHELD["en"] if _names_a_rival(line, owned) else str(line)
        for line in (lines or [])
    ]


def _line(finding: dict[str, Any]) -> str:
    """One finding as the flat string every existing reader of this report parses."""
    return f"[{finding.get('severity')}] {finding.get('column')}: {finding.get('code')} - {finding.get('message')}"


def _flat(findings: list[dict[str, Any]], severity: str, owned: str) -> list[str]:
    """The findings of one severity, flattened, and swept once more.

    The sweep runs on the assembled line and not only on the sentence inside it,
    because a line also carries the column a violation is about, and a column of
    a channel export is a channel name.
    """
    return _swept([_line(finding) for finding in findings if finding.get("severity") == severity], owned)
