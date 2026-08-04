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

**The chip beside the sentence is rendered at the read for the same reason the
sentence is.** A refusal about no column carries the place it IS about, and the
door has not always sent one: the report on disk today carries the internal key
``channels`` where a column name goes, so the dayparts card printed it as a bold
Latin chip on a Hebrew screen before anything was clicked. A stored key that the
door no longer writes is derived again from the finding's own code, which closes
every report already on disk and not only the ones written after it.
"""

from __future__ import annotations

from typing import Any

from kairos.data.loaders import CHANNELS
from kairos_api import uploads_channels, uploads_messages

__all__ = ["boundary", "channel_fields", "channel_key", "place", "rendered", "to_store"]

# The facts of a report that are the file's own and carry no sentence at all.
FACTS = ("dataset", "filename", "checked_at", "accepted", "is_valid", "rows_loaded")

# The row numbers a finding names travel with it unrendered: they are positions
# in the operator's own file, and no boundary applies to a number.
ROWS = ("rows", "rows_total")

# What a finding carries that is not a sentence and is not a number. ``scope``
# is what a finding about no column is about, and it survives the round trip
# because a stored report renders the same chip a live one does.
KEPT = ("column", "scope", "code", "severity")

# Where a refusal this destination raises is about, decided by its own code.
# Every code here is about no column at all: the file, its header row, or the
# table the loader parsed it into.
#
# **The place is rendered at the read, exactly as the sentence is.** Measured on
# the shipped card before this: the stored report on disk carries
# ``column: "channels"`` on the channel refusal, written before the door emptied
# that key, and ``channels`` is a column no dayparts export has. The read
# replayed it verbatim, so the dayparts card printed a bold Latin ``channels``
# chip on a Hebrew screen on first load with nothing clicked, and the flat line
# the assistant's read tool parses read ``[error] channels:``. Deriving the
# place from the code closes it for every report already on disk rather than
# only for the ones written next.
PLACE: dict[str, str] = {
    "unreadable_file": "file",
    "empty_file": "file",
    "too_large": "file",
    "missing_columns": "header",
    "no_recognized_channel_columns": "header",
    "no_recognized_channel_columns_unset": "header",
    "no_data_rows": "frame",
}

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


def place(finding: Any) -> tuple[str, str]:
    """The column and the scope one finding prints, with its own code deciding.

    A code in :data:`PLACE` is about no column, so the answer is that scope and
    an empty column whatever an older store put in the key. Any other code keeps
    exactly what was stored, which is what leaves a real column name, and the
    frozen contracts' own violations, precisely as they were.
    """
    if not isinstance(finding, dict):
        return "", ""
    scope = PLACE.get(str(finding.get("code") or ""))
    if scope:
        return "", scope
    return str(finding.get("column") or ""), str(finding.get("scope") or "")


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
    record = {key: finding.get(key) for key in KEPT if key in finding}
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
    """One finding as the two sentences a surface prints, both inside the boundary.

    The chip beside those sentences is rendered here too, from the code, which
    is what keeps a report stored under an older door printing the word this
    door prints now instead of a key no file of that kind carries.
    """
    if not isinstance(finding, dict):
        return {}
    record = {key: finding.get(key) for key in KEPT if key in finding}
    record["column"], scope = place(finding)
    if scope:
        record["scope"] = scope
    else:
        record.pop("scope", None)
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


def _flat(findings: list[dict[str, Any]], severity: str, owned: str) -> list[str]:
    """The findings of one severity, flattened, and swept once more.

    The line itself is :func:`uploads_messages.flat_finding`, which the live
    payload is built with too, so the line a stored report prints and the line a
    live one prints are the same line. The sweep runs on the assembled line and
    not only on the sentence inside it, because a line also carries the column a
    violation is about, and a column of a channel export is a channel name.
    """
    return _swept([uploads_messages.flat_finding(finding) for finding in findings if finding.get("severity") == severity], owned)
