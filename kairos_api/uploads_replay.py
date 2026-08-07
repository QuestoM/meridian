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

__all__ = ["at_the_door", "boundary", "channel_fields", "channel_key", "place", "rendered", "to_store"]

# The facts of a report that are the file's own and carry no sentence at all.
FACTS = ("dataset", "filename", "checked_at", "accepted", "is_valid", "rows_loaded")

# The row numbers a finding names travel with it unrendered: they are positions
# in the operator's own file, and no boundary applies to a number.
ROWS = ("rows", "rows_total")

# What a finding carries that is not a sentence and is not a number. ``scope``
# is what a finding about no column is about, and it survives the round trip
# because a stored report renders the same chip a live one does.
KEPT = ("column", "scope", "code", "severity")

# Every key a finding carries a sentence in, and the language each is read in.
# ``message`` is the sentence for a code this destination authors and the frozen
# contract's own detail for a code it does not; ``message_en`` is the English
# half authored beside that frozen detail, and ``message_he`` the Hebrew.
#
# **The boundary runs over all three at once and withholds all three together.**
# They are one finding said three ways, so a name that may not be read in one of
# them may not be read in any: an ``unknown_channel`` violation carries the
# file's own channel names into the frozen English AND into the authored English
# beside it, and withholding two of three would leave the third on the card.
SENTENCES = {"message": "en", "message_en": "en", "message_he": "he"}

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
# may not read. Split in two because the two callers below know two different
# things and neither may claim the other's. A stored report cannot tell whether
# the withheld sentence was recorded under a different channel or is a rival
# name inside the file's own content, which its wording used to claim without
# ever checking; a live check knows for certain it is the second, because
# nothing has been written yet, and its own remedy may not be "check the file
# again", which is the read that produced this sentence in the first place.
WITHHELD_STORED = {
    "en": "This stored reason names a channel this account may not read, so it is withheld. Upload the file again to read a reason checked under this account.",
    "he": "הסיבה השמורה הזו נושאת שם של ערוץ שאסור להציג לחשבון הזה, ולכן היא אינה מוצגת. העלו את הקובץ שוב כדי לקרוא סיבה שנבדקה תחת החשבון הזה.",
}

WITHHELD_LIVE = {
    "en": "This reason names a channel this account may not read, found in the file just checked, so it is withheld.",
    "he": "הסיבה הזו נושאת שם של ערוץ שאסור להציג לחשבון הזה, שנמצא בקובץ שנבדק זה עתה, ולכן היא אינה מוצגת.",
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
        # The pair authored beside it is quoted for the same reason. It was built
        # from a count measured on a frame this read no longer holds, so it is
        # kept as written rather than recomputed from nothing.
        record["message"] = finding.get("message")
        for key in ("message_en", "message_he"):
            if finding.get(key):
                record[key] = finding[key]
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
        texts = {"message": english, "message_he": hebrew}
    elif finding.get("key"):
        english, hebrew = uploads_messages.say(finding["key"], **{name: _live_value(value) for name, value in (finding.get("fields") or {}).items()})
        texts = {"message": english, "message_he": hebrew}
    else:
        # Sentences nothing here can re-author: the frozen contract's own detail,
        # the pair authored beside it, or a report stored before this format
        # existed. Quoted, every half, and swept as one.
        texts = {key: str(finding.get(key) or "") for key in SENTENCES if key == "message" or finding.get(key)}
    swept = _inside_the_boundary(texts, owned)
    record["message"] = swept.get("message", "")
    for key in ("message_en", "message_he"):
        if swept.get(key):
            record[key] = swept[key]
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


def _inside_the_boundary(texts: dict[str, str], owned: str, notice: dict[str, str] = WITHHELD_STORED) -> dict[str, str]:
    """Every half of one finding as it may be printed, or the notice replacing them.

    ``texts`` is keyed by :data:`SENTENCES`, and a rival name in any half
    withholds every half, because they are one finding said more than one way.

    ``notice`` is which of the two withheld sentences this call may honestly
    print, decided by the caller because only the caller knows which cause it
    measured: :func:`rendered` reading a stored report passes the default,
    :func:`at_the_door` sweeping a live one passes :data:`WITHHELD_LIVE`.
    """
    if any(_names_a_rival(text, owned) for text in texts.values()):
        return {key: notice[SENTENCES[key]] for key in texts}
    return dict(texts)


def _swept(lines: Any, owned: str, notice: dict[str, str] = WITHHELD_STORED) -> list[str]:
    """A flat list with every line that names a rival channel replaced."""
    return [
        notice["en"] if _names_a_rival(line, owned) else str(line)
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


def at_the_door(findings: list[dict[str, Any]], owned: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """A live payload's own findings, swept before they ever reach a screen or disk.

    The measured gap this closes: ``message`` and ``message_he`` used to leave
    :func:`kairos_api.uploads_validate.run_contract_validation` unswept, so a
    frozen contract's own detail naming a rival channel reached the door's
    response and, from there, whatever :func:`kairos_api.uploads_validate.store_report`
    went on to persist, while a stored report already read back through
    :func:`rendered` had been swept for years. One finding, shown two ways.

    This runs the same two functions :func:`rendered` uses, once, here, so a
    finding cannot leave this door carrying a name it could not leave the store
    carrying either: what is swept here is what gets persisted, so there is only
    one boundary pass to keep honest instead of two that can drift apart. The
    flat ``errors`` and ``warnings`` a caller wants beside the findings are
    rebuilt from the swept copies rather than swept a second time on their own
    text, so the sentence a finding carries and the line built from it can
    never disagree.
    """
    swept_findings: list[dict[str, Any]] = []
    for finding in findings:
        record = dict(finding)
        texts = {key: str(record.get(key) or "") for key in SENTENCES if key == "message" or record.get(key)}
        swept = _inside_the_boundary(texts, owned, WITHHELD_LIVE)
        record["message"] = swept.get("message", "")
        for key in ("message_en", "message_he"):
            if swept.get(key):
                record[key] = swept[key]
            else:
                record.pop(key, None)
        swept_findings.append(record)
    errors = [uploads_messages.flat_finding(f) for f in swept_findings if f.get("severity") == "error"]
    warnings = [uploads_messages.flat_finding(f) for f in swept_findings if f.get("severity") == "warning"]
    return swept_findings, errors, warnings
