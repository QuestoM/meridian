"""The state of every input a run reads, split out of ``uploads.py``.

Split under the file-size cap and named by the ``<parent stem>_<role>.py``
rule. Every function here is pure: it is handed the resolvers and the paths it
needs, so ``uploads.py`` keeps sole ownership of the writable locations and a
test that relocates them still relocates everything.

Three things this module adds to the status a source card used to carry.

**One closed vocabulary for the state.** Six words, and every one of them names
its own remedy on the same record: ``in_use``, ``shadowed``, ``not_read``,
``empty``, ``invalid``, ``missing``. A surface that renders a state without its
remedy has left the reader to work out what to do.

The state is derived from three things and never from the read path alone,
because a read path answers where a file is opened and not whether anything
came of it. **Content**: a file the engine reads that carries a header and zero
data rows is ``empty``, never ``in_use`` with nothing to do, since no figure
anywhere can come from it. **Siblings**: a kind that has other files on disk
that the engine does not read names them, and when one of them arrived after
the file the engine reads, the kind is ``shadowed``, because the operator's own
most recent act is not what any number rests on.

**The consequence of an upload, before it happens.** Seven codes, derived from
the real read paths rather than from optimism: ``replaces_live_input`` (the
engine reads this file, so a new one changes what the plan is computed from),
``changes_model_basis`` (the engine reads it AND the model version was measured
on it, so a new file makes that measurement stale), ``stored_not_read`` (the
file is kept and validated and nothing reads it),
``replaces_only_a_later_day`` (the engine reads this kind out of a directory by
the airing date in the filename, so whether an upload here replaces anything
depends on the day its own name carries), ``stored_without_replacing``,
which asks the third one about a named candidate rather than about a kind, so
it can name the file that will be read instead, and
``replaces_live_input_with_no_rows``, which is the first one asked about a
candidate that carries no data rows: it wins the read path and replaces the
live input with nothing, which is the one outcome the first sentence reads as
good news over. The seventh is that same shape once more, one step less
extreme: ``replaces_live_input_with_warnings`` is a candidate the engine will
read that carries a warn-severity finding, and it names how many and the field
they are about, because a file whose every row lost its clock is that same loss.

**A remedy and a consequence on one card may not disagree**, which is what the
fourth code exists for. Measured on the shipped card before it: a shadowed daily
input printed "keep uploading here and the plan will not change" and, one line
below it, "this is the live input, uploading replaces what the plan is computed
from". Both sentences were reachable because the state was derived from the
siblings and the consequence from the read path alone, and neither of them named
the airing date that decides between them. The pair is now derived together, and
:mod:`tests.test_p6_state` sweeps every card for a pair that carries both
claims.

**And a remedy may not disagree with the file's own last check.** ``in_use`` is
the right word for a live file the engine reads that carries rows, and "nothing
to do" is the wrong sentence over one whose last check came back with a warning.
Measured: committing the daily log whose 20 of 20 rows carried an unreadable
clock left exactly that card. The state is unchanged, because the engine does
read that file; the remedy names the count and the field instead.

Every sentence any of this is said in lives in :mod:`kairos_api.uploads_remedy`,
split there at the 450-line law and by subject: derivation here, words there.

**The model version and its tri-state freshness** is
:mod:`kairos_api.uploads_model`, read here for one fact only: whether the file
this kind is read from is one the version was measured on.

The row count is read through the shared fingerprinted cache, keyed on the
file's own ``(path, mtime_ns, size)``. Measured on the shipped data: counting
rows costs 98 ms per call across the seven inputs, of which the 15.7 MB spots
file is 67 ms, and every call re-read all of it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from kairos_api import (
    read_cache,
    uploads_channels,
    uploads_checks,
    uploads_model,
    uploads_remedy,
    uploads_replay,
)

# The per-file shape, keyed on file signatures, so a changed file is a changed
# fingerprint and a stale answer cannot be served.
SHAPE_NAMESPACE = "uploads.file_shape"

# Seven inputs, and headroom, so the least recently used one is never the one
# the next call asks for.
read_cache.configure(SHAPE_NAMESPACE, capacity=16)

# The six states an input can be in. A surface renders one of these and never
# invents a seventh.
STATES = ("in_use", "shadowed", "not_read", "empty", "invalid", "missing")

# The seven things an upload actually does. The first four are about a kind and
# the last three are about one named file, which is not the same question.
CONSEQUENCES = (
    "replaces_live_input",
    "changes_model_basis",
    "stored_not_read",
    "replaces_only_a_later_day",
    "stored_without_replacing",
    "replaces_live_input_with_no_rows",
    "replaces_live_input_with_warnings",
)

# The most stored-but-unread files one input names. The daily directory grows by
# one file a day forever, so the list is capped and the true count travels with
# it; a cap that hides a number is how a card starts lying by omission.
STORED_LIST_CAP = 6

# Every sentence the eight remedies, the seven consequences and the two stored
# reasons are said in lives in :mod:`kairos_api.uploads_remedy`, which this
# module reads and never re-authors. The 450-line law forced that split and the
# subject wanted it: what state an input is in is derived here, and the words it
# is read in are there.


def file_shape(
    path: Path,
    reader: Callable[[Path], tuple[list[str], int, list[str]]],
) -> tuple[list[str], int, list[str]]:
    """``(columns, rows, warnings)`` for one file, cached on its signature.

    The value is immutable to the caller by contract: it is copied on the way
    out, because the cache shares values rather than copying them.
    """
    signature = read_cache.file_signature(path)
    columns, rows, warnings = read_cache.cached(
        SHAPE_NAMESPACE,
        str(path),
        signature,
        lambda: reader(path),
    )
    return list(columns), int(rows), list(warnings)


def naming(table: dict[str, dict[str, str]], code: str, live_name: str = "", **fields: object) -> dict[str, str]:
    """A bilingual pair that names a file, or a count and a field, in its sentence.

    The Hebrew sentence carries every named run inside an isolate, or a
    left-to-right name inside a right-to-left sentence drags that sentence's own
    full stop to the wrong side of it. The English one is left plain. A filename
    is always left-to-right and is isolated as one; any other field resolves its
    own direction from its first strong character. A field whose value is a
    two-item tuple carries its own two languages, which is how a word this
    module supplies itself stays in the language of the sentence around it.
    """
    words = table.get(code) or {"en": "", "he": ""}
    both = {name: value if isinstance(value, tuple) else (value, value) for name, value in fields.items()}
    plain = {"live": live_name, **{name: str(pair[0]) for name, pair in both.items()}}
    isolated = {"live": f"⁦{live_name}⁩", **{name: f"⁨{pair[1]}⁩" for name, pair in both.items()}}
    return {"code": code, "en": words["en"].format(**plain), "he": words["he"].format(**isolated)}


def stored_reason(code: str, live_name: str) -> dict[str, str]:
    """Why one stored file is not the one the engine reads, in both languages."""
    return naming(uploads_remedy.STORED_REASONS, code, live_name)


def state_for(
    *, exists: bool, valid: bool, rows: int, in_use: bool, unread: bool, stored_after: bool
) -> str:
    """The one word this input is in, from the state it is really in.

    Read state outranks content: a file nothing reads, or one another file is
    read instead of, is that first, and how many rows it carries is a second
    question. A file the engine really does read is where content decides, and
    a live file with zero data rows is ``empty`` rather than in use, because
    every figure downstream of it would be an honest nothing.
    """
    if not exists:
        return "missing"
    if not valid:
        return "invalid"
    if unread:
        return "not_read"
    if not in_use or stored_after:
        return "shadowed"
    return "empty" if rows <= 0 else "in_use"


def consequence_for(*, in_use: bool, model_input: bool, outranked: bool = False) -> str:
    """What an upload of this kind would actually do.

    ``outranked`` is the case the first three codes cannot say: the engine reads
    this kind out of a directory and the operator's own last upload lost that
    resolver, so what the next one does is decided by the day its name carries
    and not by the read path. It is answered before the two unconditional
    verdicts because both of them would be wrong here.
    """
    if not in_use:
        return "stored_not_read"
    if outranked:
        return "replaces_only_a_later_day"
    return "changes_model_basis" if model_input else "replaces_live_input"


def labelled(table: dict[str, dict[str, str]], code: str) -> dict[str, str]:
    """A bilingual pair for a code, as a record the surface reads directly."""
    words = table.get(code) or {"en": "", "he": ""}
    return {"code": code, "en": words["en"], "he": words["he"]}


def consequence_record(
    in_use: bool, reads: str | None, models_dir: Path, root: Path, still_read: str | None = None, rows: int | None = None, findings: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """What replacing this kind's file would do, in the operator's own terms.

    ``still_read`` names the file the engine would go on reading, and it turns a
    true but useless "nothing reads this" into the sentence that says what does.
    It is passed when the question is about one candidate file, which is the
    question whose answer a person is about to act on.

    ``rows`` is that candidate's own data row count, and it is the one fact that
    outranks every other answer here: a file the engine will read that carries
    no rows replaces the live input with nothing, and "this is the live input,
    uploading replaces what the plan is computed from" is a true sentence that
    reads as good news over it. It stays None when the question is about a kind
    rather than about a file, which is an honest unknown and not a zero.

    ``findings`` are that candidate's own, and a warn-severity one is the same
    lesson a row less extreme: the file loads, so nothing refuses it, and part
    of it reaches the engine unusable anyway. It outranks the two unconditional
    verdicts for the same reason the row count does, and it stays None when the
    question is about a kind, whose next file has not been looked at yet.
    """
    if in_use and rows is not None and rows <= 0:
        return labelled(uploads_remedy.CONSEQUENCES, "replaces_live_input_with_no_rows")
    count, fields = uploads_remedy.warned_summary(findings)
    if in_use and count:
        return naming(
            uploads_remedy.CONSEQUENCES,
            "replaces_live_input_with_warnings",
            warnings=uploads_remedy.warning_count(count),
            fields=fields,
        )
    measured_on = set(uploads_model.version(models_dir, root).get("measured_on") or [])
    code = consequence_for(in_use=bool(in_use), model_input=bool(reads and reads in measured_on))
    if code == "stored_not_read" and still_read:
        return naming(uploads_remedy.CONSEQUENCES, "stored_without_replacing", still_read)
    return labelled(uploads_remedy.CONSEQUENCES, code)


def checked_the_live_file(report: Any, live: Path | None, kind: str, destination: Callable[..., Path]) -> bool:
    """Whether the last check of this kind was of the file the engine reads now.

    The report carries the name the operator's own file arrived under, and the
    destination resolver decides what that name lands as: the daily kind
    normalises the prefix its own glob requires, and every other kind ignores
    the name and writes one path. So the question is answered by running that
    same resolver rather than by comparing two strings and hoping. A report from
    a refused upload is about no file on disk at all and answers no.
    """
    name = str((report or {}).get("filename") or "").strip()
    if live is None or not name or not (report or {}).get("accepted"):
        return False
    return destination(kind, name).name == live.name


def build(
    *,
    inputs: list[dict[str, str]],
    live_path: Callable[[str], Path | None],
    destination: Callable[..., Path],
    in_use: Callable[..., tuple[bool, str]],
    engine_reads: Callable[[str], str | None],
    relative: Callable[[Path], str],
    reader: Callable[[Path], tuple[list[str], int, list[str]]],
    missing_columns: Callable[[str, list[str]], list[str]],
    stored_unread: Callable[[str], list[dict[str, Any]]],
    unread_kinds: dict[str, str],
    validation_reports: dict[str, Any],
    models_dir: Path,
    root: Path,
    required_columns: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The whole status payload: one record per input, plus the model version."""
    model = uploads_model.version(models_dir, root)
    measured_on = set(model.get("measured_on") or [])
    # Resolved once for all seven inputs: a header naming a channel this
    # operator does not own is withheld from the payload and counted in it.
    owned = uploads_channels.owned_channel()
    entries: list[dict[str, Any]] = []
    for meta in inputs:
        kind = meta["kind"]
        path = live_path(kind)
        exists = bool(path and path.exists())
        kind_in_use, in_use_reason = in_use(kind)
        reads = engine_reads(kind)
        entry: dict[str, Any] = {
            "kind": kind,
            "label_en": meta["label_en"],
            "label_he": meta["label_he"],
            "cadence": meta["cadence"],
            "filename": path.name if path else destination(kind).name,
            "path": relative(path or destination(kind)),
            "exists": exists,
            "rows": 0,
            **uploads_channels.columns_record([], owned),
            "last_modified": None,
            "valid": False,
            "in_use": kind_in_use,
            "in_use_reason": in_use_reason,
            # The file the engine ACTUALLY reads for this kind right now, so a
            # shadowed or unread upload never has to be inferred from prose.
            "engine_reads": reads,
            # The last report from an upload of this kind, or None when nothing
            # was validated (honest unknown). Stored as codes and rendered here
            # against the channel owned now, which is ``uploads_replay``'s whole
            # subject: a sentence stored whole froze the channel of its writer.
            "last_validation": uploads_replay.rendered(validation_reports.get(kind), owned),
            "warnings": [],
            "size_bytes": 0,
            # True when the file the engine reads for this kind is one the model
            # version was measured on, so replacing it moves the model's basis.
            "model_input": bool(reads and reads in measured_on),
            # Exactly what the door runs on a file of this kind, and the short
            # list of things it genuinely cannot answer.
            "checks": uploads_checks.checks_for(kind, (required_columns or {}).get(kind, [])),
        }
        if exists and path is not None:
            columns, rows, read_warnings = file_shape(path, reader)
            missing = missing_columns(kind, columns)
            entry.update(uploads_channels.columns_record(columns, owned))
            entry["rows"] = rows
            # Local time WITH the UTC offset, so "when" is unambiguous.
            entry["last_modified"] = (
                datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
            )
            entry["size_bytes"] = int(path.stat().st_size)
            entry["valid"] = not missing
            warnings = list(read_warnings)
            if missing:
                warnings.insert(0, f"Missing required columns: {', '.join(missing)}")
            if not kind_in_use and in_use_reason:
                # Surface the shadow state in warnings too, so a dashboard that
                # only renders warnings still stops short of a bare green badge.
                warnings.insert(0, in_use_reason)
            entry["warnings"] = warnings
        # The files of this kind the engine does not read. The list is capped and
        # the total is not, so a cap never hides a count.
        stored = list(stored_unread(kind) or [])
        arrived_after = [record for record in stored if record.get("arrived_after_live")]
        entry["stored_unread"] = stored[:STORED_LIST_CAP]
        entry["stored_unread_total"] = len(stored)
        state = state_for(
            exists=exists,
            valid=bool(entry["valid"]),
            rows=int(entry["rows"]),
            in_use=bool(kind_in_use),
            unread=kind in unread_kinds,
            stored_after=bool(arrived_after),
        )
        entry["state"] = state
        warnings = list(entry["warnings"])
        if state == "empty":
            warnings.insert(0, uploads_remedy.NO_ROWS_WARNING)
        if arrived_after:
            # The one an operator hits. Said in the warnings too, so the readers
            # that render nothing else still stop short of a bare green badge.
            warnings.insert(0, str(arrived_after[0]["reason"]["en"]))
        entry["warnings"] = warnings
        # The remedy and the consequence are one answer and are derived
        # together. A kind the engine reads whose own last upload lost the
        # resolver is the case where the unconditional pair contradicts itself
        # on screen, so both sentences name the file that won and the rule that
        # decided it, and the live filename is the file the engine reads.
        outranked = bool(arrived_after) and bool(kind_in_use)
        code = consequence_for(
            in_use=bool(kind_in_use), model_input=bool(entry["model_input"]), outranked=outranked
        )
        if outranked:
            entry["remedy"] = naming(uploads_remedy.REMEDIES, "shadowed_by_a_later_day", str(entry["filename"]))
            entry["consequence"] = naming(uploads_remedy.CONSEQUENCES, code, str(entry["filename"]))
        else:
            entry["remedy"] = labelled(uploads_remedy.REMEDIES, state)
            entry["consequence"] = labelled(uploads_remedy.CONSEQUENCES, code)
        # The state above is true and it is not the whole news. A live file the
        # engine reads that carries rows is ``in_use``, and the last check of
        # that same file can have come back with a warning that every figure
        # from this input now carries. Measured before this: a daily log whose
        # 20 of 20 rows lost their clock was committed and the card it landed on
        # printed the teal chip and "nothing to do", with the warning four lines
        # below it. Said in the warnings too, so a reader that renders those and
        # nothing else still stops short of a bare green badge.
        warned, warned_fields = uploads_remedy.warned_summary((entry["last_validation"] or {}).get("findings"))
        if state == "in_use" and warned and checked_the_live_file(entry["last_validation"], path, kind, destination):
            entry["remedy"] = naming(
                uploads_remedy.REMEDIES,
                "in_use_with_warnings",
                warnings=uploads_remedy.warning_count(warned),
                fields=warned_fields,
            )
            entry["warnings"] = [str(entry["remedy"]["en"]), *entry["warnings"]]
        entries.append(entry)
    summary = {state: sum(1 for entry in entries if entry["state"] == state) for state in STATES}
    return {"inputs": entries, "model": model, "summary": {**summary, "total": len(entries)}}
