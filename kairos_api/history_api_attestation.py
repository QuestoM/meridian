"""Has anything changed since, and the evidence when nothing has.

Split out of ``history_api.py`` to keep that module under the file-size cap.
It is the compliance half of History: a regulator asks whether the limits in
force are the current ones, and the only honest answer is a record with a date
and an actor on it rather than a screenshot of today's numbers.

Two rules hold here.

**An unreadable record answers unknown, never unchanged.** "Nothing changed" and
"I could not tell" are different answers and only one of them can be attested
to, so the store's own failure is reported with its reason rather than being
folded into the reassuring case.

**A preview is not a change and a run is not a change.** A preview computed an
answer and saved nothing; a run read the saved state rather than editing it. So
the verdict counts changes, restores and restore points, names the three it
counted, and still reports every other count beside them.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from kairos_api import history_api_timeline

logger = logging.getLogger(__name__)

# What counts as a change for the attestation, named in the payload so a reader
# can check the rule rather than trust the verdict.
ATTESTED_KINDS = ("change", "restore", "restore_point")


def guardrail_attestation(day: date) -> dict[str, Any]:
    """Whether any regulatory guardrail moved since ``day``, from its own store.

    An empty change list is the evidence a compliance owner needs, and it is
    only as old as the record, so the day the record itself starts is returned
    beside it.
    """
    try:
        from kairos_api import guardrail_store

        record = guardrail_store.load_record()
        changed = guardrail_store.changed_since(day, record)
        scheduled = guardrail_store.scheduled_changes(None, record)
        baseline = (record.get("baseline") or {}).get("effective_date")
        return {
            "state": "changed" if changed else "unchanged",
            "changes": changed,
            "scheduled": scheduled,
            "effective_date": guardrail_store.effective_date(None, record),
            "record_starts": baseline,
            "values": guardrail_store.current_values(record),
            "reason": None,
        }
    except Exception as error:  # noqa: BLE001 - an unreadable store is unknown, never unchanged
        logger.exception("the guardrail store could not be read for an attestation")
        return {
            "state": "unknown",
            "changes": [],
            "scheduled": [],
            "effective_date": None,
            "record_starts": None,
            "values": {},
            "reason": str(error) or "the regulatory guardrail store could not be read",
        }


def since_body(assembled: dict[str, Any], day: str) -> dict[str, Any]:
    """What changed since a calendar day, over an already assembled timeline."""
    entries = history_api_timeline.since_day(assembled["entries"], day)
    tally = history_api_timeline.counts(entries)
    changed = sum(tally[kind] for kind in ATTESTED_KINDS)
    return {
        "day": day,
        "counts": tally,
        "matched": len(entries),
        "changed": changed,
        "attested_kinds": list(ATTESTED_KINDS),
        # A count since a day is only evidence for the days the record covers.
        # Measured before this rode along: the strip attested over a seven-week
        # window against five hours of surviving record, and the one sentence
        # beside it that named a start belonged to the guardrail store.
        "record_starts": assembled.get("record_starts"),
        "examined": assembled["sources"],
        "scope": assembled["scope"],
        "guardrails": guardrail_attestation(date.fromisoformat(day)),
        "verdict": "changed" if changed else "unchanged",
    }
