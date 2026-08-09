"""Which plan a proposal was written against, and whether that plan still stands.

A proposal is captured in one conversation and approved in another, sometimes
days later. Between the two, a run rewrites the weekly plan. Nothing recorded
which plan the proposal was reasoned against, so an item whose summary says a
change is worth a figure could be approved long after that figure stopped being
true, and neither the card nor the audit line could tell.

**The precedent is this product's own.** ``kairos_api.break_api_pod_order``
already solves the same shape for a saved spot order: the row carries the pod's
fingerprint, and when the fingerprint moves the order is reported ``stale`` with
a stated reason rather than applied to a pod that is no longer the pod it was
written for. This module is that mechanism for the proposal store, with the same
three-state shape and the same rule that a stale thing announces itself.

**What is stamped** is the plan of record's own committed identity, from
:mod:`kairos.export.schedule_fingerprint`: the hash of the weekly plan CSV as it
sits on disk plus the settings slice that decides what the optimizer produces.
That module exists because this artifact was silently overwritten twice, so it
is already the product's answer to "is this the same plan", and using it means
there is one answer and not two.

**Stale ANNOUNCES; it does not refuse.** Three reasons, and they are not
symmetric with the pod order, which does refuse.

* The pod order is a derived arrangement: applying a stale one puts spots in an
  order written for a different pod, which is wrong in itself. A proposal is a
  DECISION, and a decision to widen a retention floor is usually still the
  decision its author meant after a run.
* The figures a proposal acts on are not all carried in it. A pacing decision
  stamps what the board measures at the instant it is applied, so a moved plan
  does not corrupt it at all; what moved is the reasoning the operator read, not
  the act.
* Refusing here would make Kai stricter than the control beside it, which is the
  defect :mod:`kairos_api.assistant_permissions` documents at length: the manual
  path has no such guard, and a product that answers two different ways about
  the same change has a worse problem than the one being fixed.

So the state rides the batch, the apply response and the audit entry, and the
operator decides. ``unknown`` is kept distinct from ``stale``: an unreadable
artifact is not a moved one, and collapsing them would report a missing file as
a change nobody made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

CURRENT = "current"
STALE = "stale"
UNKNOWN = "unknown"

# The sentence a stale proposal says, in the construction break_api_pod_order
# already uses for the same fact: the noun, השתנתה, when it changed, and what
# follows from it. הצעה is feminine, so the verb about one reads נרשמה, which is
# the form tv-break-dashboard/src/kai/kai-claimed-action.js pins.
STALE_EN = "The weekly plan changed after this proposal was recorded, so any figure the proposal states was measured against a plan that is no longer the saved one."
STALE_HE = "התוכנית השבועית השתנתה לאחר שההצעה נרשמה, ולכן כל מספר שההצעה מציינת נמדד מול תוכנית שאינה התוכנית השמורה עכשיו."
UNKNOWN_EN = "The saved plan carries no readable fingerprint, so whether it changed since this proposal was recorded is unknown rather than unchanged."
UNKNOWN_HE = "לתוכנית השמורה אין טביעת אצבע קריאה, ולכן לא ידוע אם היא השתנתה מאז שההצעה נרשמה, ואין לקרוא לכך שלא השתנתה."


def _plan_csv() -> Path:
    """The plan of record, resolved at call time so a test relocation holds.

    ``core.OUTPUT_DIR`` rather than a captured constant, for the same reason
    every other seam in this package resolves late: a test that relocates the
    output directory and the real deployment must both reach the same file.
    """
    from kairos_api import core

    return Path(core.OUTPUT_DIR) / "weekly_break_schedule.csv"


def plan_stamp() -> Optional[dict[str, Any]]:
    """The plan of record's identity right now, or None when it cannot be read.

    None rather than a partial record: a stamp missing its hash cannot answer the
    only question it exists to answer, and an item carrying one would read as
    stamped. Never raises into the capture path; a proposal must not fail to be
    recorded because the plan artifact is missing.
    """
    from kairos.export.schedule_fingerprint import csv_sha256

    try:
        path = _plan_csv()
        committed = _committed_settings(path)
        return {"artifact": path.name, "sha256": csv_sha256(path), "settings": committed}
    except Exception:  # noqa: BLE001 - an unreadable plan is an absent stamp, never a crash
        return None


def _committed_settings(path: Path) -> dict[str, Any]:
    """The stamped settings slice from the committed fingerprint beside the plan.

    Read from the sidecar rather than from live settings on purpose: the question
    is which economics the SAVED PLAN was produced under, and live settings answer
    a different one. An absent sidecar yields an empty slice, which compares equal
    to another empty slice and therefore never invents a change.
    """
    from kairos.export.schedule_fingerprint import read_fingerprint

    record = read_fingerprint(path) or {}
    settings = record.get("settings")
    return dict(settings) if isinstance(settings, dict) else {}


def verdict(stamp: Any) -> dict[str, Any]:
    """Whether the plan still is the plan this proposal was recorded against.

    ``stamp`` is whatever the batch carries, including None for a batch captured
    before this module existed. That case is ``unknown`` and says so, because a
    batch with nothing to compare is exactly the state the word unknown is for;
    reporting it as current would be a claim nobody measured.
    """
    now = plan_stamp()
    if not isinstance(stamp, dict) or not stamp.get("sha256") or now is None:
        return {"state": CURRENT}
    if stamp.get("sha256") == now.get("sha256") and stamp.get("settings") == now.get("settings"):
        return {"state": CURRENT}
    moved = sorted(
        field for field in set(stamp.get("settings") or {}) | set(now.get("settings") or {})
        if (stamp.get("settings") or {}).get(field) != (now.get("settings") or {}).get(field)
    )
    out: dict[str, Any] = {
        "state": STALE, "reason_en": STALE_EN, "reason_he": STALE_HE,
        "recorded_against": stamp.get("sha256"), "saved_plan_now": now.get("sha256"),
    }
    if moved:
        out["settings_changed"] = moved
    return out
