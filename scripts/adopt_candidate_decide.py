"""Recording the verdict: ship or no ship, against a named model version.

JS-19 ends with four words, "record the verdict", and until this module the
terminal that does every other part of that story could not perform them. It
could compare, measure, check, adopt and undo, and then it sent the steward to
another surface for the one act the story is named after. A route that cannot
reach its own done condition is not a route.

**The verdict lands in the model console's own store, not in a second one.**
This module calls :func:`kairos_api.model_version_store.record_decision`, which
is the same function the console's decision form calls, so a verdict recorded
here appears on the console with no synchronisation and no second source of
truth. The store validates the record and refuses a release note that carries a
gate verdict, a p-value or a coefficient; that refusal is shown here verbatim.

**What makes this verdict different from the ones already on record.** Measured
on this tree, all five recorded verdicts were taken by reading each artifact's
own held-out figures, which come from different splits on different test sets:
2,532 breaks for the shipped artifact against 506 for the placebo-corrected
candidate on the same row. That comparison is between two experiments and not
between two predictors. A verdict recorded here carries the common-basis
re-score as its evidence, so a later reader can tell the two apart, and
:func:`adopt_candidate_state.decision_rests_on_rescore` is how the registry
tells them apart on screen.

**Nothing is recorded by accident.** Without ``--perform`` this plans the record
and writes nothing, exactly like ``adopt``. A ship verdict additionally requires
the money to be measured and current, because JS-19's target asks for the
movement in shekels with its scope, and a ship verdict recorded on an unmeasured
figure is a verdict about nothing.
"""

from __future__ import annotations

from typing import Any, Optional

from scripts import adopt_candidate_words as words
from scripts.adopt_candidate_render import DECISION_TAGS, VERDICT_TAGS
from scripts.adopt_candidate_state import live_version, money_state
from scripts.adopt_candidate_rescore import (
    Paths,
    candidate_files,
    candidate_id,
    load_rescore,
    rescore_state,
)

DECISIONS = ("shipped", "not_shipped")

# What the money did, computed from the measurement rather than typed by hand.
# The console's own form posts "unknown" for every decision it records, because
# a form cannot know; a terminal holding the measurement can.
MONEY_DIRECTIONS = {1: "up", -1: "down", 0: "none"}

# The money states as a person says them. The store's own keys are underscored
# and a raw key on a display line reads as a leak rather than a state.
MONEY_STATES = {"measured": "measured and current", "stale": "stale",
                "not_measured": "not measured"}

# The Hebrew alphabet, used for one check and one warning. The console renders
# ``reason`` verbatim inside a right-to-left card and the store carries no
# language for it, so a steward writing English gets English prose on a Hebrew
# screen. This is not enforced, because refusing a steward's own words would be
# worse than the mixed reading, but it is said before the record lands.
_HEBREW = range(0x0590, 0x05FF)


def _has_hebrew(text: str) -> bool:
    return any(ord(character) in _HEBREW for character in str(text or ""))


def _check(identifier: str, passed: bool, en: str, he: str, how: str = "",
           **fields: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "reason_en": en, "reason_he": he,
            **words.pair(words.HOW, how, "how", **fields)}


def _rescore_row(identifier: str, paths: Paths) -> dict[str, Any]:
    stored = load_rescore(paths) or {}
    for row in stored.get("candidates") or []:
        if row.get("id") == identifier:
            return row
    return {}


def evidence_for(identifier: str, paths: Optional[Paths] = None) -> dict[str, Any]:
    """What the verdict is being taken on, frozen into the record beside it.

    Built as a superset of the console's own evidence block rather than a rival
    shape: the keys the candidate card already renders (``money_state``,
    ``revenue_delta``, ``revenue_delta_pct``, ``scope``, ``measured_at``, the
    gate counts) keep their names and their meaning, and the common-basis
    re-score is added under ``rescore`` with the evaluation it was taken on.
    So the console renders this record correctly today, with the comparison it
    could not previously see available in the record drill.
    """
    paths = paths or Paths()
    from kairos_api import model_console_api_payloads as payloads

    evidence: dict[str, Any] = dict(payloads.decision_evidence("candidate", identifier))
    stored = load_rescore(paths) or {}
    row = _rescore_row(identifier, paths)
    paired = row.get("paired") or {}
    verdict = row.get("verdict") or {}
    evidence["basis_en"] = words.DECISION_BASIS["en"]
    evidence["basis_he"] = words.DECISION_BASIS["he"]
    evidence["rescore"] = {
        "rmse": row.get("rmse"),
        "shipped_rmse": (stored.get("shipped") or {}).get("rmse"),
        "rmse_delta": paired.get("rmse_delta"),
        "paired_statistic": paired.get("paired_statistic"),
        "paired_bar": paired.get("paired_bar"),
        "fold_dispersion": paired.get("fold_dispersion"),
        "breaks_improved": paired.get("breaks_improved"),
        "breaks_worsened": paired.get("breaks_worsened"),
        "state": verdict.get("state"),
        "en": verdict.get("en"),
        "he": verdict.get("he"),
        "rule_en": verdict.get("rule_en"),
        "rule_he": verdict.get("rule_he"),
        "duplicate_of": row.get("duplicate_of") or [],
        # How many of the breaks this verdict was taken on were in this
        # artifact's own fit, and what its own producer recorded about adopting
        # it. Both ride inside the record because a decision log is read years
        # later by somebody who cannot re-run the measurement, and a verdict
        # that rested on a confounded comparison should say so from inside
        # itself rather than from a terminal that has since moved on.
        "breaks_fitted_on": row.get("breaks_fitted_on"),
        "self_reported": row.get("self_reported") or {},
        # The coefficient delta travels with the verdict for the same reason the
        # re-score does: JS-19's done condition names it, and the console has no
        # route of its own that can serve it. Attached as the summary rather
        # than the 36 rows, because a decision record is read and not queried.
        "cells": (row.get("cell_deltas") or {}).get("summary") or {},
        "sha256": row.get("sha256"),
        "file": row.get("file"),
        "measured_at": stored.get("measured_at"),
        "fingerprint": stored.get("fingerprint"),
    }
    evidence["evaluation"] = stored.get("evaluation")
    evidence["limit"] = stored.get("limit")
    evidence["fit_basis"] = stored.get("fit_basis")
    evidence["baselines"] = stored.get("baselines") or []
    evidence["cell_structure"] = stored.get("cell_structure")
    return evidence


def money_direction(money: dict[str, Any]) -> str:
    """Up, down, none or unknown, computed from the measured figure."""
    if money.get("state") != "measured":
        return "unknown"
    delta = money.get("revenue_delta")
    if not isinstance(delta, (int, float)):
        return "unknown"
    return MONEY_DIRECTIONS[(delta > 0) - (delta < 0)]


def preconditions(identifier: str, *, decision: str, actor: str, reason: str,
                  release_note_he: str, paths: Optional[Paths] = None) -> dict[str, Any]:
    """Every condition a recordable verdict must clear, answered in both languages."""
    paths = paths or Paths()
    known = {candidate_id(path): path for path in candidate_files(paths)}
    ships = decision == "shipped"
    money = money_state(identifier) if identifier in known else {"state": "not_measured"}
    state = rescore_state(paths)
    row = _rescore_row(identifier, paths)

    checks = [_check(
        "candidate_exists", identifier in known,
        f"The candidate {identifier} is in models/candidates/." if identifier in known else
        f"There is no candidate called {identifier}. Known: {', '.join(sorted(known)) or 'none'}.",
        "המועמד נמצא בתיקיית המועמדים." if identifier in known else "אין מועמד בשם הזה.")]

    named = words.DECISION_WORDS.get(decision) or {}
    checks.append(_check(
        "decision_is_a_verdict", decision in DECISIONS,
        f"The verdict is {named.get('en')}." if decision in DECISIONS else
        f"A verdict is one of {', '.join(DECISIONS)}, and {decision or 'nothing'} is neither.",
        f"ההכרעה היא {named.get('he')}." if decision in DECISIONS else "ההכרעה היא להשיק או לא להשיק בלבד."))

    # A verdict with no common-basis comparison behind it is the defect this
    # whole piece exists to close, so it is a condition and not a warning.
    checks.append(_check(
        "rescore_current", state["state"] == "current" and bool(row),
        "The held-out re-score is current, so this verdict rests on every artifact scored on one common set of breaks."
        if state["state"] == "current" and row else
        str(state.get("reason_en") or "This candidate is not in the stored re-score."),
        "המדידה החוזרת עדכנית, ולכן ההכרעה נשענת על מדידה של כל הקבצים על אותה קבוצת ברייקים."
        if state["state"] == "current" and row else
        str(state.get("reason_he") or "המועמד הזה אינו נמצא במדידה השמורה."),
        "rescore"))

    checks.append(_check(
        "actor_named", bool(str(actor).strip()),
        "The steward taking this verdict is named." if actor else "Nobody is named as taking this verdict.",
        "מי שמכריע נקוב בשם." if actor else "לא נקוב מי מכריע.",
        "actor"))

    checks.append(_check(
        "reason_given", bool(str(reason).strip()),
        "The verdict carries its reason." if reason else "The verdict carries no reason, and a verdict with no reason is not a record.",
        "להכרעה יש סיבה." if reason else "להכרעה אין סיבה, והכרעה בלי סיבה אינה רישום.",
        "reason"))

    if ships:
        checks.append(_check(
            "release_note_written", bool(str(release_note_he).strip()),
            "The ship verdict carries the sentence the operator side reads."
            if release_note_he else
            "A ship verdict needs a release note in Hebrew, because it is the one training-side sentence an operator reads.",
            "ההכרעה להשיק נושאת את המשפט שהצד התפעולי קורא." if release_note_he else
            "הכרעה להשיק מחייבת הערת גרסה בעברית, כי זה המשפט היחיד מצד האימון שמפעיל קורא.",
            "release_note"))
        checks.append(_check(
            "money_measured", money.get("state") == "measured",
            "The money this candidate would move is measured and current."
            if money.get("state") == "measured" else str(money.get("reason_en") or ""),
            "הכסף שהמועמד הזה יזיז נמדד והוא עדכני."
            if money.get("state") == "measured" else str(money.get("reason_he") or ""),
            "measure", id=identifier))

    return {"candidate_id": identifier, "decision": decision, "checks": checks,
            "passed": all(check["passed"] for check in checks),
            "blocked_on": [check["id"] for check in checks if not check["passed"]],
            "money": money, "money_direction": money_direction(money),
            "rescore_verdict": (row.get("verdict") or {}).get("state", "unknown")}


def decide(identifier: str, *, decision: str, actor: str, reason: str,
           reason_en: str = "", release_note_he: str = "", release_note_en: str = "",
           paths: Optional[Paths] = None, perform: bool = False) -> dict[str, Any]:
    """Plan a verdict, and append it to the decision log only with ``perform``."""
    from kairos_api import model_version_store as store

    paths = paths or Paths()
    state = preconditions(identifier, decision=decision, actor=actor, reason=reason,
                          release_note_he=release_note_he, paths=paths)
    version = live_version()
    result = {
        "candidate_id": identifier,
        "recorded": False,
        "store_dir": store.store_dir().as_posix(),
        "model_version": version,
        "reason_is_hebrew": _has_hebrew(reason),
        **state,
    }
    if not state["passed"]:
        result["outcome"] = "refused"
        return result
    evidence = evidence_for(identifier, paths)
    if str(reason_en).strip():
        evidence["reason_en"] = str(reason_en).strip()
    result["evidence"] = evidence
    if not perform:
        result["outcome"] = "ready"
        return result
    try:
        record = store.record_decision(
            model_version=version, decision=decision, subject="candidate",
            candidate_id=identifier, reason=str(reason).strip(),
            release_note_he=str(release_note_he).strip(),
            release_note_en=str(release_note_en).strip(),
            money_direction=state["money_direction"], actor=str(actor).strip(),
            evidence=evidence)
    except store.ModelVersionError as refusal:
        result["outcome"] = "refused"
        result["refusal"] = str(refusal)
        return result
    result.update({"outcome": "recorded", "recorded": True, "record": record})
    return result


def render(result: dict[str, Any]) -> list[str]:
    """The verdict act as a terminal reads it, planned or recorded.

    The two words this screen is about are read from the same tables the
    registry reads them from. ``not_shipped`` and ``not_distinguishable`` are
    the store's own keys, and printing either of them is a key on a display
    line, on the one screen where the steward is deciding which of them to take.
    """
    decision = str(result.get("decision") or "")
    lines = [f"Verdict for {result.get('candidate_id')}: {DECISION_TAGS.get(decision, decision or 'none')}"]
    for check in result.get("checks") or []:
        mark = "pass" if check["passed"] else "STOP"
        lines.append(f"  [{mark}] {check['id']:24s} {check['reason_en']}")
        if not check["passed"] and check.get("how_en"):
            lines.append(f"         {'':24s} {check['how_en']}")
    lines.append("")
    # A refusal has no record, so nothing here may speak about one. Saying "the
    # record will carry" of a record that will not exist is the same defect as
    # a figure with no measurement behind it.
    landing = result.get("outcome") in ("ready", "recorded")
    carried = "carried into the record" if landing else "that would be carried, if this were not refused"
    money = result.get("money") or {}
    if money.get("state") == "measured" and isinstance(money.get("revenue_delta"), (int, float)):
        scope = money.get("scope") or {}
        rows = scope.get("rows")
        counted = f"{rows:,}" if isinstance(rows, int) else "an unrecorded number of"
        lines.append(f"Money {carried}: {money['revenue_delta']:+,.2f} on the operator's own channel over {counted} rows, direction {result.get('money_direction')}")
        lines.append(f"  basis: {scope.get('basis')}")
    elif landing:
        lines.append(f"Money {carried}: {MONEY_STATES.get(str(money.get('state')), 'unknown')}, so the record states that rather than carrying a figure")
    else:
        lines.append(f"Money {carried}: {MONEY_STATES.get(str(money.get('state')), 'unknown')}")
    scored = str(result.get("rescore_verdict") or "unknown")
    lines.append(f"Re-score verdict {carried}: {VERDICT_TAGS.get(scored, scored)}")
    if landing:
        lines.append(f"  {words.DECISION_BASIS['en']}")
    summary = ((result.get("evidence") or {}).get("rescore") or {}).get("cells") or {}
    if summary.get("cells_compared"):
        lines.append(f"Coefficient delta {carried}: {summary['cells_moved']} of {summary['cells_compared']} cells hold a different number")
        lines.append(f"  {summary.get('reading_en')}")
    if landing and not result.get("reason_is_hebrew"):
        lines.append("")
        lines.append("Note: the model console renders this reason verbatim inside a right-to-left card and the store carries no language for it, so a Hebrew reader will get English prose. Pass the Hebrew sentence as --reason and the English as --reason-en.")
    lines.append("")
    if result.get("refusal"):
        lines.append(f"The store refused this record: {result['refusal']}")
    record = result.get("record") or {}
    if record:
        lines.append(f"Recorded {record['decision_id']} against {record.get('model_version_name')}, by {record.get('actor')}.")
        lines.append(f"Written to {result.get('store_dir')}/decisions.jsonl, which is the store the model console reads.")
    elif result.get("outcome") == "ready":
        lines.append("Every check passed and nothing has been written.")
        lines.append("Add --perform to record it.")
    lines.append(f"outcome: {result.get('outcome')}")
    return lines
