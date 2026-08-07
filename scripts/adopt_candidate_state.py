"""What the model console already knows about a candidate, read through it.

Split out of ``adopt_candidate_adoption.py`` under the naming rule of section
8.2 when that file passed the 450-line cap, and the split falls on a real seam
rather than a convenient line number: everything here asks the model console's
own modules a question and returns the answer, and nothing here decides
anything. Two callers need these answers, the adoption act and the verdict act,
and before this they lived inside one of them.

Every figure is read through P7's own store rather than off the disk, so a
figure a steward sees at this terminal and the same figure on the model console
are one figure produced by one piece of code. This module never forms a second
opinion about a measurement it did not take.
"""

from __future__ import annotations

from typing import Any, Optional


def live_version() -> dict[str, Any]:
    """The model version the tree currently holds, from the console's own reader.

    A seam rather than a direct call so a test can stand a version up without a
    whole model tree, and so this piece has exactly one place where it asks
    another piece what the live version is.
    """
    from kairos_api import model_console_artifacts as artifacts

    return artifacts.model_version()


def moved_inputs(keys: Optional[list[str]], language: str = "en") -> str:
    """The inputs that moved, in the console's own words rather than its keys.

    ``changed_inputs`` returns the store's key names, and a line reading "What
    moved: settings" puts an internal key on a display line. The model console
    publishes a label for each of the four keys in both languages, so the label
    is read from there and the two surfaces name the same thing the same way.
    """
    from kairos_api.model_console_candidates import INPUT_LABELS

    named = [str((INPUT_LABELS.get(key) or {}).get(language) or key) for key in (keys or [])]
    return ", ".join(named) or ("not recorded" if language == "en" else "לא נרשם")


def money_state(identifier: str) -> dict[str, Any]:
    """The stored money measurement and whether it is current, from P7's store.

    Read through P7's own store module rather than off the disk, so the state a
    steward sees here is the state the model console shows, computed by the same
    code, and this piece never has a second opinion about a figure it did not
    measure.

    **One function, and it used to be two.** The registry carried its own copy
    with different semantics for the stale case: this one refused to return a
    stale figure and that one returned it, so a third caller written against
    either got the other's behaviour. The refusal is the correct half and it is
    kept, under ``revenue_delta``, which is what a record and an artifact stamp
    read. The magnitude moves to ``last_known_revenue_delta``, which is what a
    screen reads, and the two names say which is which.
    """
    from kairos_api import model_console_candidates as console
    from kairos_api import model_version_store as store

    stored = store.measurement(identifier)
    path = console.candidate_path(identifier)
    if stored is None:
        return {"state": "not_measured", "revenue_delta": None,
                "last_known_revenue_delta": None, "scope": {}, "how": "measure",
                "reason_en": "The money this would move has not been measured.",
                "reason_he": "הכסף שזה יזיז לא נמדד."}
    if path is None or str(stored.get("fingerprint") or "") != console.measurement_fingerprint(path):
        moved = console.changed_inputs(path, stored) if path is not None else []
        # ``revenue_delta`` stays None so no stale figure can be carried into a
        # decision record or into an artifact stamp. The magnitude is not thrown
        # away with it: a check that says only "stale" cannot say whether the
        # figure it is refusing to use is a rounding error or a million shekels,
        # and a steward needs to know which before deciding what to re-measure.
        return {"state": "stale", "revenue_delta": None,
                "last_known_revenue_delta": (stored.get("operator_channel_delta") or {}).get("revenue_delta"),
                "changed": moved, "how": "measure",
                "measured_at": stored.get("measured_at"),
                "scope": (stored.get("scope") or {}).get("operator_channel") or {},
                "reason_en": "The stored money measurement is not current. What changed: " + moved_inputs(moved) + ".",
                "reason_he": "מדידת הכסף השמורה אינה עדכנית. מה שהשתנה: " + moved_inputs(moved, "he") + "."}
    own = stored.get("operator_channel_delta") or {}
    whole = stored.get("whole_plan_delta") or {}
    # A shipped figure is not only revenue. The plan publishes the retention sum
    # and the break count as well, so a candidate that leaves revenue alone and
    # moves either of those has still moved something an operator reads.
    moved_fields = sorted(
        f"{key} on the {scope}" for source, scope in ((own, "operator's own channel"), (whole, "whole plan"))
        for key in ("revenue_delta", "retention_sum_delta", "breaks_delta")
        if isinstance(source.get(key), (int, float)) and abs(float(source[key])) > 0)
    return {"state": "measured", "revenue_delta": own.get("revenue_delta"),
            "last_known_revenue_delta": own.get("revenue_delta"),
            "revenue_delta_pct": own.get("revenue_delta_pct"),
            "measured_at": stored.get("measured_at"),
            "scope": (stored.get("scope") or {}).get("operator_channel"),
            "moved_fields": moved_fields,
            "whole_plan_delta": whole.get("revenue_delta")}


# How many rows a held-out block was taken on, under whichever name the artifact
# that wrote it used. A metadata block that records one of these is a held-out
# re-measurement, which is how they are found here: detected from the artifact
# rather than listed, so a gate added later is carried without an edit.
HELD_OUT_SIZE_KEYS = ("n_test", "n_test_minutes", "n_test_days")


def held_out_blocks(metadata: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every held-out figure an artifact reports about itself, with its size."""
    blocks: dict[str, dict[str, Any]] = {}
    for key, value in (metadata or {}).items():
        if not isinstance(value, dict):
            continue
        unit = next((name for name in HELD_OUT_SIZE_KEYS if name in value), None)
        if unit is not None:
            blocks[key] = {"size": value[unit], "unit": unit, "figures": value}
    return blocks


def gate_evidence(shipped: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Which gates decide differently, and what each side measured that on.

    JS-19 asks for every gate delta with its held-out figure, and the two halves
    come from two places. The verdicts come from the model console's own
    ``gate_deltas``, so this terminal and that console cannot disagree about
    which gate moved. The held-out figures are read off each artifact's own
    metadata, and they are the figures that are **not** comparable: measured on
    this tree, the shipped artifact decided its series gate on 2,532 breaks and
    the placebo-corrected candidate decided the same gate on 506. That is the
    whole reason this piece scores every artifact again on one common set, so
    the two sizes are shown side by side rather than the figures alone.
    """
    from kairos_api import model_console_candidates as console

    shipped_meta = shipped.get("metadata") if isinstance(shipped.get("metadata"), dict) else {}
    candidate_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    before, after = held_out_blocks(shipped_meta), held_out_blocks(candidate_meta)
    return {
        "verdicts": console.gate_deltas(shipped_meta, candidate_meta),
        "held_out": [{"block": key,
                      "shipped_size": (before.get(key) or {}).get("size"),
                      "shipped_unit": (before.get(key) or {}).get("unit"),
                      "candidate_size": (after.get(key) or {}).get("size"),
                      "candidate_unit": (after.get(key) or {}).get("unit"),
                      "shipped_absent": key not in before,
                      "candidate_absent": key not in after,
                      "comparable": key in before and key in after
                      and (before[key]["size"] == after[key]["size"])}
                     for key in sorted(set(before) | set(after))],
    }


def recorded_decision(identifier: str) -> Optional[dict[str, Any]]:
    """The newest verdict of any kind on this candidate, ship or no ship."""
    from kairos_api import model_version_store as store

    return next((record for record in store.decisions()
                 if record.get("subject") == "candidate"
                 and record.get("candidate_id") == identifier), None)


def ship_decision(identifier: str, version_id: str) -> Optional[dict[str, Any]]:
    """The newest ship verdict on this candidate against the version on disk.

    A ship verdict recorded against an earlier version is not a verdict about
    the artifact in force, so the version is matched rather than ignored.
    """
    from kairos_api import model_version_store as store

    for record in store.decisions():
        if record.get("subject") != "candidate" or record.get("candidate_id") != identifier:
            continue
        if record.get("decision") != "shipped":
            continue
        if version_id and record.get("model_version_id") != version_id:
            continue
        return record
    return None


def decision_rests_on_rescore(record: Optional[dict[str, Any]]) -> bool:
    """Was this verdict taken on the common-basis re-score, or on something else.

    The whole reason this piece exists is that a verdict taken by reading two
    artifacts' own held-out figures compares two experiments on two different
    test sets. Measured on this tree, all five verdicts on record predate the
    common-basis comparison, so a registry that showed only "no ship" would be
    hiding what the no ship was decided on. A verdict carries the comparison in
    its evidence or it does not, and the difference is readable here.
    """
    evidence = (record or {}).get("evidence")
    if not isinstance(evidence, dict):
        return False
    rescore = evidence.get("rescore")
    return isinstance(rescore, dict) and rescore.get("rmse") is not None
