"""What each artifact was fitted on, measured against what it is scored on.

**The defect this module exists to close, stated as a measurement.** Every
payload this piece emits carries one sentence that decides how every figure on
it may be read: that each artifact was fitted on all of the breaks it is now
scored on, so the in-sample optimism is common to every row and the difference
between two rows survives it. That sentence was a constant. It was never checked
against the artifacts.

Measured on this tree, it is false. Five artifacts record
``total_breaks_measured`` 2532, and ``spotclip`` records 2336, with its own
metadata naming the cause: ``base_breaks`` 2532 and ``dropped_by_spot_clip``
196. The evaluation scores all six on the same 2,532 breaks. So 196 of the
breaks ``spotclip`` is scored on were never in its fit, and are in the fit of
every row it is compared against. The optimism is not common-mode, and the row
this is true of is the row the table ranks first.

The figure that falsifies the sentence was already being read. ``breaks_fitted_on``
is computed on every candidate row and stored in ``holdout_rescores.json``, and
nothing compared it to anything. This module makes the sentence a measurement.

**What can be said and what cannot.** The count each artifact was fitted on is
recorded. The identity of the breaks is not, on any artifact here. So the
existence of the confound is measured, its size in breaks is measured, and its
effect on the metric is not computable from anything on disk. That last one is
reported as unknown with the thing that would supply it, never as a correction
this module guessed at.

**Set identity is not count equality, and the wording keeps them apart.** Two
artifacts recording the same count were not thereby fitted on the same breaks.
What is recorded is a count, so what is stated is a count, and an artifact that
records no count at all is a third state rather than an assumed match.
"""

from __future__ import annotations

from typing import Any, Optional

from scripts import adopt_candidate_words as words

# Where an artifact records how many breaks its fit was measured over, and the
# two companion keys the clip variant uses to say what it removed and from what.
FITTED_KEY = "total_breaks_measured"
BASE_KEY = "base_breaks"
DROPPED_KEY = "dropped_by_spot_clip"

# What an artifact may record about its own out-of-sample test. This is the
# artifact's own split under its own fit, which is exactly what this piece exists
# not to compare across rows, so it is carried per row and never ranked.
SELF_TEST_KEY = "holdout_clean_test"
RECOMMENDATION_KEY = "adopt_recommended"


def _int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def basis_row(identifier: str, metadata: dict[str, Any], scored_on: Optional[int]) -> dict[str, Any]:
    """One artifact's fit basis, against the number of breaks it is scored on.

    ``state`` is the tri-state this piece uses everywhere. ``all`` means the
    artifact records a fit over as many breaks as it is scored on, ``fewer``
    means it records a fit over fewer and the shortfall is stated, and
    ``unknown`` means it records no fit basis at all, which is not the same as
    recording a match.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    fitted = _int(metadata.get(FITTED_KEY))
    base = _int(metadata.get(BASE_KEY))
    dropped = _int(metadata.get(DROPPED_KEY))
    row: dict[str, Any] = {
        "id": identifier,
        "fitted_on": fitted,
        "base": base,
        "dropped": dropped,
        "scored_on": scored_on,
        "not_fitted_on": None,
        "share_not_fitted_on": None,
    }
    if fitted is None or not scored_on:
        row["state"] = "unknown"
        return row
    shortfall = scored_on - fitted
    if shortfall <= 0:
        row["state"] = "all"
        row["not_fitted_on"] = 0
        row["share_not_fitted_on"] = 0.0
        return row
    row["state"] = "fewer"
    row["not_fitted_on"] = shortfall
    row["share_not_fitted_on"] = round(shortfall / scored_on, 9)
    return row


def self_reported(identifier: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """What the artifact's own producer recorded about adopting it.

    Deliberately not a figure this piece ranks. An artifact's own held-out test
    is its own split against its own target, and treating one as comparable with
    another is the mistake the common-basis re-score was built to stop. What is
    carried is the producer's recorded recommendation, which is not a comparison
    and is the one thing about an artifact that only its producer knows.

    Three states. ``recommended`` when a recommendation is recorded, with the
    reason and the size of the test it was taken on. ``recorded_without_a_verdict``
    when a self-test exists but reaches no recommendation. ``absent`` when the
    artifact records no self-test at all, which is most of them.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    test = metadata.get(SELF_TEST_KEY)
    if not isinstance(test, dict):
        return {"id": identifier, "state": "absent",
                **words.pair(words.SELF_TEST, "absent", "reading")}
    recommends = test.get(RECOMMENDATION_KEY)
    row = {
        "id": identifier,
        "n_test": _int(test.get("n_test")),
        "reason": test.get("reason"),
        "adopt_recommended": recommends if isinstance(recommends, bool) else None,
    }
    if not isinstance(recommends, bool):
        row["state"] = "recorded_without_a_verdict"
        return {**row, **words.pair(words.SELF_TEST, "recorded_without_a_verdict", "reading")}
    row["state"] = "recommended" if recommends else "advised_against"
    return {**row, **words.pair(words.SELF_TEST, row["state"], "reading")}


def fit_basis(rows: list[dict[str, Any]], scored_on: Optional[int]) -> dict[str, Any]:
    """Every artifact's fit basis and the one reading that follows from them.

    ``rows`` are ``basis_row`` results, shipped first. The summary names the rows
    that do not cover the evaluation rather than counting them, because the whole
    value of the finding is which row it is true of.
    """
    fewer = [row for row in rows if row.get("state") == "fewer"]
    unknown = [row for row in rows if row.get("state") == "unknown"]
    worst = max(fewer, key=lambda row: row.get("not_fitted_on") or 0, default=None)
    state = "uneven" if fewer else ("unknown" if unknown else "common")
    return {
        "state": state,
        "scored_on": scored_on,
        "rows": rows,
        "uneven": [row["id"] for row in fewer],
        "unknown": [row["id"] for row in unknown],
        "largest_shortfall": (worst or {}).get("not_fitted_on"),
        "largest_shortfall_at": (worst or {}).get("id"),
        "largest_shortfall_share": (worst or {}).get("share_not_fitted_on"),
    }


def render_fit_basis(payload: dict[str, Any]) -> list[str]:
    """The fit basis under the limit sentence, for the rows it is not true of.

    Rendered here rather than in ``adopt_candidate_render.py`` for the ordinary
    reason: that module is the terminal's renderer and it is close to the size
    cap, and these two blocks are this module's finding end to end. Every other
    block on the terminal is still rendered there.

    Only the rows that do not cover the evaluation are printed. A list of six
    rows saying five of them are fine buries the one that is not, and the
    summary above already states how many there are.
    """
    basis = payload.get("fit_basis") or {}
    if basis.get("state") == "common":
        return []
    lines = []
    for row in basis.get("rows") or []:
        if row.get("state") == "all":
            continue
        reading = words.pair(words.FIT_BASIS, row.get("state"), "reading",
                             fitted=row.get("fitted_on"), scored=row.get("scored_on"),
                             shortfall=row.get("not_fitted_on"))["reading_en"]
        lines.append(f"    {row.get('id')}: {reading}")
    return lines


def render_self_tests(payload: dict[str, Any]) -> list[str]:
    """What each artifact's own producer recorded, for the ones that recorded it.

    Printed as its own block rather than as a column, because it is not a rank
    and a column invites reading it as one. The basis sentence travels with it
    every time it is printed.
    """
    rows = [row.get("self_reported") or {} for row in payload.get("candidates") or []]
    shipped = (payload.get("shipped") or {}).get("self_reported") or {}
    rows = [row for row in ([shipped] + rows) if row.get("state") not in (None, "absent")]
    if not rows:
        return []
    lines = ["What the artifact's own producer recorded about adopting it", ""]
    for row in rows:
        lines.append(f"  {str(row.get('id')):20s} {row.get('reading_en')}")
        if row.get("reason"):
            lines.append(f"  {'':20s} its own words: {row['reason']}")
        if row.get("n_test"):
            lines.append(f"  {'':20s} taken on {row['n_test']} breaks of its own choosing")
    lines.append("")
    lines.append(f"  {SELF_TEST_BASIS_EN}")
    lines.append("")
    return lines


SELF_TEST_BASIS_EN = words.SELF_TEST_BASIS["en"]


def limit_for(basis: dict[str, Any]) -> dict[str, Any]:
    """The evaluation's own limit, selected by the measurement rather than fixed.

    This is the sentence that decides how every figure on this surface may be
    read, so it is the last sentence that should be a constant. The common case
    keeps the wording it always had; the two other cases say what is actually
    true of these artifacts, name the rows it is true of, and state that the
    effect on the metric is not computable from what any artifact records.
    """
    state = basis.get("state") or "common"
    if state == "common":
        return dict(words.IN_SAMPLE_LIMIT)
    entry = words.LIMIT_UNEVEN if state == "uneven" else words.LIMIT_UNKNOWN
    limit = dict(entry)
    limit["state"] = "in_sample_uneven" if state == "uneven" else "in_sample_unknown"
    limit["uneven"] = basis.get("uneven") or []
    limit["unknown"] = basis.get("unknown") or []
    limit["largest_shortfall"] = basis.get("largest_shortfall")
    limit["largest_shortfall_at"] = basis.get("largest_shortfall_at")
    limit["largest_shortfall_share"] = basis.get("largest_shortfall_share")
    return limit
