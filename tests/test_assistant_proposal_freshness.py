"""Which plan a proposal was recorded against, and how it says the plan moved.

A proposal is captured in one conversation and approved in another. Nothing
recorded which plan it was reasoned against, so an item whose summary quotes a
figure could be approved long after a run made that figure wrong, and neither
the card nor the audit line could tell.

These tests hold the three states apart, which is the whole of the mechanism:

* ``current`` only when the artifact AND the economics that produced it match,
* ``stale`` when either moved, carrying the Hebrew sentence an operator reads,
* ``unknown`` when there is nothing to compare, which is the state a batch
  captured before this existed is in, and which must never be reported as
  ``current`` because that is a claim nobody measured.

The stamp is read-only over ``output/``: these tests never write the plan
artifact, and the one that needs a moved plan moves the STAMP instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kairos_api import assistant_proposal_freshness as freshness

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def stamp() -> dict:
    taken = freshness.plan_stamp()
    if taken is None:
        pytest.skip("no readable saved plan, so there is nothing to stamp")
    return taken


def test_the_stamp_carries_the_artifact_its_hash_and_the_economics(stamp) -> None:
    assert stamp["artifact"] == "weekly_break_schedule.csv"
    assert len(stamp["sha256"]) == 64
    # The economics come from the COMMITTED fingerprint beside the plan, which
    # answers "what produced this plan"; live settings answer a different
    # question and would report a pending change as a moved plan.
    assert isinstance(stamp["settings"], dict)


def test_an_unchanged_plan_reads_current_and_carries_no_reason(stamp) -> None:
    verdict = freshness.verdict(stamp)
    assert verdict == {"state": freshness.CURRENT}
    # A verdict of current must not carry a reason: an empty reason on screen is
    # worse than none, and every other state's reason is its whole content.
    assert "reason_he" not in verdict


def test_a_replaced_artifact_reads_stale_and_names_both_hashes(stamp) -> None:
    verdict = freshness.verdict({**stamp, "sha256": "0" * 64})
    assert verdict["state"] == freshness.STALE
    assert verdict["reason_he"] == freshness.STALE_HE
    assert verdict["recorded_against"] == "0" * 64
    assert verdict["saved_plan_now"] == stamp["sha256"]


def test_moved_economics_read_stale_and_name_the_field(stamp) -> None:
    """The plan file can be identical while the economics under it are not."""
    if not stamp["settings"]:
        pytest.skip("the committed fingerprint carries no settings slice to move")
    field = sorted(stamp["settings"])[0]
    moved = {**stamp, "settings": {**stamp["settings"], field: "something-else"}}
    verdict = freshness.verdict(moved)
    assert verdict["state"] == freshness.STALE
    assert verdict["settings_changed"] == [field]


def test_nothing_to_compare_reads_unknown_and_never_current() -> None:
    """A batch captured before the stamp existed carries None, and says so."""
    for absent in (None, {}, {"artifact": "x"}, "not-a-stamp", 7):
        verdict = freshness.verdict(absent)
        assert verdict["state"] == freshness.UNKNOWN, absent
        assert verdict["reason_he"] == freshness.UNKNOWN_HE


def test_an_unreadable_plan_yields_no_stamp_rather_than_a_partial_one(monkeypatch) -> None:
    """Capture must never fail because the plan artifact is missing."""
    monkeypatch.setattr(freshness, "_plan_csv", lambda: ROOT / "output" / "no-such-plan.csv")
    assert freshness.plan_stamp() is None
    # And with no plan to compare against, every stamp reads unknown, not stale:
    # an absent artifact is not a changed one.
    assert freshness.verdict({"sha256": "a" * 64, "settings": {}})["state"] == freshness.UNKNOWN


def test_the_hebrew_says_the_plan_changed_and_the_proposal_was_recorded() -> None:
    """The words are the product's own, not new ones invented for this state.

    ``break_api_pod_order`` already says this about a saved spot order whose pod
    moved, and ``kai-claimed-action.js`` pins הצעה as feminine, so a sentence
    about one reads נרשמה. A new verb here would be a new term, and a term with
    no approved word does not ship.
    """
    from kairos_api import break_api_pod_order

    assert "השתנה" in break_api_pod_order.STALE_HE
    assert "השתנתה" in freshness.STALE_HE
    assert "התוכנית השבועית" in freshness.STALE_HE
    assert "ההצעה נרשמה" in freshness.STALE_HE
    for text in (freshness.STALE_HE, freshness.UNKNOWN_HE):
        assert "\n" not in text, "display text must be one string, never hard-wrapped"


def test_a_captured_batch_carries_its_stamp_and_reads_back_its_freshness(
    tmp_path, monkeypatch,
) -> None:
    """End to end: create_batch stamps, list_proposals reports, neither writes the plan."""
    from kairos_api import assistant_actions as actions

    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    batch = actions.create_batch("שאלה", [{"id": "i1", "kind": "settings", "status": "pending",
                                           "summary": "s", "payload": {}, "reason": "r"}],
                                 user="tester", model="test-model")
    assert batch["plan_stamp"] is not None
    listed = actions.list_proposals(limit=5)["batches"][0]
    assert listed["plan_freshness"]["state"] == freshness.CURRENT

    # A batch that predates the stamp is the unknown case, and the read must not
    # write the store to fix it up: a gate has to be able to diff that file.
    store = Path(tmp_path / "assistant" / "proposals.json")
    before = store.read_bytes()
    raw = store.read_text(encoding="utf-8").replace('"plan_stamp"', '"was_plan_stamp"')
    store.write_text(raw, encoding="utf-8")
    stale_listing = actions.list_proposals(limit=5)["batches"][0]
    assert stale_listing["plan_freshness"]["state"] == freshness.UNKNOWN
    assert store.read_text(encoding="utf-8") == raw
    assert before != store.read_bytes()
