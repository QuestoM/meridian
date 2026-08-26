"""Merging two records that turned out to be one party.

A merge moves money: an agency carries a rebate and priced conditions, so
folding one record into another changes what real spots cost. These tests pin
the three properties that make it safe to offer at all - the conflict table is
explicit and conservative, nothing is deleted, and every attachment moves - and
they exercise the real stores rather than mocks, because a merge that passes on
fakes and loses a campaign on disk is worse than no merge.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos_api import agencies, agency_conditions, campaigns_api_store, entity_merge


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Real store modules pointed at temp files, seeded with two records that
    are the same party: OMD (survivor, full terms) and a duplicate that carries
    one field the survivor lacks."""
    agencies_path = tmp_path / "agencies.csv"
    links_path = tmp_path / "agency_advertisers.csv"
    conditions_path = tmp_path / "agency_conditions.csv"
    campaigns_path = tmp_path / "campaigns.csv"

    seed = pd.DataFrame([
        {"agency_id": "AGY_01", "name": "OMD", "display_name": "OMD", "aliases": "או.אם.די",
         "agency_type": "מדיה מלא", "status": "active", "rebate_percent": "4",
         "commission_percent": "2", "payment_terms_days": "60", "credit_limit_ils": "500000",
         "vat_id": "513200001", "contact_name": "מיכל", "notes": ""},
        {"agency_id": "AGY_44", "name": "OMD ישראל", "display_name": "", "aliases": "",
         "agency_type": "מדיה מלא", "status": "active", "rebate_percent": "4",
         "commission_percent": "", "payment_terms_days": "45", "credit_limit_ils": "",
         "vat_id": "513200001", "contact_name": "", "notes": "נפתחה מהסכם סרוק"},
    ])
    for column in agencies.COLUMNS:
        if column not in seed.columns:
            seed[column] = ""
    seed.to_csv(agencies_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(agencies, "AGENCIES_PATH", agencies_path)

    pd.DataFrame([{"agency_id": "AGY_44", "advertiser": "מגדל", "source": "manual", "created_at": "", "created_by": ""}]) \
        .to_csv(links_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", links_path)
    pd.DataFrame(columns=agency_conditions.CONDITION_COLUMNS).to_csv(
        conditions_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", conditions_path)

    campaigns = pd.DataFrame([
        {"record_type": "campaign", "campaign_id": "CMP_900", "name": "קמפיין הכפילה",
         "advertiser": "מגדל", "agency_id": "AGY_44", "status": "active"},
        {"record_type": "campaign", "campaign_id": "CMP_901", "name": "קמפיין השורדת",
         "advertiser": "פריסבי", "agency_id": "AGY_01", "status": "active"},
    ])
    for column in campaigns_api_store.COLUMNS:
        if column not in campaigns.columns:
            campaigns[column] = ""
    campaigns.to_csv(campaigns_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", campaigns_path)
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "backups")
    return tmp_path


def test_the_plan_states_every_conflict_and_the_survivor_wins_by_default(stores):
    plan = entity_merge.plan_merge("AGY_01", "AGY_44")
    by_field = {row["field"]: row for row in plan["conflicts"]}
    # A field both carry, differently: the survivor's value stands.
    assert by_field["payment_terms_days"]["survivor_value"] == "60"
    assert by_field["payment_terms_days"]["duplicate_value"] == "45"
    assert by_field["payment_terms_days"]["winner"] == entity_merge.SURVIVOR
    assert by_field["payment_terms_days"]["kept_value"] == "60"
    # A field only the duplicate carries: it fills the gap, and says so.
    assert by_field["notes"]["winner"] == entity_merge.DUPLICATE
    assert by_field["notes"]["kept_value"] == "נפתחה מהסכם סרוק"
    # A populated survivor field is never quietly replaced.
    assert by_field["commission_percent"]["winner"] == entity_merge.SURVIVOR


def test_an_operator_override_flips_one_field_and_is_marked(stores):
    plan = entity_merge.plan_merge("AGY_01", "AGY_44", {"payment_terms_days": "duplicate"})
    row = next(r for r in plan["conflicts"] if r["field"] == "payment_terms_days")
    assert row["winner"] == entity_merge.DUPLICATE
    assert row["kept_value"] == "45"
    assert row["overridden"] is True
    assert plan["field_changes"]["payment_terms_days"] == "45"


def test_the_plan_counts_what_will_move_before_anything_moves(stores):
    plan = entity_merge.plan_merge("AGY_01", "AGY_44")
    assert plan["attachments_to_move"]["links"] == 1
    assert plan["attachments_to_move"]["campaigns"] == 1
    assert plan["attachments_total"] == 2
    # Planning is read-only: the duplicate is still active and still owns its campaign.
    frame = agencies._load_frame()
    assert frame[frame["agency_id"] == "AGY_44"].iloc[0]["status"] == "active"
    campaigns = campaigns_api_store.load_frame()
    assert (campaigns["agency_id"] == "AGY_44").sum() == 1


def test_a_merge_moves_every_attachment_and_suspends_rather_than_deletes(stores):
    result = entity_merge.apply_merge("AGY_01", "AGY_44", actor="netanel")
    assert result["moved"]["links"] == 1
    assert result["moved"]["campaigns"] == 1

    frame = agencies._load_frame()
    # NOTHING is deleted: the duplicate's row survives, suspended.
    duplicate = frame[frame["agency_id"] == "AGY_44"]
    assert len(duplicate) == 1
    assert duplicate.iloc[0]["status"] == "suspended"
    assert result["reversible"] is True

    # Both spellings still resolve to the survivor, because the names folded in.
    survivor = frame[frame["agency_id"] == "AGY_01"].iloc[0]
    aliases = str(survivor["aliases"])
    assert "OMD ישראל" in aliases
    assert "או.אם.די" in aliases
    # The gap-filling field was taken; the contested one was not.
    assert survivor["notes"] == "נפתחה מהסכם סרוק"
    assert survivor["payment_terms_days"] == "60"

    campaigns = campaigns_api_store.load_frame()
    assert (campaigns["agency_id"] == "AGY_44").sum() == 0
    assert (campaigns["agency_id"] == "AGY_01").sum() == 2
    # manual links are advertiser NAMES, and the duplicate no longer claims it.
    assert "מגדל" in agency_conditions.links_for("AGY_01")["manual"]
    assert "מגדל" not in agency_conditions.links_for("AGY_44")["manual"]
    assert "incomplete" not in result, result.get("incomplete")


def test_the_apply_replans_so_a_stale_proposal_cannot_write_old_values(stores):
    """A proposal written before an edit must not merge on the values it saw."""
    stale = entity_merge.plan_merge("AGY_01", "AGY_44")
    assert stale["field_changes"].get("notes") == "נפתחה מהסכם סרוק"
    # The operator edits the duplicate's note after the proposal was written.
    agencies.update_agency("AGY_44", agencies.AgencyUpdate(notes="הערה מעודכנת"), request=None)
    result = entity_merge.apply_merge("AGY_01", "AGY_44", actor="netanel")
    assert "notes" in result["fields_taken_from_duplicate"]
    survivor = agencies._load_frame().set_index("agency_id").loc["AGY_01"]
    assert survivor["notes"] == "הערה מעודכנת"


def test_a_merge_refuses_itself_and_an_unknown_record(stores):
    with pytest.raises(ValueError):
        entity_merge.plan_merge("AGY_01", "AGY_01")
    with pytest.raises(ValueError):
        entity_merge.plan_merge("AGY_01", "AGY_NOPE")
    with pytest.raises(ValueError):
        entity_merge.plan_merge("", "AGY_44")


def test_an_invalid_override_is_refused_before_anything_writes(stores):
    with pytest.raises(ValueError):
        entity_merge.plan_merge("AGY_01", "AGY_44", {"payment_terms_days": "whichever"})


def test_the_merge_is_a_proposal_kai_can_make_but_only_the_apply_engine_writes():
    from kairos_api import assistant_tools
    from kairos_api.assistant_propose_extra import EXTRA_APPLIERS

    assert "propose_agency_merge" in assistant_tools.PROPOSE_TOOL_NAMES
    assert assistant_tools.KIND_BY_TOOL["propose_agency_merge"] == "agency_merge"
    # The kind has exactly one applier, and no tool applies anything itself.
    assert "agency_merge" in EXTRA_APPLIERS
    assert not any(name.startswith("apply_") for name in assistant_tools.PROPOSE_TOOL_NAMES)


def test_the_proposal_carries_its_conflict_table_and_a_hebrew_summary(stores):
    from kairos_api.assistant_propose_extra import _validate_agency_merge

    payload, summary = _validate_agency_merge({
        "survivor_agency_id": "AGY_01", "duplicate_agency_id": "AGY_44",
        "reason": "אותה סוכנות, אותו ח.פ.",
    })
    assert payload["plan"]["attachments_total"] == 2
    assert payload["plan"]["conflicts"], "the card must show what the merge decides"
    assert "מיזוג סוכנויות" in summary
    assert "מושהית ולא נמחקת" in summary


# --- when KAI is the proposer ------------------------------------------------
# A model that judged two records to be one party and then proposed merging them
# is one witness testifying twice. These pin the rule that answers it: the card
# must carry WHY, and the offer needs a signal no model produced.
def test_the_card_carries_the_identity_evidence_not_only_the_conflict_table(stores):
    from kairos_api.assistant_propose_extra import _validate_agency_merge

    payload, summary = _validate_agency_merge({
        "survivor_agency_id": "AGY_01", "duplicate_agency_id": "AGY_44",
        "reason": "אותו ח.פ. בשני הכרטיסים",
    })
    evidence = payload["evidence"]
    assert evidence["signals"]["vat_match"] is True
    assert evidence["deterministic_signal"] is True
    # The operator reads the reason for the merge, not just its consequences.
    assert "הראיה" in summary
    assert "אותו ח.פ." in summary


def test_evidence_is_recomputed_from_the_stores_not_taken_from_the_proposer(stores):
    """Kai does not get to assert identity: merge_evidence resolves it again."""
    evidence = entity_merge.merge_evidence("AGY_01", "AGY_44")
    assert evidence["scored"] is True
    assert evidence["signals"]["vat_match"] is True
    assert evidence["survivor_name"] == "OMD"
    assert evidence["duplicate_name"] == "OMD ישראל"


def test_a_merge_the_model_alone_believes_in_is_refused_as_a_proposal(stores, monkeypatch):
    """A likeness with no VAT, no normalized-name match and no alias is a
    question for the operator - even when the model is confident."""
    from kairos_api import agencies, entity_merge as merge_module
    from kairos_api.assistant_propose_extra import _validate_agency_merge

    # Strip the shared VAT and rename so only a fuzzy likeness remains.
    agencies.update_agency("AGY_44", agencies.AgencyUpdate(vat_id="", name="אומדי מדיה"), request=None)

    def confident_model(kind, name, evidence, candidates):
        for candidate in candidates:
            candidate.verdict, candidate.confidence = "same", 0.99
            candidate.reason = "clearly the same house"
        return True

    from kairos_api import entity_resolution

    monkeypatch.setattr(entity_resolution, "_adjudicate", confident_model)
    with pytest.raises(ValueError) as raised:
        _validate_agency_merge({
            "survivor_agency_id": "AGY_01", "duplicate_agency_id": "AGY_44",
            "reason": "המודל בטוח שזו אותה סוכנות",
        })
    message = str(raised.value)
    assert "identity signal" in message
    assert "Ask the operator" in message
    # And nothing was written by the refusal.
    frame = merge_module._agencies_frame()
    assert frame[frame["agency_id"] == "AGY_44"].iloc[0]["status"] == "active"


def test_an_alias_already_on_the_survivor_is_evidence_enough(stores):
    """The operator themselves recorded that spelling as this agency's alias, so
    the tie is human-made and a merge may be offered on it."""
    from kairos_api import agencies
    from kairos_api.assistant_propose_extra import _validate_agency_merge

    agencies.update_agency("AGY_44", agencies.AgencyUpdate(vat_id="", name="או.אם.די"), request=None)
    payload, _summary = _validate_agency_merge({
        "survivor_agency_id": "AGY_01", "duplicate_agency_id": "AGY_44",
        "reason": "השם כבר רשום ככינוי של OMD",
    })
    assert payload["evidence"]["deterministic_signal"] is True


def test_the_reversal_path_is_named_rather_than_asserted(stores):
    plan = entity_merge.plan_merge("AGY_01", "AGY_44")
    assert "restore point" in plan["reversal"]
    assert "suspended, not deleted" in plan["reversal"]
    result = entity_merge.apply_merge("AGY_01", "AGY_44", actor="netanel")
    assert result["reversal"] == plan["reversal"]
