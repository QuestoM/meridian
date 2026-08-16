"""Destructive and high-consequence writes require one accessible review step.

These are source contracts because the interaction primitive is intentionally
shared: its browser behaviour is owned by shell/modal-primitives.jsx, while this
suite pins that every scoped writer actually routes through it and preserves the
server payload it used before the safety pass.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
SAFETY = SRC / "safety" / "ConsequenceDialog.jsx"

SURFACES = {
    "constraint": SRC / "rules" / "ConstraintBuilder.jsx",
    "restriction": SRC / "rules" / "RestrictionsPage.jsx",
    "pricing": SRC / "rules" / "PricingManager.jsx",
    "channel": SRC / "rules" / "ChannelPage.jsx",
    "calendar": SRC / "rules" / "CalendarEventsList.jsx",
    "upload": SRC / "kai" / "AssistantUpload.jsx",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_consequence_review_uses_the_canonical_modal_and_starts_on_cancel():
    source = _source(SAFETY)

    assert "{ Button } from '../studio/actions'" in source
    assert "{ Dialog } from '../studio/modal'" in source
    assert 'initialFocusRef={cancelRef}' in source
    assert 'dismissOnBackdrop={!busy}' in source
    assert 'onClose={cancel}' in source
    assert 'ref={cancelRef} variant="outlined"' in source
    assert 'variant="contained"' in source
    assert 'color="error"' in source


def test_review_names_object_scope_and_consequence_in_both_languages():
    source = _source(SAFETY)

    for label in ("'Object', 'אובייקט'", "'Scope', 'היקף'", "'Consequence', 'תוצאה'", "'Recovery', 'שחזור'"):
        assert label in source


def test_every_scoped_writer_routes_through_the_review_dialog_not_an_inline_alert():
    for name, path in SURFACES.items():
        source = _source(path)
        assert "ConsequenceDialog" in source, f"{name} bypasses the consequence review"
        assert 'role="alertdialog"' not in source, f"{name} still uses an uncontained inline alert dialog"
        assert "object={" in source and "scope={" in source and "consequence={" in source


def test_safety_pass_preserves_each_write_payload_and_target():
    assert "method: 'DELETE'" in _source(SURFACES["constraint"]).split("/api/constraints/", 1)[1]
    assert "deleteRestriction(record.restriction_id)" in _source(SURFACES["restriction"])
    assert "applyOverride({}, true)" in _source(SURFACES["pricing"])
    assert "setOperatorChannel(next)" in _source(SURFACES["channel"])
    assert "onSetActive(reviewEvent, false)" in _source(SURFACES["calendar"])

    upload = _source(SURFACES["upload"])
    delete_call = upload.split("/api/assistant/uploads/${encodeURIComponent(id)}", 1)[1]
    assert "method: 'DELETE'" in delete_call


def test_disappearing_triggers_have_a_stable_post_action_focus_target():
    for name in ("constraint", "restriction", "pricing", "channel", "upload"):
        source = _source(SURFACES[name])
        assert "focusAfterDialogClose" in source, f"{name} loses focus when the changed row or trigger disappears"


def test_pricing_reset_trigger_stays_mounted_until_the_review_closes():
    source = _source(SURFACES["pricing"])

    assert "state.has_overrides && !confirmReset" not in source
    assert "state.has_overrides && (" in source


def test_upload_review_states_that_deletion_is_permanent_and_not_restorable():
    source = _source(SURFACES["upload"])

    assert "The stored summary is removed immediately and permanently." in source
    assert "There is no in-product restore." in source
    assert "upload the original file again" in source
