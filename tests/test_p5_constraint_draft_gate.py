"""The advanced condition builder measures a complete draft before it writes."""

from pathlib import Path

from kairos_api import constraints as constraints_api
from kairos_api import constraints_effect


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tv-break-dashboard" / "src" / "rules" / "ConstraintBuilder.jsx"


def test_a_draft_effect_uses_the_engine_seam_and_writes_no_store(tmp_path, monkeypatch):
    store = tmp_path / "constraints.csv"
    seen = {}

    def measured(rows, channel=None, day=None, daily_input=None):
        seen["rows"] = rows
        return {"summary": {"matched_segments": 1}}

    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", store)
    monkeypatch.setattr(constraints_effect, "measure", measured)
    payload = constraints_api.ConstraintCreate(
        scope_type="always",
        effect="fix_offset",
        offset_seconds=60,
        where={
            "combinator": "and",
            "conditions": [{"field": "programme", "operator": "is", "value": "חדשות"}],
        },
    )

    answer = constraints_api.preview_constraint_effect(payload)

    assert answer["summary"]["matched_segments"] == 1
    assert seen["rows"][0].constraint_id == "draft-preview"
    assert seen["rows"][0].where["conditions"][0]["value"] == "חדשות"
    assert not store.exists(), "a preview must not create or rewrite the constraint store"


def test_the_surface_cannot_save_an_empty_or_unmeasured_draft():
    source = BUILDER.read_text(encoding="utf-8")

    assert "predicateComplete(body.where)" in source
    assert "POST" in source.split("/api/constraints/effect", 1)[1].split("});", 1)[0]
    assert "previewKey === JSON.stringify(body)" in source
    assert "matchedSegments === 0" in source
    assert "Measure a complete rule that matches the plan before saving it" in source
