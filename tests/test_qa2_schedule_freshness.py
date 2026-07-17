"""The schedule staleness banner must ignore cosmetic settings edits.

The settings group fingerprint used to hash the whole kairos_settings.json, so
a locale toggle or a renamed profile flipped the banner to stale and invited a
pointless recompute. The fingerprint now canonicalizes the engine-relevant
keys only (through the same KairosSettings model the API loads with), so:

  * flipping locale/direction/profile_name/notes/timezone leaves it unchanged,
  * changing revenue_weight, risk_lambda or any guardrail changes it,
  * a missing file is ABSENT and never a false fresh,
  * an explicit default and an omitted key fingerprint identically.

The stamp writer and the verdict share the one fingerprint function, so the
end-to-end write_schedule_meta -> schedule_freshness loop is covered too.
"""

from __future__ import annotations

import json
from pathlib import Path

from kairos.export.schedule_freshness import (
    ABSENT,
    SETTINGS_COSMETIC_KEYS,
    _settings_fingerprint,
    schedule_freshness,
    schedule_input_fingerprints,
    write_schedule_meta,
)


def _write_settings(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_cosmetic_keys_do_not_change_the_fingerprint(tmp_path) -> None:
    settings_path = tmp_path / "kairos_settings.json"
    _write_settings(settings_path, {"locale": "he", "revenue_weight": 60})
    baseline = _settings_fingerprint(settings_path)

    _write_settings(settings_path, {"locale": "en", "revenue_weight": 60})
    assert _settings_fingerprint(settings_path) == baseline, "a locale flip is cosmetic"

    _write_settings(
        settings_path,
        {
            "locale": "en",
            "direction": "ltr",
            "profile_name": "renamed profile",
            "notes": "different operator notes",
            "timezone": "UTC",
            "regulatory_source_url": "https://example.org/",
            "revenue_weight": 60,
        },
    )
    assert _settings_fingerprint(settings_path) == baseline, (
        "every cosmetic key must be excluded from the fingerprint"
    )


def test_engine_relevant_keys_change_the_fingerprint(tmp_path) -> None:
    settings_path = tmp_path / "kairos_settings.json"
    _write_settings(settings_path, {"locale": "he", "revenue_weight": 60})
    baseline = _settings_fingerprint(settings_path)

    _write_settings(settings_path, {"locale": "he", "revenue_weight": 61})
    changed = _settings_fingerprint(settings_path)
    assert changed != baseline, "revenue_weight is engine input"

    _write_settings(settings_path, {"locale": "he", "revenue_weight": 61, "risk_lambda": 0.3})
    assert _settings_fingerprint(settings_path) != changed, "risk_lambda is engine input"


def test_omitted_key_equals_explicit_default(tmp_path) -> None:
    """The engine runs on model defaults for omitted keys, so a file that
    spells a default out must fingerprint the same as one that omits it."""
    settings_path = tmp_path / "kairos_settings.json"
    _write_settings(settings_path, {"revenue_weight": 60})
    explicit = _settings_fingerprint(settings_path)
    _write_settings(settings_path, {})
    omitted = _settings_fingerprint(settings_path)
    assert explicit == omitted, "revenue_weight=60 is the model default"


def test_missing_settings_file_is_absent(tmp_path) -> None:
    assert _settings_fingerprint(tmp_path / "kairos_settings.json") == ABSENT


def test_group_fingerprint_uses_the_canonical_settings_hash(tmp_path) -> None:
    root = tmp_path / "root"
    _write_settings(root / "data" / "kairos_settings.json", {"revenue_weight": 42})
    prints = schedule_input_fingerprints(root)
    assert prints["settings"] == _settings_fingerprint(root / "data" / "kairos_settings.json")
    assert prints["settings"] != ABSENT


def test_locale_flip_never_flags_settings_stale_but_a_real_change_does(tmp_path) -> None:
    """End to end: stamp a schedule, flip a cosmetic key (settings must NOT be
    flagged), then change an engine key (settings MUST be flagged). Assertions
    are on the settings group membership only, so the test is independent of
    the other input groups."""
    root = tmp_path / "root"
    settings_path = root / "data" / "kairos_settings.json"
    _write_settings(settings_path, {"locale": "he", "revenue_weight": 60})
    csv_path = tmp_path / "weekly_break_schedule.csv"
    csv_path.write_text("stub", encoding="utf-8")

    write_schedule_meta(csv_path, root)
    verdict = schedule_freshness(root, csv_path=csv_path)
    assert "settings" not in verdict["changed"]

    _write_settings(settings_path, {"locale": "en", "revenue_weight": 60})
    verdict = schedule_freshness(root, csv_path=csv_path)
    assert "settings" not in verdict["changed"], (
        "a locale toggle must not flip the staleness banner"
    )

    _write_settings(settings_path, {"locale": "en", "revenue_weight": 75})
    verdict = schedule_freshness(root, csv_path=csv_path)
    assert verdict["status"] == "stale"
    assert "settings" in verdict["changed"], "a revenue_weight change is a real input change"


def test_cosmetic_key_set_matches_the_settings_model() -> None:
    """Every cosmetic key must exist on the settings model (no typos silently
    excluding nothing), and the engine levers must not be listed as cosmetic."""
    from kairos_api.core import KairosSettings

    model_fields = set(KairosSettings.model_fields.keys())
    assert SETTINGS_COSMETIC_KEYS <= model_fields
    for lever in (
        "revenue_weight",
        "risk_lambda",
        "min_retention_floor",
        "operator_channel",
        "objective_mode",
        "pricing_overrides",
        "effective_date",
        "pacing_reference_date",
    ):
        assert lever not in SETTINGS_COSMETIC_KEYS
