"""Per-group input fingerprints for the schedule freshness guard.

Split out of :mod:`kairos.export.schedule_freshness` to keep that module under
the project line limit. Every name here is re-exported by schedule_freshness
(including the leading-underscore helpers its tests exercise), so import paths
are stable; nothing else should import from this module directly.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Optional

from kairos.observability.run_log import checksum_file

# Sentinel fingerprint for an input group whose source file is not on disk. A
# group's absence is itself a fingerprint: an input present at write time and
# gone now (or vice versa) is a real change the operator should see, so we record
# it as a value rather than dropping the group.
ABSENT = "absent"

# Display-only settings keys that never change what the engine computes.
# Flipping the dashboard locale, the text direction, the profile label, the
# notes, the regulatory link or the timezone label must not flip the staleness
# banner and invite a pointless recompute. Every OTHER key in the settings
# model (revenue_weight, risk_lambda, min_retention_floor, the guardrail
# limits, operator_channel, objective_mode, pricing_overrides, the pacing
# knobs, the gold and protected-programme rules, effective_date and
# pacing_reference_date) is engine input and stays in the fingerprint. The set
# is subtractive on purpose: a NEW settings key is engine-relevant by default,
# so forgetting to classify it can only over-trigger staleness, never mask it.
SETTINGS_COSMETIC_KEYS = frozenset(
    {
        "locale",
        "direction",
        "chart_direction",
        "profile_name",
        "notes",
        "regulatory_source_url",
        "timezone",
    }
)


def _settings_fingerprint(settings_path: Path) -> str:
    """Fingerprint only the engine-relevant settings, canonically serialized.

    The old whole-file hash flipped the staleness banner on cosmetic edits (a
    locale toggle, a renamed profile). Instead, parse the file through the same
    ``KairosSettings`` model the API loads it with (so missing keys take the
    same defaults the engine would run on, and unknown keys are ignored exactly
    as pydantic ignores them), drop the :data:`SETTINGS_COSMETIC_KEYS`, and
    hash a canonical sorted-key JSON of the rest. A missing file is
    :data:`ABSENT`; an unparseable file mirrors ``_load_settings`` and
    fingerprints the pure defaults, because those ARE what the engine runs on.
    If the settings model itself cannot be imported (engine used standalone),
    fall back to hashing the whole file: over-sensitive, never a false fresh.
    """
    if not settings_path.exists():
        return ABSENT
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    try:
        from kairos_api.core import KairosSettings, _model_dump

        try:
            settings = KairosSettings(**raw) if isinstance(raw, dict) else KairosSettings()
        except (TypeError, ValueError):
            # Mirrors kairos_api.core._load_settings: on an invalid settings
            # file the engine runs on pure defaults, so fingerprint those.
            settings = KairosSettings()
        dumped = _model_dump(settings)
    except Exception:  # pragma: no cover - defensive: never block the stamp
        return checksum_file(settings_path) or ABSENT
    canonical = {
        key: value for key, value in dumped.items() if key not in SETTINGS_COSMETIC_KEYS
    }
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _constant_path(module: str, attr: str, fallback: Path) -> Path:
    """Resolve a loader's own path constant by import, never a hardcoded guess.

    Mirrors the try/except the settings/constraints/overrides groups use: import
    the loader's canonical path constant so this module fingerprints the exact
    file the engine reads, and fall back to the conventional location only if the
    import fails, so a loader refactor never silently blinds the freshness check.
    """
    try:
        mod = importlib.import_module(module)
        return Path(getattr(mod, attr))
    except Exception:  # pragma: no cover - defensive: never block the stamp
        return fallback


def _advertiser_fingerprint(root: Path) -> str:
    """Return one combined sha256 over the two advertiser demand-engine CSVs.

    The baseline-rules CSV and the scoped-conditions CSV are hashed together, each
    file contributing its digest or the :data:`ABSENT` sentinel, so a change in
    either file (or either one appearing or disappearing) changes the combined
    value. Unlike the reference-workbook trio, these two files exist
    independently (conditions ships as a header-only seed), so the whole group is
    :data:`ABSENT` only when NEITHER file is on disk; that keeps a demand engine
    with no data on disk stable-not-stale, matching the identity no-op the engine
    itself falls back to.
    """
    try:
        from kairos.optimize.advertiser_rules import (
            DEFAULT_CONDITIONS_PATH,
            DEFAULT_RULES_PATH,
        )

        entries = [
            ("advertiser_rules.csv", Path(DEFAULT_RULES_PATH)),
            ("advertiser_conditions.csv", Path(DEFAULT_CONDITIONS_PATH)),
        ]
    except Exception:  # pragma: no cover - defensive
        entries = [
            ("advertiser_rules.csv", root / "data" / "advertiser_rules.csv"),
            ("advertiser_conditions.csv", root / "data" / "advertiser_conditions.csv"),
        ]

    combined = hashlib.sha256()
    present = False
    for name, path in entries:
        digest = checksum_file(path)
        if digest is None:
            digest = ABSENT
        else:
            present = True
        combined.update(f"{name}:{digest}\n".encode("utf-8"))
    return combined.hexdigest() if present else ABSENT


def _coefficients_fingerprint(root: Path) -> str:
    """Return the coefficients ``computed_at`` as the group's fingerprint."""
    try:
        from kairos.model.measure import read_coefficients_metadata

        metadata = read_coefficients_metadata(
            root / "models" / "tv_break_coefficients.json"
        )
    except Exception:  # pragma: no cover - defensive
        return ABSENT
    computed_at = metadata.get("computed_at") if isinstance(metadata, dict) else None
    if computed_at is None:
        return ABSENT
    text = str(computed_at).strip()
    return text or ABSENT


def _data_fingerprint(root: Path) -> str:
    """Return one combined sha256 over the three reference tables the engine reads.

    Mirrors the loaders' resolution: each table is hashed as its xlsx when
    present, else the uploaded CSV fallback the loader adopts when the xlsx is
    absent. With all three xlsx present (the shipped state) the entries equal
    hashing the xlsx alone, so the value is unchanged; a table with neither file
    makes the group :data:`ABSENT`. Paths come from the loaders constants.
    """
    try:
        from kairos.data.loaders import REFERENCE_CSV_FALLBACK, REFERENCE_DIR

        reference_dir = Path(REFERENCE_DIR)
        fallback_map = {k: Path(v) for k, v in REFERENCE_CSV_FALLBACK.items()}
    except Exception:  # pragma: no cover - defensive
        reference_dir = root / "data" / "reference"
        fallback_map = {f"{s}.xlsx": root / "data" / f"{s}.csv" for s in ("Programmes", "Spots", "Dayparts")}

    combined = hashlib.sha256()
    for name in ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx"):
        digest = checksum_file(reference_dir / name) or checksum_file(
            fallback_map.get(name, reference_dir / name)
        )
        if digest is None:
            return ABSENT
        combined.update(f"{name}:{digest}\n".encode("utf-8"))
    return combined.hexdigest()


def _events_fingerprint(root: Path) -> Optional[str]:
    """The calendar-events store's fingerprint, ONLY while the events layer is on.

    The event pricing layer (``pricing_activation.events``, shipped OFF) turns
    operator-stored calendar events into forecast price multipliers, so once it
    is active an edited event changes the next recompute and the saved schedule
    must read stale. While the layer is OFF the store is never read by the
    engine, so this returns ``None`` and the group is omitted entirely; that
    mirrors the cosmetic-keys pattern and keeps the off-state sidecar
    byte-identical to the pre-events stamp (an events edit with the layer off
    never invites a pointless recompute).

    Activation is read exactly the way the engine seam reads it
    (:func:`kairos.optimize.pricing.pricing_from_settings`): the saved settings'
    ``pricing_overrides`` deep-merged onto the YAML rate card. When activation
    cannot be established at all, the group stays omitted (never a guessed
    "stale"); note the settings group already flags the activation flip itself,
    because ``pricing_overrides`` is engine input.
    """
    try:
        raw = json.loads((root / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
        overrides = raw.get("pricing_overrides") if isinstance(raw, dict) else None
    except (OSError, ValueError):
        overrides = None
    try:
        from kairos.optimize.pricing import PricingModel

        model = PricingModel.from_config(overrides if isinstance(overrides, dict) else {})
    except Exception:  # pragma: no cover - defensive: never block the stamp
        return None
    if not model.enable_events:
        return None
    events_path = _constant_path(
        "kairos.optimize.event_pricing",
        "DEFAULT_EVENTS_PATH",
        root / "data" / "calendar_events.csv",
    )
    return checksum_file(events_path) or ABSENT


def _audience_model_fingerprint(root: Path) -> Optional[str]:
    """The audience-model artifact's fingerprint, ONLY while its activation is on.

    Mirrors :func:`_events_fingerprint`: the ``audience_model_activation``
    settings flag (shipped OFF) makes forward-dated segments take their
    baseline from ``models/audience_model.json``, so once it is on a retrained
    artifact changes the next recompute and the saved schedule must read stale.
    While the flag is OFF the engine never reads the artifact, so this returns
    ``None`` and the group is omitted entirely, keeping the off-state sidecar
    byte-identical to the pre-model stamp. Activation is read exactly the way
    the transform seam reads it (:func:`kairos.data.audience_overlay.
    audience_model_active`); when it cannot be established the group stays
    omitted (never a guessed "stale"), and the settings group already flags the
    activation flip itself because the flag is engine input. On with no
    artifact on disk records :data:`ABSENT`, an honest value, never an error.
    """
    try:
        from kairos.data.audience_overlay import audience_model_active

        active = audience_model_active(root / "data" / "kairos_settings.json")
    except Exception:  # pragma: no cover - defensive: never block the stamp
        return None
    if not active:
        return None
    return checksum_file(root / "models" / "audience_model.json") or ABSENT
