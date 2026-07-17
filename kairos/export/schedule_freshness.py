"""Schedule freshness: tell honestly whether the saved schedule is current.

The dashboard's Schedule, Reports, Overview, Forecasts, Break-operations and
Break-library screens all render from one saved file,
``output/weekly_break_schedule.csv``. That file is written only by
:func:`kairos.export.schedule.write_weekly_schedule` (the recompute endpoint and
the export CLI). The CSV itself carries no timestamp and no record of the inputs
it was built from, so when the operator edits settings, constraints, overrides,
the coefficients, the reference data, the impact model, the inventory, the
advertiser rules, the campaign flights, or the program classifications, those
screens keep showing the previous snapshot with no signal that it is out of
date. A number presented as current
that is actually stale is a dishonesty risk, so this module detects staleness and
surfaces it.

The mechanism mirrors the coefficient freshness guard
(:mod:`kairos.model.freshness`): a sidecar JSON file written next to the CSV
records, at write time, ``computed_at`` (UTC ISO-8601) and a content fingerprint
for each INPUT GROUP that feeds the schedule. :func:`schedule_freshness`
recomputes those fingerprints now and compares, returning one of three honest
states and never inventing a "fresh":

  * ``fresh``    a sidecar stamp exists and every input group resolves and matches
                 the fingerprint recorded at write time.
  * ``stale``    a sidecar stamp exists but at least one input group differs from
                 what it was when the schedule was stamped; the changed group
                 labels are listed so the operator knows what to recompute.
  * ``unknown``  no sidecar stamp exists yet (no schedule has been written since
                 this feature shipped), or the sidecar cannot be read. This is
                 "cannot verify", never a false "fresh".

The fingerprints are written INTO a sidecar JSON, never into the CSV: the CSV's
column schema is a contract the dashboard readers and an export endpoint depend
on, so it must not gain a metadata column. The comparison is read-only and reads
no clock (``computed_at`` is only echoed from the sidecar), so the verdict is
deterministic given the filesystem.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kairos.observability.run_log import checksum_file

logger = logging.getLogger(__name__)

# Repo root, resolved the same way the sibling export module does (this file is
# kairos/export/schedule_freshness.py, so two parents up is the build root).
ROOT = Path(__file__).resolve().parents[2]

# The saved schedule the dashboard reads, and its metadata sidecar. The sidecar
# sits next to the CSV with a ".meta.json" suffix so it travels with the file and
# never pollutes the CSV's column schema.
DEFAULT_SCHEDULE_PATH = ROOT / "output" / "weekly_break_schedule.csv"
SCHEDULE_META_PATH = DEFAULT_SCHEDULE_PATH.with_suffix(
    DEFAULT_SCHEDULE_PATH.suffix + ".meta.json"
)

# Sidecar keys (mirrors the coefficient guard's metadata vocabulary).
COMPUTED_AT_KEY = "computed_at"
FINGERPRINTS_KEY = "fingerprints"

# Sentinel fingerprint for an input group whose source file is not on disk. A
# group's absence is itself a fingerprint: an input present at write time and
# gone now (or vice versa) is a real change the operator should see, so we record
# it as a value rather than dropping the group.
ABSENT = "absent"

# Clean, operator-facing label for each internal input group, in a stable order.
GROUP_LABELS = {
    "settings": "settings",
    "constraints": "constraints",
    "overrides": "overrides",
    "coefficients": "coefficients",
    "data": "data",
    "impact_model": "the impact model",
    "inventory": "inventory data",
    "advertiser": "advertiser rules",
    "campaigns": "campaign flights",
    "classifications": "program classifications",
}
_GROUP_ORDER = (
    "settings",
    "constraints",
    "overrides",
    "coefficients",
    "data",
    "impact_model",
    "inventory",
    "advertiser",
    "campaigns",
    "classifications",
)


def _meta_path(csv_path: str | Path) -> Path:
    """Return the sidecar metadata path for a given schedule CSV path."""
    target = Path(csv_path)
    return target.with_suffix(target.suffix + ".meta.json")


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


def schedule_input_fingerprints(root: str | Path) -> dict[str, str]:
    """Map each schedule input group to a content fingerprint computed now.

    The groups and their canonical sources (located by importing the existing
    constants, never hardcoded):

      * ``settings``     ``data/kairos_settings.json`` (the file the API's
                         ``_load_settings`` reads; one file carries
                         operator_channel, guardrails, pricing_overrides,
                         revenue_weight, risk_lambda and min_retention_floor).
                         Fingerprinted over the engine-relevant keys only (see
                         :func:`_settings_fingerprint`), so a cosmetic edit
                         such as a locale toggle never reads as stale.
      * ``constraints``  the scoped-constraint CSV
                         (``DEFAULT_CONSTRAINTS_PATH``).
      * ``overrides``    the manual-overrides CSV (``DEFAULT_OVERRIDES_PATH``,
                         what ``OverrideSet.from_csv`` reads).
      * ``coefficients`` the coefficients metadata ``computed_at`` string (read
                         via ``read_coefficients_metadata``), used as the
                         fingerprint value rather than a file hash, so the
                         schedule tracks the deltas it was actually built with.
      * ``data``         the three reference tables (``Spots/Programmes/Dayparts``)
                         combined into one sha256, each hashed as the loaders
                         resolve it: the xlsx, else the uploaded CSV fallback.
      * ``impact_model`` the fitted posterior pickle
                         (``DEFAULT_IMPACT_MODEL_PATH``) that scores every segment;
                         the ``coefficients`` group hashes the JSON delta version,
                         not this pickle, so a retrained posterior is tracked here.
      * ``inventory``    the booked-spot inventory CSV (``DEFAULT_INVENTORY_PATH``)
                         the placement steer reads.
      * ``advertiser``   the two demand-engine CSVs (``DEFAULT_RULES_PATH`` +
                         ``DEFAULT_CONDITIONS_PATH``) combined into one sha256.
      * ``campaigns``    the pacing flight file (``DEFAULT_CAMPAIGNS_PATH``).
      * ``classifications`` the AI genre-override cache
                         (``AI_CLASSIFICATIONS_PATH``), which shifts program class
                         and therefore premiums and coefficients.

    A group whose source file is missing is recorded as :data:`ABSENT` (so
    freshness can tell a present-then-gone input from an unchanged one), never
    dropped. The function is read-only apart from hashing and reads no clock.
    """
    root = Path(root)
    prints: dict[str, str] = {}

    # settings: one JSON file under data/, hashed over engine-relevant keys only
    # so a cosmetic edit (locale, profile name, notes) never invites a recompute.
    settings_path = root / "data" / "kairos_settings.json"
    prints["settings"] = _settings_fingerprint(settings_path)

    # constraints: the scoped-constraint store's canonical CSV.
    try:
        from kairos.optimize.constraints_store import DEFAULT_CONSTRAINTS_PATH

        constraints_path = Path(DEFAULT_CONSTRAINTS_PATH)
    except Exception:  # pragma: no cover - defensive: never block the stamp
        constraints_path = root / "data" / "kairos_constraints.csv"
    prints["constraints"] = checksum_file(constraints_path) or ABSENT

    # overrides: the manual-overrides CSV that OverrideSet.from_csv reads.
    try:
        from kairos.optimize.overrides import DEFAULT_OVERRIDES_PATH

        overrides_path = Path(DEFAULT_OVERRIDES_PATH)
    except Exception:  # pragma: no cover - defensive
        overrides_path = root / "data" / "manual_overrides.csv"
    prints["overrides"] = checksum_file(overrides_path) or ABSENT

    # coefficients: use the metadata computed_at string as the fingerprint, so a
    # recompute (which restamps computed_at) registers as a changed input. Absent
    # when there is no metadata or no timestamp, which is honest: we cannot tie
    # the schedule to a coefficient version we cannot name.
    prints["coefficients"] = _coefficients_fingerprint(root)

    # data: one combined hash over the three reference workbooks. If any is
    # missing the whole group is ABSENT, because the measured schedule cannot be
    # tied to a reference snapshot that is not fully present.
    prints["data"] = _data_fingerprint(root)

    # impact model: the fitted posterior pickle the schedule scores every segment
    # with. The coefficients group above tracks the JSON delta version, NOT the
    # pickle, so a retrained posterior would otherwise move the schedule while
    # freshness stayed green. Hash the pickle's bytes; ABSENT when not on disk.
    prints["impact_model"] = checksum_file(
        _constant_path(
            "kairos.service",
            "DEFAULT_IMPACT_MODEL_PATH",
            root / "models" / "tv_break_posterior.pkl",
        )
    ) or ABSENT

    # inventory: the booked-spot inventory CSV the placement steer reads. Present
    # with real rows today, so a genuine live input; ABSENT when not uploaded.
    prints["inventory"] = checksum_file(
        _constant_path(
            "kairos.optimize.inventory",
            "DEFAULT_INVENTORY_PATH",
            root / "data" / "Spots - inventory.csv",
        )
    ) or ABSENT

    # advertiser: the two coupled demand-engine CSVs (baseline rules + scoped
    # conditions) combined into one sha256 so a change in either registers.
    prints["advertiser"] = _advertiser_fingerprint(root)

    # campaigns: the pacing flight file. Header-only today (an inert seed) but a
    # genuine future input; hashing its bytes keeps a stable value while it stays
    # header-only and flips the moment real flights land. ABSENT when not on disk.
    prints["campaigns"] = checksum_file(
        _constant_path(
            "kairos.optimize.pacing",
            "DEFAULT_CAMPAIGNS_PATH",
            root / "data" / "campaign_flights.csv",
        )
    ) or ABSENT

    # classifications: the AI genre-override cache. Absent today (latent), but when
    # present it changes program class -> premiums and coefficients, so it belongs
    # in the fingerprint. ABSENT until the cache is written.
    prints["classifications"] = checksum_file(
        _constant_path(
            "kairos.service",
            "AI_CLASSIFICATIONS_PATH",
            root / "models" / "ai_program_classifications.json",
        )
    ) or ABSENT

    return prints


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
    import hashlib

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
    import hashlib

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


def write_schedule_meta(csv_path: str | Path, root: str | Path) -> None:
    """Stamp the sidecar next to ``csv_path`` with the current input fingerprints.

    Writes ``{"computed_at": <now UTC ISO>, "fingerprints": {...}}`` so a later
    :func:`schedule_freshness` call can prove whether the saved schedule still
    matches its inputs. The clock is read here on purpose: this is the write path,
    where stamping "when" is the point. The sidecar is written atomically via a
    temporary file so a reader never sees a half-written stamp.
    """
    csv_path = Path(csv_path)
    meta_path = _meta_path(csv_path)
    payload = {
        COMPUTED_AT_KEY: datetime.now(timezone.utc).isoformat(),
        FINGERPRINTS_KEY: schedule_input_fingerprints(root),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(meta_path)


def _read_meta(meta_path: Path) -> Optional[dict[str, Any]]:
    """Read the sidecar JSON, or ``None`` when absent or unreadable.

    An unreadable sidecar is treated the same as an absent one: we cannot verify
    freshness, so the caller reports ``unknown`` rather than guessing.
    """
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read schedule meta sidecar at %s; ignoring.", meta_path)
        return None
    return payload if isinstance(payload, dict) else None


def schedule_freshness(root: str | Path, csv_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Compare the saved schedule's stamped inputs against the inputs on disk now.

    ``root`` is the repo root the input paths resolve against. ``csv_path``
    defaults to ``output/weekly_break_schedule.csv`` under that root; the sidecar
    checked is its ``.meta.json``.

    Returns the frozen contract shape::

        {
          "status": "fresh" | "stale" | "unknown",
          "computed_at": "<ISO-8601 UTC>" | None,
          "changed": ["settings", "constraints", ...]
        }

    Rules (conservative, never fabricating a "fresh"):

      * No sidecar (or an unreadable one) -> ``unknown``, ``computed_at`` None,
        ``changed`` empty. No schedule has been stamped, so freshness is unknown.
      * A sidecar exists -> recompute each group's fingerprint and compare with
        the stamped value. Any group that differs (a hash change, or a present
        / absent flip) goes into ``changed``. Status is ``fresh`` when ``changed``
        is empty, else ``stale``.

    The function is read-only and reads no clock.
    """
    root = Path(root)
    target_csv = Path(csv_path) if csv_path is not None else (root / "output" / "weekly_break_schedule.csv")
    meta_path = _meta_path(target_csv)

    meta = _read_meta(meta_path)
    if meta is None:
        # No stamp (or unreadable): we honestly cannot say the schedule is fresh.
        return {"status": "unknown", "computed_at": None, "changed": []}

    computed_at = _coerce_str(meta.get(COMPUTED_AT_KEY))
    stamped = meta.get(FINGERPRINTS_KEY)
    if not isinstance(stamped, dict):
        # A sidecar without a usable fingerprints block cannot be compared.
        return {"status": "unknown", "computed_at": computed_at, "changed": []}

    current = schedule_input_fingerprints(root)

    changed: list[str] = []
    # Compare every group we know about plus any extra group the stamp carried, so
    # a stamp written by a newer build is still compared honestly.
    group_keys = list(_GROUP_ORDER)
    for key in stamped:
        if key not in group_keys:
            group_keys.append(str(key))

    for key in group_keys:
        stamped_value = stamped.get(key)
        current_value = current.get(key, ABSENT)
        if stamped_value is None:
            # The stamp did not record this group at all: it is a group this build
            # knows about but the older stamp predates. We cannot prove it changed,
            # so we do not flag it (conservative: never invent a "stale").
            continue
        if str(current_value) != str(stamped_value):
            changed.append(GROUP_LABELS.get(key, str(key)))

    if changed:
        return {"status": "stale", "computed_at": computed_at, "changed": changed}
    return {"status": "fresh", "computed_at": computed_at, "changed": []}


def _coerce_str(value: Any) -> Optional[str]:
    """Return ``value`` as a non-empty string, else ``None`` (no fabrication)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
