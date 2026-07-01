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


def schedule_input_fingerprints(root: str | Path) -> dict[str, str]:
    """Map each schedule input group to a content fingerprint computed now.

    The groups and their canonical sources (located by importing the existing
    constants, never hardcoded):

      * ``settings``     ``data/kairos_settings.json`` (the file the API's
                         ``_load_settings`` reads; one file carries
                         operator_channel, guardrails, pricing_overrides,
                         revenue_weight, risk_lambda and min_retention_floor).
      * ``constraints``  the scoped-constraint CSV
                         (``DEFAULT_CONSTRAINTS_PATH``).
      * ``overrides``    the manual-overrides CSV (``DEFAULT_OVERRIDES_PATH``,
                         what ``OverrideSet.from_csv`` reads).
      * ``coefficients`` the coefficients metadata ``computed_at`` string (read
                         via ``read_coefficients_metadata``), used as the
                         fingerprint value rather than a file hash, so the
                         schedule tracks the deltas it was actually built with.
      * ``data``         the reference workbooks
                         (``Spots/Programmes/Dayparts`` xlsx) combined into one
                         sha256, the same files the coefficient guard
                         fingerprints.
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

    # settings: one JSON file under data/.
    settings_path = root / "data" / "kairos_settings.json"
    prints["settings"] = checksum_file(settings_path) or ABSENT

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
    """Return one combined sha256 over the three reference workbooks.

    Resolves the reference directory from the loaders constant (not a guess). The
    per-file digests are concatenated in a fixed name order and re-hashed, so the
    combined value changes if any workbook's bytes change. If a workbook is
    missing the group is :data:`ABSENT`.
    """
    import hashlib

    try:
        from kairos.data.loaders import REFERENCE_DIR

        reference_dir = Path(REFERENCE_DIR)
    except Exception:  # pragma: no cover - defensive
        reference_dir = root / "data" / "reference"

    combined = hashlib.sha256()
    for name in ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx"):
        digest = checksum_file(reference_dir / name)
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
