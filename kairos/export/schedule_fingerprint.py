"""A committed fingerprint for the saved weekly plan.

The plan CSV is the plan of record. It carries real money, it is read by every
money surface, and it is the artifact the golden test compares against. It was
silently overwritten twice on 2026-08-07 with a stale copy taken from a temp
mirror, and both times the only thing that caught it was a person hashing the
file by hand before committing.

There was already a freshness sidecar next to the CSV, written by the same
function that writes the CSV. It could not help, for one reason: ``output/
*.meta.json`` is in .gitignore, so it never travels with the artifact. On a
fresh checkout, in CI, and after any clone, ``schedule_freshness`` finds no
sidecar and answers "unknown". A guard that answers "unknown" is not a guard,
and this repository has now met that same failure three times in one day: a
poller that ran thirty times reporting success over zero work, a smoke test
reading two entry files that no longer exist, and this.

So this fingerprint is COMMITTED, deliberately, and it is tiny. It records what
the artifact was when it was written and under which economic settings, so a
test can answer two questions that matter and cost nothing to check:

* Has the artifact been replaced since it was exported? A hash mismatch means
  something wrote this file without running the exporter, which is exactly the
  overwrite that happened twice.
* Do the settings that produced it still match the settings on disk? A mismatch
  means the plan of record was computed under economics nobody is using any
  more, which is the defect that moved 15.8M and put the operator's own front
  page into a declared licence breach.

It deliberately does NOT re-run the engine. That would be the strongest check
and it takes minutes; this one takes milliseconds and runs on every suite, which
is what makes it a guard rather than a ceremony.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

FINGERPRINT_SUFFIX = ".fingerprint.json"

# The settings that change what the optimizer produces. A change to any of these
# without a re-export means the saved plan is not this configuration's plan.
STAMPED_SETTINGS = (
    "revenue_weight", "min_retention_floor", "operator_channel", "risk_lambda",
    "pricing_overrides",
)
PRICING_CONFIG = "config/optimization_weights.yaml"

# Settings that do NOT change the plan, and are guarded anyway, because they are
# in the same shared writable file and an agent that walks the UI writes them.
#
# This list exists because the guard above shipped without it and the very next
# commit walked through the hole: locale and direction were committed as en and
# ltr, which would have shipped an Israeli Hebrew right-to-left product booting
# in English. That is the third time this one file has been polluted in a day,
# and the second time by these two fields specifically.
#
# The lesson is not "add locale". It is that a guard scoped to what the author
# was thinking about at the time protects only that, and the file is the unit of
# risk, not the field. Anything here must hold its expected value exactly.
PINNED_SETTINGS = {
    "locale": "he",
    "direction": "rtl",
}

# The OTHER shared writable store, and the one that got away.
#
# data/manual_overrides.csv holds the decisions an operator pins by hand, and the
# optimizer honours every row whose status is active. It is written by the same
# browser the settings are written by, so it carries the same risk, and on
# 2026-08-01 the same walk that changed revenue_weight also wrote one gold mark
# into it. The settings were restored and guarded; this file was not, so the row
# survived the restore and moved 131,878.70 ILS on 2024-11-03 for another eight
# days without anything noticing.
#
# The lesson recorded above was "the file is the unit of risk, not the field".
# It was still too narrow. The unit of risk is EVERY shared writable store the
# plan is computed from, and there were two.
#
# Only ACTIVE rows are digested, because only active rows bend the plan. That is
# deliberate and it is what makes the guard usable: retiring a bad row changes
# the digest and correctly demands a re-export, while the row itself stays on
# disk as a record rather than being deleted to make a test pass.
OVERRIDE_STORE = "data/manual_overrides.csv"


def active_override_digest(root: str | Path) -> str:
    """A hash of the override rows the optimizer would actually honour.

    Absent file and no active rows hash the same, on purpose: both mean the plan
    was computed with nothing pinned, which is one state and not two.
    """
    path = Path(root) / OVERRIDE_STORE
    rows: list[str] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        text = ""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        header = [column.strip() for column in lines[0].split(",")]
        try:
            status = header.index("status")
        except ValueError:
            status = -1
        for line in lines[1:]:
            cells = list(csv.reader([line]))[0] if line else []
            if status < 0 or (len(cells) > status and cells[status].strip().lower() == "active"):
                rows.append(line)
    digest = hashlib.sha256()
    for line in sorted(rows):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def csv_sha256(path: str | Path) -> str:
    """The hash of the artifact exactly as it sits on disk, bytes in, no parsing."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pricing_config_digest(root: str | Path) -> str:
    """Hash the shipped pricing config, including QH activation/provenance."""
    path = Path(root) / PRICING_CONFIG
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def fingerprint_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(path.name + FINGERPRINT_SUFFIX)


def settings_slice(settings: Any) -> dict[str, Any]:
    """The stamped settings, read from a mapping or an object, missing ones as None."""
    out: dict[str, Any] = {}
    for key in STAMPED_SETTINGS:
        if isinstance(settings, dict):
            out[key] = settings.get(key)
        else:
            out[key] = getattr(settings, key, None)
    return out


def build_fingerprint(
    csv_path: str | Path,
    settings: Any,
    revenue_provenance: Optional[dict[str, Any]] = None,
    *,
    pricing_config_sha256: Optional[str] = None,
    active_overrides_sha256: Optional[str] = None,
    run_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The record to commit beside the CSV. Counts are read back from the file."""
    path = Path(csv_path)
    rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for index, _ in enumerate(handle):
            rows = index  # header is line 0, so the final index is the data row count
    # The repository root, from this module rather than from the caller, so the
    # digest is taken over the same store the optimizer read.
    root = Path(__file__).resolve().parents[2]
    return {
        "artifact": path.name,
        "sha256": csv_sha256(path),
        "rows": rows,
        "settings": settings_slice(settings),
        "pricing_config_sha256": (
            pricing_config_sha256
            if pricing_config_sha256 is not None else pricing_config_digest(root)
        ),
        "revenue_provenance": dict(revenue_provenance or {}),
        "run_context": dict(run_context or {}),
        "active_overrides": (
            active_overrides_sha256
            if active_overrides_sha256 is not None else active_override_digest(root)
        ),
        "note": (
            "Committed on purpose. The freshness sidecar beside this file is gitignored "
            "and answers unknown on a fresh checkout, so it cannot guard the artifact. "
            "tests/test_plan_artifact_fingerprint.py compares this against the file, "
            "against data/kairos_settings.json, and against the active rows of "
            "data/manual_overrides.csv."
        ),
    }


def write_fingerprint(
    csv_path: str | Path,
    settings: Any,
    revenue_provenance: Optional[dict[str, Any]] = None,
    *,
    pricing_config_sha256: Optional[str] = None,
    active_overrides_sha256: Optional[str] = None,
    run_context: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Stamp the fingerprint beside the CSV. Never raises into the write path."""
    try:
        target = fingerprint_path(csv_path)
        payload = build_fingerprint(
            csv_path,
            settings,
            revenue_provenance,
            pricing_config_sha256=pricing_config_sha256,
            active_overrides_sha256=active_overrides_sha256,
            run_context=run_context,
        )
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
    except Exception:
        return None


def read_fingerprint(csv_path: str | Path) -> Optional[dict[str, Any]]:
    """The committed record, or None when it is absent or unreadable."""
    try:
        return json.loads(fingerprint_path(csv_path).read_text(encoding="utf-8"))
    except Exception:
        return None
