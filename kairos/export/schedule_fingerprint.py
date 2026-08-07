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

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

FINGERPRINT_SUFFIX = ".fingerprint.json"

# The settings that change what the optimizer produces. A change to any of these
# without a re-export means the saved plan is not this configuration's plan.
STAMPED_SETTINGS = ("revenue_weight", "min_retention_floor", "operator_channel", "risk_lambda")

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


def csv_sha256(path: str | Path) -> str:
    """The hash of the artifact exactly as it sits on disk, bytes in, no parsing."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(path.name + FINGERPRINT_SUFFIX)


def _settings_slice(settings: Any) -> dict[str, Any]:
    """The stamped settings, read from a mapping or an object, missing ones as None."""
    out: dict[str, Any] = {}
    for key in STAMPED_SETTINGS:
        if isinstance(settings, dict):
            out[key] = settings.get(key)
        else:
            out[key] = getattr(settings, key, None)
    return out


def build_fingerprint(csv_path: str | Path, settings: Any) -> dict[str, Any]:
    """The record to commit beside the CSV. Counts are read back from the file."""
    path = Path(csv_path)
    rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for index, _ in enumerate(handle):
            rows = index  # header is line 0, so the final index is the data row count
    return {
        "artifact": path.name,
        "sha256": csv_sha256(path),
        "rows": rows,
        "settings": _settings_slice(settings),
        "note": (
            "Committed on purpose. The freshness sidecar beside this file is gitignored "
            "and answers unknown on a fresh checkout, so it cannot guard the artifact. "
            "tests/test_plan_artifact_fingerprint.py compares this against the file and "
            "against data/kairos_settings.json."
        ),
    }


def write_fingerprint(csv_path: str | Path, settings: Any) -> Optional[Path]:
    """Stamp the fingerprint beside the CSV. Never raises into the write path."""
    try:
        target = fingerprint_path(csv_path)
        payload = build_fingerprint(csv_path, settings)
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
