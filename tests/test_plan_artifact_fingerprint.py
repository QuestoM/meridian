"""The guard on the plan of record.

output/weekly_break_schedule.csv carries real money and is what every money
surface reads. On 2026-08-07 it was silently overwritten twice with a stale copy
taken from a temp mirror, and both times the only thing that caught it was a
person hashing the file by hand before committing. That is not a guard, that is
a habit, and a habit is exactly what fails at four in the morning.

The freshness sidecar that already sits beside the CSV cannot do this job, for
one reason: output/*.meta.json is gitignored, so it never travels with the
artifact and answers "unknown" on any fresh checkout. A guard that answers
"unknown" is not a guard. This repository has met that same shape three times in
one day, so this one is committed and it fails loudly.

Two questions, both cheap, neither of which re-runs the engine:

  * Is the artifact still the bytes the exporter wrote?
  * Were those bytes produced under the settings that are on disk now?

Deliberately NOT here: a full re-derivation. That is the strongest check and it
takes minutes. tests/golden_weekly_schedule.py already does it. This one runs in
milliseconds on every suite, which is what makes it a guard rather than a
ceremony.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kairos.export.schedule_fingerprint import (
    STAMPED_SETTINGS,
    csv_sha256,
    fingerprint_path,
    read_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "output" / "weekly_break_schedule.csv"
SETTINGS = ROOT / "data" / "kairos_settings.json"


def _settings_on_disk() -> dict:
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def test_the_fingerprint_travels_with_the_artifact():
    """It must be committed, or it cannot guard anything on a fresh checkout."""
    assert CSV.exists(), "the plan of record is missing"
    stamp = fingerprint_path(CSV)
    assert stamp.exists(), (
        f"{stamp.name} is missing. It is committed on purpose: the freshness sidecar "
        "beside it is gitignored and reads unknown on a fresh checkout, so without this "
        "file nothing guards the artifact."
    )


def test_the_artifact_is_the_bytes_the_exporter_wrote():
    """A hash mismatch means something replaced this file without running the exporter.

    That is exactly the failure this guard exists for. If the plan legitimately
    changed, the exporter rewrites the fingerprint in the same call, so a
    mismatch always means the file moved some other way.
    """
    recorded = read_fingerprint(CSV)
    if recorded is None:
        pytest.fail("the fingerprint is unreadable, so the artifact is unguarded")
    actual = csv_sha256(CSV)
    assert actual == recorded.get("sha256"), (
        "output/weekly_break_schedule.csv does not match its committed fingerprint.\n"
        f"  on disk:   {actual}\n"
        f"  stamped:   {recorded.get('sha256')}\n"
        "Something wrote this file without running the exporter. Do NOT re-stamp the "
        "fingerprint to make this pass. Find out what wrote it. If the plan really did "
        "change, run scripts/export_schedule.py, which rewrites both together."
    )


def test_the_artifact_was_produced_under_the_settings_on_disk():
    """A settings mismatch means the plan of record is not this configuration's plan.

    This is the defect that moved 15,844,833 ILS and put the operator's own front
    page into a declared licence breach: a critic's browser walk changed
    revenue_weight and min_retention_floor, the change was committed, and the
    saved plan was never re-exported under it.
    """
    recorded = read_fingerprint(CSV)
    if recorded is None:
        pytest.fail("the fingerprint is unreadable, so the artifact is unguarded")
    stamped = recorded.get("settings") or {}
    live = _settings_on_disk()
    drifted = [
        (key, stamped.get(key), live.get(key))
        for key in STAMPED_SETTINGS
        if stamped.get(key) != live.get(key)
    ]
    assert not drifted, (
        "the saved plan was computed under settings that are no longer on disk:\n"
        + "\n".join(f"  {k}: plan has {was!r}, settings say {now!r}" for k, was, now in drifted)
        + "\nRe-export with scripts/export_schedule.py, or restore the settings. Do not "
        "edit the fingerprint."
    )
