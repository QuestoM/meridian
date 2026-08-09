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
    active_override_digest,
    PINNED_SETTINGS,
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


def test_the_plan_was_computed_under_the_overrides_that_are_active_now():
    """The other shared writable store, and the one that got away.

    data/manual_overrides.csv holds what an operator pins by hand, the optimizer
    honours every active row, and the same browser writes it that writes the
    settings. On 2026-08-01 one walk changed revenue_weight AND wrote a single
    gold mark into this file. The settings were restored and guarded; this file
    was not, so that row survived the restore and moved 131,878.70 ILS on
    2024-11-03 for another eight days with nothing noticing.

    The guard above learned that the file is the unit of risk rather than the
    field. This one is the next size up: the unit of risk is every shared
    writable store the plan is computed from, and there were two.
    """
    recorded = read_fingerprint(CSV)
    if recorded is None:
        pytest.fail("the fingerprint is unreadable, so the artifact is unguarded")
    stamped = recorded.get("active_overrides")
    if stamped is None:
        pytest.fail(
            "this fingerprint predates the override digest. Re-export with "
            "scripts/export_schedule.py; do not hand-edit the fingerprint."
        )
    live = active_override_digest(ROOT)
    assert stamped == live, (
        "the saved plan was computed under a different set of active overrides:\n"
        f"  plan has: {stamped}\n"
        f"  on disk:  {live}\n"
        "An override was added, retired or edited since the export. Re-run "
        "scripts/export_schedule.py. If you did not make that change, find out who did "
        "before re-exporting: this is exactly how a browser walk moved 131,878.70 ILS."
    )


def test_the_shipping_locale_is_not_a_test_leftover():
    """locale and direction do not change the plan, and are guarded anyway.

    This test exists because the guard above shipped without it and the very next
    commit walked through the hole: a critic switched the product to English
    left-to-right to measure something, never switched back, and it was committed.
    That would have shipped an Israeli Hebrew right-to-left product booting in
    English. Third pollution of this one file in a day, second by these two
    fields.

    The lesson is not "also check locale". It is that a guard scoped to whatever
    the author had in mind protects only that. The FILE is the unit of risk, not
    the field, because any agent that walks the UI writes the whole thing back.
    """
    live = _settings_on_disk()
    wrong = [
        (key, want, live.get(key)) for key, want in PINNED_SETTINGS.items() if live.get(key) != want
    ]
    assert not wrong, (
        "data/kairos_settings.json is a shared writable store and an agent left a test "
        "value in it:\n"
        + "\n".join(f"  {k}: expected {want!r}, found {got!r}" for k, want, got in wrong)
        + "\nThis is almost always a browser walk that measured in the other locale and "
        "did not restore. Restore it; do not change the expected value."
    )


def test_the_committed_artifact_is_the_one_on_disk():
    """The plan artifact has been silently rewritten THREE times in one session.

    Each time the shape was identical: 69 rows changed, breaks added, and the
    fingerprint re-stamped in the same call so the pair stayed consistent while
    the plan was wrong. Each time it was found hours later by an agent measuring
    something else, and each time the restore held only until the next run.

    THE CAUSE IS NOT CARELESSNESS, IT IS A DEFAULT. ``kairos/export/schedule.py``
    sets ``DEFAULT_OUTPUT_PATH`` to the shipped artifact itself, so anything that
    calls the exporter without naming a path overwrites the real plan. Every
    brief in this campaign says read-only on ``output/``, and agents obeyed it and
    polluted the file anyway, because they never knew they had written to it.

    So the guard is not another instruction. It is this test. The artifact is
    version-controlled, so the honest check is whether the bytes on disk are the
    bytes that were committed, and it fails within one suite run rather than
    within one afternoon.

    A REAL RECOMPUTE MAKES THIS FAIL, AND THAT IS CORRECT. Changing the plan is a
    deliberate act with money attached; it should be committed, not left dirty for
    the next reader to discover. Recompute, check the golden, commit, and this
    goes green with the new plan as the new baseline.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout, so there is no committed baseline to compare against")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "output/weekly_break_schedule.csv"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("the plan artifact is not tracked on this tree")
    dirty = subprocess.run(
        ["git", "diff", "--name-only", "--",
         "output/weekly_break_schedule.csv", "output/weekly_break_schedule.csv.fingerprint.json"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    changed = [line for line in dirty.stdout.splitlines() if line.strip()]
    assert not changed, (
        "the plan artifact on disk is not the one that was committed: "
        f"{changed}. Either something re-exported it without meaning to, which is "
        "what the exporter's default path makes easy, or a recompute was run and "
        "not committed. Run `git diff --stat output/` to see which. If it was "
        "unintended: `git checkout -- output/`. If it was deliberate: verify the "
        "golden and commit it, because a changed plan is a change to money. AND "
        "BEFORE YOU RESTORE, CHECK WHICH ONE IS STALE: on 2026-08-09 a fresh "
        "export reproduced the on-disk file byte for byte and the COMMITTED one "
        "was the out-of-date side, so restoring put an old plan back four times. "
        "Export to a scratch path and compare before deciding which way to go."
    )


def test_writing_the_shipped_plan_takes_a_deliberate_argument():
    """A default cannot be fixed by an instruction, and this one was not.

    ``write_weekly_schedule``'s path defaulted to the real, committed,
    version-controlled artifact the dashboard reads. So any caller that named no
    path replaced the operator's plan, and on 2026-08-09 that happened FOUR times
    in one session under briefs that every one of those callers obeyed. They were
    told read-only on ``output/``; none of them knew it had written.

    What the accidental writes did is worth knowing, because it is not what it
    looked like. Same row count, same break count, same total ad seconds to the
    second, and a redistribution of breaks inside a channel-day that left the plan
    claiming 17,966.31 LESS revenue. Not corruption: a re-optimisation landing on
    a marginally worse local optimum and silently replacing a committed one.

    So the default is a refusal now. Two callers legitimately mean the real plan,
    the recompute endpoint and the export script, and both say so.
    """
    import pytest as _pytest
    from kairos.export.schedule import DEFAULT_OUTPUT_PATH, write_weekly_schedule

    with _pytest.raises(ValueError) as raised:
        write_weekly_schedule(frame=None)
    message = str(raised.value)
    assert "shipped plan" in message
    assert "replace_shipped_plan=True" in message, (
        "the refusal must name the way through it, or the next caller guesses"
    )
    assert str(DEFAULT_OUTPUT_PATH) in message, (
        "the refusal must name the file it protected, or the reader has to go and find it"
    )


def test_the_two_callers_that_really_do_mean_the_shipped_plan_say_so():
    """Named explicitly, because the flag is worthless if it spreads by copying.

    Replacing the plan is the whole job of exactly two callers. Any third one is
    a decision somebody should have to make on purpose.
    """
    root = Path(__file__).resolve().parents[1]
    callers: dict[str, list[str]] = {}
    for folder in ("kairos", "kairos_api", "scripts"):
        for module in sorted((root / folder).rglob("*.py")):
            text = module.read_text(encoding="utf-8")
            if "write_weekly_schedule(" not in text or "def write_weekly_schedule" in text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "write_weekly_schedule(" in line and "import" not in line:
                    callers[f"{module.relative_to(root)}:{line_no}"] = [line.strip()]
    unflagged = {
        where: line for where, line in callers.items()
        if "replace_shipped_plan" not in line[0] and "path=" not in line[0]
    }
    assert not unflagged, (
        "a caller writes the schedule without naming a path and without saying it "
        f"means the shipped plan: {unflagged}. It will raise at runtime, which is "
        "the point, but it should be corrected here rather than discovered there."
    )
