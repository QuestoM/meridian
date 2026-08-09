"""Remove each rule, run the tests, count what fails, put the rule back.

A check whose failure mode is silence is not a check. This harness answers the
question a green suite cannot: would anything actually FAIL if the rule it
claims to protect were deleted. It deletes one guarantee at a time, runs the
tests, reports how many fail, and restores the file.

Not named ``test_*`` on purpose, exactly like :mod:`tests.lever_probe`: pytest
must never collect it. It MUTATES SOURCE FILES, and a harness that ran as part
of an ordinary suite run would mutate the tree under whatever else is running.

Run it directly:

    ~/.venvs/meridian/bin/python -u tests/bite_harness.py

THE FOUR SAFETY RULES, AND THE INCIDENT THAT PAID FOR EACH
==========================================================

Every one of these was written after the failure it prevents, on 2026-08-09,
during the Kai coverage wave. They are here rather than in a review thread
because a harness whose safety rules live in a chat log gets rebuilt without
them.

**1. Refuse to start unless the baseline is wholly green.**
A run killed by a timeout left one mutation on disk. The next run could not find
that anchor, reported it as a harmless SKIP, and produced fifteen plausible
mutation counts every one of which was measured against a tree that already had
a live defect in it. A run wrong in that direction reads exactly like a run that
is right. So the baseline is measured first and a non-green one aborts.

**2. An anchor that is not found is a LOUD FAILURE, never a skip.**
It means the file does not say what this harness thinks it says. Either a killed
run left a mutation behind, or the rule has quietly moved and the mutation is no
longer testing it. Both make the run untrustworthy, and both used to print the
word SKIP and be scrolled past.

**3. Restore on any signal, not only in a finally block.**
``finally`` does not survive SIGKILL and did not survive the timeout. Every file
is read into memory before the first mutation and written back from a signal
handler. Measured working: two instances killed with SIGTERM mid-run both left
the tree clean.

**4. Re-run the baseline AFTER restoring, and fail if the tree did not come
back.** Proving the tree recovered is the last thing the harness does, because
the one time it did not, the residue reached a commit. A fabricated money figure
(``"revenue_delta": -1.0``) sat on an operator-facing revenue preview on main
for about two hours, because a whole directory was staged while this harness was
mid-run.

**And run ONE instance at a time.** Two concurrent runs mutate the same files and
each restores from its own snapshot, so whichever finishes last wins and can
write back a file containing the other's mutation. This was nearly done once.

ADDING A MUTATION
=================
Append ``(label, path relative to the repo root, exact text to find, text to
replace it with)``. The find text must be unique in the file and must be the
rule itself rather than a comment about it. A mutation that scores 0 is a rule
NOTHING CHECKS, which is the finding this harness exists to produce; two of the
tests written in the wave that created it scored 0 on the first run, and both
were tests that fell silent rather than failing.
"""

from __future__ import annotations

import pathlib
import signal
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(pathlib.Path.home() / ".venvs" / "meridian" / "bin" / "python")

# The suites whose rules the mutations below belong to. Keep this narrow: the
# harness runs it once per mutation, so a broad suite makes the run cost minutes
# per rule and nobody runs it.
SUITE = [
    "tests/test_assistant_pacing_propose.py",
    "tests/test_assistant_proposal_freshness.py",
    "tests/test_assistant_coverage_reads.py",
]

# (label, relative path, find, replace)
MUTATIONS: list[tuple[str, str, str, str]] = [
    ("version store forgets the ledger's logical name",
     "kairos_api/version_store.py",
     '_KNOWN_LOGICAL = _LOGICAL_ORDER + ("campaigns", "plan_targets", "make_goods")',
     '_KNOWN_LOGICAL = _LOGICAL_ORDER + ("campaigns", "plan_targets")'),

    ("the applier drops the approving account and records nobody",
     "kairos_api/assistant_pacing_propose.py",
     'result = api._write_decision(kind, str(payload.get("campaign_id") or ""),\n'
     '                                 str(payload.get("note") or ""), actor)',
     'result = api._write_decision(kind, str(payload.get("campaign_id") or ""),\n'
     '                                 str(payload.get("note") or ""), "")'),

    ("a move records nobody as the offerer",
     "kairos_api/assistant_pacing_propose.py",
     'result = api._move_decision(str(payload.get("make_good_id") or ""), move, actor)',
     'result = api._move_decision(str(payload.get("make_good_id") or ""), move, "")'),

    ("the validator stops checking whether a shortfall is owed",
     "kairos_api/assistant_pacing_propose.py",
     '    if deficit is None:\n'
     '        raise ValueError(api.REFUSED_RAISE.get(why, (read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE))[1])',
     '    if deficit is None:\n'
     '        deficit = {"deficit_value": 0, "unit": "rating_points", "deficit_kind": "to_date"}'),

    ("the validator stops refusing an unknown campaign",
     "kairos_api/assistant_pacing_propose.py",
     '    if row is None:\n        raise ValueError(read.UNKNOWN_CAMPAIGN_HE)',
     '    if row is None:\n        raise ValueError("not found")'),

    ("the validator registers only when the router loads (the real bug)",
     "kairos_api/assistant_propose_tools.py",
     '_PROPOSE_VALIDATORS["propose_pacing_decision"] = validate_pacing_decision',
     'pass  # validator not registered at import'),

    ("a batch stops recording which plan it was reasoned against",
     "kairos_api/assistant_actions.py",
     '"conversation_id": conversation_id, "plan_stamp": plan_stamp(),',
     '"conversation_id": conversation_id,'),

    ("nothing to compare reads as current instead of unknown",
     "kairos_api/assistant_proposal_freshness.py",
     '        return {"state": UNKNOWN, "reason_en": UNKNOWN_EN, "reason_he": UNKNOWN_HE}',
     '        return {"state": CURRENT}'),

    ("a moved plan stops being reported stale",
     "kairos_api/assistant_proposal_freshness.py",
     '    if stamp.get("sha256") == now.get("sha256") and stamp.get("settings") == now.get("settings"):',
     '    if True:'),

    ("the restriction preview blends its two money bases into one total",
     "kairos_api/assistant_read_tools_restriction.py",
     '        "scored": body.get("scored"),',
     '        "revenue_delta": -1.0,\n        "scored": body.get("scored"),'),

    ("the restriction change list stops being capped",
     "kairos_api/assistant_read_tools_restriction.py",
     '        "changes": changes[:MAX_CHANGES],',
     '        "changes": changes,'),

    ("the roster is handed to an account that is not an administrator",
     "kairos_api/assistant_read_tools_accounts.py",
     '    if str(record.get("role") or "") not in {"admin"}:',
     '    if False:'),

    ("the licence limits stop being listed",
     "kairos_api/assistant_read_tools_accounts.py",
     '        "keys": list(guardrail_store.GUARDRAIL_KEYS),',
     '        "keys": [],'),

    ("the client mapper drops the freshness verdict (THE INERT LEVER, as found)",
     "tv-break-dashboard/src/kai/assistant-panel-state.js",
     "    items, restorePoints, planFreshness: freshness,",
     "    items, restorePoints,"),

    ("the card stops reading the verdict, so it can never show it",
     "tv-break-dashboard/src/kai/AssistantProposalCard.jsx",
     "  const freshness = batch && batch.planFreshness;",
     "  const freshness = null;"),

    ("the card writes its own copy of the server's sentence",
     "tv-break-dashboard/src/kai/AssistantProposalCard.jsx",
     "  return pageText(locale, freshness.reasonEn, freshness.reasonHe);",
     "  return pageText(locale, 'The plan changed.', 'התוכנית השבועית השתנתה.');"),

    ("pending proposals are merged into the ledger's own index",
     "kairos_api/pacing_alerts_api.py",
     '    payload["proposed"] = _proposed_index()',
     '    payload["proposed"] = payload["make_goods"]'),
]


def _restore_all(saved: dict[pathlib.Path, bytes]) -> None:
    for path, data in saved.items():
        path.write_bytes(data)


def _baseline() -> str:
    done = subprocess.run([PY, "-m", "pytest", *SUITE, "-q", "-p", "no:randomly"],
                          cwd=ROOT, capture_output=True, text=True)
    lines = done.stdout.strip().splitlines()
    return lines[-1] if lines else "no output"


def _is_green(line: str) -> bool:
    return "failed" not in line and "error" not in line.lower() and "passed" in line


def _failure_count(tail: str) -> int:
    """Failures from a pytest summary line, or -1 when it did not even collect."""
    tokens = tail.split()
    for index, token in enumerate(tokens[:-1]):
        if token.isdigit() and tokens[index + 1].startswith("failed"):
            return int(token)
    return -1 if "error" in tail.lower() else 0


def run() -> int:
    saved = {ROOT / rel: (ROOT / rel).read_bytes() for _, rel, _, _ in MUTATIONS}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_restore_all(saved), sys.exit(2)))
    try:
        return _run(saved)
    finally:
        _restore_all(saved)


def _run(saved: dict[pathlib.Path, bytes]) -> int:
    line = _baseline()
    print(f"BASELINE: {line}\n")
    if not _is_green(line):
        print("REFUSING TO RUN. The baseline is not green, so every count below would be "
              "measured against a tree that is already broken. A killed run may have left "
              "a mutation on disk; fix the baseline first. (Safety rule 1.)")
        return 1

    print(f"{'failures':>8}  mutation")
    print("-" * 78)
    unbitten: list[str] = []
    for label, rel, find, replace in MUTATIONS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        if find not in text:
            # Safety rule 2: never a skip.
            print(f"{'ANCHOR!':>8}  {label}  (anchor missing in {rel})")
            unbitten.append(f"{label} [ANCHOR MISSING in {rel}]")
            continue
        path.write_text(text.replace(find, replace, 1), encoding="utf-8")
        done = subprocess.run([PY, "-m", "pytest", *SUITE, "-q", "-p", "no:randomly", "--tb=no"],
                              cwd=ROOT, capture_output=True, text=True)
        tail = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "no output"
        path.write_bytes(saved[path])
        count = _failure_count(tail)
        print(f"{count:>8}  {label}")
        if count == 0:
            unbitten.append(label)

    print("-" * 78)
    _restore_all(saved)
    after = _baseline()
    print(f"BASELINE AFTER RESTORE: {after}")
    if not _is_green(after):
        # Safety rule 4. This is the one that, when it was absent, put a
        # fabricated money figure on main.
        print("THE TREE DID NOT COME BACK. A mutation is still on disk. Fix it before "
              "trusting anything above, and before staging anything.")
        return 1
    if unbitten:
        print("NOT BITTEN (a rule nothing checks):")
        for label in unbitten:
            print("   -", label)
        return 1
    print(f"every one of the {len(MUTATIONS)} mutations was caught")
    return 0


if __name__ == "__main__":
    sys.exit(run())
