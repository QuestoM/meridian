"""The engine numbers, and the test suite.

The golden weekly schedule is the sharpest engine check in the repository and
it is not collected by a plain pytest run, because it carries no test prefix.
It has to be invoked by name or it silently does not run, which is the worst
possible failure mode for a safety net, so it is invoked by name here.

The suite is run on both sides rather than against a remembered number. A
remembered pass count ages badly, and the whole point of this harness is to stop
trusting numbers whose provenance nobody re-checked.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from result import Result
from materialise import isolated_env

CSV_HASH = re.compile(r"full-CSV sha256:\s+([0-9a-f]{64})")
AGG_HASH = re.compile(r"aggregate sha256:\s+([0-9a-f]{64})")
MATCHES = re.compile(r"matches golden:\s+(True|False)")
GOLDEN_SCRIPT = Path("tests") / "golden_weekly_schedule.py"

COUNT = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)")
FAILURE_LINE = re.compile(r"^(FAILED|ERROR) (\S+)(.*)$")


def _golden(python: str, tree: Path, scratch: Path, timeout: int) -> tuple[dict, str]:
    script = tree / GOLDEN_SCRIPT
    if not script.is_file():
        return {}, "no %s in this tree" % GOLDEN_SCRIPT
    try:
        proc = subprocess.run([python, str(script)], cwd=str(tree), env=isolated_env(scratch),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, "timed out after %ds" % timeout
    out = proc.stdout + proc.stderr
    csv_hash = CSV_HASH.search(out)
    agg_hash = AGG_HASH.search(out)
    if not csv_hash:
        tail = out.strip().splitlines()[-6:]
        return {}, "no CSV hash in the output: %s" % " / ".join(tail)
    return {
        "csv_sha256": csv_hash.group(1),
        "aggregate_sha256": agg_hash.group(1) if agg_hash else None,
        "matches_own_golden": [m == "True" for m in MATCHES.findall(out)],
        "returncode": proc.returncode,
    }, ""


def check_engine_golden(python: str, ref: Path, work: Path, scratch: Path, timeout: int) -> Result:
    r = Result("engine", "Engine numbers, the golden weekly schedule")
    started = time.time()

    ref_out, ref_err = _golden(python, ref, scratch / "golden-ref", timeout)
    if ref_err:
        return r.cannot_check("reference side did not produce a hash: %s" % ref_err)
    work_out, work_err = _golden(python, work, scratch / "golden-work", timeout)
    if work_err:
        return r.cannot_check("working side did not produce a hash: %s" % work_err)
    r.seconds = time.time() - started

    r.note("reference CSV sha256: %s" % ref_out["csv_sha256"])
    r.note("working   CSV sha256: %s" % work_out["csv_sha256"])
    if ref_out.get("aggregate_sha256"):
        r.note("reference aggregate: %s" % ref_out["aggregate_sha256"])
        r.note("working   aggregate: %s" % work_out["aggregate_sha256"])
    for side, out in (("reference", ref_out), ("working", work_out)):
        if out["matches_own_golden"] and not all(out["matches_own_golden"]):
            r.note("%s does NOT reproduce its own committed golden" % side)

    same_csv = ref_out["csv_sha256"] == work_out["csv_sha256"]
    same_agg = ref_out.get("aggregate_sha256") == work_out.get("aggregate_sha256")
    measurements = {"reference_csv_sha256": ref_out["csv_sha256"],
                    "working_csv_sha256": work_out["csv_sha256"],
                    "reference_returncode": ref_out["returncode"],
                    "working_returncode": work_out["returncode"]}
    if ref_out["returncode"] != 0:
        return r.cannot_check(
            "the reference golden script exited %d, so it is not a valid comparison baseline"
            % ref_out["returncode"]
        )
    if work_out["returncode"] != 0:
        return r.failed(
            "the working golden script exited %d; its shipped artifact did not pass its own gate"
            % work_out["returncode"],
            **measurements,
        )
    if not ref_out["matches_own_golden"] or not all(ref_out["matches_own_golden"]):
        return r.cannot_check("the reference does not reproduce its own committed golden")
    if not work_out["matches_own_golden"] or not all(work_out["matches_own_golden"]):
        return r.failed(
            "the working tree does not reproduce its own committed golden",
            **measurements,
        )
    if same_csv and same_agg:
        return r.passed("byte identical on both sides", **measurements)
    return r.failed("the schedule moved: CSV hash %s, aggregate hash %s"
                    % ("same" if same_csv else "DIFFERENT", "same" if same_agg else "DIFFERENT"),
                    **measurements)


def _pytest(python: str, tree: Path, scratch: Path, timeout: int) -> tuple[dict, str]:
    # Scope: tests/ only. A bare pytest at the repo root also collects the
    # vendored Google Meridian library under meridian/, whose own suite carries
    # hundreds of failures unrelated to this product and identical on both
    # trees, which drowned the real signal and made the gate always fail. The
    # product's suite is tests/, which is what every count in this campaign
    # refers to.
    try:
        proc = subprocess.run([python, "-m", "pytest", "tests/", "-q", "--no-header",
                               "-p", "no:cacheprovider"],
                              cwd=str(tree), env=isolated_env(scratch),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, "timed out after %ds" % timeout
    out = proc.stdout + proc.stderr
    counts = {k: int(n) for n, k in COUNT.findall(out)}
    failures = []
    for line in out.splitlines():
        m = FAILURE_LINE.match(line.strip())
        if m:
            failures.append(line.strip())
    if not counts:
        tail = out.strip().splitlines()[-8:]
        return {}, "no result counts in the output: %s" % " / ".join(tail)
    return {"counts": counts, "failures": failures, "returncode": proc.returncode}, ""


def check_test_suite(python: str, ref: Path, work: Path, scratch: Path,
                     timeout: int, both: bool) -> Result:
    r = Result("suite", "The test suite")
    started = time.time()

    work_out, work_err = _pytest(python, work, scratch / "pytest-work", timeout)
    if work_err:
        return r.cannot_check("the working tree's suite did not report counts: %s" % work_err)

    ref_out: dict = {}
    if both:
        ref_out, ref_err = _pytest(python, ref, scratch / "pytest-ref", timeout)
        if ref_err:
            r.note("reference side did not report counts: %s" % ref_err)
            ref_out = {}
    r.seconds = time.time() - started

    wc = work_out["counts"]
    r.note("working tree: %s" % ", ".join("%d %s" % (v, k) for k, v in sorted(wc.items())))
    if ref_out:
        rc = ref_out["counts"]
        r.note("reference:    %s" % ", ".join("%d %s" % (v, k) for k, v in sorted(rc.items())))
        new = [f for f in work_out["failures"] if f not in set(ref_out["failures"])]
        pre_existing = [f for f in work_out["failures"] if f in set(ref_out["failures"])]
        if pre_existing:
            r.note("%d failure(s) also fail on the reference, so they are not this wave's doing" % len(pre_existing))
    else:
        new = work_out["failures"]
        r.note("no reference run to compare against, so every failure below is listed as new")

    for line in new[:40]:
        r.note("new failure: %s" % line)
    if len(new) > 40:
        r.note("... and %d more" % (len(new) - 40))

    measurements = {"working_counts": wc, "reference_counts": ref_out.get("counts", {}),
                    "new_failures": len(new)}
    broken = wc.get("failed", 0) + wc.get("error", 0) + wc.get("errors", 0)
    if new:
        return r.failed("%d failing test(s) that the reference does not have" % len(new), **measurements)
    if broken:
        return r.passed("%d failing, all of which fail on the reference too" % broken, **measurements)
    return r.passed("green: %d passed, %d skipped" % (wc.get("passed", 0), wc.get("skipped", 0)), **measurements)
