#!/usr/bin/env python3
"""Prove, or fail to prove, that the working tree is behaviourally identical to a commit.

A wave that is a pure refactor makes one claim: nothing changed. That claim is
cheap to assert and tedious to prove, which is exactly the shape of claim that
goes unchecked until something breaks in front of a customer. This harness makes
it provable in one command, and makes the failure to prove it visible rather
than silent.

It never mutates the shared tree or the index. The reference comes out of the
object database with `git archive`, the working tree is copied before anything
runs against it, and everything lands in a temporary directory that is removed
on the way out.

Exit codes are the point of the thing:
  0  every requested check ran and passed
  1  at least one check failed
  2  nothing failed, but at least one check could not run, so the proof is incomplete
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks_api import check_api_surface  # noqa: E402
from checks_bodies import check_response_bodies  # noqa: E402
from checks_engine import check_engine_golden, check_test_suite  # noqa: E402
from checks_files import check_moved_files  # noqa: E402
from checks_frontend import check_frontend_text  # noqa: E402
from materialise import dependency_sets_match, link_node_modules, materialise  # noqa: E402
from result import Result, exit_code, render  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ALL_CHECKS = ("api", "bodies", "engine", "moved", "suite", "frontend")
DECLARED_SURFACE = {"paths": 90, "operations": 113, "writes": 56}


def default_python() -> str:
    for candidate in (Path.home() / ".venvs" / "meridian" / "bin" / "python", Path(sys.executable)):
        if Path(candidate).exists():
            return str(candidate)
    return sys.executable


def _git(repo: Path, args: list[str]) -> str:
    p = subprocess.run(["git"] + args, cwd=repo, capture_output=True, text=True)
    return p.stdout.strip()


def tree_census(repo: Path) -> dict:
    """What the tree carries beyond its last commit, counted, never changed."""
    modified = [x for x in _git(repo, ["diff", "--name-only", "HEAD"]).splitlines() if x]
    untracked = [x for x in _git(repo, ["ls-files", "--others", "--exclude-standard"]).splitlines() if x]
    product = [x for x in modified + untracked if not x.startswith("docs/")]
    return {"head": _git(repo, ["rev-parse", "--short=8", "HEAD"]),
            "modified": len(modified), "untracked": len(untracked),
            "product_in_flight": len(product), "sample": sorted(product)[:5]}


def tree_state(repo: Path, census: dict, dirty_allowed: bool) -> str:
    """Describe what is being verified, without changing any of it."""
    line = ("HEAD %s, working tree carries %d modified and %d untracked file(s)"
            % (census["head"], census["modified"], census["untracked"]))
    if census["product_in_flight"] and dirty_allowed:
        line += ("\nMEASURED AGAINST A MOVING TARGET: %d product file(s) are uncommitted, so every "
                 "result below describes this instant and belongs to no wave."
                 % census["product_in_flight"])
    return line


def refuse_if_moving(census: dict, allow: bool) -> str | None:
    """A gate that measures a tree being edited underneath it reports a moment, not a wave.

    The working tree is the sum of every wave that touched it plus whatever is in
    flight this second. Measuring there is legitimate, but it has to be a decision
    somebody made, not something a reader is left to notice afterwards.
    """
    if not census["product_in_flight"] or allow:
        return None
    return ("refusing to run: %d product file(s) are uncommitted, so a wave is still in flight.\n"
            "  A result measured now describes this instant and belongs to no wave, which is the\n"
            "  one thing a wave-close gate must not report. First few: %s\n"
            "  Run against the committed tree instead, or pass --allow-dirty to proceed anyway and\n"
            "  have every result labelled as measured against a moving target."
            % (census["product_in_flight"], ", ".join(census["sample"])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", default="5a80a709", help="commit the tree must match")
    ap.add_argument("--only", default="", help="comma separated subset of: %s" % ", ".join(ALL_CHECKS))
    ap.add_argument("--python", default=default_python(), help="interpreter that has the dependencies")
    ap.add_argument("--suite-both", action="store_true",
                    help="run the suite on the reference too, instead of on the working tree alone")
    ap.add_argument("--bodies-timeout", type=int, default=1800)
    ap.add_argument("--route-deadline", type=float, default=30.0,
                    help="per-route seconds before a GET is recorded as unproven rather than hanging the run")
    ap.add_argument("--golden-timeout", type=int, default=900)
    ap.add_argument("--suite-timeout", type=int, default=3600)
    ap.add_argument("--build-timeout", type=int, default=900)
    ap.add_argument("--settle", type=float, default=30.0, help="seconds a route may take to stop changing before it is called unsettled")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="exit 0 when checks could not run; off by default so a gap cannot pass as a pass")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even though a wave is in flight; every result is labelled as measured "
                         "against a moving target")
    ap.add_argument("--self-check", action="store_true",
                    help="frontend only: compare the reference against itself across two cold "
                         "browser starts, to prove the check can reproduce its own baseline")
    ap.add_argument("--keep", action="store_true", help="leave the temporary trees in place")
    ap.add_argument("--json", default="", help="also write the full result to this path")
    args = ap.parse_args(argv)

    requested = [c.strip() for c in args.only.split(",") if c.strip()] or list(ALL_CHECKS)
    unknown = [c for c in requested if c not in ALL_CHECKS]
    if unknown:
        ap.error("unknown check(s): %s" % ", ".join(unknown))

    started = time.time()
    census = tree_census(REPO)
    refusal = refuse_if_moving(census, args.allow_dirty or args.self_check)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    needs_copy = bool({"api", "bodies", "engine", "suite", "frontend"} & set(requested))
    print("materialising %s and the working tree ..." % args.reference, file=sys.stderr)
    try:
        m = materialise(REPO, args.reference, args.keep, needs_copy)
    except (RuntimeError, OSError) as exc:
        # A gate that cannot set itself up says so in one line and fails; a traceback
        # here reads as a bug in the harness rather than an answer about the tree.
        print("could not verify: %s" % exc, file=sys.stderr)
        return 2
    results: list[Result] = []

    try:
        if needs_copy and "frontend" in requested:
            if dependency_sets_match(REPO, m.ref):
                link_node_modules(REPO, m.ref)
                link_node_modules(REPO, m.work)
            else:
                print("dependency sets differ; the frontend check will report it", file=sys.stderr)

        for name in requested:
            print("running check: %s ..." % name, file=sys.stderr)
            if name == "api":
                results.append(check_api_surface(args.python, m.ref, m.work, m.scratch, DECLARED_SURFACE))
            elif name == "bodies":
                results.append(check_response_bodies(args.python, m.ref, m.work, m.scratch, args.bodies_timeout, args.route_deadline))
            elif name == "engine":
                results.append(check_engine_golden(args.python, m.ref, m.work, m.scratch, args.golden_timeout))
            elif name == "moved":
                results.append(check_moved_files(REPO, m.ref, REPO))
            elif name == "suite":
                results.append(check_test_suite(args.python, m.ref, m.work, m.scratch,
                                                args.suite_timeout, args.suite_both))
            elif name == "frontend":
                results.append(check_frontend_text(args.python, m.ref, m.work, m.scratch, args.build_timeout,
                                                   args.settle, args.self_check))

        for skipped in [c for c in ALL_CHECKS if c not in requested]:
            r = Result(skipped, "%s (not requested)" % skipped, requested=False)
            results.append(r.cannot_check("not requested on this run"))

        note = tree_state(REPO, census, args.allow_dirty or args.self_check) + \
            "\nverified from a copy, so nothing here touched the shared tree"
        print(render(results, args.reference, note))
        if args.json:
            Path(args.json).write_text(json.dumps({
                "reference": args.reference,
                "tree": note,
                "seconds": round(time.time() - started, 1),
                "results": [r.as_dict() for r in results],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        if m.keep:
            print("\ntemporary trees kept at %s" % m.root)
        return exit_code(results, args.allow_unchecked)
    finally:
        m.cleanup()


if __name__ == "__main__":
    sys.exit(main())
